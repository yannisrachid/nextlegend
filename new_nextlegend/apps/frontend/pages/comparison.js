import { useEffect, useMemo, useState } from "react";
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { fetchJson, fetchJsonCached } from "@/lib/api";
import { METRIC_LABELS } from "@/lib/metricLabels";

const DEFAULT_RADAR_METRICS = [
  "goals_per_90",
  "xa_per_90",
  "accurate_passes_percent",
  "passes_to_penalty_area_per_90",
  "progressive_passes_per_90",
  "progressive_runs_per_90",
  "successful_dribbles_percent",
  "def_duels_won_percent",
  "interceptions_padj",
  "aerial_duels_won_percent",
];

const PLAYER_STYLES = [
  { label: "Player 1", color: "#22d3ee" },
  { label: "Player 2", color: "#f97316" },
  { label: "Player 3", color: "#34d399" },
];

const EXCLUDED_METRIC_PREFIXES = [
  "calendar",
  "page_number",
  "row_number",
  "player",
  "player_id",
  "birth_year",
  "age",
  "matches_played",
  "minutes_played",
  "team",
  "competition",
  "assigned_role",
];

const Card = ({ children, className = "" }) => (
  <div className={`glass-panel rounded-xl p-4 border border-white/5 ${className}`}>
    {children}
  </div>
);

const Label = ({ children }) => (
  <label className="text-xs uppercase tracking-[0.2em] text-slate-400">
    {children}
  </label>
);

const formatMetricLabel = (key) => {
  if (METRIC_LABELS[key]) return METRIC_LABELS[key];
  if (key.startsWith("summary_")) {
    const label = key
      .slice("summary_".length)
      .split("_")
      .map((chunk) => chunk.charAt(0).toUpperCase() + chunk.slice(1))
      .join(" ");
    return `Summary ${label}`;
  }
  return key.replace(/_/g, " ");
};

const normalizeMetricKey = (value) => {
  if (!value) return "";
  return String(value)
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]/gi, "")
    .toLowerCase();
};

const formatValue = (value, digits = 2) => {
  if (value === null || value === undefined || value === "") return "--";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return String(value);
  const abs = Math.abs(numeric);
  if (abs >= 100) return numeric.toFixed(0);
  return numeric.toFixed(digits);
};

const formatScore = (value) => {
  if (value === null || value === undefined || value === "") return "--";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "--";
  return numeric.toFixed(0);
};

const formatSummaryLabel = (key) => {
  const label = formatMetricLabel(key);
  return label.startsWith("Summary ") ? label.slice("Summary ".length) : label;
};

const sortRows = (rows, sortConfig) => {
  const sorted = [...rows];
  const { key, dir } = sortConfig;
  const getValue = (row) => {
    if (key === "metric") return row.metricLabel;
    if (key.startsWith("p")) {
      const idx = Number(key.slice(1));
      return row.values[idx];
    }
    return row.metricLabel;
  };
  sorted.sort((a, b) => {
    const left = getValue(a);
    const right = getValue(b);
    if (left == null && right == null) return 0;
    if (left == null) return 1;
    if (right == null) return -1;
    if (key === "metric") {
      const order = String(left).localeCompare(String(right), undefined, { sensitivity: "base" });
      return dir === "desc" ? -order : order;
    }
    const leftNum = Number(left);
    const rightNum = Number(right);
    if (Number.isFinite(leftNum) && Number.isFinite(rightNum)) {
      const order = leftNum - rightNum;
      return dir === "desc" ? -order : order;
    }
    const order = String(left).localeCompare(String(right), undefined, { sensitivity: "base" });
    return dir === "desc" ? -order : order;
  });
  return sorted;
};

const getRowMax = (values) => {
  const numeric = values
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value));
  if (numeric.length === 0) return null;
  return Math.max(...numeric);
};

const makeSortIndicator = (sortConfig, key) => {
  if (sortConfig.key !== key) return "";
  return sortConfig.dir === "asc" ? "^" : "v";
};

const seasonSortKey = (value) => {
  const text = String(value || "").trim();
  if (!text) return 0;
  const matchRange = text.match(/(20\d{2})\s*[/-]\s*((?:20)?\d{2,4})/);
  if (matchRange) {
    const start = Number(matchRange[1]);
    const rawEnd = matchRange[2];
    const end =
      rawEnd.length === 2
        ? Math.floor(start / 100) * 100 + Number(rawEnd)
        : Number(rawEnd.slice(-4));
    return end * 10000 + start;
  }
  const matchYear = text.match(/20\d{2}/);
  return matchYear ? Number(matchYear[0]) * 10000 + Number(matchYear[0]) : 0;
};

const sortSeasonsDesc = (items) =>
  Array.from(new Set((items || []).filter(Boolean))).sort((a, b) => {
    const diff = seasonSortKey(b) - seasonSortKey(a);
    if (diff !== 0) return diff;
    return String(b).localeCompare(String(a), undefined, { sensitivity: "base" });
  });

const selectRadarMetrics = (reports, maxCount = 10) => {
  const fallback = DEFAULT_RADAR_METRICS;
  if (!reports || reports.length === 0) return fallback;

  const roleCounts = {};
  const roleLists = new Map();
  reports.forEach((report) => {
    const role = report?.player?.assigned_role || "";
    if (role) {
      roleCounts[role] = (roleCounts[role] || 0) + 1;
    }
    if (Array.isArray(report?.radar_metrics) && report.radar_metrics.length > 0) {
      roleLists.set(role, report.radar_metrics);
    }
  });

  const preferredRole = Object.entries(roleCounts).find(([, count]) => count >= 2)?.[0];
  if (preferredRole && roleLists.has(preferredRole)) {
    return roleLists.get(preferredRole).slice(0, maxCount);
  }

  const lists = reports
    .map((report) => report?.radar_metrics || [])
    .filter((list) => list.length > 0)
    .map((list) => Array.from(new Set(list)));

  if (lists.length === 0) return fallback;

  const combined = [];
  let idx = 0;
  let added = true;
  while (combined.length < maxCount && added) {
    added = false;
    lists.forEach((list) => {
      if (combined.length >= maxCount) return;
      const key = list[idx];
      if (key && !combined.includes(key)) {
        combined.push(key);
        added = true;
      }
    });
    idx += 1;
  }

  if (combined.length < maxCount) {
    fallback.forEach((key) => {
      if (combined.length >= maxCount) return;
      if (!combined.includes(key)) combined.push(key);
    });
  }

  return combined.slice(0, maxCount);
};

const usePlayerSearch = (onSelect, seasonFilter) => {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [showResults, setShowResults] = useState(false);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    if (!query || query.trim().length < 2) {
      setResults([]);
      return;
    }
    const handle = setTimeout(async () => {
      try {
        const res = await fetchJson("/players", {
          q: query.trim(),
          season: seasonFilter || undefined,
        });
        const unique = new Map();
        (res || []).forEach((item) => {
          const normalize = (value) =>
            String(value || "")
              .trim()
              .toLowerCase();
          const key = [
            normalize(item.name),
            normalize(item.team),
            normalize(item.competition_name),
            normalize(item.calendar),
          ].join("|");
          if (!unique.has(key)) {
            unique.set(key, item);
          }
        });
        setResults(Array.from(unique.values()));
      } catch (err) {
        console.error(err);
      }
    }, 200);
    return () => clearTimeout(handle);
  }, [query, seasonFilter]);

  const options = useMemo(() => {
    return results.map((player) => ({
      id: String(player.id),
      seasonId: player.player_season_id ? String(player.player_season_id) : "",
      label: `${player.name} - ${player.team || "--"} - ${player.competition_name || "--"} - ${player.calendar || "--"}`,
    }));
  }, [results]);

  const handleSelect = (player) => {
    setSelected(player);
    setQuery(player.label);
    setShowResults(false);
    onSelect(player);
  };

  const clearSelection = () => {
    setSelected(null);
    onSelect(null);
  };

  return {
    query,
    setQuery,
    showResults,
    setShowResults,
    options,
    selected,
    handleSelect,
    clearSelection,
  };
};

const ComparisonRadarTooltip = ({ active, payload, players, contextLabel }) => {
  if (!active || !payload || payload.length === 0) return null;
  const point = payload[0]?.payload;
  if (!point) return null;
  return (
    <div className="rounded-md border border-slate-700 bg-slate-900/95 px-3 py-2 text-xs text-slate-100 shadow-xl">
      <div className="font-semibold text-white">{point.metric}</div>
      <div className="text-slate-400">{contextLabel}</div>
      <div className="mt-2 space-y-1">
        {players.map((player, idx) => {
          const valueKey = `p${idx}`;
          const rawKey = `p${idx}_raw`;
          return (
            <div key={player.report.player.player_id} className="flex items-center justify-between gap-3">
              <span className="flex items-center gap-2">
                <span
                  className="h-2 w-2 rounded-full"
                  style={{ backgroundColor: player.color }}
                />
                <span className="text-slate-200">{player.report.player.name}</span>
              </span>
              <span className="text-slate-300">
                {Number(point[valueKey] ?? 0).toFixed(0)} / {formatValue(point[rawKey])}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default function ComparisonPage() {
  const [error, setError] = useState("");
  const [compareLoading, setCompareLoading] = useState(false);
  const [comparison, setComparison] = useState([]);
  const [seasons, setSeasons] = useState([]);
  const [selectedSeason, setSelectedSeason] = useState("");
  const [radarContext, setRadarContext] = useState("global");
  const [showAllMetrics, setShowAllMetrics] = useState(false);
  const [radarSort, setRadarSort] = useState({ key: "metric", dir: "asc" });
  const [allMetricsSort, setAllMetricsSort] = useState({ key: "metric", dir: "asc" });
  const [summarySort, setSummarySort] = useState({ key: "metric", dir: "asc" });

  const [selectedPlayers, setSelectedPlayers] = useState([null, null, null]);

  useEffect(() => {
    fetchJsonCached("/meta/seasons")
      .then((data) => setSeasons(sortSeasonsDesc(data || [])))
      .catch(() => setSeasons([]));
  }, []);

  const search1 = usePlayerSearch((player) => {
    setSelectedPlayers((prev) => [player, prev[1], prev[2]]);
  }, selectedSeason);
  const search2 = usePlayerSearch((player) => {
    setSelectedPlayers((prev) => [prev[0], player, prev[2]]);
  }, selectedSeason);
  const search3 = usePlayerSearch((player) => {
    setSelectedPlayers((prev) => [prev[0], prev[1], player]);
  }, selectedSeason);

  const activeSelections = useMemo(() => {
    return selectedPlayers
      .map((player, index) => (player ? { ...player, slot: index } : null))
      .filter(Boolean);
  }, [selectedPlayers]);

  const handleCompare = async () => {
    setError("");
    if (activeSelections.length < 2) {
      setError("Select at least two players to compare.");
      return;
    }
    setCompareLoading(true);
    try {
      const reports = await Promise.all(
        activeSelections.map((player) =>
          fetchJson(`/players/${player.id}/report`, {
            player_season_id: player.seasonId || undefined,
          })
        )
      );
      const merged = activeSelections.map((player, index) => ({
        ...player,
        report: reports[index],
        color: PLAYER_STYLES[player.slot]?.color || "#22d3ee",
      }));
      setComparison(merged);
    } catch (err) {
      setError(err.message || "Failed to load comparison");
    } finally {
      setCompareLoading(false);
    }
  };

  const updateSort = (setSort, key) => {
    setSort((prev) => {
      if (prev.key === key) {
        return { key, dir: prev.dir === "asc" ? "desc" : "asc" };
      }
      return { key, dir: "asc" };
    });
  };

  const radarMetricKeys = useMemo(
    () => selectRadarMetrics(comparison.map((item) => item.report)),
    [comparison]
  );

  const radarData = useMemo(() => {
    if (comparison.length === 0) return [];
    return radarMetricKeys.map((key) => {
      const entry = {
        metric: formatMetricLabel(key),
      };
      comparison.forEach((item, idx) => {
        const metrics = item.report?.metrics || {};
        const leagueKey = `${key}_pct_league`;
        const globalKey = `${key}_pct_global`;
        const leagueValue = metrics[leagueKey];
        const globalValue = metrics[globalKey];
        const display =
          radarContext === "league"
            ? Number(leagueValue ?? globalValue ?? 0)
            : Number(globalValue ?? leagueValue ?? 0);
        entry[`p${idx}`] = Number(display) || 0;
        entry[`p${idx}_raw`] = metrics[key];
      });
      return entry;
    });
  }, [comparison, radarMetricKeys, radarContext]);

  const radarComparisonMetrics = useMemo(() => {
    return radarMetricKeys;
  }, [radarMetricKeys]);

  const roleProfileKeys = useMemo(() => {
    const keys = new Set();
    comparison.forEach((item) => {
      const profiles = item.report?.role_scores || [];
      profiles.forEach((profile) => {
        const normalized = normalizeMetricKey(profile.profile);
        if (normalized) {
          keys.add(normalized);
        }
      });
    });
    return keys;
  }, [comparison]);

  const allMetricsKeys = useMemo(() => {
    if (comparison.length === 0) return [];
    const keys = new Set();
    comparison.forEach((item) => {
      const metrics = item.report?.metrics || {};
      Object.keys(metrics).forEach((key) => {
        if (key.startsWith("summary_")) return;
        if (key.endsWith("_pct_league") || key.endsWith("_pct_global")) return;
        if (EXCLUDED_METRIC_PREFIXES.some((prefix) => key.startsWith(prefix))) return;
        if (key.startsWith("tm_")) return;
        if (key.toLowerCase().includes("profile")) return;
        if (key.includes(" - ")) return;
        if (roleProfileKeys.has(normalizeMetricKey(key))) return;
        keys.add(key);
      });
    });
    return Array.from(keys).sort((a, b) =>
      formatMetricLabel(a).localeCompare(formatMetricLabel(b), undefined, { sensitivity: "base" })
    );
  }, [comparison, roleProfileKeys]);

  const summaryMetrics = useMemo(() => {
    if (comparison.length === 0) return [];
    const keys = new Set();
    comparison.forEach((item) => {
      const metrics = item.report?.metrics || {};
      Object.keys(metrics).forEach((key) => {
        if (!key.startsWith("summary_")) return;
        if (key.includes("_pct_")) return;
        keys.add(key);
      });
    });
    return Array.from(keys).sort((a, b) =>
      formatMetricLabel(a).localeCompare(formatMetricLabel(b), undefined, { sensitivity: "base" })
    );
  }, [comparison]);

  const radarRows = useMemo(() => {
    return radarComparisonMetrics.map((key) => ({
      metricKey: key,
      metricLabel: formatMetricLabel(key),
      values: comparison.map((item) => item.report?.metrics?.[key]),
    }));
  }, [radarComparisonMetrics, comparison]);

  const allMetricsRows = useMemo(() => {
    return allMetricsKeys.map((key) => ({
      metricKey: key,
      metricLabel: formatMetricLabel(key),
      values: comparison.map((item) => item.report?.metrics?.[key]),
    }));
  }, [allMetricsKeys, comparison]);

  const summaryRows = useMemo(() => {
    return summaryMetrics.map((key) => ({
      metricKey: key,
      metricLabel: formatSummaryLabel(key),
      values: comparison.map((item) => item.report?.metrics?.[key]),
    }));
  }, [summaryMetrics, comparison]);

  const sortedRadarRows = useMemo(
    () => sortRows(radarRows, radarSort),
    [radarRows, radarSort]
  );
  const sortedAllMetricsRows = useMemo(
    () => sortRows(allMetricsRows, allMetricsSort),
    [allMetricsRows, allMetricsSort]
  );
  const sortedSummaryRows = useMemo(
    () => sortRows(summaryRows, summarySort),
    [summaryRows, summarySort]
  );

  const comparisonContextLabel =
    radarContext === "league" ? "League percentile" : "Global percentile";

  return (
    <main className="nl-page py-10 px-4">
      <div className="max-w-6xl mx-auto space-y-6">
        <header className="flex flex-col gap-2">
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
            Comparison
          </p>
          <h1 className="text-4xl font-bold text-white tracking-tight">
            Player comparison board
          </h1>
          <p className="text-slate-300 max-w-3xl">
            Compare two or three profiles with shared radars, role context and a complete metric table for recruitment discussions.
          </p>
        </header>

        <Card className="relative z-30">
          <div className="mb-4 max-w-xs">
            <div className="flex flex-col gap-2">
              <Label>Season Filter</Label>
              <select
                className="bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100"
                value={selectedSeason}
                onChange={(e) => {
                  setSelectedSeason(e.target.value);
                  setSelectedPlayers([null, null, null]);
                  [search1, search2, search3].forEach((search) => {
                    search.setQuery("");
                    search.clearSelection();
                  });
                  setComparison([]);
                }}
              >
                <option value="">All seasons</option>
                {seasons.map((season) => (
                  <option key={season} value={season}>
                    {season}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {[search1, search2, search3].map((search, index) => (
              <div key={`search-${index}`} className="relative">
                <div className="flex flex-col gap-2">
                  <Label>{PLAYER_STYLES[index].label}</Label>
                  <input
                    className="bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100"
                    placeholder="Start typing a player name..."
                    value={search.query}
                    onChange={(e) => {
                      search.setQuery(e.target.value);
                      if (search.selected) search.clearSelection();
                      search.setShowResults(true);
                    }}
                    onFocus={() => {
                      if (search.selected) {
                        search.setQuery("");
                        search.clearSelection();
                      }
                      search.setShowResults(true);
                    }}
                    onClick={() => {
                      if (search.selected) {
                        search.setQuery("");
                        search.clearSelection();
                        search.setShowResults(true);
                      }
                    }}
                    onBlur={() => setTimeout(() => search.setShowResults(false), 150)}
                  />
                </div>
                {search.showResults && search.query.trim().length >= 2 ? (
                  <div className="absolute z-50 mt-2 w-full max-h-72 overflow-auto rounded-lg border border-slate-700 bg-slate-900/95 shadow-xl">
                    {search.options.length === 0 ? (
                      <div className="px-3 py-2 text-sm text-slate-400">
                        No matches found.
                      </div>
                    ) : (
                      search.options.map((player) => (
                        <button
                          key={`${player.id}-${player.seasonId || "latest"}`}
                          type="button"
                          className="w-full text-left px-3 py-2 text-sm text-slate-200 hover:bg-slate-800/80"
                          onMouseDown={(e) => e.preventDefault()}
                          onClick={() => search.handleSelect(player)}
                        >
                          {player.label}
                        </button>
                      ))
                    )}
                  </div>
                ) : null}
              </div>
            ))}
          </div>
          <div className="flex items-center justify-between flex-wrap gap-3 mt-4">
            <p className="text-sm text-slate-400">
              Select at least two players, then compare.
            </p>
            <button
              type="button"
              onClick={handleCompare}
              disabled={activeSelections.length < 2 || compareLoading}
              className={`px-5 py-2 rounded-md text-sm font-semibold uppercase tracking-[0.2em] border transition ${
                activeSelections.length < 2 || compareLoading
                  ? "border-slate-700 text-slate-500 cursor-not-allowed"
                  : "border-emerald-400/60 text-emerald-200 hover:bg-emerald-400/10"
              }`}
            >
              {compareLoading ? "Comparing..." : "Compare"}
            </button>
          </div>
        </Card>

        {error && (
          <Card>
            <p className="text-danger">Error: {error}</p>
          </Card>
        )}

        {comparison.length === 0 ? (
          <Card>
            <p className="text-slate-400">
              Select two or three players to build a side-by-side recruitment view.
            </p>
          </Card>
        ) : (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <Card className="lg:col-span-1 space-y-4">
                <h3 className="text-lg font-semibold text-white">Players</h3>
                {comparison.map((item) => (
                  <div key={item.report.player.player_id} className="border border-white/10 rounded-lg p-3">
                    <div className="flex items-center justify-between">
                      <p className="text-sm text-slate-400">Name</p>
                      <p className="text-sm font-semibold" style={{ color: item.color }}>
                        {item.report.player.name}
                      </p>
                    </div>
                    <div className="mt-2 space-y-1 text-sm text-slate-300">
                      <div className="flex items-center justify-between">
                        <span>Club</span>
                        <span>{item.report.player.team || "--"}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Position</span>
                        <span>{item.report.player.position || "--"}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Age</span>
                        <span>{formatScore(item.report.player.age)}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Minutes</span>
                        <span>{formatScore(item.report.player.minutes_played)}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>League percentile</span>
                        <span>{formatScore(item.report.player.assigned_role_pct_league)}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Score v2</span>
                        <span>{formatScore(item.report.player.assigned_role_pct_global)}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </Card>

              <Card className="lg:col-span-2">
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <h3 className="text-lg font-semibold text-white">
                    Percentile Radar
                  </h3>
                  <div className="flex items-center gap-2">
                    {[
                      { key: "global", label: "Global" },
                      { key: "league", label: "League" },
                    ].map((option) => (
                      <button
                        key={option.key}
                        type="button"
                        onClick={() => setRadarContext(option.key)}
                        className={`px-3 py-1 rounded-md text-xs uppercase tracking-[0.2em] border ${
                          radarContext === option.key
                            ? "border-emerald-400/70 bg-emerald-400/20 text-emerald-200"
                            : "border-slate-700 bg-slate-900/60 text-slate-300"
                        }`}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="h-96">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={radarData} outerRadius="88%">
                      <PolarGrid stroke="#334155" strokeDasharray="3 3" />
                      <PolarAngleAxis dataKey="metric" tick={{ fill: "#94a3b8", fontSize: 10 }} />
                      <PolarRadiusAxis
                        angle={90}
                        domain={[0, 100]}
                        ticks={[0, 25, 50, 75, 100]}
                        tick={{ fill: "#94a3b8", fontSize: 8 }}
                        tickLine={{ stroke: "#475569", strokeOpacity: 0.6 }}
                        axisLine={{ stroke: "#475569", strokeOpacity: 0.6 }}
                      />
                      <Tooltip
                        content={
                          <ComparisonRadarTooltip
                            players={comparison}
                            contextLabel={comparisonContextLabel}
                          />
                        }
                        cursor={false}
                      />
                      {comparison.map((item, idx) => (
                        <Radar
                          key={item.report.player.player_id}
                          name={item.report.player.name}
                          dataKey={`p${idx}`}
                          stroke={item.color}
                          fill="transparent"
                          fillOpacity={0}
                          dot={{ r: 4, stroke: item.color, strokeWidth: 1 }}
                          activeDot={{ r: 6 }}
                        />
                      ))}
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </div>

            <Card>
              <h3 className="text-lg font-semibold text-white mb-3">
                Metrics Comparison
              </h3>
              <div className="max-h-[520px] overflow-auto pr-2">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-slate-900/95">
                    <tr className="text-xs uppercase text-slate-400 border-b border-white/5">
                      <th
                        className="text-left py-2 cursor-pointer select-none"
                        onClick={() => updateSort(setRadarSort, "metric")}
                      >
                        Metric {makeSortIndicator(radarSort, "metric")}
                      </th>
                      {comparison.map((item, index) => (
                        <th
                          key={item.report.player.player_id}
                          className="text-left py-2 cursor-pointer select-none"
                          style={{ color: item.color }}
                          onClick={() => updateSort(setRadarSort, `p${index}`)}
                        >
                          {item.report.player.name} {makeSortIndicator(radarSort, `p${index}`)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {sortedRadarRows.map((row) => {
                      const rowMax = getRowMax(row.values);
                      return (
                        <tr key={row.metricKey} className="border-b border-white/5">
                          <td className="py-2 text-slate-200">{row.metricLabel}</td>
                          {comparison.map((item, index) => {
                            const value = row.values[index];
                            const numeric = Number(value);
                            const isBest = rowMax !== null && Number.isFinite(numeric) && numeric === rowMax;
                            return (
                              <td
                                key={`${item.report.player.player_id}-${row.metricKey}`}
                                className={`py-2 ${isBest ? "text-emerald-300 font-semibold" : "text-slate-300"}`}
                              >
                                {formatValue(value)}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Card>

            <Card>
              <div className="flex items-center justify-between gap-2 mb-3">
                <h3 className="text-lg font-semibold text-white">
                  All Metrics
                </h3>
                <button
                  type="button"
                  onClick={() => setShowAllMetrics((prev) => !prev)}
                  className="px-3 py-1 rounded-md text-xs uppercase tracking-[0.2em] border border-slate-700 text-slate-200 hover:border-slate-500"
                >
                  {showAllMetrics ? "Hide" : "Show"}
                </button>
              </div>
              {showAllMetrics ? (
                <div className="max-h-[420px] overflow-auto pr-4">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-slate-900/95">
                      <tr className="text-xs uppercase text-slate-400 border-b border-white/5">
                        <th
                          className="text-left py-2 cursor-pointer select-none"
                          onClick={() => updateSort(setAllMetricsSort, "metric")}
                        >
                          Metric {makeSortIndicator(allMetricsSort, "metric")}
                        </th>
                        {comparison.map((item, index) => (
                          <th
                            key={item.report.player.player_id}
                            className="text-left py-2 cursor-pointer select-none"
                            style={{ color: item.color }}
                            onClick={() => updateSort(setAllMetricsSort, `p${index}`)}
                          >
                            {item.report.player.name}{" "}
                            {makeSortIndicator(allMetricsSort, `p${index}`)}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {sortedAllMetricsRows.map((row) => {
                        const rowMax = getRowMax(row.values);
                        return (
                          <tr key={row.metricKey} className="border-b border-white/5">
                            <td className="py-2 text-slate-200">{row.metricLabel}</td>
                            {comparison.map((item, index) => {
                              const value = row.values[index];
                              const numeric = Number(value);
                              const isBest =
                                rowMax !== null && Number.isFinite(numeric) && numeric === rowMax;
                              return (
                                <td
                                  key={`${item.report.player.player_id}-${row.metricKey}`}
                                  className={`py-2 ${isBest ? "text-emerald-300 font-semibold" : "text-slate-300"}`}
                                >
                                  {formatValue(value)}
                                </td>
                              );
                            })}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-sm text-slate-400">
                  Expand to browse all metrics with the same colors and layout.
                </p>
              )}
            </Card>

            <Card>
              <h3 className="text-lg font-semibold text-white mb-3">
                Summary Scores
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-xs uppercase text-slate-400 border-b border-white/5">
                      <th
                        className="text-left py-2 cursor-pointer select-none"
                        onClick={() => updateSort(setSummarySort, "metric")}
                      >
                        Metric {makeSortIndicator(summarySort, "metric")}
                      </th>
                      {comparison.map((item, index) => (
                        <th
                          key={item.report.player.player_id}
                          className="text-left py-2 cursor-pointer select-none"
                          style={{ color: item.color }}
                          onClick={() => updateSort(setSummarySort, `p${index}`)}
                        >
                          {item.report.player.name}{" "}
                          {makeSortIndicator(summarySort, `p${index}`)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {sortedSummaryRows.map((row) => {
                      const rowMax = getRowMax(row.values);
                      return (
                        <tr key={row.metricKey} className="border-b border-white/5">
                          <td className="py-2 text-slate-200">{row.metricLabel}</td>
                          {comparison.map((item, index) => {
                            const value = row.values[index];
                            const numeric = Number(value);
                            const isBest =
                              rowMax !== null && Number.isFinite(numeric) && numeric === rowMax;
                            return (
                              <td
                                key={`${item.report.player.player_id}-${row.metricKey}`}
                                className={`py-2 ${isBest ? "text-emerald-300 font-semibold" : "text-slate-300"}`}
                              >
                                {formatValue(value)}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Card>
          </>
        )}
      </div>
    </main>
  );
}
