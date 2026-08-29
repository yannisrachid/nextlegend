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
import {
  DEFAULT_SCOUTING_SEASON,
  withDefaultSeason,
} from "@/lib/scoutingFilters";

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

const Select = ({ value, onChange, children }) => (
  <select
    className="bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100"
    value={value}
    onChange={onChange}
  >
    {children}
  </select>
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

const formatSummaryLabel = (key) => {
  const label = formatMetricLabel(key);
  return label.startsWith("Summary ") ? label.slice("Summary ".length) : label;
};

const formatValue = (value, digits = 2) => {
  if (value === null || value === undefined || value === "") return "--";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return String(value);
  const abs = Math.abs(numeric);
  if (abs >= 100) return numeric.toFixed(0);
  return numeric.toFixed(digits);
};

const clampPercentile = (value) => {
  if (!Number.isFinite(value)) return 0;
  return Math.min(100, Math.max(0, value));
};

const normalizeMetric = (value, maxValue) => {
  if (!Number.isFinite(value) || !Number.isFinite(maxValue) || maxValue <= 0) return 0;
  return (value / maxValue) * 100;
};

const ProjectionRadarTooltip = ({ active, payload, label, isPercentile, valueKey, rawKey }) => {
  if (!active || !payload || payload.length === 0) return null;
  const point = payload[0]?.payload;
  if (!point) return null;
  return (
    <div className="rounded-md border border-slate-700 bg-slate-900/95 px-3 py-2 text-xs text-slate-100 shadow-xl">
      <div className="font-semibold text-white">{point.metric}</div>
      <div className="text-slate-400">{label}</div>
      <div className="text-slate-300 mt-2">
        {isPercentile ? (
          <>Percentile: {Number(point[valueKey] ?? 0).toFixed(0)}</>
        ) : (
          <>Value: {formatValue(point[rawKey])}</>
        )}
      </div>
      {!isPercentile ? (
        <div className="text-slate-500 mt-1">
          Radar scaled per metric.
        </div>
      ) : null}
    </div>
  );
};

export default function ProjectionPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [playerQuery, setPlayerQuery] = useState("");
  const [playerResults, setPlayerResults] = useState([]);
  const [selectedPlayerId, setSelectedPlayerId] = useState("");
  const [selectedPlayerSeasonId, setSelectedPlayerSeasonId] = useState("");
  const [seasons, setSeasons] = useState([]);
  const [selectedSeason, setSelectedSeason] = useState(DEFAULT_SCOUTING_SEASON);
  const [showResults, setShowResults] = useState(false);
  const [report, setReport] = useState(null);
  const [translationLeagues, setTranslationLeagues] = useState([]);
  const [targetLeague, setTargetLeague] = useState("");
  const [translationCoeff, setTranslationCoeff] = useState({ overall_coeff: 1.0 });
  const [percentileView, setPercentileView] = useState(true);

  useEffect(() => {
    fetchJsonCached("/meta/league-translation/leagues")
      .then((data) => setTranslationLeagues(data || []))
      .catch((err) => console.error(err));
  }, []);

  useEffect(() => {
    fetchJsonCached("/meta/seasons")
      .then((data) => setSeasons(withDefaultSeason(data || [])))
      .catch(() => setSeasons([]));
  }, []);

  useEffect(() => {
    if (!playerQuery || playerQuery.trim().length < 2) {
      setPlayerResults([]);
      setSelectedPlayerId("");
      setSelectedPlayerSeasonId("");
      return;
    }
    const handle = setTimeout(async () => {
      try {
        const res = await fetchJson("/players", {
          q: playerQuery.trim(),
          season: selectedSeason || undefined,
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
        setPlayerResults(Array.from(unique.values()));
      } catch (err) {
        console.error(err);
      }
    }, 200);
    return () => clearTimeout(handle);
  }, [playerQuery, selectedSeason]);

  const playerOptions = useMemo(() => {
    return playerResults.map((player) => ({
      id: String(player.id),
      seasonId: player.player_season_id ? String(player.player_season_id) : "",
      label: `${player.name} - ${player.team || "--"} - ${player.competition_name || "--"} - ${player.calendar || "--"}`,
    }));
  }, [playerResults]);

  const handlePlayerSelect = (player) => {
    setSelectedPlayerId(player.id);
    setSelectedPlayerSeasonId(player.seasonId || "");
    setPlayerQuery(player.label);
    setShowResults(false);
  };

  useEffect(() => {
    if (!selectedPlayerId) {
      setReport(null);
      return;
    }
    const loadReport = async () => {
      setLoading(true);
      setError("");
      try {
        const data = await fetchJson(`/players/${selectedPlayerId}/report`, {
          player_season_id: selectedPlayerSeasonId || undefined,
        });
        setReport(data);
      } catch (err) {
        setError(err.message || "Failed to load report");
      } finally {
        setLoading(false);
      }
    };
    loadReport();
  }, [selectedPlayerId, selectedPlayerSeasonId]);

  useEffect(() => {
    if (!report || translationLeagues.length === 0) return;
    const currentLeague = report.player?.competition_name;
    if (!currentLeague) return;
    if (!translationLeagues.includes(currentLeague)) return;
    const options = translationLeagues.filter((league) => league !== currentLeague);
    if (options.length === 0) return;
    setTargetLeague((prev) => (prev && prev !== currentLeague ? prev : options[0]));
  }, [report, translationLeagues]);

  useEffect(() => {
    const sourceLeague = report?.player?.competition_name;
    if (!sourceLeague || !targetLeague) return;
    fetchJson("/meta/league-translation", { source: sourceLeague, target: targetLeague })
      .then((data) => setTranslationCoeff(data || { overall_coeff: 1.0 }))
      .catch((err) => {
        console.error(err);
        setTranslationCoeff({ overall_coeff: 1.0 });
      });
  }, [report, targetLeague]);

  const metrics = report?.metrics || {};
  const radarMetricKeys = Array.isArray(report?.radar_metrics) && report.radar_metrics.length > 0
    ? report.radar_metrics
    : DEFAULT_RADAR_METRICS;
  const coeff = Number(translationCoeff?.overall_coeff ?? 1.0) || 1.0;
  const currentScore = report?.player?.global_score_adjusted;
  const projectedScore =
    currentScore != null && Number.isFinite(Number(currentScore))
      ? Number(currentScore) * coeff
      : null;

  const radarData = useMemo(() => {
    return radarMetricKeys.map((key) => {
      const rawValue = metrics[key];
      const leaguePct = metrics[`${key}_pct_league`];
      const globalPct = metrics[`${key}_pct_global`];
      const basePercentile = leaguePct ?? globalPct ?? 0;
      const projectedRaw = rawValue != null ? Number(rawValue) * coeff : null;
      const projectedPercentile = clampPercentile(Number(basePercentile) * coeff);
      const currentDisplay = percentileView ? Number(basePercentile) : Number(rawValue);
      const projectedDisplay = percentileView ? projectedPercentile : Number(projectedRaw);
      const maxValue = percentileView
        ? 100
        : Math.max(
            Number.isFinite(currentDisplay) ? currentDisplay : 0,
            Number.isFinite(projectedDisplay) ? projectedDisplay : 0,
            0
          );
      return {
        metric: formatMetricLabel(key),
        current: percentileView ? clampPercentile(currentDisplay) : normalizeMetric(currentDisplay, maxValue),
        projected: percentileView ? clampPercentile(projectedDisplay) : normalizeMetric(projectedDisplay, maxValue),
        current_raw: rawValue,
        projected_raw: projectedRaw,
        current_pct: basePercentile,
        projected_pct: projectedPercentile,
      };
    });
  }, [radarMetricKeys, metrics, coeff, percentileView]);

  const summaryKeys = useMemo(() => {
    return Object.keys(metrics)
      .filter((key) => key.startsWith("summary_"))
      .filter((key) => !key.includes("_pct_"))
      .sort((a, b) =>
        formatSummaryLabel(a).localeCompare(formatSummaryLabel(b), undefined, { sensitivity: "base" })
      );
  }, [metrics]);

  const summaryRows = useMemo(() => {
    return summaryKeys.map((key) => {
      const value = metrics[key];
      const projected = value != null ? Number(value) * coeff : null;
      return {
        key,
        label: formatSummaryLabel(key),
        current: value,
        projected,
      };
    });
  }, [summaryKeys, metrics, coeff]);

  const currentLeague = report?.player?.competition_name || "";
  const projectionOptions = translationLeagues.filter((league) => league !== currentLeague);

  return (
    <main className="nl-page py-10 px-4">
      <div className="max-w-6xl mx-auto space-y-6">
        <header className="flex flex-col gap-2">
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
            Projection
          </p>
          <h1 className="text-4xl font-bold text-white tracking-tight">
            League fit projection
          </h1>
          <p className="text-slate-300 max-w-3xl">
            Estimate how a player’s output may translate when the competitive context changes.
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <Card className="relative z-30 lg:col-span-2 nl-filter-bar p-0">
            <div className="flex flex-col gap-4 border-b border-white/10 px-4 py-4 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <p className="nl-kicker">Projection setup</p>
                <h2 className="mt-1 text-lg font-semibold text-white">Select the player baseline</h2>
              </div>
              <span className="rounded-md border border-[#3A8967]/30 bg-[#2F7D5C]/15 px-3 py-2 text-xs font-semibold text-[#8CC7A7]">
                {selectedSeason || "All seasons"}
              </span>
            </div>
            <div className="relative space-y-4 p-4">
              <div className="grid grid-cols-1 gap-3 lg:grid-cols-[220px_minmax(0,1fr)]">
              <div className="flex flex-col gap-2">
                <Label>Season</Label>
                <Select
                  id="projection-season"
                  value={selectedSeason}
                  onChange={(e) => {
                    setSelectedSeason(e.target.value);
                    setPlayerQuery("");
                    setSelectedPlayerId("");
                    setSelectedPlayerSeasonId("");
                    setReport(null);
                    setPlayerResults([]);
                  }}
                >
                  <option value="">All seasons</option>
                  {seasons.map((season) => (
                    <option key={season} value={season}>
                      {season}
                    </option>
                  ))}
                </Select>
              </div>
              <div className="flex flex-col gap-2">
                <Label>Player</Label>
                <input
                  className="bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100"
                  placeholder="Start typing a player name..."
                  value={playerQuery}
                  onChange={(e) => {
                    setPlayerQuery(e.target.value);
                    setSelectedPlayerId("");
                    setSelectedPlayerSeasonId("");
                    setShowResults(true);
                  }}
                  onFocus={() => {
                    if (selectedPlayerId) {
                      setPlayerQuery("");
                      setSelectedPlayerId("");
                      setSelectedPlayerSeasonId("");
                      setReport(null);
                    }
                    setShowResults(true);
                  }}
                  onClick={() => {
                    if (selectedPlayerId) {
                      setPlayerQuery("");
                      setSelectedPlayerId("");
                      setSelectedPlayerSeasonId("");
                      setReport(null);
                      setShowResults(true);
                    }
                  }}
                  onBlur={() => setTimeout(() => setShowResults(false), 150)}
                />
              </div>
              </div>
              {showResults && playerQuery.trim().length >= 2 ? (
                <div className="absolute z-50 mt-2 w-full max-h-72 overflow-auto rounded-lg border border-slate-700 bg-slate-900/95 shadow-xl">
                  {playerOptions.length === 0 ? (
                    <div className="px-3 py-2 text-sm text-slate-400">
                      No matches found.
                    </div>
                  ) : (
                    playerOptions.map((player) => (
                      <button
                        key={`${player.id}-${player.seasonId || "latest"}`}
                        type="button"
                        className="w-full text-left px-3 py-2 text-sm text-slate-200 hover:bg-slate-800/80"
                        onMouseDown={(e) => e.preventDefault()}
                        onClick={() => handlePlayerSelect(player)}
                      >
                        {player.label}
                      </button>
                    ))
                  )}
                </div>
              ) : null}
            </div>
          </Card>

          <Card className="space-y-4">
            <div className="flex flex-col gap-2">
              <Label>Projection league</Label>
              <Select
                value={targetLeague}
                onChange={(e) => setTargetLeague(e.target.value)}
              >
                <option value="">Select league</option>
                {projectionOptions.map((league) => (
                  <option key={league} value={league}>
                    {league}
                  </option>
                ))}
              </Select>
            </div>
            <label className="flex items-center gap-2 text-sm text-slate-300">
              <input
                type="checkbox"
                className="accent-emerald-400"
                checked={percentileView}
                onChange={(e) => setPercentileView(e.target.checked)}
              />
              Percentile view
            </label>
            <div className="text-xs text-slate-400">
              Translation coeff: {coeff.toFixed(3)}
            </div>
            <div className="rounded-lg border border-slate-700/70 bg-slate-900/50 px-3 py-2 text-xs text-slate-300">
              <div className="flex items-center justify-between">
                <span>Current score</span>
                <span className="text-slate-100">
                  {formatValue(currentScore, 2)}
                </span>
              </div>
              <div className="flex items-center justify-between mt-1">
                <span>Projected score</span>
                <span className="text-emerald-200">
                  {formatValue(projectedScore, 2)}
                </span>
              </div>
            </div>
          </Card>
        </div>

        {error && (
          <Card>
            <p className="text-danger">Error: {error}</p>
          </Card>
        )}

        {loading ? (
          <Card>
            <p className="text-slate-400">Loading player...</p>
          </Card>
        ) : report ? (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto_1fr] gap-4 items-center">
              <Card>
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-white">{currentLeague}</h3>
                  <span className="text-xs text-slate-400">Current</span>
                </div>
                <div className="h-80 mt-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={radarData} outerRadius="85%">
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
                          <ProjectionRadarTooltip
                            label={percentileView ? "Percentiles" : "Raw"}
                            isPercentile={percentileView}
                            valueKey="current"
                            rawKey="current_raw"
                          />
                        }
                        cursor={false}
                      />
                      <Radar
                        name="Current"
                        dataKey="current"
                        stroke="#7bd389"
                        fill="rgba(123, 211, 137, 0.25)"
                        fillOpacity={percentileView ? 0.25 : 0}
                        dot={{ r: 4, stroke: "#7bd389", strokeWidth: 1 }}
                        activeDot={{ r: 6 }}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </Card>

              <div className="text-4xl text-emerald-300 px-2">-&gt;</div>

              <Card>
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-white">{targetLeague || "Projection"}</h3>
                  <span className="text-xs text-slate-400">Projected</span>
                </div>
                <div className="h-80 mt-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={radarData} outerRadius="85%">
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
                          <ProjectionRadarTooltip
                            label={percentileView ? "Projected percentiles" : "Projected raw"}
                            isPercentile={percentileView}
                            valueKey="projected"
                            rawKey="projected_raw"
                          />
                        }
                        cursor={false}
                      />
                      <Radar
                        name="Projected"
                        dataKey="projected"
                        stroke="#60a5fa"
                        fill="rgba(96, 165, 250, 0.2)"
                        fillOpacity={percentileView ? 0.2 : 0}
                        dot={{ r: 4, stroke: "#60a5fa", strokeWidth: 1 }}
                        activeDot={{ r: 6 }}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </div>

            <Card>
              <h3 className="text-lg font-semibold text-white mb-3">
                Aggregated summary scores
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-xs uppercase text-slate-400 border-b border-white/5">
                      <th className="text-left py-2">Metric</th>
                      <th className="text-left py-2">{currentLeague || "Current"}</th>
                      <th className="text-left py-2">{targetLeague || "Projected"}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {summaryRows.map((row) => (
                      <tr key={row.key} className="border-b border-white/5">
                        <td className="py-2 text-slate-200">{row.label}</td>
                        <td className="py-2 text-slate-300">{formatValue(row.current)}</td>
                        <td className="py-2 text-slate-300">{formatValue(row.projected)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </>
        ) : (
          <Card>
            <p className="text-slate-400">
              Select a player to see projection results.
            </p>
          </Card>
        )}
      </div>
    </main>
  );
}
