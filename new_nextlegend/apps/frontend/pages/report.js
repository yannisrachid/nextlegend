import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/router";
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

const TM_BASE_URL = "https://www.transfermarkt.com";
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

const Select = ({ value, onChange, children }) => (
  <select
    className="bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100"
    value={value}
    onChange={onChange}
  >
    {children}
  </select>
);

const Badge = ({ children }) => (
  <span className="px-2 py-1 rounded-full bg-slate-800 text-xs text-slate-200 border border-white/5">
    {children}
  </span>
);

const RadarTooltip = ({ active, payload }) => {
  if (!active || !payload || payload.length === 0) return null;
  const point = payload[0]?.payload;
  if (!point) return null;
  return (
    <div className="rounded-md border border-slate-700 bg-slate-900/95 px-3 py-2 text-xs text-slate-100 shadow-xl">
      <div className="font-semibold text-white">{point.metric}</div>
      <div className="text-slate-300">
        {point.contextLabel}: {Number(payload[0]?.value ?? point.value).toFixed(0)}
      </div>
      <div className="text-slate-400">
        Value: {point.raw != null ? Number(point.raw).toFixed(2) : "—"}
      </div>
    </div>
  );
};

const toAbsoluteUrl = (value) => {
  if (!value) return "";
  const url = String(value).trim();
  if (!url) return "";
  if (url.startsWith("http://") || url.startsWith("https://")) {
    return url;
  }
  if (url.startsWith("/")) {
    return `${TM_BASE_URL}${url}`;
  }
  return url;
};

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

const formatCompactNumber = (value) => {
  if (value === null || value === undefined || value === "") return "";
  let numeric = value;
  if (typeof numeric === "string") {
    const raw = numeric.trim();
    if (!raw) return "";
    const lower = raw.toLowerCase();
    const match = lower.match(/([0-9]+(?:\\.[0-9]+)?)/);
    if (!match) return raw;
    numeric = Number(match[1]);
    if (!Number.isFinite(numeric)) return raw;
    if (lower.includes("bn") || lower.includes("b")) {
      numeric *= 1e9;
    } else if (lower.includes("m")) {
      numeric *= 1e6;
    } else if (lower.includes("k")) {
      numeric *= 1e3;
    }
  }
  if (typeof numeric !== "number" || !Number.isFinite(numeric)) {
    return String(value);
  }
  const abs = Math.abs(numeric);
  const format = (num, suffix) => {
    const rounded = num >= 10 ? Math.round(num) : Math.round(num * 10) / 10;
    const label = rounded % 1 === 0 ? rounded.toFixed(0) : rounded.toFixed(1);
    return `${label} ${suffix}`;
  };
  if (abs >= 1e9) return format(abs / 1e9, "B");
  if (abs >= 1e6) return format(abs / 1e6, "M");
  if (abs >= 1e3) return format(abs / 1e3, "K");
  return `${Math.round(abs)}`;
};

const getInitials = (value) => {
  if (!value) return "—";
  const parts = String(value).trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "—";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase();
};

export default function ReportPage() {
  const router = useRouter();
  const hydratedQuery = useRef(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [playerQuery, setPlayerQuery] = useState("");
  const [playerResults, setPlayerResults] = useState([]);
  const [selectedPlayerId, setSelectedPlayerId] = useState("");
  const [showResults, setShowResults] = useState(false);
  const [report, setReport] = useState(null);
  const [similarities, setSimilarities] = useState([]);
  const [similarLoading, setSimilarLoading] = useState(false);
  const [sortConfig, setSortConfig] = useState({ key: "league", dir: "desc" });
  const [showAllMetrics, setShowAllMetrics] = useState(false);
  const [similarSort, setSimilarSort] = useState({ key: "similarity", dir: "desc" });
  const [similarPage, setSimilarPage] = useState(0);
  const [similarHasNext, setSimilarHasNext] = useState(false);
  const [similarFilters, setSimilarFilters] = useState({
    ageMin: "",
    ageMax: "",
    big5Only: false,
  });
  const [radarContext, setRadarContext] = useState("global");

  const similarLimit = 10;

  useEffect(() => {
    if (!playerQuery || playerQuery.trim().length < 2) {
      setPlayerResults([]);
      setSelectedPlayerId("");
      return;
    }
    const handle = setTimeout(async () => {
      try {
        const res = await fetchJson("/players", { q: playerQuery.trim() });
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
  }, [playerQuery]);

  useEffect(() => {
    if (!selectedPlayerId) {
      setReport(null);
      setSimilarities([]);
      setSimilarPage(0);
      setSimilarHasNext(false);
      setSimilarLoading(false);
      return;
    }
    const loadReport = async () => {
      setLoading(true);
      setError("");
      try {
        const data = await fetchJson(`/players/${selectedPlayerId}/report`);
        setReport(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    loadReport();
  }, [selectedPlayerId]);

  useEffect(() => {
    if (!selectedPlayerId) {
      return;
    }
    const loadSimilarities = async () => {
      setSimilarLoading(true);
      setError("");
      try {
        const sims = await fetchJson(
          `/players/${selectedPlayerId}/similarities`,
          {
            limit: similarLimit,
            offset: similarPage * similarLimit,
            age_min: similarFilters.ageMin || undefined,
            age_max: similarFilters.ageMax || undefined,
            big5_only: similarFilters.big5Only ? "true" : undefined,
          }
        );
        setSimilarities(sims);
        setSimilarHasNext(sims.length === similarLimit);
      } catch (err) {
        setError(err.message);
      } finally {
        setSimilarLoading(false);
      }
    };
    loadSimilarities();
  }, [selectedPlayerId, similarPage, similarFilters]);

  const playerOptions = useMemo(() => {
    return playerResults.map((p) => ({
      id: String(p.id),
      label: `${p.name} - ${p.team || "—"} - ${p.competition_name || "—"} - ${p.calendar || "—"}`,
    }));
  }, [playerResults]);

  const handlePlayerSelect = (player) => {
    setSelectedPlayerId(player.id);
    setPlayerQuery(player.label);
    setShowResults(false);
    setSimilarPage(0);
  };

  useEffect(() => {
    if (!router.isReady || hydratedQuery.current) return;
    const queryId = router.query.player_id || router.query.playerId;
    if (!queryId) return;
    const playerId = String(queryId);
    hydratedQuery.current = true;
    setSelectedPlayerId(playerId);
    fetchJson(`/players/${playerId}`)
      .then((data) => {
        if (!data) return;
        const label = `${data.name} - ${data.team || "—"} - ${data.competition_name || "—"} - ${data.calendar || "—"}`;
        setPlayerQuery(label);
      })
      .catch(() => {});
  }, [router.isReady, router.query, setSelectedPlayerId]);

  const metrics = report?.metrics || {};
  const rawMetrics = report?.raw_metrics || metrics;
  const tmFields = report?.tm_fields || {};
  const tmProfileUrl = toAbsoluteUrl(tmFields.tm_profile_url);
  const tmAgentUrl = toAbsoluteUrl(tmFields.tm_agent_url);
  const tmPhotoUrl = toAbsoluteUrl(tmFields.tm_profile_image_url || tmFields.profile_image_url);
  const tmDetails = [
    { label: "Market value", value: formatCompactNumber(tmFields.tm_market_value) },
    { label: "Contract expires", value: tmFields.tm_club_contract_expires },
    { label: "Birth date", value: tmFields.tm_birth_date },
    { label: "Birth city", value: tmFields.tm_birth_city },
    { label: "Birth country", value: tmFields.tm_birth_country },
    { label: "Citizenship", value: tmFields.tm_citizenship },
    { label: "Foot", value: tmFields.tm_foot },
    { label: "Outfitter", value: tmFields.tm_outfitter },
  ];
  const hasTmData = Object.values(tmFields).some(
    (value) => value !== null && value !== undefined && String(value).trim() !== ""
  );
  const radarMetricKeys = useMemo(() => {
    if (Array.isArray(report?.radar_metrics) && report.radar_metrics.length > 0) {
      return report.radar_metrics;
    }
    return DEFAULT_RADAR_METRICS;
  }, [report]);
  const radarStats = useMemo(
    () =>
      radarMetricKeys.map((key) => ({
        key,
        label: formatMetricLabel(key),
      })),
    [radarMetricKeys]
  );
  const radarData = useMemo(() => {
    return radarStats.map((stat) => {
      const leagueKey = `${stat.key}_pct_league`;
      const globalKey = `${stat.key}_pct_global`;
      const leagueValue = metrics[leagueKey];
      const globalValue = metrics[globalKey];
      const leagueDisplay = leagueValue ?? globalValue ?? 0;
      const globalDisplay = globalValue ?? leagueValue ?? 0;
      return {
        metric: stat.label,
        value:
          radarContext === "league"
            ? Number(leagueDisplay) || 0
            : Number(globalDisplay) || 0,
        league: metrics[leagueKey],
        global: metrics[globalKey],
        leagueDisplay: Number(leagueDisplay) || 0,
        globalDisplay: Number(globalDisplay) || 0,
        raw: rawMetrics[stat.key],
        contextLabel: radarContext === "league" ? "League percentile" : "Global percentile",
      };
    });
  }, [metrics, rawMetrics, radarContext, radarStats]);

  const sortRows = (rows) => {
    const sorted = [...rows];
    const key = sortConfig.key;
    const dir = sortConfig.dir;
    const valueFor = (row) => {
      if (key === "metric") return row.metric;
      if (key === "raw") return row.raw;
      if (key === "league") return row.league;
      if (key === "global") return row.global;
      return row.metric;
    };
    sorted.sort((a, b) => {
      const left = valueFor(a);
      const right = valueFor(b);
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

  const tableData = useMemo(() => {
    return sortRows(radarData);
  }, [radarData, sortConfig]);

  const allMetricsData = useMemo(() => {
    const keys = Object.keys(metrics || {}).filter(
      (key) =>
        !key.endsWith("_pct_league") &&
        !key.endsWith("_pct_global") &&
        !EXCLUDED_METRIC_PREFIXES.some((prefix) => key.startsWith(prefix))
    );
    const rows = keys
      .map((key) => {
        const raw = metrics[key];
        const league = metrics[`${key}_pct_league`];
        const global = metrics[`${key}_pct_global`];
        const hasValue =
          raw !== null && raw !== undefined
          || league !== null && league !== undefined
          || global !== null && global !== undefined;
        if (!hasValue) return null;
        return {
          metric: formatMetricLabel(key),
          metricKey: key,
          raw,
          league,
          global,
        };
      })
      .filter(Boolean);
    return sortRows(rows);
  }, [metrics, sortConfig]);

  const handleSort = (key) => {
    setSortConfig((prev) => {
      if (prev.key === key) {
        return { key, dir: prev.dir === "asc" ? "desc" : "asc" };
      }
      return { key, dir: "asc" };
    });
  };

  const sortIndicator = (key) => {
    if (sortConfig.key !== key) return "";
    return sortConfig.dir === "asc" ? "▲" : "▼";
  };

  const percentileStyle = (value) => {
    const num = Number(value);
    if (!Number.isFinite(num)) {
      return { className: "text-slate-400", style: {} };
    }
    const pct = Math.max(0, Math.min(100, num));
    const hue = Math.round((pct / 100) * 120);
    return {
      className: "text-slate-100",
      style: { color: `hsl(${hue}, 70%, 55%)` },
    };
  };

  const handleSimilarSort = (key) => {
    setSimilarSort((prev) => {
      if (prev.key === key) {
        return { key, dir: prev.dir === "asc" ? "desc" : "asc" };
      }
      return { key, dir: "desc" };
    });
  };

  const similarSortIndicator = (key) => {
    if (similarSort.key !== key) return "";
    return similarSort.dir === "asc" ? "▲" : "▼";
  };

  const sortedSimilarities = useMemo(() => {
    const rows = [...similarities];
    const key = similarSort.key;
    const dir = similarSort.dir;
    const valueFor = (row) => {
      if (key === "player") return row.player_b_name || "";
      if (key === "similarity") return row.similarity;
      if (key === "adjusted") return row.global_score_adjusted;
      if (key === "league") return row.assigned_role_pct_league;
      if (key === "global") return row.assigned_role_pct_global;
      if (key === "age") return row.age;
      return row.similarity;
    };
    rows.sort((a, b) => {
      const left = valueFor(a);
      const right = valueFor(b);
      if (left == null && right == null) return 0;
      if (left == null) return 1;
      if (right == null) return -1;
      if (key === "player") {
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
    return rows;
  }, [similarities, similarSort]);

  const topRoles = useMemo(() => {
    if (!report?.role_scores) return [];
    return report.role_scores.slice(0, 3);
  }, [report]);

  return (
    <main className="min-h-screen bg-hero-pattern text-slate-100 py-10 px-4">
      <div className="max-w-6xl mx-auto space-y-6">
        <header className="flex flex-col gap-2">
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
            Report
          </p>
          <h1 className="text-4xl font-bold text-white tracking-tight">
            Player Report
          </h1>
          <p className="text-slate-300 max-w-3xl">
            League → Team → Player. Synthetic view with radar, role fit and
            similar players from the v2 pipeline.
          </p>
        </header>

        <Card className="relative z-30">
          <div className="relative">
            <div className="flex flex-col gap-2">
              <Label>Player</Label>
              <input
                className="bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100"
                placeholder="Start typing a player name..."
                value={playerQuery}
                onChange={(e) => {
                  setPlayerQuery(e.target.value);
                  setSelectedPlayerId("");
                  setShowResults(true);
                }}
                onFocus={() => {
                  if (selectedPlayerId) {
                    setPlayerQuery("");
                    setSelectedPlayerId("");
                    setReport(null);
                    setSimilarities([]);
                  }
                  setShowResults(true);
                }}
                onClick={() => {
                  if (selectedPlayerId) {
                    setPlayerQuery("");
                    setSelectedPlayerId("");
                    setReport(null);
                    setSimilarities([]);
                    setShowResults(true);
                  }
                }}
                onBlur={() => setTimeout(() => setShowResults(false), 150)}
              />
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
                      key={player.id}
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

        {error && (
          <Card>
            <p className="text-danger">Error: {error}</p>
          </Card>
        )}

        {loading ? (
          <Card>
            <p className="text-slate-400">Loading report…</p>
          </Card>
        ) : report ? (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <Card className="lg:col-span-1 space-y-4">
                <div className="flex items-start gap-3">
                  {tmPhotoUrl ? (
                    <img
                      src={tmPhotoUrl}
                      alt={report.player.name}
                      className="h-24 w-24 rounded-full object-cover border border-white/10"
                    />
                  ) : (
                    <div className="h-24 w-24 rounded-full bg-slate-800 border border-white/10 flex items-center justify-center text-slate-200 font-semibold">
                      {getInitials(report.player.name)}
                    </div>
                  )}
                  <div>
                    <p className="text-sm text-slate-400">Player</p>
                    <h2 className="text-2xl font-semibold text-white">
                      {report.player.name}
                    </h2>
                    <p className="text-slate-400">
                      {report.player.team} • {report.player.competition_name}
                    </p>
                    {tmProfileUrl ? (
                      <a
                        href={tmProfileUrl}
                        target="_blank"
                        rel="noreferrer"
                        className="text-xs text-primary hover:text-primary/80"
                      >
                        Transfermarkt profile
                      </a>
                    ) : null}
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  {report.player.assigned_role && (
                    <Badge>{report.player.assigned_role}</Badge>
                  )}
                  {report.player.position && (
                    <Badge>{report.player.position}</Badge>
                  )}
                  {report.player.age && <Badge>{report.player.age} yrs</Badge>}
                  <Badge>{report.player.minutes_played} mins</Badge>
                </div>
                {hasTmData ? (
                  <div className="space-y-2 border-t border-white/5 pt-3">
                    <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                      Transfermarkt
                    </p>
                    {(tmFields.tm_agent_name || tmAgentUrl) && (
                      <p className="text-sm text-slate-300">
                        Agent:{" "}
                        {tmAgentUrl ? (
                          <a
                            href={tmAgentUrl}
                            target="_blank"
                            rel="noreferrer"
                            className="text-primary hover:text-primary/80"
                          >
                            {tmFields.tm_agent_name || "Profile"}
                          </a>
                        ) : (
                          tmFields.tm_agent_name
                        )}
                      </p>
                    )}
                    <div className="grid grid-cols-1 gap-1 text-sm text-slate-300">
                      {tmDetails
                        .filter(
                          (item) =>
                            item.value !== null &&
                            item.value !== undefined &&
                            String(item.value).trim() !== ""
                        )
                        .map((item) => (
                          <div key={item.label} className="flex items-center justify-between gap-2">
                            <span className="text-slate-400">{item.label}</span>
                            <span className="text-slate-100">{item.value}</span>
                          </div>
                        ))}
                    </div>
                  </div>
                ) : (
                  <p className="text-xs text-slate-500">
                    Transfermarkt data not available yet.
                  </p>
                )}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <p className="text-xs uppercase text-slate-400">
                      Global score
                    </p>
                    <p className="text-2xl font-bold text-primary">
                      {report.player.global_score_adjusted?.toFixed(1) ?? "—"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs uppercase text-slate-400">
                      Role pct (L/G)
                    </p>
                    <p className="text-lg font-semibold">
                      {report.player.assigned_role_pct_league?.toFixed(0) ?? "—"} /
                      {report.player.assigned_role_pct_global?.toFixed(0) ?? "—"}
                    </p>
                  </div>
                </div>
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
                            ? "border-emerald-400/70 bg-emerald-400/15 text-emerald-200"
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
                    <RadarChart
                      key={`radar-${radarContext}`}
                      data={radarData}
                      outerRadius="92%"
                    >
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
                      <Tooltip content={<RadarTooltip />} cursor={false} />
                      <Radar
                        name="Percentile"
                        dataKey="value"
                        stroke="#7bd389"
                        fill="rgba(123, 211, 137, 0.25)"
                        fillOpacity={0.35}
                        dot={{ r: 4, stroke: "#7bd389", strokeWidth: 1 }}
                        activeDot={{ r: 6 }}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </div>

            <Card>
              <h3 className="text-lg font-semibold text-white mb-3">
                Role Fit (Top 3)
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {topRoles.map((role, index) => (
                  <div
                    key={role.profile}
                    className={`border rounded-lg p-3 bg-slate-900/50 ${
                      index === 0 ? "border-emerald-400/80" : "border-white/5"
                    }`}
                  >
                    <p className="text-sm text-slate-400">Profile</p>
                    <p className="text-base font-semibold text-white">
                      {role.profile}
                    </p>
                    <p className="text-sm text-slate-300 mt-2">
                      League: {role.pct_league?.toFixed(0) ?? "—"} • Global:{" "}
                      {role.pct_global?.toFixed(0) ?? "—"}
                    </p>
                  </div>
                ))}
              </div>
            </Card>

            <Card>
              <h3 className="text-lg font-semibold text-white mb-3">
                Percentile Overview
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-xs uppercase text-slate-400 border-b border-white/5">
                      <th
                        className="text-left py-2 cursor-pointer select-none"
                        onClick={() => handleSort("metric")}
                      >
                        Metric {sortIndicator("metric")}
                      </th>
                      <th
                        className="text-right py-2 cursor-pointer select-none"
                        onClick={() => handleSort("raw")}
                      >
                        Value {sortIndicator("raw")}
                      </th>
                      <th
                        className="text-right py-2 cursor-pointer select-none"
                        onClick={() => handleSort("league")}
                      >
                        League percentile {sortIndicator("league")}
                      </th>
                      <th
                        className="text-right py-2 cursor-pointer select-none"
                        onClick={() => handleSort("global")}
                      >
                        Global percentile {sortIndicator("global")}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {tableData.map((row) => (
                      <tr key={row.metric} className="border-b border-white/5">
                        <td className="py-2 text-slate-200">{row.metric}</td>
                        <td className="py-2 text-right text-slate-100">
                          {row.raw != null ? Number(row.raw).toFixed(2) : "—"}
                        </td>
                        {(() => {
                          const style = percentileStyle(row.league);
                          return (
                            <td className={`py-2 text-right ${style.className}`} style={style.style}>
                              {row.league != null ? Number(row.league).toFixed(0) : "—"}
                            </td>
                          );
                        })()}
                        {(() => {
                          const style = percentileStyle(row.global);
                          return (
                            <td className={`py-2 text-right ${style.className}`} style={style.style}>
                              {row.global != null ? Number(row.global).toFixed(0) : "—"}
                            </td>
                          );
                        })()}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>

            <Card>
              <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
                <h3 className="text-lg font-semibold text-white">
                  All Metrics
                </h3>
                <button
                  type="button"
                  className="text-xs uppercase tracking-[0.2em] text-slate-300 hover:text-white"
                  onClick={() => setShowAllMetrics((prev) => !prev)}
                >
                  {showAllMetrics ? "Hide full table" : "Show full table"}
                </button>
              </div>
              {showAllMetrics ? (
                <div className="overflow-x-auto">
                  <div className="max-h-96 overflow-y-auto pr-3">
                    <table className="w-full text-sm">
                      <thead className="sticky top-0 bg-slate-950/95">
                        <tr className="text-xs uppercase text-slate-400 border-b border-white/5">
                          <th
                            className="text-left py-2 cursor-pointer select-none"
                            onClick={() => handleSort("metric")}
                          >
                            Metric {sortIndicator("metric")}
                          </th>
                          <th
                            className="text-right py-2 cursor-pointer select-none"
                            onClick={() => handleSort("raw")}
                          >
                            Value {sortIndicator("raw")}
                          </th>
                          <th
                            className="text-right py-2 cursor-pointer select-none"
                            onClick={() => handleSort("league")}
                          >
                            League percentile {sortIndicator("league")}
                          </th>
                          <th
                            className="text-right py-2 cursor-pointer select-none"
                            onClick={() => handleSort("global")}
                          >
                            Global percentile {sortIndicator("global")}
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {allMetricsData.length === 0 ? (
                          <tr>
                            <td colSpan={4} className="py-4 text-center text-slate-400">
                              No metrics available.
                            </td>
                          </tr>
                        ) : (
                          allMetricsData.map((row) => (
                            <tr key={row.metricKey} className="border-b border-white/5">
                              <td className="py-2 text-slate-200">{row.metric}</td>
                              <td className="py-2 text-right text-slate-100">
                                {row.raw != null ? Number(row.raw).toFixed(2) : "—"}
                              </td>
                              {(() => {
                                const style = percentileStyle(row.league);
                                return (
                                  <td className={`py-2 text-right ${style.className}`} style={style.style}>
                                    {row.league != null ? Number(row.league).toFixed(0) : "—"}
                                  </td>
                                );
                              })()}
                              {(() => {
                                const style = percentileStyle(row.global);
                                return (
                                  <td className={`py-2 text-right ${style.className}`} style={style.style}>
                                    {row.global != null ? Number(row.global).toFixed(0) : "—"}
                                  </td>
                                );
                              })()}
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : (
                <p className="text-slate-400 text-sm">
                  Expand to browse all metrics with percentiles.
                </p>
              )}
            </Card>

            <Card>
              <h3 className="text-lg font-semibold text-white mb-3">
                Similar Players
              </h3>
              <div className="flex flex-wrap items-end gap-4 mb-4">
                <div className="flex flex-col gap-2">
                  <Label>Age min</Label>
                  <input
                    type="number"
                    min={0}
                    className="bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100 w-28"
                    value={similarFilters.ageMin}
                    onChange={(e) => {
                      setSimilarFilters((prev) => ({ ...prev, ageMin: e.target.value }));
                      setSimilarPage(0);
                    }}
                  />
                </div>
                <div className="flex flex-col gap-2">
                  <Label>Age max</Label>
                  <input
                    type="number"
                    min={0}
                    className="bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100 w-28"
                    value={similarFilters.ageMax}
                    onChange={(e) => {
                      setSimilarFilters((prev) => ({ ...prev, ageMax: e.target.value }));
                      setSimilarPage(0);
                    }}
                  />
                </div>
                <label className="flex items-center gap-2 text-sm text-slate-300">
                  <input
                    type="checkbox"
                    className="accent-emerald-400"
                    checked={similarFilters.big5Only}
                    onChange={(e) => {
                      setSimilarFilters((prev) => ({ ...prev, big5Only: e.target.checked }));
                      setSimilarPage(0);
                    }}
                  />
                  5 Big Leagues Only
                </label>
              </div>
              {similarLoading ? (
                <p className="text-slate-400 text-sm mb-2">Loading similar players…</p>
              ) : null}
              {similarities.length === 0 ? (
                <p className="text-slate-400">No similar players found.</p>
              ) : (
                <div className="space-y-3">
                  <div className="hidden md:grid grid-cols-[minmax(0,2.3fr)_repeat(5,minmax(0,1fr))] text-xs uppercase text-slate-400 border-b border-white/5 pb-2">
                    <button
                      type="button"
                      className="text-left cursor-pointer select-none"
                      onClick={() => handleSimilarSort("player")}
                    >
                      Player {similarSortIndicator("player")}
                    </button>
                    <button
                      type="button"
                      className="text-right cursor-pointer select-none"
                      onClick={() => handleSimilarSort("age")}
                    >
                      Age {similarSortIndicator("age")}
                    </button>
                    <button
                      type="button"
                      className="text-right cursor-pointer select-none"
                      onClick={() => handleSimilarSort("similarity")}
                    >
                      Similarity {similarSortIndicator("similarity")}
                    </button>
                    <button
                      type="button"
                      className="text-right cursor-pointer select-none"
                      onClick={() => handleSimilarSort("adjusted")}
                    >
                      Adjusted score {similarSortIndicator("adjusted")}
                    </button>
                    <button
                      type="button"
                      className="text-right cursor-pointer select-none"
                      onClick={() => handleSimilarSort("league")}
                    >
                      League pct {similarSortIndicator("league")}
                    </button>
                    <button
                      type="button"
                      className="text-right cursor-pointer select-none"
                      onClick={() => handleSimilarSort("global")}
                    >
                      Global pct {similarSortIndicator("global")}
                    </button>
                  </div>
                  <div className="space-y-3">
                    {sortedSimilarities.map((sim) => {
                      const tmFields = sim.tm_fields || {};
                      const tmPhotoUrl = toAbsoluteUrl(
                        tmFields.tm_profile_image_url || tmFields.profile_image_url
                      );
                      const tmProfileUrl = toAbsoluteUrl(
                        tmFields.tm_profile_url || sim.tm_profile_url
                      );
                      const reportUrl = `/report?player_id=${sim.player_b_id}`;
                      const leagueStyle = percentileStyle(sim.assigned_role_pct_league);
                      const globalStyle = percentileStyle(sim.assigned_role_pct_global);
                      return (
                        <div
                          key={`${sim.player_b_id}-${sim.profile}`}
                          role="button"
                          tabIndex={0}
                          onClick={() => window.open(reportUrl, "_blank", "noopener,noreferrer")}
                          onKeyDown={(event) => {
                            if (event.key === "Enter" || event.key === " ") {
                              event.preventDefault();
                              window.open(reportUrl, "_blank", "noopener,noreferrer");
                            }
                          }}
                          className="grid grid-cols-1 md:grid-cols-[minmax(0,2.3fr)_repeat(5,minmax(0,1fr))] gap-3 items-center border border-white/5 rounded-lg p-3 bg-slate-900/50 cursor-pointer focus:outline-none focus:ring-2 focus:ring-emerald-400/60"
                        >
                          <div className="flex items-center gap-3">
                            {tmPhotoUrl ? (
                              <img
                                src={tmPhotoUrl}
                                alt={sim.player_b_name}
                                className="h-12 w-12 rounded-full object-cover border border-white/10"
                              />
                            ) : (
                              <div className="h-12 w-12 rounded-full bg-slate-800 border border-white/10 flex items-center justify-center text-slate-200 font-semibold">
                                {getInitials(sim.player_b_name)}
                              </div>
                            )}
                            <div>
                              <p className="text-base font-semibold text-white">
                                {sim.player_b_name}
                              </p>
                              <p className="text-sm text-slate-400">
                                {sim.team || "—"} • {sim.competition_name || "—"}
                              </p>
                              {tmProfileUrl ? (
                                <a
                                  href={tmProfileUrl}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="text-xs text-primary hover:text-primary/80"
                                  onClick={(event) => event.stopPropagation()}
                                >
                                  Transfermarkt profile
                                </a>
                              ) : null}
                            </div>
                          </div>
                          <div className="text-right text-sm text-slate-100">
                            {sim.age != null ? Number(sim.age).toFixed(0) : "—"}
                          </div>
                          <div className="text-right text-sm text-slate-100">
                            {sim.similarity != null
                              ? `${(Number(sim.similarity) * 100).toFixed(1)}%`
                              : "—"}
                          </div>
                          <div className="text-right text-sm text-slate-100">
                            {sim.global_score_adjusted != null
                              ? Number(sim.global_score_adjusted).toFixed(1)
                              : "—"}
                          </div>
                          <div className={`text-right text-sm ${leagueStyle.className}`} style={leagueStyle.style}>
                            {sim.assigned_role_pct_league != null
                              ? Number(sim.assigned_role_pct_league).toFixed(0)
                              : "—"}
                          </div>
                          <div className={`text-right text-sm ${globalStyle.className}`} style={globalStyle.style}>
                            {sim.assigned_role_pct_global != null
                              ? Number(sim.assigned_role_pct_global).toFixed(0)
                              : "—"}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <div className="flex items-center justify-between pt-2">
                    <button
                      type="button"
                      className="px-3 py-2 rounded-md border border-slate-700 bg-slate-900/60 disabled:opacity-50"
                      disabled={similarPage === 0}
                      onClick={() => setSimilarPage((prev) => Math.max(0, prev - 1))}
                    >
                      Prev
                    </button>
                    <span className="text-xs text-slate-400">
                      Page {similarPage + 1}
                    </span>
                    <button
                      type="button"
                      className="px-3 py-2 rounded-md border border-slate-700 bg-slate-900/60 disabled:opacity-50"
                      disabled={!similarHasNext}
                      onClick={() => setSimilarPage((prev) => prev + 1)}
                    >
                      Next
                    </button>
                  </div>
                </div>
              )}
            </Card>
          </>
        ) : (
          <Card>
            <p className="text-slate-400">
              Start typing a player name to see matching results.
            </p>
          </Card>
        )}
      </div>
    </main>
  );
}
