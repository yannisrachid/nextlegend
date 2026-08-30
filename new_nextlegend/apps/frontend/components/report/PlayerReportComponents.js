import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import ClubLogo from "@/components/ClubLogo";
import { clampPercentile, strengthLevel, weaknessLevel } from "@/lib/percentileColors";
import {
  getMetricConfig,
  getPositionMeta,
  getRadarMetricKeys,
  getRelevantMetricKeys,
  lowerIsBetter,
  metricGroupOrderForPosition,
  metricGroupOrder,
  metricGroups,
  normalizePositionGroup,
  profileCategories,
} from "@/lib/reportMetrics";

const TM_BASE_URL = "https://www.transfermarkt.com";

export const ReportCard = ({ children, className = "" }) => (
  <section className={`surface-panel rounded-lg p-4 ${className}`}>
    {children}
  </section>
);

const Kicker = ({ children }) => (
  <p className="text-[11px] font-black uppercase tracking-[0.18em] text-[#8CC7A7]">{children}</p>
);

const valueExists = (value) => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));

const toAbsoluteUrl = (value) => {
  if (!value) return "";
  const url = String(value).trim();
  if (!url) return "";
  if (url.startsWith("http://") || url.startsWith("https://")) return url;
  if (url.startsWith("/")) return `${TM_BASE_URL}${url}`;
  return url;
};

const getInitials = (value) => {
  const parts = String(value || "").trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "-";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase();
};

export const formatValue = (value, format = "number") => {
  if (!valueExists(value)) return "-";
  const numeric = Number(value);
  if (format === "integer") return numeric.toFixed(0);
  if (format === "score") return numeric.toFixed(1);
  if (format === "percent") return `${numeric.toFixed(numeric >= 10 ? 0 : 1)}%`;
  if (Math.abs(numeric) >= 100) return numeric.toFixed(0);
  if (Math.abs(numeric) >= 10) return numeric.toFixed(1);
  return numeric.toFixed(2);
};

const metricRaw = (metrics, key) => metrics?.[key];
const metricPct = (metrics, key, context = "global") => metrics?.[`${key}_pct_${context}`] ?? metrics?.[`${key}_pct_global`] ?? metrics?.[`${key}_pct_league`];

const metricAvailable = (metrics, key) => {
  if (!metrics) return false;
  return valueExists(metrics[key]) || valueExists(metrics[`${key}_pct_global`]) || valueExists(metrics[`${key}_pct_league`]);
};

const metricRowsForGroup = (metrics, groupKey, context) => {
  const group = metricGroups[groupKey];
  if (!group) return [];
  return group.metrics
    .filter((metric) => metricAvailable(metrics, metric.key))
    .map((metric) => ({
      ...metric,
      raw: metricRaw(metrics, metric.key),
      percentile: metricPct(metrics, metric.key, context),
    }));
};

const averageMetric = (report, context, referenceGroup, key) =>
  report?.average_contexts?.[context]?.[referenceGroup]?.metrics?.[key] ?? null;

const averageSampleSize = (report, context, referenceGroup) =>
  report?.average_contexts?.[context]?.[referenceGroup]?.sample_size ?? 0;

const averageMinMinutes = (report, context, referenceGroup) =>
  report?.average_contexts?.[context]?.[referenceGroup]?.min_minutes ?? null;

const average = (values) => {
  const numeric = values.map(Number).filter(Number.isFinite);
  if (!numeric.length) return null;
  return numeric.reduce((sum, value) => sum + value, 0) / numeric.length;
};

export const buildProfileCategories = (metrics, context = "global") =>
  profileCategories.map((category) => {
    const values = category.metrics.map((key) => metricPct(metrics, key, context)).filter(valueExists);
    return {
      ...category,
      score: average(values),
      sampleSize: values.length,
    };
  });

const insightForCategory = (category) => {
  if (!valueExists(category.score)) return null;
  const score = Number(category.score);
  const label = category.label.toLowerCase();
  if (score >= 85) return `Outstanding ${label} profile for his position group.`;
  if (score >= 70) return `Positive ${label} signal against positional peers.`;
  if (score <= 30) return `${category.label} is a relative limitation in this profile.`;
  if (score <= 45) return `${category.label} sits below the ideal benchmark.`;
  return null;
};

const scoreBarTone = (value) => {
  const score = clampPercentile(value);
  if (score === null) return { color: "#94a3b8", track: "rgba(148,163,184,0.14)" };
  if (score < 40) return { color: "#ef4444", track: "rgba(239,68,68,0.14)" };
  if (score < 70) return { color: "#d97706", track: "rgba(217,119,6,0.16)" };
  return { color: "#559A78", track: "rgba(85,154,120,0.16)" };
};

const percentileBadgeStyle = (value) => {
  const pct = clampPercentile(value);
  if (pct === null) return "border-white/10 bg-white/[0.04] text-slate-400";
  if (pct >= 95) return "border-cyan-400/25 bg-cyan-400/10 text-cyan-200";
  if (pct >= 80) return "border-[#3A8967]/35 bg-[#2F7D5C]/15 text-[#8CC7A7]";
  if (pct >= 60) return "border-emerald-400/25 bg-emerald-500/10 text-emerald-200";
  if (pct >= 40) return "border-amber-400/25 bg-amber-500/10 text-amber-200";
  if (pct >= 20) return "border-orange-400/25 bg-orange-500/10 text-orange-200";
  return "border-rose-400/25 bg-rose-500/10 text-rose-200";
};

export const buildCharacteristics = (metrics, positionGroup, context = "global") => {
  const relevantKeys = getRelevantMetricKeys(positionGroup);
  const rows = [];
  relevantKeys.forEach((key) => {
    const config = getMetricConfig(key);
    const percentile = metricPct(metrics, key, context);
    if (!valueExists(percentile)) return;
    rows.push({
      key,
      label: config.label,
      description: config.description,
      raw: metricRaw(metrics, key),
      percentile: Number(percentile),
      lowerIsBetter: config.lowerIsBetter,
    });
  });
  const strengths = rows
    .map((row) => ({ ...row, level: strengthLevel(row.percentile) }))
    .filter((row) => row.level)
    .sort((a, b) => b.percentile - a.percentile);
  const weaknesses = rows
    .map((row) => ({ ...row, level: weaknessLevel(row.percentile) }))
    .filter((row) => row.level)
    .sort((a, b) => a.percentile - b.percentile);
  return { strengths, weaknesses };
};

export function PlayerSearch({ query, results, visible, onQueryChange, onFocus, onSelect }) {
  return (
    <div className="relative z-[9999]">
      <label htmlFor="report-player-search" className="text-[11px] font-black uppercase tracking-[0.18em] text-[#8CC7A7]">Player search</label>
      <input
        id="report-player-search"
        className="nl-field mt-2 w-full rounded-lg px-4 py-3 text-sm font-semibold"
        placeholder="Search a player..."
        value={query}
        onChange={(event) => onQueryChange(event.target.value)}
        onFocus={onFocus}
      />
      {visible && query.trim().length >= 2 ? (
        <div className="absolute z-[9999] mt-2 max-h-80 w-full overflow-auto rounded-lg border border-white/10 bg-[#080B0A] shadow-2xl">
          {results.length ? results.map((player) => (
            <button
              key={`${player.id}-${player.player_season_id || "latest"}`}
              type="button"
              className="flex w-full items-center justify-between gap-3 border-b border-white/5 px-4 py-3 text-left text-sm text-slate-300 transition last:border-b-0 hover:bg-[#2F7D5C]/12 hover:text-white"
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => onSelect(player)}
            >
              <span className="min-w-0">
                <span className="block truncate font-black text-white">{player.name}</span>
                <span className="block truncate text-xs text-slate-500">{[player.team, player.competition_name, player.calendar].filter(Boolean).join(" - ")}</span>
              </span>
              <span className="shrink-0 rounded-md border border-[#3A8967]/30 bg-[#2F7D5C]/15 px-2 py-1 text-[10px] font-black uppercase tracking-[0.12em] text-[#8CC7A7]">Select</span>
            </button>
          )) : <div className="px-4 py-3 text-sm text-slate-500">No matches found.</div>}
        </div>
      ) : null}
    </div>
  );
}

export function SeasonSelector({ seasons, selectedSeasonId, onSelect }) {
  if (!seasons?.length) return null;
  return (
    <div className="flex gap-3 overflow-x-auto pb-2">
      {seasons.map((season) => {
        const active = String(season.player_season_id) === String(selectedSeasonId);
        return (
          <button
            key={season.player_season_id}
            type="button"
            onClick={() => onSelect(String(season.player_season_id))}
            className={`flex min-w-[210px] items-center gap-3 rounded-lg border px-3 py-3 text-left transition ${active ? "border-[#3A8967]/45 bg-[#2F7D5C]/18 shadow-sm" : "border-white/10 bg-white/[0.035] hover:border-[#3A8967]/35 hover:bg-white/[0.055]"}`}
          >
            <ClubLogo name={season.team} className="h-10 w-10 rounded-xl" />
            <span className="min-w-0">
              <span className={`block truncate text-sm font-black ${active ? "text-[#DDF3E8]" : "text-white"}`}>{season.team || "Unknown club"}</span>
              <span className="block truncate text-xs text-slate-500">{season.calendar || "-"} - {season.competition_name || "-"}</span>
            </span>
          </button>
        );
      })}
    </div>
  );
}

export function SeasonStatistics({ report, metrics }) {
  const player = report?.player || {};
  const stats = [
    { label: "Season", value: player.calendar, format: "text" },
    { label: "Club", value: player.team, format: "text" },
    { label: "Competition", value: player.competition_name, format: "text" },
    { label: "Minutes", value: player.minutes_played, format: "integer" },
    { label: "Matches", value: player.matches_played, format: "integer" },
    { label: "Goals", value: metrics.goals, format: "integer" },
    { label: "xG", value: metrics.xg, format: "number" },
    { label: "Assists", value: metrics.assists, format: "integer" },
    { label: "xA", value: metrics.xa, format: "number" },
    { label: "Prog passes /90", value: metrics.progressive_passes_per_90, format: "number" },
    { label: "Prog runs /90", value: metrics.progressive_runs_per_90, format: "number" },
    { label: "Def actions /90", value: metrics.successful_def_actions_per_90, format: "number" },
  ];
  return (
    <ReportCard className="border-[#3A8967]/25 bg-[#080B0A]">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-4">
        <div>
          <Kicker>Statistics</Kicker>
          <h2 className="mt-2 text-2xl font-black tracking-tight text-white">Season executive summary</h2>
        </div>
        <div className="rounded-lg border border-[#3A8967]/35 bg-[#2F7D5C]/15 px-5 py-3 text-right shadow-sm">
          <p className="text-[10px] font-black uppercase tracking-[0.16em] text-[#8CC7A7]">Score</p>
          <p className="text-4xl font-black tabular-nums text-[#DDF3E8]">{formatValue(player.global_score_adjusted, "score")}</p>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-6">
        {stats.map((stat) => (
          <div key={stat.label} className="min-w-0 rounded-lg border border-white/10 bg-white/[0.035] p-3">
            <p className="truncate text-[10px] font-black uppercase tracking-[0.18em] text-slate-500">{stat.label}</p>
            <p className="mt-2 truncate text-lg font-black tabular-nums text-white">{stat.format === "text" ? (stat.value || "-") : formatValue(stat.value, stat.format)}</p>
          </div>
        ))}
      </div>
    </ReportCard>
  );
}

export function PositionCard({ player }) {
  const meta = getPositionMeta(player?.assigned_role, player?.position);
  return (
    <ReportCard className="min-h-[360px]">
      <div className="flex items-start justify-between gap-3">
        <div>
          <Kicker>Position</Kicker>
          <h3 className="mt-2 text-xl font-black text-white">{meta.label}</h3>
        </div>
        <span className="rounded-md border border-[#3A8967]/35 bg-[#2F7D5C]/15 px-3 py-1 text-sm font-black text-[#8CC7A7]">{meta.short}</span>
      </div>
      <div className="relative mx-auto mt-5 h-[260px] max-w-[210px] overflow-hidden rounded-lg border border-[#3A8967]/25 bg-[#06100C] shadow-inner">
        <div className="absolute inset-3 rounded-lg border border-[#3A8967]/35" />
        <div className="absolute left-1/2 top-1/2 h-20 w-20 -translate-x-1/2 -translate-y-1/2 rounded-full border border-[#3A8967]/35" />
        <div className="absolute left-3 right-3 top-1/2 h-px bg-[#3A8967]/35" />
        <div className="absolute left-1/2 top-3 h-10 w-24 -translate-x-1/2 rounded-b-full border-x border-b border-[#3A8967]/35" />
        <div className="absolute bottom-3 left-1/2 h-10 w-24 -translate-x-1/2 rounded-t-full border-x border-t border-[#3A8967]/35" />
        <div
          className="absolute h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-[#559A78] shadow-[0_0_20px_rgba(85,154,120,0.35)]"
          style={{ left: `${meta.pitch.x}%`, top: `${meta.pitch.y}%` }}
        />
      </div>
      <p className="mt-4 text-sm leading-6 text-slate-600">Main scoring group for this season. Secondary positions can be layered later without changing the component contract.</p>
    </ReportCard>
  );
}

export function PlayerProfileCard({ categories }) {
  const insights = categories.map(insightForCategory).filter(Boolean).slice(0, 5);
  return (
    <ReportCard className="min-h-[360px]">
      <Kicker>Profile</Kicker>
      <h3 className="mt-2 text-xl font-black text-white">Performance dimensions</h3>
      <div className="mt-5 space-y-3">
        {categories.map((category) => {
          const score = clampPercentile(category.score);
          const tone = scoreBarTone(category.score);
          return (
            <div key={category.key}>
              <div className="mb-1 flex items-center justify-between text-xs">
                <span className="font-black uppercase tracking-[0.12em] text-slate-300">{category.label}</span>
                <span className="font-black tabular-nums" style={{ color: tone.color }}>{score !== null ? Math.round(score) : "-"}</span>
              </div>
              <div className="h-2.5 overflow-hidden rounded-full border border-white/10" style={{ backgroundColor: tone.track }}>
                <div className="h-full rounded-full transition-[width] duration-500 ease-out" style={{ width: `${score ?? 0}%`, backgroundColor: tone.color }} />
              </div>
            </div>
          );
        })}
      </div>
      <div className="mt-5 space-y-2 border-t border-white/10 pt-4">
        {(insights.length ? insights : ["Profile summary will improve as more percentile metrics become available."]).map((item) => (
          <p key={item} className="text-sm leading-5 text-slate-600">- {item}</p>
        ))}
      </div>
    </ReportCard>
  );
}

function CharacteristicList({ title, count, rows, type }) {
  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <p className="text-[11px] font-black uppercase tracking-[0.18em] text-[#8CC7A7]">{title}</p>
        <span className="rounded-md border border-white/10 bg-white/[0.035] px-2 py-0.5 text-xs font-black text-white">{count}</span>
      </div>
      <div className="space-y-2">
        {rows.length ? rows.slice(0, 5).map((row) => {
          return (
            <div key={`${type}-${row.key}`} className="flex items-center justify-between gap-3 rounded-lg border border-white/10 bg-white/[0.035] px-3 py-2">
              <div className="min-w-0">
                <p className="truncate text-sm font-bold text-white">{row.label}</p>
                <p className="text-xs text-slate-500">{formatValue(row.raw, getMetricConfig(row.key).format)} raw</p>
              </div>
              <span className={`shrink-0 rounded-md border px-2 py-1 text-[10px] font-black uppercase tracking-[0.1em] ${percentileBadgeStyle(row.percentile)}`}>{row.level}</span>
            </div>
          );
        }) : <p className="text-sm text-slate-500">No significant signal in this section.</p>}
      </div>
    </div>
  );
}

export function CharacteristicsCard({ characteristics }) {
  return (
    <ReportCard className="min-h-[360px]">
      <Kicker>Characteristics</Kicker>
      <h3 className="mt-2 text-xl font-black text-white">Strengths and weaknesses</h3>
      <div className="mt-5 grid gap-5 md:grid-cols-2 lg:grid-cols-1">
        <CharacteristicList title="Strengths" count={characteristics.strengths.length} rows={characteristics.strengths} type="strength" />
        <CharacteristicList title="Weaknesses" count={characteristics.weaknesses.length} rows={characteristics.weaknesses} type="weakness" />
      </div>
    </ReportCard>
  );
}

function MetricGroupCard({ groupKey, metrics, rawMode, context }) {
  const rows = metricRowsForGroup(metrics, groupKey, context);
  const group = metricGroups[groupKey];
  return (
    <ReportCard className="min-h-[310px]">
      <div className="mb-4 flex items-center justify-between">
        <h4 className="text-sm font-black uppercase tracking-[0.18em] text-white">{group.label}</h4>
        <span className="text-xs font-black text-slate-500">{rows.length}</span>
      </div>
      <div className="space-y-3">
        {rows.length ? rows.map((metric) => {
          const percentile = clampPercentile(metric.percentile);
          const tone = scoreBarTone(metric.percentile);
          return (
            <div key={metric.key}>
              <div className="mb-1 flex items-center justify-between gap-3 text-xs">
                <span className="min-w-0 truncate font-bold text-slate-300">{metric.label}</span>
                <span className="shrink-0 font-black tabular-nums" style={{ color: tone.color }}>{rawMode ? formatValue(metric.raw, metric.format) : formatValue(percentile, "integer")}</span>
              </div>
              <div className="h-2.5 overflow-hidden rounded-full border border-white/10" style={{ backgroundColor: tone.track }}>
                <div className="h-full rounded-full transition-[width] duration-500 ease-out" style={{ width: `${percentile ?? 0}%`, backgroundColor: tone.color }} />
              </div>
            </div>
          );
        }) : <p className="text-sm text-slate-500">No available metrics.</p>}
      </div>
    </ReportCard>
  );
}

export function AdvancedCharacteristics({ metrics, player, rawMode, setRawMode, context, setContext }) {
  const positionGroup = normalizePositionGroup(player?.assigned_role, player?.position);
  const positionMeta = getPositionMeta(player?.assigned_role, player?.position);
  const groupOrder = metricGroupOrderForPosition(positionGroup);
  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <Kicker>Advanced characteristics</Kicker>
          <h3 className="mt-2 text-2xl font-black text-white">Detailed metric percentiles</h3>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
            An 80 means the player is better than 80% of {positionMeta.label.toLowerCase()} in the selected {context === "league" ? "league" : "global"} context during {player?.calendar || "the selected season"}.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <div className="rounded-md border border-white/10 bg-white/[0.035] p-1">
            {["global", "league"].map((item) => (
              <button key={item} type="button" onClick={() => setContext(item)} className={`rounded-md px-3 py-1 text-xs font-black uppercase tracking-[0.12em] ${context === item ? "bg-[#2F7D5C]/35 text-[#DDF3E8]" : "text-slate-400 hover:text-white"}`}>{item}</button>
            ))}
          </div>
          <button type="button" onClick={() => setRawMode(!rawMode)} className={`rounded-md border px-4 py-2 text-xs font-black uppercase tracking-[0.14em] ${rawMode ? "border-[#3A8967]/35 bg-[#2F7D5C]/15 text-[#8CC7A7]" : "border-white/10 text-slate-400 hover:text-white"}`}>Raw values</button>
        </div>
      </div>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {groupOrder.map((groupKey) => (
          <MetricGroupCard key={groupKey} groupKey={groupKey} metrics={metrics} rawMode={rawMode} context={context} />
        ))}
      </div>
    </div>
  );
}

const RadarTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-white/10 bg-[#080B0A] px-3 py-2 text-xs shadow-xl">
      <p className="font-black text-white">{label}</p>
      {payload.map((entry) => (
        <p key={entry.dataKey} style={{ color: entry.color }}>{entry.name}: {formatValue(entry.value, "integer")}</p>
      ))}
    </div>
  );
};

const seasonRadarColors = ["#0f766e", "#2563eb", "#d97706", "#7c3aed", "#dc2626", "#0891b2"];

export function PlayerRadarComparison({ report, comparisonReport, context }) {
  const player = report?.player || {};
  const metrics = report?.metrics || {};
  const positionGroup = normalizePositionGroup(player.assigned_role, player.position);
  const comparison = comparisonReport?.player;
  const comparisonMetrics = comparisonReport?.metrics || {};
  const radarKeys = getRadarMetricKeys(positionGroup, report?.radar_metrics || []).filter((key) => metricAvailable(metrics, key));
  const sampleSize = averageSampleSize(report, context, positionGroup);
  const minMinutes = averageMinMinutes(report, context, positionGroup);
  const data = radarKeys.map((key) => ({
    metric: getMetricConfig(key).label,
    player: clampPercentile(metricPct(metrics, key, context)) ?? 0,
    comparison: comparison
      ? clampPercentile(metricPct(comparisonMetrics, key, context)) ?? 0
      : clampPercentile(averageMetric(report, context, positionGroup, key)?.percentile) ?? 0,
  }));
  const comparisonLabel = comparison?.name || "Cohort average";
  return (
    <ReportCard className="min-h-[460px]">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <Kicker>Visual comparison</Kicker>
          <h3 className="mt-2 text-xl font-black text-white">{player.name || "Player"} vs {comparisonLabel}</h3>
          {!comparison ? <p className="mt-1 text-xs font-semibold text-slate-500">Benchmark sample: {sampleSize || "-"} {positionGroup.toLowerCase()} with available metric data{minMinutes ? `, ${Math.round(Number(minMinutes))}+ minutes` : ""}.</p> : null}
        </div>
        <div className="flex items-center gap-3 text-xs font-black uppercase tracking-[0.12em]">
          <span className="text-[#8CC7A7]">Player</span>
          <span className="text-slate-500">{comparisonLabel}</span>
        </div>
      </div>
      <div className="mt-4 h-[360px]">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={data} outerRadius="82%">
            <PolarGrid stroke="rgba(255,255,255,0.12)" strokeDasharray="3 3" />
            <PolarAngleAxis dataKey="metric" tick={{ fill: "#A0A8A3", fontSize: 10 }} />
            <PolarRadiusAxis domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} tick={{ fill: "#6F7772", fontSize: 9 }} axisLine={false} tickLine={false} />
            <Tooltip content={<RadarTooltip />} />
            <Radar name={comparisonLabel} dataKey="comparison" stroke="#94a3b8" fill="rgba(148,163,184,0.18)" fillOpacity={0.45} strokeWidth={1.5} />
            <Radar name={player.name || "Player"} dataKey="player" stroke="#559A78" fill="rgba(85,154,120,0.18)" fillOpacity={0.55} strokeWidth={2.5} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </ReportCard>
  );
}

export function PlayerSeasonRadarComparison({ report, context }) {
  const player = report?.player || {};
  const metrics = report?.metrics || {};
  const positionGroup = normalizePositionGroup(player.assigned_role, player.position);
  const radarKeys = getRadarMetricKeys(positionGroup, report?.radar_metrics || []).filter((key) => metricAvailable(metrics, key));
  const seasons = (report?.season_metric_history || []).slice(-4);
  const data = radarKeys.map((key) => {
    const row = { metric: getMetricConfig(key).label };
    seasons.forEach((season, index) => {
      row[`s${index}`] = clampPercentile(metricPct(season.metrics || {}, key, context));
    });
    return row;
  });
  return (
    <ReportCard className="min-h-[420px]">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <Kicker>Season radar</Kicker>
          <h3 className="mt-2 text-xl font-black text-white">Metric profile by season</h3>
          <p className="mt-1 text-xs font-semibold text-slate-500">Latest 4 available seasons, using {context} percentiles.</p>
        </div>
        <div className="flex max-w-sm flex-wrap justify-end gap-2">
          {seasons.map((season, index) => (
            <span key={season.player_season_id || `${season.calendar}-${index}`} className="inline-flex items-center gap-1.5 rounded-md border border-white/10 bg-white/[0.035] px-2 py-1 text-[10px] font-black uppercase tracking-[0.1em] text-slate-400">
              <span className="h-2 w-2 rounded-full" style={{ backgroundColor: seasonRadarColors[index % seasonRadarColors.length] }} />
              {season.calendar || `S${index + 1}`}
            </span>
          ))}
        </div>
      </div>
      <div className="mt-4 h-[320px]">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={data} outerRadius="82%">
            <PolarGrid stroke="rgba(255,255,255,0.12)" strokeDasharray="3 3" />
            <PolarAngleAxis dataKey="metric" tick={{ fill: "#A0A8A3", fontSize: 10 }} />
            <PolarRadiusAxis domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} tick={{ fill: "#6F7772", fontSize: 9 }} axisLine={false} tickLine={false} />
            <Tooltip content={<RadarTooltip />} />
            {seasons.map((season, index) => {
              const color = seasonRadarColors[index % seasonRadarColors.length];
              return (
                <Radar
                  key={season.player_season_id || `${season.calendar}-${index}`}
                  name={`${season.calendar || `Season ${index + 1}`}${season.team ? ` - ${season.team}` : ""}`}
                  dataKey={`s${index}`}
                  stroke={color}
                  fill={color}
                  fillOpacity={index === seasons.length - 1 ? 0.18 : 0.08}
                  strokeWidth={index === seasons.length - 1 ? 2.4 : 1.6}
                  connectNulls
                />
              );
            })}
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </ReportCard>
  );
}

export function PlayerStatsComparison({ report, comparisonReport, context, onSearch, query, results, showResults, onSelect, loading }) {
  const player = report?.player || {};
  const positionGroup = normalizePositionGroup(player.assigned_role, player.position);
  const keys = getRadarMetricKeys(positionGroup, report?.radar_metrics || []).filter((key) => metricAvailable(report?.metrics || {}, key));
  const comparison = comparisonReport?.player;
  const comparisonLabel = comparison?.name || "Cohort average";
  const sampleSize = averageSampleSize(report, context, positionGroup);
  const minMinutes = averageMinMinutes(report, context, positionGroup);
  return (
    <ReportCard className="min-h-[460px]">
      <Kicker>Stats comparison</Kicker>
      <h3 className="mt-2 text-xl font-black text-white">{player.name || "Player"} vs {comparisonLabel}</h3>
      {!comparison ? <p className="mt-1 text-xs font-semibold text-slate-500">Default comparison uses average raw values from {sampleSize || "-"} {positionGroup.toLowerCase()} with available metric data{minMinutes ? `, ${Math.round(Number(minMinutes))}+ minutes` : ""}.</p> : null}
      <div className="relative mt-4">
        <input
          className="nl-field w-full rounded-lg px-4 py-3 text-sm font-semibold"
          placeholder="Compare with a player..."
          value={query}
          onChange={(event) => onSearch(event.target.value)}
        />
        {showResults ? (
          <div className="absolute z-40 mt-2 max-h-72 w-full overflow-auto rounded-lg border border-white/10 bg-[#080B0A] shadow-xl">
            {results.length ? results.map((item) => (
              <button key={`${item.id}-${item.player_season_id}`} type="button" onMouseDown={(event) => event.preventDefault()} onClick={() => onSelect(item)} className="block w-full border-b border-white/5 px-4 py-3 text-left text-sm text-slate-300 last:border-b-0 hover:bg-[#2F7D5C]/12 hover:text-white">
                <span className="block font-black text-white">{item.name}</span>
                <span className="block text-xs text-slate-500">{[item.team, item.competition_name, item.calendar].filter(Boolean).join(" - ")}</span>
              </button>
            )) : <p className="px-4 py-3 text-sm text-slate-500">No matches found.</p>}
          </div>
        ) : null}
      </div>
      {loading ? <p className="mt-3 text-sm text-slate-500">Loading comparison...</p> : null}
      <div className="mt-5 overflow-hidden rounded-lg border border-white/10">
        <div className="grid grid-cols-[1.15fr_0.8fr_0.8fr] bg-white/[0.035] px-3 py-2 text-[10px] font-black uppercase tracking-[0.14em] text-[#8CC7A7]">
          <span>Stat</span><span className="text-right">{player.name || "Player A"}</span><span className="text-right">{comparisonLabel}</span>
        </div>
        {keys.map((key) => {
          const config = getMetricConfig(key);
          const a = report?.metrics?.[key];
          const b = comparison ? comparisonReport?.metrics?.[key] : averageMetric(report, context, positionGroup, key)?.raw;
          const aNum = Number(a);
          const bNum = Number(b);
          const comparable = Number.isFinite(aNum) && Number.isFinite(bNum);
          const lower = config.lowerIsBetter || lowerIsBetter.has(key);
          const aBetter = comparable && (lower ? aNum < bNum : aNum > bNum);
          const bBetter = comparable && (lower ? bNum < aNum : bNum > aNum);
          return (
            <div key={key} className="grid grid-cols-[1.15fr_0.8fr_0.8fr] border-t border-white/10 px-3 py-2 text-sm">
              <span className="min-w-0 truncate font-semibold text-slate-300">{config.label}</span>
              <span className={`text-right font-black tabular-nums ${aBetter ? "text-[#8CC7A7]" : bBetter ? "text-rose-300" : "text-white"}`}>{formatValue(a, config.format)}</span>
              <span className={`text-right font-black tabular-nums ${bBetter ? "text-[#8CC7A7]" : aBetter ? "text-rose-300" : "text-white"}`}>{formatValue(b, config.format)}</span>
            </div>
          );
        })}
      </div>
      {comparison ? <p className="mt-3 text-xs text-slate-500">Comparison season: {comparison.calendar || "fallback"} - {comparison.team || "-"}</p> : null}
    </ReportCard>
  );
}

export function SimilarPlayersCard({ similarities, loading, onOpen }) {
  const sortedSimilarities = [...(similarities || [])].sort((a, b) => {
    const aScore = Number(a.global_score_adjusted);
    const bScore = Number(b.global_score_adjusted);
    if (Number.isFinite(aScore) && Number.isFinite(bScore)) return bScore - aScore;
    if (Number.isFinite(aScore)) return -1;
    if (Number.isFinite(bScore)) return 1;
    return String(a.player_b_name || "").localeCompare(String(b.player_b_name || ""));
  });
  return (
    <ReportCard>
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <Kicker>Similar players</Kicker>
          <h3 className="mt-2 text-xl font-black text-white">Top 10 statistical neighbors</h3>
          <p className="mt-1 text-xs font-semibold text-slate-500">Sorted by score, descending. Click a row to open the report in a new tab.</p>
        </div>
        <span className="rounded-md border border-[#3A8967]/30 bg-[#2F7D5C]/15 px-3 py-1 text-xs font-black text-[#8CC7A7]">{sortedSimilarities.length}</span>
      </div>
      {loading ? <p className="text-sm text-slate-500">Loading similar players...</p> : null}
      {!loading && !sortedSimilarities.length ? <p className="text-sm text-slate-500">No similar players found.</p> : null}
      {sortedSimilarities.length ? (
        <div className="space-y-2">
          <div className="hidden grid-cols-[52px_minmax(0,1.5fr)_minmax(0,1fr)_minmax(0,1fr)_90px_90px] gap-3 px-4 text-[10px] font-black uppercase tracking-[0.14em] text-slate-500 md:grid">
            <span>#</span>
            <span>Player</span>
            <span>Club</span>
            <span>Context</span>
            <span className="text-right">Age</span>
            <span className="text-right">Score</span>
          </div>
          {sortedSimilarities.map((sim, index) => {
            const tmFields = sim.tm_fields || {};
            const photoUrl = toAbsoluteUrl(
              tmFields.app_photo_url || tmFields.tm_profile_image_url || tmFields.profile_image_url
            );
            return (
              <button
                key={`${sim.player_b_id}-${sim.player_b_name}`}
                type="button"
                onClick={() => onOpen(sim)}
                className="grid w-full gap-3 rounded-lg border border-white/10 bg-white/[0.025] p-4 text-left shadow-sm transition hover:-translate-y-0.5 hover:border-[#3A8967]/35 hover:bg-[#2F7D5C]/10 hover:shadow-md md:grid-cols-[52px_minmax(0,1.5fr)_minmax(0,1fr)_minmax(0,1fr)_90px_90px] md:items-center"
              >
                <div className="flex h-9 w-9 items-center justify-center rounded-md border border-[#3A8967]/35 bg-[#2F7D5C]/18 text-sm font-black text-[#8CC7A7]">
                  {index + 1}
                </div>
                <div className="flex min-w-0 items-center gap-3">
                  {photoUrl ? (
                    <img src={photoUrl} alt={sim.player_b_name || "Player"} className="h-12 w-12 shrink-0 rounded-md border border-white/10 object-cover" loading="lazy" />
                  ) : (
                    <span className="flex h-12 w-12 shrink-0 items-center justify-center rounded-md border border-white/10 bg-white/[0.04] text-sm font-black text-slate-300">
                      {getInitials(sim.player_b_name)}
                    </span>
                  )}
                  <div className="min-w-0">
                    <p className="truncate text-sm font-black text-white">{sim.player_b_name || "-"}</p>
                    <p className="mt-1 truncate text-xs font-semibold text-slate-500">{sim.profile || "Similar profile"}</p>
                  </div>
                </div>
                <div className="flex min-w-0 items-center gap-2">
                  <ClubLogo name={sim.team} className="h-9 w-9 rounded-md" />
                  <span className="truncate text-sm font-bold text-slate-300">{sim.team || "-"}</span>
                </div>
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold text-slate-300">{sim.competition_name || "-"}</p>
                  <p className="mt-1 truncate text-xs text-slate-500">{sim.calendar || "-"}</p>
                </div>
                <div className="rounded-md bg-white/[0.035] p-2 md:bg-transparent md:p-0 md:text-right">
                  <p className="text-[10px] font-black uppercase tracking-[0.12em] text-slate-400 md:hidden">Age</p>
                  <p className="font-black text-white">{formatValue(sim.age, "integer")}</p>
                </div>
                <div className="rounded-md bg-[#2F7D5C]/15 p-2 md:bg-transparent md:p-0 md:text-right">
                  <p className="text-[10px] font-black uppercase tracking-[0.12em] text-slate-400 md:hidden">Score</p>
                  <p className="text-lg font-black text-[#8CC7A7]">{formatValue(sim.global_score_adjusted, "score")}</p>
                </div>
              </button>
            );
          })}
        </div>
      ) : null}
    </ReportCard>
  );
}

export const availableMetricsSummary = (metrics = {}) => {
  const allConfigured = metricGroupOrder.flatMap((groupKey) => metricGroups[groupKey].metrics.map((metric) => metric.key));
  const available = allConfigured.filter((key) => metricAvailable(metrics, key));
  const missing = allConfigured.filter((key) => !metricAvailable(metrics, key));
  return { available, missing };
};
