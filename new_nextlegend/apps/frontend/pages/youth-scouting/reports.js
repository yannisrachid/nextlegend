import { useRouter } from "next/router";
import { useEffect, useMemo, useRef, useState } from "react";
import ClubLogo from "@/components/ClubLogo";
import {
  AdvancedCharacteristics,
  availableMetricsSummary,
  buildCharacteristics,
  buildProfileCategories,
  CharacteristicsCard,
  PlayerProfileCard,
  PlayerRadarComparison,
  PlayerSearch,
  PlayerSeasonRadarComparison,
  PlayerStatsComparison,
  PositionCard,
  ReportCard,
  SeasonSelector,
  SeasonStatistics,
  SimilarPlayersCard,
  formatValue,
} from "@/components/report/PlayerReportComponents";
import { apiUrl, deleteJson, fetchJson, patchJson, postForm, postJson } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { normalizePositionGroup } from "@/lib/reportMetrics";

const DEFAULT_SEASON = 2027;

const selectedLabel = (item) =>
  [item?.name || item?.display_name, item?.team || item?.club_name, item?.competition_name || item?.championship, item?.calendar || formatYouthSeason(item?.season)]
    .filter(Boolean)
    .join(" - ");

const formatYouthSeason = (season) => {
  const numeric = Number(season);
  return Number.isFinite(numeric) ? `${numeric - 1}/${numeric}` : season || "-";
};

const YOUTH_TO_REPORT_METRICS = [
  ["rating", "rating"],
  ["minutes_played", "minutes_played"],
  ["games_count", "games_count"],
  ["total_goals", "goals"],
  ["total_assists", "assists"],
  ["goals_per_90", "goals_per_90"],
  ["assists_per_90", "assists_per_90"],
  ["shots_per_90", "shots_per_90"],
  ["shots_on_target_per_90", "shots_on_target_per_90"],
  ["shots_inside_area_per_90", "shots_inside_area_per_90"],
  ["shots_outside_area_per_90", "shots_outside_area_per_90"],
  ["key_passes_per_90", "key_passes_per_90"],
  ["crosses_total_per_90", "crosses_per_90"],
  ["crosses_success_per_90", "accurate_crosses_per_90"],
  ["passes_total_per_90", "passes_per_90"],
  ["passes_success_per_90", "accurate_passes_per_90"],
  ["passes_accuracy_pct", "accurate_passes_percent"],
  ["forward_passes_total_per_90", "forward_passes_per_90"],
  ["forward_passes_success_per_90", "progressive_passes_per_90"],
  ["takeons_total_per_90", "dribbles_per_90"],
  ["takeons_success_per_90", "successful_dribbles_per_90"],
  ["tackles_total_per_90", "def_duels_per_90"],
  ["tackles_success_per_90", "successful_def_actions_per_90"],
  ["aerial_duels_total_per_90", "aerial_duels_per_90"],
  ["aerial_duels_success_per_90", "successful_aerial_duels_per_90"],
  ["recoveries_per_90", "recoveries_per_90"],
  ["interceptions_per_90", "interceptions_per_90"],
  ["clearances_per_90", "clearances_per_90"],
  ["blocks_per_90", "blocked_shots_per_90"],
  ["goals_conceded_per_90", "goals_conceded_per_90"],
  ["catches_per_90", "exits_per_90"],
  ["punches_per_90", "punches_per_90"],
  ["goal_kicks_success_per_90", "successful_goal_kicks_per_90"],
  ["goal_kicks_total_per_90", "goal_kicks_per_90"],
  ["aerial_clearances_success_per_90", "aerial_clearances_success_per_90"],
  ["aerial_clearances_total_per_90", "aerial_duels_gk_per_90"],
];

const DERIVED_METRICS = [
  {
    key: "shots_on_target_percent",
    raw: (m) => ratio(m.shots_on_target_per_90, m.shots_per_90),
    percentileKeys: ["shots_on_target_per_90"],
  },
  {
    key: "goal_conversion_rate",
    raw: (m) => ratio(m.goals_per_90, m.shots_per_90),
    percentileKeys: ["goals_per_90", "shots_on_target_per_90"],
  },
  {
    key: "accurate_crosses_percent",
    raw: (m) => ratio(m.crosses_success_per_90, m.crosses_total_per_90),
    percentileKeys: ["crosses_success_per_90"],
  },
  {
    key: "successful_dribbles_percent",
    raw: (m) => ratio(m.takeons_success_per_90, m.takeons_total_per_90),
    percentileKeys: ["takeons_success_per_90"],
  },
  {
    key: "accurate_progressive_passes_percent",
    raw: (m) => ratio(m.forward_passes_success_per_90, m.forward_passes_total_per_90),
    percentileKeys: ["forward_passes_success_per_90"],
  },
  {
    key: "def_duels_won_percent",
    raw: (m) => ratio(m.tackles_success_per_90, m.tackles_total_per_90),
    percentileKeys: ["tackles_success_per_90"],
  },
  {
    key: "aerial_duels_won_percent",
    raw: (m) => ratio(m.aerial_duels_success_per_90, m.aerial_duels_total_per_90),
    percentileKeys: ["aerial_duels_success_per_90"],
  },
  {
    key: "interceptions_padj",
    raw: (m) => m.interceptions_per_90,
    percentileKeys: ["interceptions_per_90"],
  },
];

const EXTRA_RADAR_METRICS = [
  "rating",
  "shots_on_target_per_90",
  "shots_inside_area_per_90",
  "recoveries_per_90",
  "clearances_per_90",
  "blocked_shots_per_90",
];

const YOUTH_UNAVAILABLE_METRICS = new Set(["xg", "xa", "xg_per_90", "xa_per_90"]);

const SCOUTING_DETAIL_RATING_FIELDS = [
  ["technical_rating", "Technical"],
  ["physical_rating", "Physical"],
  ["tactical_rating", "Tactical"],
  ["mental_rating", "Mental"],
];

const MATCH_POSITIONS = [
  "GK",
  "RB",
  "CB",
  "LB",
  "RWB",
  "LWB",
  "DM",
  "CM",
  "AM",
  "RW",
  "LW",
  "ST",
];

const MATCH_POSITION_ALIASES = {
  G: "GK",
  GK: "GK",
  RB: "RB",
  RCB: "CB",
  CB: "CB",
  LCB: "CB",
  LB: "LB",
  RWB: "RWB",
  LWB: "LWB",
  DM: "DM",
  DMF: "DM",
  RDMF: "DM",
  LDMF: "DM",
  CM: "CM",
  CMF: "CM",
  RCMF: "CM",
  LCMF: "CM",
  AM: "AM",
  AMF: "AM",
  RAMF: "AM",
  LAMF: "AM",
  RW: "RW",
  LW: "LW",
  CF: "ST",
  ST: "ST",
  SS: "ST",
  FW: "ST",
  ATT: "ST",
};

const QUALITATIVE_TAGS_BY_POSITION = {
  GK: [
    "SHOT-STOPPER",
    "COMMANDING",
    "SAVIOR",
    "PENALTY KILLER",
    "CLEAN SHEET",
    "SWEEPER KEEPER",
    "RELIABLE HANDS",
    "BIG SAVE",
    "SHAKY",
    "ERROR-PRONE",
    "COOL UNDER PRESSURE",
    "DISTRIBUTOR",
  ],
  CB: [
    "SOLID",
    "DOMINANT",
    "AERIAL THREAT",
    "INTERCEPTOR",
    "BRICK WALL",
    "CLEAN SHEET HERO",
    "COMPOSED ON BALL",
    "LAST-DITCH TACKLE",
    "CAUGHT OUT",
    "ERROR LEADING TO GOAL",
    "LEADER",
    "NO-NONSENSE",
  ],
  FULLBACK: [
    "OVERLAPPING",
    "ENERGETIC",
    "SOLID DEFENSIVELY",
    "CROSSING THREAT",
    "RELENTLESS",
    "CAUGHT OUT OF POSITION",
    "ONE-ON-ONE SPECIALIST",
    "ASSIST PROVIDER",
    "TIRELESS",
    "EXPOSED",
  ],
  WINGBACK: [
    "EXPLOSIVE",
    "ENGINE",
    "CROSSING THREAT",
    "BOX-TO-BOX",
    "OVERLAPPING",
    "DEFENSIVE LIABILITY",
    "ASSIST PROVIDER",
    "RELENTLESS",
    "GAME CHANGER",
    "TIRELESS",
  ],
  DM: [
    "SHIELD",
    "INTERCEPTOR",
    "DISCIPLINED",
    "BALL WINNER",
    "COMPOSED",
    "DEEP PLAYMAKER",
    "PRESSING MACHINE",
    "CARELESS ON BALL",
    "ANCHOR",
    "BOOKED / RECKLESS",
  ],
  CM: [
    "ENGINE",
    "BOX-TO-BOX",
    "PLAYMAKER",
    "CONSISTENT",
    "VISIONARY",
    "WORK RATE",
    "GAME CONTROLLER",
    "SLOPPY PASSING",
    "INEFFECTIVE",
    "TWO-WAY THREAT",
  ],
  AM: [
    "CREATIVE",
    "VISIONARY",
    "DECISIVE",
    "CLINICAL",
    "KEY PASS MACHINE",
    "UNPREDICTABLE",
    "GAME CHANGER",
    "QUIET / INVISIBLE",
    "SKILLFUL",
    "LAST-THIRD THREAT",
  ],
  WINGER: [
    "EXPLOSIVE",
    "SKILLFUL",
    "DRIBBLER",
    "FLASHY",
    "CLINICAL",
    "ASSIST MACHINE",
    "UNPREDICTABLE",
    "WASTEFUL",
    "ONE-ON-ONE THREAT",
    "CUT-INSIDE THREAT",
  ],
  ST: [
    "CLINICAL",
    "POACHER",
    "SCORER",
    "HOLD-UP PLAY",
    "AERIAL THREAT",
    "RELENTLESS",
    "WASTEFUL",
    "ISOLATED",
    "GAME CHANGER",
    "COLD-BLOODED",
  ],
};

function normalizeMatchPosition(position) {
  const key = String(position || "").trim().toUpperCase();
  return MATCH_POSITION_ALIASES[key] || (MATCH_POSITIONS.includes(key) ? key : "");
}

function qualitativeTagKey(position) {
  const normalized = normalizeMatchPosition(position);
  if (normalized === "RB" || normalized === "LB") return "FULLBACK";
  if (normalized === "RWB" || normalized === "LWB") return "WINGBACK";
  if (normalized === "RW" || normalized === "LW") return "WINGER";
  return normalized;
}

function qualitativeTagsForPosition(position) {
  return QUALITATIVE_TAGS_BY_POSITION[qualitativeTagKey(position)] || [];
}

function parseQualitativeTags(value) {
  if (Array.isArray(value)) {
    return value.map((tag) => String(tag).trim().toUpperCase()).filter(Boolean);
  }
  return String(value || "")
    .split(/[,;]/)
    .map((tag) => tag.trim().toUpperCase())
    .filter(Boolean);
}

function filterTagsForPosition(tags, position) {
  const allowed = new Set(qualitativeTagsForPosition(position));
  return parseQualitativeTags(tags).filter((tag) => allowed.has(tag));
}

const todayIso = () => new Date().toISOString().slice(0, 10);

const initials = (name) => String(name || "?")
  .split(/\s+/)
  .filter(Boolean)
  .slice(0, 2)
  .map((part) => part[0])
  .join("")
  .toUpperCase() || "?";

const scoutPhotoUrl = (photoKey) => {
  const raw = String(photoKey || "").trim();
  if (!raw) return "";
  if (/^(https?:|data:|blob:)/i.test(raw)) return raw;
  if (raw.startsWith("/hd-players/files/")) return apiUrl(raw);
  return apiUrl(`/hd-players/files/${encodeURIComponent(raw)}`);
};

const reporterDisplayName = (username, users = []) => {
  const key = String(username || "").trim().toLowerCase();
  if (!key) return "";
  const found = users.find((user) => String(user.username || "").toLowerCase() === key);
  return found?.display_name || username;
};

const canManageScoutingReport = (item, me) => {
  if (!item) return Boolean(me?.username);
  const username = String(me?.username || "").toLowerCase();
  return username && (String(item.scout || "").toLowerCase() === username || (username === "yrachid" && me?.role === "admin"));
};

const emptyMatch = (report) => ({
  source: "nextlegend_ui",
  team_a: "",
  score_a: "",
  score_b: "",
  team_b: "",
  competition: "",
  match_date: todayIso(),
  player_rating: "",
  minutes_played: "",
  position: normalizeMatchPosition(report?.player?.position),
  observations: "",
  qualitative_tags: [],
});

const emptyScoutingForm = (report) => ({
  player_name: report?.player?.name || "",
  club: report?.player?.team || "",
  year_of_birth: report?.player?.birth_year || "",
  position: report?.player?.position || "",
  nationality: report?.player?.nationality || "",
  portal_url: report?.source?.player?.provider_player_url || "",
  scout: "",
  star_rating: "",
  potential_star_rating: "",
  technical_rating: "",
  physical_rating: "",
  tactical_rating: "",
  mental_rating: "",
  potential_rating: "",
  overall_rating: "",
  technical_notes: "",
  physical_notes: "",
  tactical_notes: "",
  mental_notes: "",
  game_intelligence: "",
  strengths: "",
  weaknesses: "",
  development_projection: "",
  comparison: "",
  overall_comments: "",
  matches_observed: [],
  photo_key: "",
  photo_url: "",
});

const valueExists = (value) => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));

function ratio(numerator, denominator) {
  const top = Number(numerator);
  const bottom = Number(denominator);
  if (!Number.isFinite(top) || !Number.isFinite(bottom) || bottom <= 0) return null;
  return (top / bottom) * 100;
}

function average(values) {
  const numeric = values.map(Number).filter(Number.isFinite);
  if (!numeric.length) return null;
  return numeric.reduce((sum, value) => sum + value, 0) / numeric.length;
}

function reportSeasonId(season) {
  return season?.player_season_id ?? season?.id ?? season?.player_id;
}

function sortReportSeasons(seasons = []) {
  const bySeason = new Map();
  seasons.forEach((season) => {
    const seasonId = reportSeasonId(season);
    if (!seasonId) return;
    const seasonKey = Number(season.season);
    const key = Number.isFinite(seasonKey) ? String(seasonKey) : String(season.calendar || seasonId);
    const existing = bySeason.get(key);
    if (!existing) {
      bySeason.set(key, season);
      return;
    }
    const existingMinutes = Number(existing.minutes_played ?? -1);
    const currentMinutes = Number(season.minutes_played ?? -1);
    const existingScore = Number(existing.global_score_adjusted ?? existing.score ?? -1);
    const currentScore = Number(season.global_score_adjusted ?? season.score ?? -1);
    if (
      currentMinutes > existingMinutes ||
      (currentMinutes === existingMinutes && currentScore > existingScore) ||
      (currentMinutes === existingMinutes && currentScore === existingScore && Number(seasonId) > Number(reportSeasonId(existing)))
    ) {
      bySeason.set(key, season);
    }
  });
  return Array.from(bySeason.values()).sort((a, b) => {
    const seasonDiff = Number(b.season || 0) - Number(a.season || 0);
    if (seasonDiff) return seasonDiff;
    return Number(reportSeasonId(b) || 0) - Number(reportSeasonId(a) || 0);
  });
}

function pctFor(percentiles, youthKey, context = "global") {
  const row = percentiles?.[youthKey] || {};
  return context === "league" ? row.championship ?? row.global_position : row.global_position;
}

function setReportMetric(target, key, raw, globalPct, leaguePct) {
  if (valueExists(raw)) target[key] = Number(raw);
  if (valueExists(globalPct)) target[`${key}_pct_global`] = Number(globalPct);
  if (valueExists(leaguePct)) target[`${key}_pct_league`] = Number(leaguePct);
}

function buildMetrics(report) {
  const sourceMetrics = report?.metrics || {};
  const sourcePercentiles = report?.metric_percentiles || {};
  const output = {};
  YOUTH_TO_REPORT_METRICS.forEach(([sourceKey, targetKey]) => {
    setReportMetric(
      output,
      targetKey,
      sourceMetrics[sourceKey],
      pctFor(sourcePercentiles, sourceKey, "global"),
      pctFor(sourcePercentiles, sourceKey, "league")
    );
  });
  DERIVED_METRICS.forEach((metric) => {
    const globalPct = average(metric.percentileKeys.map((key) => pctFor(sourcePercentiles, key, "global")));
    const leaguePct = average(metric.percentileKeys.map((key) => pctFor(sourcePercentiles, key, "league")));
    setReportMetric(output, metric.key, metric.raw(sourceMetrics), globalPct, leaguePct);
  });
  output.xg = null;
  output.xa = null;
  output.xg_per_90 = null;
  output.xa_per_90 = null;
  output.progressive_runs_per_90 = null;
  output.touches_in_penalty_area_per_90 = sourceMetrics.shots_inside_area_per_90 ?? null;
  output.touches_in_penalty_area_per_90_pct_global = pctFor(sourcePercentiles, "shots_inside_area_per_90", "global");
  output.touches_in_penalty_area_per_90_pct_league = pctFor(sourcePercentiles, "shots_inside_area_per_90", "league");
  return output;
}

function buildAverageContexts(report, adaptedPositionGroup) {
  const sourceContexts = report?.average_contexts || {};
  const output = { global: {}, league: {} };
  ["global", "league"].forEach((context) => {
    const source = sourceContexts?.[context]?.[report?.player?.position_group] || {};
    const sourceMetrics = source.metrics || {};
    const metrics = {};
    YOUTH_TO_REPORT_METRICS.forEach(([sourceKey, targetKey]) => {
      const row = sourceMetrics[sourceKey] || {};
      metrics[targetKey] = {
        raw: row.raw ?? null,
        percentile: row.percentile ?? null,
      };
    });
    DERIVED_METRICS.forEach((metric) => {
      const rawValues = Object.fromEntries(
        Object.entries(sourceMetrics).map(([key, row]) => [key, row?.raw])
      );
      metrics[metric.key] = {
        raw: metric.raw(rawValues),
        percentile: average(metric.percentileKeys.map((key) => sourceMetrics[key]?.percentile)),
      };
    });
    metrics.touches_in_penalty_area_per_90 = {
      raw: sourceMetrics.shots_inside_area_per_90?.raw ?? null,
      percentile: sourceMetrics.shots_inside_area_per_90?.percentile ?? null,
    };
    output[context][adaptedPositionGroup] = {
      sample_size: source.sample_size || 0,
      min_minutes: source.min_minutes ?? null,
      metrics,
    };
  });
  return output;
}

function adaptPlayerRow(player) {
  if (!player) return null;
  const adaptedPositionGroup = player.position_group || normalizePositionGroup(null, player.position);
  return {
    id: player.id,
    player_id: player.id,
    player_season_id: player.id,
    name: player.display_name,
    team: player.club_name,
    competition_name: player.championship,
    calendar: player.calendar || formatYouthSeason(player.season || DEFAULT_SEASON),
    season: player.season,
    assigned_role: adaptedPositionGroup,
    position: player.primary_position || player.position,
    age: player.age,
    birth_year: player.birth_year,
    birth_date: player.birth_date,
    nationality: player.nationality_label || player.nationality_code,
    minutes_played: player.minutes_played,
    matches_played: player.games_count,
    global_score_adjusted: player.score,
  };
}

function buildSeasonMetricRow(player) {
  const adaptedPlayer = adaptPlayerRow(player);
  if (!adaptedPlayer) return null;
  return {
    ...adaptedPlayer,
    metrics: buildMetrics({
      metrics: player.metrics || {},
      metric_percentiles: player.metric_percentiles || {},
    }),
  };
}

function filterYouthMetricSummary(summary) {
  return {
    available: (summary?.available || []).filter((key) => !YOUTH_UNAVAILABLE_METRICS.has(key)),
    missing: (summary?.missing || []).filter((key) => !YOUTH_UNAVAILABLE_METRICS.has(key)),
  };
}

function buildAdaptedReport(rawReport) {
  if (!rawReport?.player) return null;
  const player = rawReport.player;
  const adaptedPositionGroup = player.position_group || normalizePositionGroup(null, player.position);
  const metrics = buildMetrics(rawReport);
  const adaptedPlayer = adaptPlayerRow(player);
  const availableSeasons = sortReportSeasons((rawReport.available_seasons || [player]).map(adaptPlayerRow).filter(Boolean));
  const seasonMetricHistory = (rawReport.season_metric_history || rawReport.score_history || [player]).map(buildSeasonMetricRow).filter(Boolean);
  const scoreHistory = sortReportSeasons((rawReport.score_history || [player]).map(adaptPlayerRow).filter(Boolean)).reverse();
  const radarMetrics = Array.from(new Set([
    ...YOUTH_TO_REPORT_METRICS.map(([, target]) => target),
    ...DERIVED_METRICS.map((item) => item.key),
    ...EXTRA_RADAR_METRICS,
  ])).filter((key) => !YOUTH_UNAVAILABLE_METRICS.has(key) && (valueExists(metrics[key]) || valueExists(metrics[`${key}_pct_global`])));
  return {
    player: adaptedPlayer,
    metrics,
    radar_metrics: radarMetrics,
    available_seasons: availableSeasons,
    season_metric_history: seasonMetricHistory,
    score_history: scoreHistory,
    score_snapshots: [],
    average_contexts: buildAverageContexts(rawReport, adaptedPositionGroup),
    source: rawReport,
  };
}

function adaptSearchRows(rows = []) {
  return rows.map((item) => ({
    ...item,
    id: item.id,
    player_season_id: item.id,
    name: item.display_name,
    team: item.club_name,
    competition_name: item.championship,
    calendar: item.calendar || formatYouthSeason(item.season || DEFAULT_SEASON),
  }));
}

function adaptSimilarRows(rows = []) {
  return rows.map((item) => ({
    ...item,
    player_b_id: item.id,
    player_b_name: item.display_name,
    team: item.club_name,
    competition_name: item.championship,
    calendar: item.calendar || formatYouthSeason(item.season || DEFAULT_SEASON),
    age: item.birth_year && item.season ? Number(item.season) - Number(item.birth_year) : null,
    profile: [item.position_group, item.age_category, item.nationality_label || item.nationality_code].filter(Boolean).join(" - "),
    global_score_adjusted: item.score,
  }));
}

function YouthScoreHistory({ data, onSelect }) {
  const sorted = [...(data || [])].sort((a, b) => Number(a.season || 0) - Number(b.season || 0));
  const values = sorted.map((row) => Number(row.global_score_adjusted)).filter(Number.isFinite);
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 100;
  const range = Math.max(1, max - min);
  const points = sorted.map((row, index) => {
    const score = Number(row.global_score_adjusted);
    const x = sorted.length <= 1 ? 50 : 12 + (index / (sorted.length - 1)) * 76;
    const y = Number.isFinite(score) ? 82 - ((score - min) / range) * 56 : 82;
    return { ...row, x, y, score };
  });
  const path = points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
  const grid = [26, 40, 54, 68, 82];
  return (
    <ReportCard>
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="nl-kicker">Score history</p>
          <h3 className="mt-2 text-xl font-black text-white">Youth rating evolution</h3>
          <p className="mt-1 text-sm text-[#A0A8A3]">Season-by-season evolution for this player across imported Eyeball campaigns.</p>
        </div>
      </div>
      <div className="rounded-lg border border-white/10 bg-white/[0.025] p-4">
        {points.length ? (
          <>
            <svg viewBox="0 0 100 106" className="h-64 w-full overflow-visible" role="img" aria-label="Youth score history">
              <defs>
                <linearGradient id="youth-score-line" x1="0" x2="1" y1="0" y2="0">
                  <stop offset="0%" stopColor="#2F7D5C" />
                  <stop offset="100%" stopColor="#8CC7A7" />
                </linearGradient>
                <linearGradient id="youth-score-area" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor="#559A78" stopOpacity="0.22" />
                  <stop offset="100%" stopColor="#559A78" stopOpacity="0" />
                </linearGradient>
              </defs>
              <path d="M 12 82 H 88" stroke="rgba(255,255,255,0.16)" strokeWidth="0.7" />
              {grid.map((y) => (
                <path key={y} d={`M 12 ${y} H 88`} stroke="rgba(255,255,255,0.07)" strokeWidth="0.45" />
              ))}
              {path && points.length > 1 ? (
                <path
                  d={`${path} L ${points[points.length - 1].x} 82 L ${points[0].x} 82 Z`}
                  fill="url(#youth-score-area)"
                />
              ) : null}
              {path ? <path d={path} fill="none" stroke="url(#youth-score-line)" strokeWidth="2.8" strokeLinecap="round" strokeLinejoin="round" /> : null}
              {points.map((point) => (
                <g key={point.player_season_id || point.calendar}>
                  <circle cx={point.x} cy={point.y} r="3.2" fill="#8CC7A7" stroke="#06100C" strokeWidth="1.4" />
                  <text x={point.x} y={point.y - 8} textAnchor="middle" className="fill-white text-[5px] font-black">
                    {Number.isFinite(point.score) ? point.score.toFixed(1) : "-"}
                  </text>
                  <text x={point.x} y="97" textAnchor="middle" className="fill-[#A0A8A3] text-[4.6px] font-bold">
                    {point.calendar}
                  </text>
                </g>
              ))}
            </svg>
            <div className="mt-3 grid gap-2 md:grid-cols-2">
              {points.map((row) => {
                const seasonId = reportSeasonId(row);
                const Wrapper = seasonId && onSelect ? "button" : "div";
                return (
                <Wrapper
                  key={seasonId || row.calendar}
                  type={Wrapper === "button" ? "button" : undefined}
                  onClick={seasonId && onSelect ? () => onSelect(String(seasonId)) : undefined}
                  className="flex w-full items-center justify-between gap-4 rounded-md border border-white/10 bg-white/[0.025] px-3 py-2 text-left transition hover:border-[#3A8967]/35 hover:bg-[#2F7D5C]/10"
                >
                  <div className="min-w-0">
                    <p className="text-sm font-black text-white">{row.calendar}</p>
                    <p className="truncate text-xs text-[#6F7772]">{[row.team, row.competition_name].filter(Boolean).join(" - ") || "-"}</p>
                  </div>
                  <p className="text-xl font-black text-[#8CC7A7]">{formatValue(row.global_score_adjusted, "score")}</p>
                </Wrapper>
              );})}
            </div>
          </>
        ) : <p className="text-sm text-slate-500">No imported history for this player yet.</p>}
      </div>
    </ReportCard>
  );
}

function Stars({ value, onChange, disabled = false }) {
  const numeric = Number(value);
  const current = Number.isFinite(numeric) ? Math.max(0, Math.min(5, Math.round(numeric))) : 0;
  return (
    <div className="flex items-center gap-1">
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          key={star}
          type="button"
          disabled={disabled}
          onClick={() => onChange(star)}
          className={`text-2xl leading-none transition ${star <= current ? "text-[#F2C75C]" : "text-white/20"} ${disabled ? "cursor-default" : "hover:text-[#F2C75C]"}`}
          aria-label={`${star} stars`}
        >
          ★
        </button>
      ))}
    </div>
  );
}

function wrapScoutExportText(ctx, text, x, y, maxWidth, lineHeight, maxLines = 4) {
  const words = String(text || "-").split(/\s+/).filter(Boolean);
  const lines = [];
  let line = "";
  words.forEach((word) => {
    const candidate = line ? `${line} ${word}` : word;
    if (ctx.measureText(candidate).width <= maxWidth || !line) {
      line = candidate;
    } else {
      lines.push(line);
      line = word;
    }
  });
  if (line) lines.push(line);
  const clipped = lines.slice(0, maxLines);
  clipped.forEach((item, index) => {
    const suffix = index === maxLines - 1 && lines.length > maxLines ? "..." : "";
    ctx.fillText(`${item}${suffix}`, x, y + index * lineHeight);
  });
  return y + clipped.length * lineHeight;
}

function drawScoutExportRoundedRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function drawScoutExportPanel(ctx, x, y, width, height, options = {}) {
  ctx.fillStyle = options.fill || "#0A0C0B";
  drawScoutExportRoundedRect(ctx, x, y, width, height, options.radius || 24);
  ctx.fill();
  ctx.strokeStyle = options.stroke || "rgba(255,255,255,0.10)";
  ctx.lineWidth = options.lineWidth || 2;
  ctx.stroke();
}

async function loadScoutExportImage(url) {
  if (!url || typeof window === "undefined") return null;
  try {
    const response = await fetch(url, { credentials: "include" });
    if (!response.ok) return null;
    const blob = await response.blob();
    const objectUrl = window.URL.createObjectURL(blob);
    return await new Promise((resolve) => {
      const image = new Image();
      image.onload = () => resolve({ image, objectUrl });
      image.onerror = () => {
        window.URL.revokeObjectURL(objectUrl);
        resolve(null);
      };
      image.src = objectUrl;
    });
  } catch {
    return null;
  }
}

function drawScoutExportPhoto(ctx, loaded, x, y, width, height, fallbackName) {
  ctx.save();
  drawScoutExportRoundedRect(ctx, x, y, width, height, 28);
  ctx.clip();
  if (loaded?.image) {
    const image = loaded.image;
    const scale = Math.max(width / image.width, height / image.height);
    const sw = width / scale;
    const sh = height / scale;
    const sx = (image.width - sw) / 2;
    const sy = (image.height - sh) / 2;
    ctx.drawImage(image, sx, sy, sw, sh, x, y, width, height);
  } else {
    const gradient = ctx.createLinearGradient(x, y, x + width, y + height);
    gradient.addColorStop(0, "#0A0C0B");
    gradient.addColorStop(1, "#17231D");
    ctx.fillStyle = gradient;
    ctx.fillRect(x, y, width, height);
    ctx.fillStyle = "rgba(140,199,167,0.18)";
    ctx.font = "900 88px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(initials(fallbackName), x + width / 2, y + height / 2);
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
  }
  ctx.restore();
  ctx.strokeStyle = "rgba(140,199,167,0.24)";
  ctx.lineWidth = 2;
  drawScoutExportRoundedRect(ctx, x, y, width, height, 28);
  ctx.stroke();
}

function drawScoutExportStars(ctx, value, x, y, size = 30) {
  const numeric = Number(value);
  const current = Number.isFinite(numeric) ? Math.max(0, Math.min(5, Math.round(numeric))) : 0;
  ctx.font = `900 ${size}px Arial`;
  for (let star = 1; star <= 5; star += 1) {
    ctx.fillStyle = star <= current ? "#F2C75C" : "rgba(255,255,255,0.18)";
    ctx.fillText("★", x + (star - 1) * (size * 0.9), y);
  }
}

function drawScoutExportTags(ctx, tags, x, y, maxWidth) {
  let cursorX = x;
  let cursorY = y;
  const height = 25;
  const gap = 8;
  ctx.font = "900 13px Arial";
  tags.slice(0, 6).forEach((tag) => {
    const label = String(tag || "").toUpperCase();
    const width = Math.min(maxWidth, Math.ceil(ctx.measureText(label).width) + 24);
    if (cursorX + width > x + maxWidth) {
      cursorX = x;
      cursorY += height + gap;
    }
    if (cursorY > y + height + gap) return;
    drawScoutExportRoundedRect(ctx, cursorX, cursorY, width, height, 8);
    ctx.fillStyle = "rgba(47,125,92,0.18)";
    ctx.fill();
    ctx.strokeStyle = "rgba(140,199,167,0.28)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = "#8CC7A7";
    ctx.fillText(label, cursorX + 12, cursorY + 17);
    cursorX += width + gap;
  });
}

function drawScoutExportMetric(ctx, label, value, x, y, width) {
  drawScoutExportPanel(ctx, x, y, width, 112, { fill: "#070807", radius: 18 });
  ctx.fillStyle = "#6F7772";
  ctx.font = "900 18px Arial";
  ctx.fillText(String(label || "").toUpperCase(), x + 24, y + 34);
  drawScoutExportStars(ctx, value, x + 24, y + 82, 32);
  ctx.fillStyle = "#F3F5F4";
  ctx.font = "900 24px Arial";
  ctx.textAlign = "right";
  ctx.fillText(value ? `${value}/5` : "-", x + width - 24, y + 80);
  ctx.textAlign = "left";
}

function downloadYouthScoutCanvas(canvas, filename) {
  const link = document.createElement("a");
  link.download = filename;
  link.href = canvas.toDataURL("image/png");
  link.click();
}

async function drawYouthScoutObservationPng({ report, form, reporterName, photoUrl }) {
  const canvas = document.createElement("canvas");
  canvas.width = 1600;
  canvas.height = 2160;
  const ctx = canvas.getContext("2d");
  const player = report?.player || {};
  const photo = await loadScoutExportImage(photoUrl);
  const generatedAt = new Intl.DateTimeFormat("en", { day: "2-digit", month: "short", year: "numeric" }).format(new Date());
  const matches = (form.matches_observed || []).filter((match) =>
    Object.values(match).some((value) => value !== null && value !== undefined && String(value).trim() !== "")
  );

  ctx.fillStyle = "#050706";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const glow = ctx.createRadialGradient(1220, 80, 20, 1220, 80, 720);
  glow.addColorStop(0, "rgba(85,154,120,0.34)");
  glow.addColorStop(0.42, "rgba(47,125,92,0.12)");
  glow.addColorStop(1, "rgba(5,7,6,0)");
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, canvas.width, 720);

  ctx.fillStyle = "#8CC7A7";
  ctx.font = "900 25px Arial";
  ctx.fillText("SCOUT OBSERVATIONS", 80, 90);
  ctx.fillStyle = "#F3F5F4";
  ctx.font = "900 72px Arial";
  wrapScoutExportText(ctx, player.name || form.player_name || "Youth player", 80, 180, 980, 78, 2);
  ctx.fillStyle = "#A0A8A3";
  ctx.font = "800 26px Arial";
  wrapScoutExportText(
    ctx,
    [player.position, player.team || form.club, player.competition_name, player.calendar].filter(Boolean).join("  |  "),
    82,
    332,
    980,
    34,
    2
  );

  drawScoutExportPhoto(ctx, photo, 1180, 70, 300, 300, player.name || form.player_name || "Youth player");
  if (photo?.objectUrl) window.URL.revokeObjectURL(photo.objectUrl);

  drawScoutExportMetric(ctx, "Scouting note", form.star_rating, 80, 430, 450);
  drawScoutExportMetric(ctx, "Potential", form.potential_star_rating, 560, 430, 450);
  drawScoutExportPanel(ctx, 1040, 430, 440, 112, { fill: "#070807", radius: 18 });
  ctx.fillStyle = "#6F7772";
  ctx.font = "900 18px Arial";
  ctx.fillText("REPORTER", 1064, 464);
  ctx.fillStyle = "#F3F5F4";
  ctx.font = "900 24px Arial";
  wrapScoutExportText(ctx, reporterName || form.scout || "-", 1064, 506, 376, 30, 2);

  let y = 610;
  drawScoutExportPanel(ctx, 80, y, 1400, 300, { fill: "rgba(255,255,255,0.035)", radius: 24 });
  ctx.fillStyle = "#8CC7A7";
  ctx.font = "900 22px Arial";
  ctx.fillText("RESUME", 116, y + 50);
  ctx.fillStyle = "#F3F5F4";
  ctx.font = "700 28px Arial";
  wrapScoutExportText(ctx, form.overall_comments || "No resume available yet.", 116, y + 105, 1328, 39, 5);

  y += 360;
  drawScoutExportPanel(ctx, 80, y, 1400, 270, { fill: "rgba(255,255,255,0.035)", radius: 24 });
  ctx.fillStyle = "#8CC7A7";
  ctx.font = "900 22px Arial";
  ctx.fillText("DETAILED ASSESSMENT", 116, y + 50);
  SCOUTING_DETAIL_RATING_FIELDS.forEach(([key, label], index) => {
    const col = index % 4;
    const x = 116 + col * 340;
    drawScoutExportMetric(ctx, label, form[key], x, y + 88, 300);
  });

  y += 330;
  drawScoutExportPanel(ctx, 80, y, 1400, 710, { fill: "rgba(255,255,255,0.035)", radius: 24 });
  ctx.fillStyle = "#8CC7A7";
  ctx.font = "900 22px Arial";
  ctx.fillText("MATCHES OBSERVED", 116, y + 50);
  ctx.fillStyle = "#6F7772";
  ctx.font = "800 18px Arial";
  ctx.fillText(matches.length ? `Showing ${Math.min(matches.length, 4)} of ${matches.length} recorded observations.` : "No match observation recorded yet.", 116, y + 82);

  let matchY = y + 122;
  matches.slice(0, 4).forEach((match, index) => {
    drawScoutExportPanel(ctx, 116, matchY, 1328, 136, { fill: "#070807", radius: 18 });
    ctx.fillStyle = "#F3F5F4";
    ctx.font = "900 24px Arial";
    wrapScoutExportText(ctx, `${match.team_a || "-"} ${match.score_a !== "" && match.score_a !== null ? match.score_a : ""} - ${match.score_b !== "" && match.score_b !== null ? match.score_b : ""} ${match.team_b || "-"}`, 142, matchY + 38, 540, 28, 1);
    ctx.fillStyle = "#A0A8A3";
    ctx.font = "800 18px Arial";
    wrapScoutExportText(ctx, [match.match_date, match.competition, match.position, match.minutes_played ? `${match.minutes_played} min` : ""].filter(Boolean).join(" | "), 142, matchY + 72, 620, 24, 1);
    drawScoutExportStars(ctx, match.player_rating, 760, matchY + 58, 28);
    ctx.fillStyle = "#F3F5F4";
    ctx.font = "700 20px Arial";
    wrapScoutExportText(ctx, match.observations || `Observation ${index + 1}`, 760, matchY + 88, 640, 25, 1);
    const tags = parseQualitativeTags(match.qualitative_tags);
    if (tags.length) {
      drawScoutExportTags(ctx, tags, 760, matchY + 104, 640);
    }
    matchY += 150;
  });

  ctx.fillStyle = "#6F7772";
  ctx.font = "700 17px Arial";
  ctx.fillText("Source: Next Legend youth scouting report. Qualitative observations must be validated through live/video review.", 80, canvas.height - 58);
  return canvas;
}

function scoutingFormFromReport(item, report) {
  const base = emptyScoutingForm(report);
  if (!item) return base;
  return {
    ...base,
    player_name: item.player_name || base.player_name,
    club: item.club || "",
    year_of_birth: item.year_of_birth || "",
    position: item.position || "",
    nationality: item.nationality || "",
    scout: item.scout || "",
    portal_url: item.portal_url || "",
    star_rating: item.star_rating ?? item.ratings?.stars ?? "",
    potential_star_rating: item.potential_star_rating ?? item.ratings?.potential_stars ?? "",
    technical_rating: detailRatingForForm(item.technical_rating),
    physical_rating: detailRatingForForm(item.physical_rating),
    tactical_rating: detailRatingForForm(item.tactical_rating),
    mental_rating: detailRatingForForm(item.mental_rating),
    potential_rating: item.potential_rating ?? "",
    overall_rating: item.overall_rating ?? "",
    technical_notes: item.technical_notes || "",
    physical_notes: item.physical_notes || "",
    tactical_notes: item.tactical_notes || "",
    mental_notes: item.mental_notes || "",
    game_intelligence: item.game_intelligence || "",
    strengths: item.strengths || "",
    weaknesses: item.weaknesses || "",
    development_projection: item.development_projection || "",
    comparison: item.comparison || "",
    overall_comments: item.overall_comments || "",
    photo_key: item.photo_key || "",
    photo_url: scoutPhotoUrl(item.photo_key),
    matches_observed: (item.matches_observed || []).map((match) => {
      const normalizedPosition = normalizeMatchPosition(match.position || item.position || report?.player?.position);
      return {
        source: match.source || "",
        team_a: match.team_a || "",
        score_a: match.score_a ?? "",
        score_b: match.score_b ?? "",
        team_b: match.team_b || "",
        competition: match.competition || "",
        match_date: match.match_date || "",
        player_rating: matchRatingForForm(match.player_rating),
        minutes_played: match.minutes_played ?? "",
        position: normalizedPosition,
        observations: match.observations || "",
        qualitative_tags: filterTagsForPosition(match.qualitative_tags, normalizedPosition),
      };
    }),
  };
}

function numberOrNull(value) {
  if (value === "" || value === null || value === undefined) return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : NaN;
}

function matchRatingForForm(value) {
  const numeric = numberOrNull(value);
  if (numeric === null || !Number.isFinite(numeric)) return "";
  const fivePointValue = numeric > 5 ? Math.round(numeric / 2) : numeric;
  return Math.max(1, Math.min(5, fivePointValue));
}

function detailRatingForForm(value) {
  return matchRatingForForm(value);
}

function validateScoutingForm(form) {
  if (!String(form.player_name || "").trim()) return "Player name is required.";
  const year = numberOrNull(form.year_of_birth);
  if (year !== null && (!Number.isFinite(year) || year < 1990 || year > 2035)) {
    return "Year of birth must be valid.";
  }
  const stars = numberOrNull(form.star_rating);
  if (stars !== null && (!Number.isFinite(stars) || stars < 1 || stars > 5)) {
    return "Scouting note must be between 1 and 5.";
  }
  const potentialStars = numberOrNull(form.potential_star_rating);
  if (potentialStars !== null && (!Number.isFinite(potentialStars) || potentialStars < 1 || potentialStars > 5)) {
    return "Potential must be between 1 and 5.";
  }
  for (const [key, label] of SCOUTING_DETAIL_RATING_FIELDS) {
    const rating = numberOrNull(form[key]);
    if (rating !== null && (!Number.isFinite(rating) || rating < 1 || rating > 5)) {
      return `${label} rating must be between 1 and 5.`;
    }
  }
  for (const [idx, match] of (form.matches_observed || []).entries()) {
    const label = `Match ${idx + 1}`;
    const touched = Object.values(match).some((value) => value !== null && value !== undefined && String(value).trim() !== "");
    if (!touched) continue;
    for (const [key, fieldLabel] of [["team_a", "Team A"], ["team_b", "Team B"], ["competition", "Competition"], ["match_date", "Match date"], ["position", "Position"], ["observations", "Observations"]]) {
      if (!String(match[key] || "").trim()) return `${label} ${fieldLabel} is required.`;
    }
    for (const [key, max, fieldLabel] of [["score_a", 30, "Score A"], ["score_b", 30, "Score B"], ["player_rating", 5, "Player rating"], ["minutes_played", 130, "Minutes played"]]) {
      const value = numberOrNull(match[key]);
      if (key === "player_rating" && value === null) return `${label} Player rating is required.`;
      const min = key === "player_rating" ? 1 : 0;
      if (value !== null && (!Number.isFinite(value) || value < min || value > max)) {
        return `${label} ${fieldLabel} must be between ${min} and ${max}.`;
      }
    }
  }
  return "";
}

function validateScoutingMatch(match, index = 0) {
  const label = `Match ${index + 1}`;
  for (const [key, fieldLabel] of [["team_a", "Opponent A"], ["team_b", "Opponent B"], ["competition", "Competition"], ["match_date", "Match date"], ["position", "Position"], ["observations", "Observations"]]) {
    if (!String(match?.[key] || "").trim()) return `${label} ${fieldLabel} is required.`;
  }
  for (const [key, max, fieldLabel] of [["score_a", 30, "Score A"], ["score_b", 30, "Score B"], ["player_rating", 5, "Player rating"], ["minutes_played", 130, "Minutes played"]]) {
    const value = numberOrNull(match?.[key]);
    if (key === "player_rating" && value === null) return `${label} Player rating is required.`;
    const min = key === "player_rating" ? 1 : 0;
    if (value !== null && (!Number.isFinite(value) || value < min || value > max)) {
      return `${label} ${fieldLabel} must be between ${min} and ${max}.`;
    }
  }
  const allowedTags = new Set(qualitativeTagsForPosition(match.position));
  const invalidTag = parseQualitativeTags(match.qualitative_tags).find((tag) => !allowedTags.has(tag));
  if (invalidTag) return `${label} Qualitative tag ${invalidTag} is not valid for ${match.position}.`;
  return "";
}

function scoutingPayload(form, youthId) {
  const payload = {
    youth_id: youthId ? Number(youthId) : undefined,
    player_name: String(form.player_name || "").trim(),
    club: String(form.club || "").trim() || null,
    year_of_birth: numberOrNull(form.year_of_birth),
    position: String(form.position || "").trim() || null,
    nationality: String(form.nationality || "").trim() || null,
    portal_url: String(form.portal_url || "").trim() || null,
    photo_key: String(form.photo_key || "").trim() || null,
    star_rating: numberOrNull(form.star_rating),
    potential_star_rating: numberOrNull(form.potential_star_rating),
    scout: String(form.scout || "").trim() || null,
    matches_observed: (form.matches_observed || [])
      .map((match) => ({
        source: match.source || null,
        team_a: String(match.team_a || "").trim(),
        score_a: numberOrNull(match.score_a),
        score_b: numberOrNull(match.score_b),
        team_b: String(match.team_b || "").trim(),
        competition: String(match.competition || "").trim(),
        match_date: String(match.match_date || "").trim(),
        player_rating: numberOrNull(match.player_rating),
        minutes_played: numberOrNull(match.minutes_played),
        position: String(match.position || "").trim(),
        observations: String(match.observations || "").trim(),
        qualitative_tags: filterTagsForPosition(match.qualitative_tags, match.position),
      }))
      .filter((match) => Object.values(match).some((value) => value !== null && value !== "")),
  };
  payload.overall_comments = form.overall_comments || null;
  SCOUTING_DETAIL_RATING_FIELDS.forEach(([key]) => {
    payload[key] = numberOrNull(form[key]);
  });
  return payload;
}

function YouthReportPhotoUploader({ report, youthId, me }) {
  const [item, setItem] = useState(null);
  const [photoBusy, setPhotoBusy] = useState(false);
  const [photoDragging, setPhotoDragging] = useState(false);
  const [error, setError] = useState("");
  const editable = canManageScoutingReport(item, me);
  const photoUrl = scoutPhotoUrl(item?.photo_key) || "";
  const player = report?.player || {};

  const loadPhotoReport = async () => {
    if (!youthId) return;
    try {
      const data = await fetchJson("/youth/scouting-reports", { youth_id: youthId, limit: 1 });
      setItem((data.items || [])[0] || null);
    } catch {
      setItem(null);
    }
  };

  useEffect(() => {
    loadPhotoReport();
  }, [youthId]);

  const uploadPhoto = async (file) => {
    if (!file || !editable || photoBusy) return;
    setPhotoBusy(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const uploaded = await postForm("/youth/scouting-reports/upload", formData, { youth_id: youthId, purpose: "photo" });
      const photoKey = uploaded.file_key || "";
      const payload = {
        youth_id: youthId ? Number(youthId) : undefined,
        player_name: player.name || "",
        club: player.team || null,
        year_of_birth: player.birth_year || null,
        position: player.position || null,
        nationality: player.nationality || null,
        portal_url: report?.source?.player?.provider_player_url || null,
        scout: item?.scout || me?.username || null,
        photo_key: photoKey,
      };
      const saved = item?.id
        ? await patchJson(`/youth/scouting-reports/${item.id}`, { photo_key: photoKey })
        : await postJson("/youth/scouting-reports", payload);
      setItem(saved.report || null);
      if (typeof window !== "undefined") {
        window.dispatchEvent(new CustomEvent("youth-scouting-report-updated", { detail: { youthId } }));
      }
    } catch (err) {
      setError(err.message || "Unable to upload player photo.");
    } finally {
      setPhotoBusy(false);
      setPhotoDragging(false);
    }
  };

  return (
    <div className="w-full max-w-[170px]">
      <div
        className={`relative flex h-[210px] items-center justify-center overflow-hidden rounded-lg border border-[#3A8967]/25 bg-[#06100C] p-2 shadow-sm ${
          photoDragging ? "ring-2 ring-[#559A78]" : ""
        }`}
        onDragOver={(event) => {
          event.preventDefault();
          if (editable) setPhotoDragging(true);
        }}
        onDragLeave={() => setPhotoDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          uploadPhoto(event.dataTransfer.files?.[0]);
        }}
      >
        <div className="flex h-full w-full items-center justify-center overflow-hidden rounded-md border border-white/10 bg-white/[0.04] text-4xl font-black text-[#8CC7A7]">
          {photoUrl ? (
            <img src={photoUrl} alt="" className="h-full w-full object-cover" />
          ) : (
            initials(player.name || "YP")
          )}
        </div>
        {photoBusy ? (
          <div className="absolute inset-0 flex items-center justify-center bg-black/60 text-[10px] font-black uppercase tracking-[0.14em] text-white">
            Uploading...
          </div>
        ) : null}
      </div>
      <div className="mt-2 flex flex-wrap gap-2">
        <label className={`nl-button-primary cursor-pointer px-3 py-2 text-[10px] uppercase tracking-[0.1em] ${!editable || photoBusy ? "pointer-events-none opacity-50" : ""}`}>
          Upload photo
          <input type="file" accept="image/*" className="sr-only" disabled={!editable || photoBusy} onChange={(event) => uploadPhoto(event.target.files?.[0])} />
        </label>
        {photoUrl ? (
          <a href={photoUrl} target="_blank" rel="noreferrer" className="nl-button-secondary px-3 py-2 text-[10px] uppercase tracking-[0.1em]">
            Open
          </a>
        ) : null}
      </div>
      <div className="mt-3 flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.035] px-3 py-2 shadow-sm">
        <ClubLogo name={player.team} className="h-8 w-8 rounded-md" />
        <span className="min-w-0 truncate text-xs font-black text-slate-300">{player.team || "No club"}</span>
      </div>
      <p className="mt-2 text-[11px] leading-5 text-[#6F7772]">Drop an image on the player photo or upload one manually.</p>
      {error ? <p className="mt-2 text-xs font-semibold text-rose-200">{error}</p> : null}
    </div>
  );
}

function YouthScoutingReportsPanel({ report, youthId, me }) {
  const [items, setItems] = useState([]);
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [form, setForm] = useState(() => emptyScoutingForm(report));
  const [exportBusy, setExportBusy] = useState(false);
  const [editingMatchIndex, setEditingMatchIndex] = useState(null);
  const activeReport = items[0] || null;
  const persistedMatchCount = activeReport?.matches_observed?.length || 0;
  const currentPhotoUrl = scoutPhotoUrl(form.photo_key) || form.photo_url || "";

  const reload = async () => {
    if (!youthId) return;
    setLoading(true);
    setError("");
    try {
      const data = await fetchJson("/youth/scouting-reports", { youth_id: youthId, limit: 50 });
      setItems(data.items || []);
    } catch (err) {
      setError(err.message || "Unable to load scout observations.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    reload();
  }, [youthId]);

  useEffect(() => {
    const handler = (event) => {
      if (!event?.detail?.youthId || String(event.detail.youthId) === String(youthId)) {
        reload();
      }
    };
    if (typeof window !== "undefined") {
      window.addEventListener("youth-scouting-report-updated", handler);
    }
    return () => {
      if (typeof window !== "undefined") {
        window.removeEventListener("youth-scouting-report-updated", handler);
      }
    };
  }, [youthId]);

  useEffect(() => {
    let cancelled = false;
    fetchJson("/users/options")
      .then((data) => {
        if (!cancelled) setUsers(data.items || []);
      })
      .catch(() => {
        if (!cancelled) setUsers([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const nextForm = activeReport ? scoutingFormFromReport(activeReport, report) : emptyScoutingForm(report);
    if (!nextForm.scout && me?.username) nextForm.scout = me.username;
    setForm(nextForm);
    setEditingMatchIndex(null);
  }, [activeReport?.id, report?.player?.player_season_id, me?.username]);

  const saveReport = async (overrides = {}, includeMatches = false, options = {}) => {
    const nextForm = { ...form, scout: form.scout || me?.username || "", ...overrides };
    const validation = options.skipMatchValidation
      ? validateScoutingForm({ ...nextForm, matches_observed: [] })
      : includeMatches && Number.isInteger(options.matchIndex)
      ? validateScoutingMatch(nextForm.matches_observed?.[options.matchIndex], options.matchIndex)
      : validateScoutingForm(includeMatches ? nextForm : { ...nextForm, matches_observed: [] });
    if (validation) {
      setError(validation);
      return null;
    }
    const payload = scoutingPayload(nextForm, youthId);
    if (!includeMatches) delete payload.matches_observed;
    setSaving(true);
    setError("");
    setMessage("");
    try {
      if (activeReport?.id) {
        const res = await patchJson(`/youth/scouting-reports/${activeReport.id}`, payload);
        setItems((current) => current.map((item) => (item.id === activeReport.id ? res.report : item)));
        setMessage("Saved automatically.");
        return res.report;
      }
      const res = await postJson("/youth/scouting-reports", payload);
      setItems((current) => [res.report, ...current]);
      setMessage("Scout observation created.");
      return res.report;
    } catch (err) {
      setError(err.message || "Unable to save scout observation.");
      return null;
    } finally {
      setSaving(false);
    }
  };

  const updateMatch = (index, key, value) => {
    setForm((current) => ({
      ...current,
      matches_observed: current.matches_observed.map((match, idx) => idx === index ? { ...match, [key]: value } : match),
    }));
  };

  const updateMatchPosition = (index, value) => {
    const normalizedPosition = normalizeMatchPosition(value);
    setForm((current) => ({
      ...current,
      matches_observed: current.matches_observed.map((match, idx) => (
        idx === index
          ? { ...match, position: normalizedPosition, qualitative_tags: filterTagsForPosition(match.qualitative_tags, normalizedPosition) }
          : match
      )),
    }));
  };

  const toggleQualitativeTag = (index, tag) => {
    setForm((current) => ({
      ...current,
      matches_observed: current.matches_observed.map((match, idx) => {
        if (idx !== index) return match;
        const selected = new Set(parseQualitativeTags(match.qualitative_tags));
        if (selected.has(tag)) selected.delete(tag);
        else selected.add(tag);
        return { ...match, qualitative_tags: Array.from(selected) };
      }),
    }));
  };

  const addMatch = () => {
    if (editingMatchIndex !== null) return;
    const nextIndex = form.matches_observed.length;
    setForm((current) => ({
      ...current,
      matches_observed: [...current.matches_observed, { ...emptyMatch(report), team_a: current.club || report?.player?.team || "" }],
    }));
    setEditingMatchIndex(nextIndex);
  };

  const swapTeams = (index) => {
    setForm((current) => ({
      ...current,
      matches_observed: current.matches_observed.map((match, idx) => (
        idx === index ? { ...match, team_a: match.team_b, team_b: match.team_a, score_a: match.score_b, score_b: match.score_a } : match
      )),
    }));
  };

  const removeMatch = (index) => {
    const nextMatches = (form.matches_observed || []).filter((_, itemIdx) => itemIdx !== index);
    setForm((current) => ({
      ...current,
      matches_observed: current.matches_observed.filter((_, itemIdx) => itemIdx !== index),
    }));
    saveReport({ matches_observed: nextMatches }, true, { skipMatchValidation: true });
    setEditingMatchIndex(null);
  };

  const discardNewMatch = (index) => {
    setForm((current) => ({
      ...current,
      matches_observed: current.matches_observed.filter((_, itemIdx) => itemIdx !== index),
    }));
    setEditingMatchIndex(null);
  };

  const persistMatch = async (index) => {
    const nextMatches = (form.matches_observed || []).map((match, idx) => (
      idx === index ? { ...match, source: "nextlegend_ui" } : match
    ));
    setForm((current) => ({ ...current, matches_observed: nextMatches }));
    const saved = await saveReport({ matches_observed: nextMatches }, true, { matchIndex: index });
    if (saved) setEditingMatchIndex(null);
  };

  const editable = canManageScoutingReport(activeReport, me);

  const exportScoutObservations = async () => {
    if (typeof document === "undefined" || exportBusy) return;
    setExportBusy(true);
    setError("");
    try {
      const canvas = await drawYouthScoutObservationPng({
        report,
        form,
        photoUrl: currentPhotoUrl,
        reporterName: reporterDisplayName(form.scout || me?.username, users),
      });
      const slug = String(report?.player?.name || form.player_name || "youth-player")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-|-$/g, "");
      downloadYouthScoutCanvas(canvas, `scout-observations-${slug || "youth-player"}.png`);
    } catch (err) {
      setError(err.message || "Unable to export scout observations.");
    } finally {
      setExportBusy(false);
    }
  };

  return (
    <ReportCard className="relative z-10 scroll-mt-8" id="human-scouting-reports">
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div>
          <h3 className="text-xl font-black uppercase tracking-[0.08em] text-white">Scout observations</h3>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-[#A0A8A3]">
            Structured scout assessment combining qualitative notes, observed-match evidence and accountable reporting.
          </p>
        </div>
        <button type="button" className="nl-button-primary px-4 py-2 text-xs uppercase tracking-[0.14em]" onClick={exportScoutObservations} disabled={exportBusy}>
          {exportBusy ? "Exporting..." : "Export"}
        </button>
      </div>

      {error ? <p className="mt-4 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm font-semibold text-rose-200">{error}</p> : null}
      {message ? <p className="mt-4 rounded-md border border-[#3A8967]/35 bg-[#2F7D5C]/15 px-3 py-2 text-sm font-semibold text-[#8CC7A7]">{message}</p> : null}

      {loading ? (
        <div className="mt-5 rounded-lg border border-white/10 bg-white/[0.025] p-4 text-sm text-[#A0A8A3]">Loading scout observations...</div>
      ) : null}

      {!editable ? (
        <p className="mt-5 rounded-md border border-amber-400/25 bg-amber-400/10 px-3 py-2 text-sm font-semibold text-amber-100">
          This report is read-only for your account.
        </p>
      ) : null}

      <div className="mt-5 grid gap-4 xl:grid-cols-[1fr_1fr_260px]">
          <div className="rounded-lg border border-white/10 bg-white/[0.025] p-4">
            <p className="text-[10px] font-black uppercase tracking-[0.16em] text-[#6F7772]">Scouting note</p>
            <div className="mt-2 flex items-center justify-between gap-3">
              <Stars
                value={form.star_rating}
                disabled={!editable || saving}
                onChange={(value) => {
                  setForm((current) => ({ ...current, star_rating: value }));
                  saveReport({ star_rating: value }, false);
                }}
              />
              <span className="text-sm font-black text-white">{form.star_rating ? `${form.star_rating}/5` : "Not rated"}</span>
            </div>
          </div>
          <div className="rounded-lg border border-white/10 bg-white/[0.025] p-4">
            <p className="text-[10px] font-black uppercase tracking-[0.16em] text-[#6F7772]">Potential</p>
            <div className="mt-2 flex items-center justify-between gap-3">
              <Stars
                value={form.potential_star_rating}
                disabled={!editable || saving}
                onChange={(value) => {
                  setForm((current) => ({ ...current, potential_star_rating: value }));
                  saveReport({ potential_star_rating: value }, false);
                }}
              />
              <span className="text-sm font-black text-white">{form.potential_star_rating ? `${form.potential_star_rating}/5` : "Not rated"}</span>
            </div>
          </div>
          <label className="rounded-lg border border-white/10 bg-white/[0.025] p-4">
            <span className="text-[10px] font-black uppercase tracking-[0.16em] text-[#6F7772]">Reporter</span>
            <select
              className="nl-field mt-2"
              value={form.scout || me?.username || ""}
              disabled={!editable || saving}
              onChange={(event) => {
                const value = event.target.value;
                setForm((current) => ({ ...current, scout: value }));
                saveReport({ scout: value }, false);
              }}
            >
              {[...users, ...(form.scout && !users.some((user) => user.username === form.scout) ? [{ username: form.scout, display_name: form.scout }] : [])].map((user) => (
                <option key={user.username} value={user.username}>
                  {user.display_name || user.username}
                </option>
              ))}
            </select>
          </label>
        </div>

      <div className="mt-4 rounded-lg border border-white/10 bg-white/[0.025] p-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-[10px] font-black uppercase tracking-[0.16em] text-[#6F7772]">Resume</p>
            <p className="mt-1 text-sm text-[#A0A8A3]">General scout observations and decision-useful context.</p>
          </div>
          <p className="text-xs font-semibold text-[#6F7772]">{saving ? "Saving..." : "Autosaved"}</p>
        </div>
        <textarea
          className="nl-field mt-4 min-h-[180px] text-sm leading-6"
          value={form.overall_comments || ""}
          disabled={!editable || saving}
          placeholder="Write the scout resume here..."
          onChange={(event) => setForm((current) => ({ ...current, overall_comments: event.target.value }))}
          onBlur={() => saveReport({}, false)}
        />
      </div>

      <div className="mt-4 rounded-lg border border-white/10 bg-white/[0.025] p-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-[10px] font-black uppercase tracking-[0.16em] text-[#6F7772]">Detailed assessment</p>
            <p className="mt-1 text-sm text-[#A0A8A3]">Four core scouting dimensions rated on a five-star scale.</p>
          </div>
          <p className="text-xs font-semibold text-[#6F7772]">{saving ? "Saving..." : "Autosaved"}</p>
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {SCOUTING_DETAIL_RATING_FIELDS.map(([key, label]) => (
            <div key={key} className="rounded-lg border border-white/10 bg-[#070807] p-4">
              <p className="text-[10px] font-black uppercase tracking-[0.14em] text-[#6F7772]">{label}</p>
              <div className="mt-3 flex items-center justify-between gap-3">
                <Stars
                  value={form[key]}
                  disabled={!editable || saving}
                  onChange={(value) => {
                    setForm((current) => ({ ...current, [key]: value }));
                    saveReport({ [key]: value }, false);
                  }}
                />
                <span className="text-sm font-black text-white">{form[key] ? `${form[key]}/5` : "Not rated"}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4 rounded-lg border border-white/10 bg-white/[0.025] p-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-[10px] font-black uppercase tracking-[0.16em] text-[#6F7772]">Matches observed</p>
            <p className="mt-1 text-sm text-[#A0A8A3]">Every match must include a date, competition, teams, position, rating and observations.</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button type="button" className="nl-button-secondary px-3 py-2 text-xs uppercase tracking-[0.14em]" disabled={!editable || saving || editingMatchIndex !== null} onClick={addMatch}>
              Add
            </button>
            <p className="self-center text-xs font-semibold text-[#6F7772]">{saving ? "Saving..." : editingMatchIndex !== null ? "Editing draft" : "Locked after save"}</p>
          </div>
        </div>
        <div className="mt-4 space-y-3">
          {form.matches_observed.length ? form.matches_observed.map((match, idx) => {
            const isEditing = editingMatchIndex === idx;
            const isNew = idx >= persistedMatchCount;
            const locked = !isEditing;
            const selectedTags = parseQualitativeTags(match.qualitative_tags);
            return (
            <div key={`match-${idx}`} className={`rounded-lg border p-4 transition ${isEditing ? "border-[#3A8967]/45 bg-[#0A0C0B]" : "border-white/10 bg-[#070807]"}`}>
              <div className="mb-3 flex items-center justify-between gap-3">
                <p className="text-[10px] font-black uppercase tracking-[0.14em] text-[#6F7772]">
                  {isNew ? "New match draft" : `Match ${idx + 1}`}
                </p>
                <span className={`rounded-md border px-2 py-1 text-[10px] font-black uppercase tracking-[0.14em] ${isEditing ? "border-[#3A8967]/35 bg-[#2F7D5C]/15 text-[#8CC7A7]" : "border-white/10 bg-white/[0.035] text-[#6F7772]"}`}>
                  {isEditing ? "Editing" : "Locked"}
                </span>
              </div>
              <div className="grid gap-3 xl:grid-cols-[1fr_112px_auto_112px_1fr]">
                <label>
                  <span className="mb-1 block text-[10px] font-black uppercase tracking-[0.14em] text-[#6F7772]">Opponent A</span>
                  <input className="nl-field" disabled={!editable || saving || locked} value={match.team_a || ""} onChange={(event) => updateMatch(idx, "team_a", event.target.value)} />
                </label>
                <label>
                  <span className="mb-1 block text-[10px] font-black uppercase tracking-[0.14em] text-[#6F7772]">Score A</span>
                  <input className="nl-field" type="number" min="0" max="30" disabled={!editable || saving || locked} value={match.score_a ?? ""} onChange={(event) => updateMatch(idx, "score_a", event.target.value)} />
                </label>
                <div className="flex items-end justify-center">
                  <button type="button" className="nl-button-secondary h-10 px-3 text-base" disabled={!editable || saving || locked} onClick={() => swapTeams(idx)} aria-label="Swap teams">
                    ⇄
                  </button>
                </div>
                <label>
                  <span className="mb-1 block text-[10px] font-black uppercase tracking-[0.14em] text-[#6F7772]">Score B</span>
                  <input className="nl-field" type="number" min="0" max="30" disabled={!editable || saving || locked} value={match.score_b ?? ""} onChange={(event) => updateMatch(idx, "score_b", event.target.value)} />
                </label>
                <label>
                  <span className="mb-1 block text-[10px] font-black uppercase tracking-[0.14em] text-[#6F7772]">Opponent B</span>
                  <input className="nl-field" disabled={!editable || saving || locked} value={match.team_b || ""} onChange={(event) => updateMatch(idx, "team_b", event.target.value)} />
                </label>
              </div>
              <div className="mt-3 grid gap-3 md:grid-cols-2 xl:grid-cols-5">
                <label>
                  <span className="mb-1 block text-[10px] font-black uppercase tracking-[0.14em] text-[#6F7772]">Match date</span>
                  <input className="nl-field" type="date" disabled={!editable || saving || locked} value={match.match_date || todayIso()} onChange={(event) => updateMatch(idx, "match_date", event.target.value)} />
                </label>
                <label>
                  <span className="mb-1 block text-[10px] font-black uppercase tracking-[0.14em] text-[#6F7772]">Competition</span>
                  <input className="nl-field" disabled={!editable || saving || locked} value={match.competition || ""} onChange={(event) => updateMatch(idx, "competition", event.target.value)} />
                </label>
                <label>
                  <span className="mb-1 block text-[10px] font-black uppercase tracking-[0.14em] text-[#6F7772]">Position</span>
                  <select className="nl-field" disabled={!editable || saving || locked} value={match.position || ""} onChange={(event) => updateMatchPosition(idx, event.target.value)}>
                    <option value="">Select position</option>
                    {MATCH_POSITIONS.map((position) => (
                      <option key={position} value={position}>{position}</option>
                    ))}
                  </select>
                </label>
                <label>
                  <span className="mb-1 block text-[10px] font-black uppercase tracking-[0.14em] text-[#6F7772]">Minutes</span>
                  <input className="nl-field" type="number" min="0" max="130" disabled={!editable || saving || locked} value={match.minutes_played ?? ""} onChange={(event) => updateMatch(idx, "minutes_played", event.target.value)} />
                </label>
                <div>
                  <span className="mb-1 block text-[10px] font-black uppercase tracking-[0.14em] text-[#6F7772]">Match rating</span>
                  <Stars value={match.player_rating} disabled={!editable || saving || locked} onChange={(value) => updateMatch(idx, "player_rating", value)} />
                </div>
              </div>
              <div className="mt-3 grid gap-3 xl:grid-cols-[1fr_420px_auto]">
                <label>
                  <span className="mb-1 block text-[10px] font-black uppercase tracking-[0.14em] text-[#6F7772]">Observations</span>
                  <textarea className="nl-field min-h-[92px]" disabled={!editable || saving || locked} value={match.observations || ""} onChange={(event) => updateMatch(idx, "observations", event.target.value)} />
                </label>
                <div>
                  <span className="mb-1 block text-[10px] font-black uppercase tracking-[0.14em] text-[#6F7772]">Qualitative tags</span>
                  <div className="min-h-[92px] rounded-md border border-white/10 bg-white/[0.025] p-2">
                    {locked ? (
                      selectedTags.length ? (
                        <div className="flex flex-wrap content-start gap-1.5">
                          {selectedTags.map((tag) => (
                            <span
                              key={tag}
                              className="rounded-md border border-[#3A8967]/35 bg-[#2F7D5C]/15 px-2 py-1 text-[9px] font-black uppercase tracking-[0.08em] text-[#8CC7A7]"
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <p className="px-2 py-3 text-sm text-[#6F7772]">No qualitative tag selected.</p>
                      )
                    ) : match.position ? (
                      <div className="flex flex-wrap gap-2">
                        {qualitativeTagsForPosition(match.position).map((tag) => {
                          const selected = selectedTags.includes(tag);
                          return (
                            <button
                              key={tag}
                              type="button"
                              disabled={!editable || saving || locked}
                              onClick={() => toggleQualitativeTag(idx, tag)}
                              className={`rounded-md border px-2.5 py-1.5 text-[10px] font-black uppercase tracking-[0.1em] transition ${
                                selected
                                  ? "border-[#3A8967]/45 bg-[#2F7D5C]/20 text-[#8CC7A7]"
                                  : "border-white/10 bg-white/[0.035] text-[#A0A8A3] hover:border-[#3A8967]/30 hover:text-white"
                              }`}
                            >
                              {tag}
                            </button>
                          );
                        })}
                      </div>
                    ) : (
                      <p className="px-2 py-3 text-sm text-[#6F7772]">Select a position to display available tags.</p>
                    )}
                  </div>
                </div>
                <div className="flex flex-col justify-end gap-2">
                  {isEditing ? (
                    <button
                      type="button"
                      className="nl-button-primary w-full px-3 py-2 text-xs uppercase tracking-[0.14em]"
                      disabled={!editable || saving}
                      onClick={() => persistMatch(idx)}
                    >
                      {saving ? "Saving..." : isNew ? "Add" : "Edit"}
                    </button>
                  ) : (
                    <button
                      type="button"
                      className="nl-button-secondary w-full px-3 py-2 text-xs uppercase tracking-[0.14em]"
                      disabled={!editable || saving || editingMatchIndex !== null}
                      onClick={() => setEditingMatchIndex(idx)}
                    >
                      Edit
                    </button>
                  )}
                  <button
                    type="button"
                    className="nl-button-secondary w-full px-3 py-2 text-xs uppercase tracking-[0.14em] text-rose-200"
                    disabled={!editable || saving || (editingMatchIndex !== null && editingMatchIndex !== idx)}
                    onClick={() => (isNew ? discardNewMatch(idx) : removeMatch(idx))}
                  >
                    Delete
                  </button>
                </div>
              </div>
            </div>
            );
          }) : (
            <div className="rounded-lg border border-dashed border-white/15 bg-white/[0.02] p-6 text-center">
              <p className="text-sm font-black text-white">No observed match yet.</p>
              <p className="mt-1 text-sm text-[#A0A8A3]">Use Add to capture the first live scouting observation for this player.</p>
            </div>
          )}
        </div>
      </div>
    </ReportCard>
  );
}

export default function YouthReportsPage() {
  const router = useRouter();
  const { me } = useAuth();
  const hydratedQuery = useRef(false);
  const [playerQuery, setPlayerQuery] = useState("");
  const [playerResults, setPlayerResults] = useState([]);
  const [showPlayerResults, setShowPlayerResults] = useState(false);
  const [meta, setMeta] = useState(null);
  const [searchSeason, setSearchSeason] = useState(DEFAULT_SEASON);
  const [selectedPlayerId, setSelectedPlayerId] = useState("");
  const [rawReport, setRawReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [percentileContext, setPercentileContext] = useState("global");
  const [rawMode, setRawMode] = useState(false);
  const [compareQuery, setCompareQuery] = useState("");
  const [compareResults, setCompareResults] = useState([]);
  const [showCompareResults, setShowCompareResults] = useState(false);
  const [selectedComparisonLabel, setSelectedComparisonLabel] = useState("");
  const [comparisonReport, setComparisonReport] = useState(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [prospectLoading, setProspectLoading] = useState(false);
  const [isProspect, setIsProspect] = useState(false);
  const [prospectMessage, setProspectMessage] = useState("");

  useEffect(() => {
    let cancelled = false;
    fetchJson("/youth/meta")
      .then((data) => {
        if (cancelled) return;
        setMeta(data);
      })
      .catch((err) => console.error(err));
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!router.isReady || hydratedQuery.current) return;
    const youthId = router.query.youth_id || router.query.id;
    if (!youthId) return;
    hydratedQuery.current = true;
    setSelectedPlayerId(String(youthId));
  }, [router.isReady, router.query]);

  useEffect(() => {
    if (playerQuery.trim().length < 2 || selectedPlayerId) {
      setPlayerResults([]);
      return;
    }
    const handle = setTimeout(async () => {
      try {
        const rows = await fetchJson("/youth/players", {
          q: playerQuery.trim(),
          season: searchSeason,
          limit: 12,
        });
        setPlayerResults(adaptSearchRows(rows || []));
      } catch (err) {
        console.error(err);
      }
    }, 180);
    return () => clearTimeout(handle);
  }, [playerQuery, selectedPlayerId, searchSeason]);

  useEffect(() => {
    if (!selectedPlayerId) {
      setRawReport(null);
      setComparisonReport(null);
      setIsProspect(false);
      setProspectMessage("");
      return;
    }
    let cancelled = false;
    const loadReport = async () => {
      setLoading(true);
      setError("");
      try {
        const data = await fetchJson(`/youth/players/${selectedPlayerId}/report`);
        if (!cancelled) {
          setRawReport(data);
          setPlayerQuery(selectedLabel(data.player));
        }
      } catch (err) {
        if (!cancelled) setError(err.message || "Unable to load Youth report.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    loadReport();
    return () => {
      cancelled = true;
    };
  }, [selectedPlayerId]);

  useEffect(() => {
    if (!selectedPlayerId) {
      setIsProspect(false);
      setProspectMessage("");
      return;
    }
    let cancelled = false;
    fetchJson(`/youth/prospects/${selectedPlayerId}`)
      .then((res) => {
        if (!cancelled) setIsProspect(Boolean(res?.is_prospect));
      })
      .catch(() => {
        if (!cancelled) setIsProspect(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedPlayerId]);

  const report = useMemo(() => buildAdaptedReport(rawReport), [rawReport]);
  const selectedSeasonId = report?.player?.player_season_id || "";
  const seasons = report?.available_seasons || [];
  const scoreHistory = report?.score_history || [];

  useEffect(() => {
    if (compareQuery.trim().length < 2) {
      setCompareResults([]);
      setShowCompareResults(false);
      return;
    }
    if (selectedComparisonLabel && compareQuery === selectedComparisonLabel) {
      setShowCompareResults(false);
      return;
    }
    const handle = setTimeout(async () => {
      try {
        const rows = await fetchJson("/youth/players", {
          q: compareQuery.trim(),
          season: report?.player?.season || searchSeason,
          limit: 12,
        });
        setCompareResults(adaptSearchRows(rows || []));
        setShowCompareResults(true);
      } catch (err) {
        console.error(err);
      }
    }, 180);
    return () => clearTimeout(handle);
  }, [compareQuery, selectedComparisonLabel, report?.player?.season, searchSeason]);

  const metrics = report?.metrics || {};
  const positionGroup = normalizePositionGroup(report?.player?.assigned_role, report?.player?.position);
  const profileCategoriesData = useMemo(() => buildProfileCategories(metrics, percentileContext), [metrics, percentileContext]);
  const characteristics = useMemo(() => buildCharacteristics(metrics, positionGroup, percentileContext), [metrics, positionGroup, percentileContext]);
  const similarities = useMemo(() => adaptSimilarRows(rawReport?.similar_players || []), [rawReport?.similar_players]);
  const metricsSummary = useMemo(() => filterYouthMetricSummary(availableMetricsSummary(metrics)), [metrics]);

  const selectPlayer = (item) => {
    setSelectedPlayerId(String(item.id));
    setPlayerQuery(selectedLabel(item));
    setShowPlayerResults(false);
    setComparisonReport(null);
    setSelectedComparisonLabel("");
    setCompareQuery("");
    router.replace({ pathname: "/youth-scouting/reports", query: { youth_id: item.id } }, undefined, { shallow: true });
  };

  const selectSeason = (seasonId) => {
    if (!seasonId) return;
    setSelectedPlayerId(String(seasonId));
    setComparisonReport(null);
    setSelectedComparisonLabel("");
    setCompareQuery("");
    router.replace({ pathname: "/youth-scouting/reports", query: { youth_id: seasonId } }, undefined, { shallow: true });
  };

  const selectComparisonPlayer = async (item) => {
    const label = selectedLabel(item);
    setSelectedComparisonLabel(label);
    setCompareQuery(label);
    setShowCompareResults(false);
    setComparisonLoading(true);
    try {
      const data = await fetchJson(`/youth/players/${item.id}/report`);
      setComparisonReport(buildAdaptedReport(data));
    } catch (err) {
      console.error(err);
      setComparisonReport(null);
    } finally {
      setComparisonLoading(false);
    }
  };

  const toggleProspect = async () => {
    if (!selectedPlayerId || prospectLoading) return;
    setProspectLoading(true);
    setProspectMessage("");
    try {
      if (isProspect) {
        await deleteJson(`/youth/prospects/${selectedPlayerId}`);
        setIsProspect(false);
        setProspectMessage("Youth player removed from prospects.");
      } else {
        const res = await postJson("/youth/prospects", { youth_id: Number(selectedPlayerId) });
        setIsProspect(true);
        setProspectMessage(res?.added ? "Youth player added to prospects." : "Youth player is already in prospects.");
      }
    } catch (err) {
      setProspectMessage(err.message || "Unable to update youth prospect status.");
    } finally {
      setProspectLoading(false);
    }
  };

  const searchSeasons = meta?.seasons?.length ? meta.seasons : [DEFAULT_SEASON];

  return (
    <main className="nl-page px-4 py-8 text-slate-900 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-[1540px] space-y-6">
        <header className="surface-panel relative z-50 overflow-visible rounded-lg p-5 md:p-7">
          <div className="grid gap-5 2xl:grid-cols-[minmax(0,1fr)_420px] 2xl:items-start">
            <div className={report ? "grid gap-5 lg:grid-cols-[170px_minmax(0,1fr)]" : ""}>
              {report ? (
                <YouthReportPhotoUploader report={report} youthId={selectedPlayerId} me={me} />
              ) : null}

              <div>
                <p className="nl-kicker">Player report</p>
                <h1 className="mt-3 max-w-4xl text-4xl font-black tracking-[-0.04em] text-white md:text-6xl">
                  {report?.player?.name || "Youth scouting dossier"}
                </h1>
                <p className="mt-3 max-w-3xl text-sm leading-6 text-[#A0A8A3] md:text-base">
                  Position-based youth report using Eyeball statistics, competition context, raw metrics and positional percentiles.
                </p>
                {report ? (
                  <>
                    <div className="mt-5 flex flex-wrap items-center gap-2">
                      <span className="rounded-md border border-[#3A8967]/35 bg-[#2F7D5C]/15 px-3 py-1 text-xs font-black uppercase tracking-[0.14em] text-[#8CC7A7]">{positionGroup}</span>
                      <span className="rounded-md border border-white/10 bg-white/[0.035] px-3 py-1 text-xs font-bold text-slate-300">{report.player.team || "No club"}</span>
                      <span className="rounded-md border border-white/10 bg-white/[0.035] px-3 py-1 text-xs font-bold text-slate-300">{report.player.competition_name || "No competition"}</span>
                      <span className="rounded-md border border-white/10 bg-white/[0.035] px-3 py-1 text-xs font-bold text-slate-300">{report.player.nationality || "No nationality"}</span>
                    </div>
                    <div className="mt-5 flex flex-wrap gap-2">
                      <button
                        type="button"
                        onClick={toggleProspect}
                        disabled={prospectLoading}
                        className={isProspect ? "nl-button-secondary px-4 py-2 text-xs uppercase tracking-[0.14em]" : "nl-button-primary px-4 py-2 text-xs uppercase tracking-[0.14em]"}
                      >
                        {prospectLoading ? "Updating..." : isProspect ? "Remove prospect" : "Add prospect"}
                      </button>
                      {rawReport?.player?.provider_player_url ? (
                        <a href={rawReport.player.provider_player_url} target="_blank" rel="noreferrer" className="nl-button-primary px-4 py-2 text-xs uppercase tracking-[0.14em]">
                          Eyeball profile
                        </a>
                      ) : null}
                    </div>
                    {prospectMessage ? (
                      <p className={`mt-3 text-xs font-semibold ${isProspect ? "text-[#8CC7A7]" : "text-amber-200"}`}>
                        {prospectMessage}
                      </p>
                    ) : null}
                    <div className="mt-5 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
                      {[
                        ["Birth date", report.player.birth_date],
                        ["Birth year", report.player.birth_year],
                        ["Age", report.player.age],
                        ["Nationality", report.player.nationality],
                        ["Category", rawReport?.player?.age_category],
                        ["Strong foot", rawReport?.player?.strong_foot],
                      ].map(([label, value]) => (
                        <div key={label} className="rounded-lg border border-white/10 bg-white/[0.035] px-3 py-2">
                          <p className="text-[10px] font-black uppercase tracking-[0.14em] text-slate-400">{label}</p>
                          <p className="mt-1 truncate text-sm font-black text-white">{value || "-"}</p>
                        </div>
                      ))}
                    </div>
                  </>
                ) : null}
              </div>
            </div>

            <div className="relative z-[9999] space-y-3 2xl:pt-2">
              <div className="rounded-lg border border-white/10 bg-white/[0.035] p-3">
                <label htmlFor="youth-report-search-season" className="text-[10px] font-black uppercase tracking-[0.16em] text-slate-500">
                  Search season
                </label>
                <select
                  id="youth-report-search-season"
                  className="nl-field mt-2"
                  value={searchSeason}
                  onChange={(event) => {
                    setSearchSeason(event.target.value);
                    setPlayerResults([]);
                    if (!selectedPlayerId) {
                      setShowPlayerResults(playerQuery.trim().length >= 2);
                    }
                  }}
                >
                  <option value="">All seasons</option>
                  {searchSeasons.map((season) => (
                    <option key={season} value={season}>{formatYouthSeason(season)}</option>
                  ))}
                </select>
              </div>
              <PlayerSearch
                query={playerQuery}
                results={playerResults}
                visible={showPlayerResults}
                onFocus={() => {
                  if (selectedPlayerId) {
                    setPlayerQuery("");
                    setSelectedPlayerId("");
                  }
                  setShowPlayerResults(true);
                }}
                onQueryChange={(value) => {
                  setPlayerQuery(value);
                  setSelectedPlayerId("");
                  setRawReport(null);
                  setShowPlayerResults(true);
                }}
                onSelect={selectPlayer}
              />
              {report && seasons.length > 1 ? (
                <div className="rounded-lg border border-[#3A8967]/25 bg-[#2F7D5C]/10 p-3">
                  <label htmlFor="youth-report-season" className="text-[10px] font-black uppercase tracking-[0.16em] text-[#8CC7A7]">
                    Report season
                  </label>
                  <select
                    id="youth-report-season"
                    className="nl-field mt-2"
                    value={selectedSeasonId}
                    onChange={(event) => selectSeason(event.target.value)}
                  >
                    {seasons.map((season) => {
                      const seasonId = reportSeasonId(season);
                      return (
                        <option key={seasonId} value={seasonId}>
                          {[season.calendar, season.team, season.competition_name].filter(Boolean).join(" - ")}
                        </option>
                      );
                    })}
                  </select>
                </div>
              ) : null}
            </div>
          </div>
        </header>

        {error ? <ReportCard><p className="text-sm font-semibold text-red-300">{error}</p></ReportCard> : null}
        {loading ? <ReportCard><p className="text-sm text-[#A0A8A3]">Loading report...</p></ReportCard> : null}

        {report ? (
          <>
            {seasons.length > 1 ? (
              <ReportCard className="relative z-10 overflow-visible">
                <div className="mb-3 flex flex-col gap-1 md:flex-row md:items-end md:justify-between">
                  <div>
                    <p className="nl-kicker">Available reports</p>
                    <h2 className="mt-2 text-xl font-black text-white">Season history</h2>
                  </div>
                  <p className="text-xs font-semibold text-slate-500">Click a season to open the full report.</p>
                </div>
                <SeasonSelector seasons={seasons} selectedSeasonId={selectedSeasonId} onSelect={selectSeason} />
              </ReportCard>
            ) : (
              <SeasonSelector seasons={seasons} selectedSeasonId={selectedSeasonId} onSelect={selectSeason} />
            )}
            <SeasonStatistics report={report} metrics={metrics} hideExpectedMetrics />

            <div className="grid gap-4 xl:grid-cols-[0.82fr_1.18fr_1fr]">
              <PositionCard player={report.player} />
              <PlayerProfileCard categories={profileCategoriesData} />
              <CharacteristicsCard characteristics={characteristics} />
            </div>

            <AdvancedCharacteristics
              metrics={metrics}
              player={report.player}
              rawMode={rawMode}
              setRawMode={setRawMode}
              context={percentileContext}
              setContext={setPercentileContext}
            />

            <div className="grid gap-4 xl:grid-cols-2">
              <PlayerRadarComparison report={report} comparisonReport={comparisonReport} context={percentileContext} />
              <PlayerStatsComparison
                report={report}
                comparisonReport={comparisonReport}
                context={percentileContext}
                query={compareQuery}
                results={compareResults}
                showResults={showCompareResults}
                loading={comparisonLoading}
                onSearch={(value) => {
                  setCompareQuery(value);
                  setSelectedComparisonLabel("");
                  setShowCompareResults(value.trim().length >= 2);
                }}
                onSelect={selectComparisonPlayer}
              />
            </div>

            <SimilarPlayersCard
              similarities={similarities}
              loading={false}
              onOpen={(sim) => {
                if (sim?.player_b_id) {
                  window.open(`/youth-scouting/reports?youth_id=${sim.player_b_id}`, "_blank", "noopener,noreferrer");
                }
              }}
            />

            <div className="grid gap-4 xl:grid-cols-2">
              <PlayerSeasonRadarComparison report={report} context={percentileContext} />
              <YouthScoreHistory data={scoreHistory} onSelect={selectSeason} />
            </div>

            <ReportCard>
              <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                <div>
                  <p className="text-[11px] font-black uppercase tracking-[0.22em] text-[#8CC7A7]">Data coverage</p>
                  <p className="mt-2 text-sm text-[#A0A8A3]">Configured report metrics available: {metricsSummary.available.length}. Missing or unavailable for this youth provider: {metricsSummary.missing.length}.</p>
                </div>
                <p className="text-xs text-[#6F7772]">Missing Eyeball values are shown as "-" and are never coerced to zero.</p>
              </div>
            </ReportCard>

            <YouthScoutingReportsPanel report={report} youthId={selectedPlayerId} me={me} />
          </>
        ) : !loading ? (
          <ReportCard className="py-14 text-center">
            <p className="text-lg font-black text-white">Search a player to open a youth scouting report.</p>
            <p className="mt-2 text-sm text-[#A0A8A3]">All statistics and percentiles are loaded from the Eyeball youth table.</p>
          </ReportCard>
        ) : null}
      </div>
    </main>
  );
}
