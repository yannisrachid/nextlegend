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
import { deleteJson, fetchJson, postJson } from "@/lib/api";
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
    const x = sorted.length <= 1 ? 50 : 8 + (index / (sorted.length - 1)) * 84;
    const y = Number.isFinite(score) ? 82 - ((score - min) / range) * 60 : 82;
    return { ...row, x, y, score };
  });
  const path = points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
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
            <svg viewBox="0 0 100 92" className="h-32 w-full overflow-visible" role="img" aria-label="Youth score history">
              <path d="M 8 82 H 92" stroke="rgba(255,255,255,0.12)" strokeWidth="0.7" />
              {path ? <path d={path} fill="none" stroke="#559A78" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" /> : null}
              {points.map((point) => (
                <g key={point.player_season_id || point.calendar}>
                  <circle cx={point.x} cy={point.y} r="2.6" fill="#8CC7A7" stroke="#06100C" strokeWidth="1.2" />
                  <text x={point.x} y={point.y - 7} textAnchor="middle" className="fill-white text-[5px] font-black">
                    {Number.isFinite(point.score) ? point.score.toFixed(1) : "-"}
                  </text>
                </g>
              ))}
            </svg>
            <div className="mt-3 space-y-2">
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

function YouthContextCard({ player }) {
  return (
    <ReportCard>
      <p className="nl-kicker">Youth context</p>
      <h3 className="mt-2 text-xl font-black text-white">Identity and cohort</h3>
      <div className="mt-4 grid grid-cols-2 gap-3">
        {[
          ["Birth date", player.birth_date],
          ["Birth year", player.birth_year],
          ["Age", player.age],
          ["Nationality", player.nationality],
          ["Minutes", formatValue(player.minutes_played, "integer")],
          ["Matches", formatValue(player.matches_played, "integer")],
          ["Provider", "Eyeball"],
        ].map(([label, value]) => (
          <div key={label} className="rounded-lg border border-white/10 bg-white/[0.035] p-3">
            <p className="text-[10px] font-black uppercase tracking-[0.16em] text-[#6F7772]">{label}</p>
            <p className="mt-1 truncate text-sm font-black text-white">{value || "-"}</p>
          </div>
        ))}
      </div>
    </ReportCard>
  );
}

export default function YouthReportsPage() {
  const router = useRouter();
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
                <div className="w-full max-w-[170px]">
                  <div className="flex h-[210px] items-center justify-center overflow-hidden rounded-lg border border-[#3A8967]/25 bg-[#06100C] p-2 shadow-sm">
                    <div className="flex h-full w-full items-center justify-center rounded-md border border-white/10 bg-white/[0.04] text-4xl font-black text-[#8CC7A7]">
                      {String(report.player.name || "YP").split(/\s+/).map((part) => part[0]).join("").slice(0, 2).toUpperCase()}
                    </div>
                  </div>
                  <div className="mt-3 flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.035] px-3 py-2 shadow-sm">
                    <ClubLogo name={report.player.team} className="h-8 w-8 rounded-md" />
                    <span className="min-w-0 truncate text-xs font-black text-slate-300">{report.player.team || "No club"}</span>
                  </div>
                </div>
              ) : null}

              <div>
                <p className="nl-kicker">Player report</p>
                <h1 className="mt-3 max-w-4xl text-4xl font-black tracking-[-0.04em] text-white md:text-6xl">
                  {report?.player?.name || "Youth scouting dossier"}
                </h1>
                <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600 md:text-base">
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
                      <a href="/youth-scouting/ranking" className="nl-button-secondary px-4 py-2 text-xs uppercase tracking-[0.14em]">
                        Back to ranking
                      </a>
                      <button
                        type="button"
                        onClick={toggleProspect}
                        disabled={prospectLoading}
                        className={isProspect ? "nl-button-secondary px-4 py-2 text-xs uppercase tracking-[0.14em]" : "nl-button-primary px-4 py-2 text-xs uppercase tracking-[0.14em]"}
                      >
                        {prospectLoading ? "Updating..." : isProspect ? "Remove prospect" : "Add prospect"}
                      </button>
                      <a href="/youth-scouting/prospects" className="nl-button-secondary px-4 py-2 text-xs uppercase tracking-[0.14em]">
                        Youth prospects
                      </a>
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
        {loading ? <ReportCard><p className="text-sm text-slate-600">Loading report...</p></ReportCard> : null}

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

            <div className="grid gap-4 xl:grid-cols-3">
              <PlayerSeasonRadarComparison report={report} context={percentileContext} />
              <YouthScoreHistory data={scoreHistory} onSelect={selectSeason} />
              <YouthContextCard player={report.player} />
            </div>

            <ReportCard>
              <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                <div>
                  <p className="text-[11px] font-black uppercase tracking-[0.22em] text-[#8CC7A7]">Data coverage</p>
                  <p className="mt-2 text-sm text-slate-600">Configured report metrics available: {metricsSummary.available.length}. Missing or unavailable for this youth provider: {metricsSummary.missing.length}.</p>
                </div>
                <p className="text-xs text-slate-500">Missing Eyeball values are shown as "-" and are never coerced to zero.</p>
              </div>
            </ReportCard>
          </>
        ) : !loading ? (
          <ReportCard className="py-14 text-center">
            <p className="text-lg font-black text-slate-950">Search a player to open a youth scouting report.</p>
            <p className="mt-2 text-sm text-slate-500">All statistics and percentiles are loaded from the Eyeball youth table.</p>
          </ReportCard>
        ) : null}
      </div>
    </main>
  );
}
