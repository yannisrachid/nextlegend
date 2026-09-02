import { METRIC_LABELS } from "@/lib/metricLabels";

export const POSITION_GROUPS = {
  GK: { short: "GK", label: "Goalkeepers", pitch: { x: 50, y: 88 } },
  Goalkeepers: { short: "GK", label: "Goalkeepers", pitch: { x: 50, y: 88 } },
  "Centre Backs": { short: "CB", label: "Centre backs", pitch: { x: 50, y: 70 } },
  Fullbacks: { short: "FB", label: "Fullbacks", pitch: { x: 50, y: 64 } },
  "Left Backs": { short: "LB", label: "Left backs", pitch: { x: 18, y: 64 } },
  "Right Backs": { short: "RB", label: "Right backs", pitch: { x: 82, y: 64 } },
  "Defensive Midfielders": { short: "DM", label: "Defensive midfielders", pitch: { x: 50, y: 54 } },
  "Central Midfielders": { short: "CM", label: "Central midfielders", pitch: { x: 50, y: 43 } },
  "Attacking Midfielders": { short: "AM", label: "Attacking midfielders", pitch: { x: 50, y: 30 } },
  "Left Wingers": { short: "LW", label: "Left wingers", pitch: { x: 20, y: 22 } },
  "Right Wingers": { short: "RW", label: "Right wingers", pitch: { x: 80, y: 22 } },
  Wingers: { short: "W", label: "Wingers", pitch: { x: 50, y: 22 } },
  "Centre Forwards": { short: "CF", label: "Centre forwards", pitch: { x: 50, y: 14 } },
  Forwards: { short: "FW", label: "Forwards", pitch: { x: 50, y: 14 } },
};

const POSITION_CODE_MAP = {
  G: "Goalkeepers", GB: "Goalkeepers", GK: "Goalkeepers",
  DC: "Centre Backs", CB: "Centre Backs", LCB: "Centre Backs", RCB: "Centre Backs",
  DG: "Left Backs", LB: "Left Backs", LWB: "Left Backs", DD: "Right Backs", RB: "Right Backs", RWB: "Right Backs",
  MDC: "Defensive Midfielders", DM: "Defensive Midfielders", DMF: "Defensive Midfielders",
  MC: "Central Midfielders", CM: "Central Midfielders", CMF: "Central Midfielders", LCMF: "Central Midfielders", RCMF: "Central Midfielders",
  MOC: "Attacking Midfielders", AM: "Attacking Midfielders", AMF: "Attacking Midfielders",
  MG: "Left Wingers", AG: "Left Wingers", LW: "Left Wingers", LWF: "Left Wingers", MD: "Right Wingers", AD: "Right Wingers", RW: "Right Wingers", RWF: "Right Wingers",
  ATT: "Centre Forwards", BU: "Centre Forwards", CF: "Centre Forwards", ST: "Centre Forwards",
};

export const lowerIsBetter = new Set([
  "fouls_per_90", "yellow_cards_per_90", "red_cards_per_90", "goals_conceded_per_90", "shots_against_per_90", "xg_against_per_90",
]);

const m = (key, options = {}) => ({
  key,
  label: options.label || METRIC_LABELS[key] || key.replace(/_/g, " "),
  format: options.format || (key.includes("percent") || key.endsWith("_rate") ? "percent" : "number"),
  lowerIsBetter: options.lowerIsBetter ?? lowerIsBetter.has(key),
  positions: options.positions || null,
  description: options.description || options.label || METRIC_LABELS[key] || key.replace(/_/g, " "),
});

export const metricGroups = {
  goalkeeping: {
    label: "Goalkeeping",
    metrics: [
      m("goals_prevented_per_90", { label: "Goals prevented /90" }),
      m("save_percent", { label: "Save rate %" }),
      m("goals_conceded_per_90", { label: "Goals conceded /90", lowerIsBetter: true }),
      m("xg_against_per_90", { label: "xG against /90", lowerIsBetter: true }),
      m("shots_against_per_90", { label: "Shots against /90", lowerIsBetter: true }),
      m("exits_per_90", { label: "Exits /90" }),
      m("aerial_duels_gk_per_90", { label: "GK aerials /90" }),
    ],
  },
  finishing: { label: "Finishing", metrics: [m("goals_per_90", { label: "Goals /90" }), m("non_penalty_goals_per_90", { label: "Non-penalty goals /90" }), m("xg_per_90", { label: "xG /90" }), m("shots_per_90", { label: "Shots /90" }), m("shots_on_target_percent", { label: "Shots on target %" }), m("goal_conversion_rate", { label: "Conversion" }), m("touches_in_penalty_area_per_90", { label: "Box touches /90" }), m("headed_goals_per_90", { label: "Headed goals /90" })] },
  creation: { label: "Creation", metrics: [m("assists_per_90", { label: "Assists /90" }), m("xa_per_90", { label: "xA /90" }), m("shot_assists_per_90", { label: "Shot assists /90" }), m("key_passes_per_90", { label: "Key passes /90" }), m("passes_to_penalty_area_per_90", { label: "Passes to penalty area /90" }), m("crosses_per_90", { label: "Crosses /90" }), m("accurate_crosses_percent", { label: "Cross accuracy %" }), m("smart_passes_per_90", { label: "Smart passes /90" })] },
  technique: { label: "Technique", metrics: [m("dribbles_per_90", { label: "Dribbles /90" }), m("successful_dribbles_percent", { label: "Successful dribbles %" }), m("progressive_runs_per_90", { label: "Progressive runs /90" }), m("accelerations_per_90", { label: "Accelerations /90" }), m("offensive_duels_per_90", { label: "Offensive duels /90" }), m("offensive_duels_won_percent", { label: "Offensive duels won %" }), m("fouls_suffered_per_90", { label: "Fouls suffered /90" })] },
  buildUp: { label: "Build-up", metrics: [m("passes_per_90", { label: "Passes /90" }), m("accurate_passes_percent", { label: "Pass accuracy %" }), m("forward_passes_per_90", { label: "Forward passes /90" }), m("backward_passes_per_90", { label: "Backward passes /90" }), m("lateral_passes_per_90", { label: "Lateral passes /90" }), m("long_passes_per_90", { label: "Long passes /90" }), m("accurate_long_passes_percent", { label: "Long pass accuracy %" }), m("progressive_passes_per_90", { label: "Progressive passes /90" }), m("accurate_progressive_passes_percent", { label: "Progressive pass accuracy %" }), m("passes_to_final_third_per_90", { label: "Passes to final third /90" })] },
  defense: { label: "Defense", metrics: [m("successful_def_actions_per_90", { label: "Defensive actions /90" }), m("def_duels_per_90", { label: "Defensive duels /90" }), m("def_duels_won_percent", { label: "Defensive duels won %" }), m("interceptions_per_90", { label: "Interceptions /90" }), m("interceptions_padj", { label: "Interceptions PAdj" }), m("sliding_tackles_per_90", { label: "Sliding tackles /90" }), m("blocked_shots_per_90", { label: "Blocked shots /90" }), m("fouls_per_90", { label: "Fouls /90", lowerIsBetter: true })] },
  aerial: { label: "Aerial", metrics: [m("aerial_duels_per_90", { label: "Aerial duels /90" }), m("aerial_duels_won_percent", { label: "Aerial duels won %" }), m("headed_goals", { label: "Headed goals" }), m("headed_goals_per_90", { label: "Headed goals /90" }), m("aerial_duels_gk_per_90", { label: "GK aerials /90" })] },
};

export const metricGroupOrder = ["finishing", "creation", "technique", "buildUp", "defense", "aerial"];

export const metricGroupOrderForPosition = (positionGroup) =>
  positionGroup === "Goalkeepers"
    ? ["goalkeeping", "buildUp", "aerial", "defense"]
    : metricGroupOrder;

export const profileCategories = [
  { key: "finishing", label: "Finishing", metrics: ["goals_per_90", "non_penalty_goals_per_90", "xg_per_90", "shots_per_90", "shots_on_target_percent", "goal_conversion_rate"] },
  { key: "creation", label: "Creation", metrics: ["assists_per_90", "xa_per_90", "shot_assists_per_90", "key_passes_per_90", "passes_to_penalty_area_per_90", "smart_passes_per_90"] },
  { key: "technique", label: "Technique", metrics: ["successful_dribbles_percent", "dribbles_per_90", "progressive_runs_per_90", "accelerations_per_90", "offensive_duels_won_percent"] },
  { key: "buildUp", label: "Build-up", metrics: ["passes_per_90", "accurate_passes_percent", "forward_passes_per_90", "progressive_passes_per_90", "accurate_progressive_passes_percent", "passes_to_final_third_per_90"] },
  { key: "aerial", label: "Aerial", metrics: ["aerial_duels_per_90", "aerial_duels_won_percent", "headed_goals_per_90"] },
  { key: "defense", label: "Defense", metrics: ["successful_def_actions_per_90", "def_duels_per_90", "def_duels_won_percent", "interceptions_per_90", "interceptions_padj", "sliding_tackles_per_90", "fouls_per_90"] },
];

export const radarMetricsByPosition = {
  Goalkeepers: ["goals_prevented_per_90", "def_duels_won_percent", "save_percent", "aerial_duels_won_percent", "aerial_duels_gk_per_90", "accurate_passes_percent", "passes_per_90", "accurate_progressive_passes_percent", "progressive_passes_per_90"],
  "Centre Backs": ["successful_def_actions_per_90", "def_duels_won_percent", "interceptions_padj", "aerial_duels_won_percent", "passes_per_90", "accurate_passes_percent", "progressive_passes_per_90", "accurate_long_passes_percent"],
  Fullbacks: ["def_duels_won_percent", "successful_def_actions_per_90", "interceptions_per_90", "progressive_runs_per_90", "successful_dribbles_percent", "progressive_passes_per_90", "passes_to_final_third_per_90", "key_passes_per_90", "accurate_crosses_percent"],
  "Left Backs": ["def_duels_won_percent", "successful_def_actions_per_90", "successful_attacks_per_90", "interceptions_per_90", "progressive_runs_per_90", "successful_dribbles_percent", "progressive_passes_per_90", "passes_to_final_third_per_90", "passes_to_penalty_area_per_90", "xa_per_90"],
  "Right Backs": ["def_duels_won_percent", "successful_def_actions_per_90", "successful_attacks_per_90", "interceptions_per_90", "progressive_runs_per_90", "successful_dribbles_percent", "progressive_passes_per_90", "passes_to_final_third_per_90", "passes_to_penalty_area_per_90", "xa_per_90"],
  "Defensive Midfielders": ["successful_def_actions_per_90", "interceptions_per_90", "def_duels_won_percent", "def_duels_per_90", "aerial_duels_won_percent", "passes_per_90", "accurate_passes_percent", "progressive_passes_per_90", "accurate_progressive_passes_percent", "passes_to_final_third_per_90"],
  "Central Midfielders": ["progressive_passes_per_90", "accurate_progressive_passes_percent", "passes_to_final_third_per_90", "progressive_runs_per_90", "successful_dribbles_percent", "successful_def_actions_per_90", "def_duels_won_percent", "xa_per_90", "key_passes_per_90", "xg_per_90"],
  "Attacking Midfielders": ["xa_per_90", "key_passes_per_90", "smart_passes_per_90", "through_passes_per_90", "deep_completions_per_90", "passes_to_penalty_area_per_90", "progressive_passes_per_90", "progressive_runs_per_90", "touches_in_penalty_area_per_90", "xg_per_90"],
  "Left Wingers": ["progressive_runs_per_90", "dribbles_per_90", "successful_dribbles_percent", "xa_per_90", "key_passes_per_90", "passes_to_penalty_area_per_90", "touches_in_penalty_area_per_90", "xg_per_90", "goals_per_90", "accurate_crosses_percent"],
  "Right Wingers": ["progressive_runs_per_90", "dribbles_per_90", "successful_dribbles_percent", "xa_per_90", "key_passes_per_90", "passes_to_penalty_area_per_90", "touches_in_penalty_area_per_90", "xg_per_90", "goals_per_90", "accurate_crosses_percent"],
  Wingers: ["progressive_runs_per_90", "dribbles_per_90", "successful_dribbles_percent", "key_passes_per_90", "passes_to_penalty_area_per_90", "goals_per_90", "assists_per_90", "accurate_crosses_percent"],
  "Centre Forwards": ["xg_per_90", "goals_per_90", "shots_per_90", "shots_on_target_percent", "goal_conversion_rate", "touches_in_penalty_area_per_90", "aerial_duels_won_percent", "aerial_duels_per_90", "passes_received_per_90", "xa_per_90"],
  Forwards: ["goals_per_90", "shots_per_90", "shots_on_target_percent", "goal_conversion_rate", "touches_in_penalty_area_per_90", "aerial_duels_won_percent", "aerial_duels_per_90", "assists_per_90"],
};

export const positionRelevantMetricGroups = {
  Goalkeepers: ["goalkeeping", "buildUp", "aerial"],
  "Centre Backs": ["defense", "aerial", "buildUp"],
  Fullbacks: ["defense", "buildUp", "technique", "creation"],
  "Left Backs": ["defense", "buildUp", "technique", "creation"],
  "Right Backs": ["defense", "buildUp", "technique", "creation"],
  "Defensive Midfielders": ["buildUp", "defense", "technique", "aerial"],
  "Central Midfielders": ["buildUp", "creation", "technique", "defense"],
  "Attacking Midfielders": ["creation", "technique", "finishing", "buildUp"],
  "Left Wingers": ["technique", "creation", "finishing", "buildUp"],
  "Right Wingers": ["technique", "creation", "finishing", "buildUp"],
  Wingers: ["technique", "creation", "finishing", "buildUp"],
  "Centre Forwards": ["finishing", "creation", "aerial", "technique"],
  Forwards: ["finishing", "creation", "aerial", "technique"],
};

export const normalizePositionGroup = (assignedRole, position) => {
  if (assignedRole === "GK") return "Goalkeepers";
  if (assignedRole && POSITION_GROUPS[assignedRole]) return assignedRole;
  const text = String(position || assignedRole || "").toUpperCase().trim();
  const firstCode = text.split(/[ ,/]+/).find(Boolean);
  return POSITION_CODE_MAP[firstCode] || assignedRole || "Central Midfielders";
};

export const getPositionMeta = (assignedRole, position) => {
  const group = normalizePositionGroup(assignedRole, position);
  return POSITION_GROUPS[group] || { short: String(position || "POS").slice(0, 3).toUpperCase(), label: group, pitch: { x: 50, y: 50 } };
};

export const getMetricConfig = (key) => {
  for (const group of Object.values(metricGroups)) {
    const found = group.metrics.find((metric) => metric.key === key);
    if (found) return found;
  }
  return m(key);
};

export const getRelevantMetricKeys = (positionGroup) => {
  const groups = positionRelevantMetricGroups[positionGroup] || metricGroupOrder;
  const keys = new Set();
  groups.forEach((groupKey) => {
    (metricGroups[groupKey]?.metrics || []).forEach((metric) => keys.add(metric.key));
  });
  return keys;
};

export const getRadarMetricKeys = (positionGroup, fallback = []) => {
  const configured = radarMetricsByPosition[positionGroup] || fallback || [];
  return Array.from(new Set(configured)).slice(0, 10);
};
