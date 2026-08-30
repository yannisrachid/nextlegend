export const DEFAULT_SCOUTING_COMPETITION = "Big 5 Leagues";
export const DEFAULT_SCOUTING_SEASON = "2026/2027";
export const DEFAULT_MIN_MINUTES = 90;
export const DEFAULT_AGE_MIN = 15;
export const DEFAULT_AGE_MAX = 45;
export const DEFAULT_LIMIT = 30;

const POSITION_ORDER = [
  "GK",
  "RB",
  "RCB",
  "CB",
  "LCB",
  "LB",
  "RWB",
  "LWB",
  "DMF",
  "RDMF",
  "LDMF",
  "RMF",
  "RCMF",
  "CMF",
  "LCMF",
  "LMF",
  "RAMF",
  "AMF",
  "LAMF",
  "RW",
  "LW",
  "RWF",
  "LWF",
  "SS",
  "CF",
];

const ROLE_ORDER_HINTS = [
  ["goalkeeper", "keeper"],
  ["centre back", "center back", "central defender", "ball-playing centre back", "ball playing centre back"],
  ["right back", "right backs", "right fullback", "right fullbacks", "full-back right", "rwb"],
  ["left back", "left backs", "left fullback", "left fullbacks", "full-back left", "lwb"],
  ["defensive midfielder", "defensive midfielders", "holding midfielder", "dmf"],
  ["central midfielder", "central midfielders", "centre midfielder", "centre midfielders", "cmf", "box-to-box"],
  ["attacking midfielder", "attacking midfielders", "amf", "playmaker"],
  ["right winger", "right wingers", "right midfielder", "right midfielders", "rw", "rmf"],
  ["left winger", "left wingers", "left midfielder", "left midfielders", "lw", "lmf"],
  ["forward", "forwards", "striker", "strikers", "centre forward", "centre forwards", "center forward", "center forwards", "cf"],
];

const EXCLUDED_OPTION_VALUES = new Set(["NA", "<NA>", "N/A", "NONE", "NULL", "-", "--"]);
const COMPETITION_ORDER = [
  DEFAULT_SCOUTING_COMPETITION,
  "Big 8 Leagues",
  "Big 10 Competitions",
  "First Divisions Only",
  "Lower Divisions Only",
  "Second Divisions Only",
];

export const normalizeOption = (value) => String(value || "").trim();

export const uniqueOptions = (values = []) => {
  const seen = new Set();
  return values.filter((value) => {
    const normalized = normalizeOption(value);
    if (!normalized || seen.has(normalized)) return false;
    seen.add(normalized);
    return true;
  });
};

const isValidOption = (value) => {
  const normalized = normalizeOption(value).toUpperCase();
  return Boolean(normalized) && !EXCLUDED_OPTION_VALUES.has(normalized);
};

export const seasonSortKey = (value) => {
  const raw = normalizeOption(value);
  const years = raw.match(/\d{4}/g);
  if (!years?.length) return 0;
  const start = Number(years[0]) || 0;
  const end = Number(years[years.length - 1]) || start;
  return end * 10000 + start;
};

export const sortSeasonsDesc = (values = []) =>
  uniqueOptions(values).sort((a, b) => {
    const diff = seasonSortKey(b) - seasonSortKey(a);
    return diff || String(b).localeCompare(String(a), undefined, { sensitivity: "base" });
  });

export const withDefaultSeason = (values = []) =>
  sortSeasonsDesc([DEFAULT_SCOUTING_SEASON, ...values]);

export const positionSortValue = (value) => {
  const normalized = normalizeOption(value).toUpperCase();
  const index = POSITION_ORDER.indexOf(normalized);
  if (index >= 0) return index;
  if (normalized.includes("GK")) return 0;
  if (normalized.includes("RB")) return 1;
  if (normalized.includes("CB")) return 3;
  if (normalized.includes("LB")) return 5;
  if (normalized.includes("WB")) return 6;
  if (normalized.includes("DM")) return 8;
  if (normalized.includes("CM")) return 12;
  if (normalized.includes("AM")) return 17;
  if (normalized.includes("RW")) return 19;
  if (normalized.includes("LW")) return 20;
  if (normalized.includes("CF") || normalized.includes("ST")) return 24;
  return 99;
};

export const sortPositions = (values = []) =>
  uniqueOptions(values)
    .filter(isValidOption)
    .sort((a, b) => {
      const diff = positionSortValue(a) - positionSortValue(b);
      return diff || String(a).localeCompare(String(b), undefined, { sensitivity: "base" });
    });

const roleSortValue = (value) => {
  const normalized = normalizeOption(value).toLowerCase();
  const index = ROLE_ORDER_HINTS.findIndex((hints) =>
    hints.some((hint) => (hint.length <= 3 ? normalized === hint : normalized.includes(hint)))
  );
  return index >= 0 ? index : 99;
};

export const sortRoles = (values = []) =>
  uniqueOptions(values)
    .filter(isValidOption)
    .sort((a, b) => {
      const diff = roleSortValue(a) - roleSortValue(b);
      return diff || String(a).localeCompare(String(b), undefined, { sensitivity: "base" });
    });

export const compareCompetitions = (a, b) => {
  const left = normalizeOption(typeof a === "string" ? a : a?.name);
  const right = normalizeOption(typeof b === "string" ? b : b?.name);
  const leftIndex = COMPETITION_ORDER.indexOf(left);
  const rightIndex = COMPETITION_ORDER.indexOf(right);
  if (leftIndex >= 0 || rightIndex >= 0) {
    return (leftIndex >= 0 ? leftIndex : 99) - (rightIndex >= 0 ? rightIndex : 99);
  }
  return left.localeCompare(right, undefined, { sensitivity: "base" });
};

export const sortCompetitionNames = (values = []) =>
  uniqueOptions([DEFAULT_SCOUTING_COMPETITION, ...values]).sort(compareCompetitions);

export const parseIntegerInput = (value, fallback = 0) => {
  const raw = String(value ?? "").replace(/[^\d]/g, "");
  if (!raw) return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
};

export const formatFilterValue = (value, fallback) => normalizeOption(value) || fallback;
