import { useEffect, useMemo, useState } from "react";
import ClubLogo from "@/components/ClubLogo";
import { fetchJson, fetchJsonCached } from "@/lib/api";

const TM_BASE_URL = "https://www.transfermarkt.com";
const DEFAULT_COMPETITION = "Big 5 Leagues";
const DEFAULT_SEASON = "2026/2027";
const DEFAULT_MIN_MINUTES = 90;
const DEFAULT_AGE_MIN = 15;
const DEFAULT_AGE_MAX = 45;
const DEFAULT_LIMIT = 30;

const DEFAULT_FILTERS = {
  competition: DEFAULT_COMPETITION,
  season: DEFAULT_SEASON,
  role: "",
  position: "",
  team: "",
  min_minutes: DEFAULT_MIN_MINUTES,
  age_min: DEFAULT_AGE_MIN,
  age_max: DEFAULT_AGE_MAX,
  limit: DEFAULT_LIMIT,
};

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

const EXCLUDED_POSITION_VALUES = new Set(["NA", "<NA>", "N/A", "NONE", "NULL", "-", "--"]);

const Card = ({ children, className = "", ...props }) => (
  <div
    className={`surface-panel rounded-lg p-4 ${className}`}
    {...props}
  >
    {children}
  </div>
);

const Badge = ({ children }) => (
  <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-semibold text-slate-700">
    {children}
  </span>
);

const Label = ({ children, htmlFor }) => (
  <label htmlFor={htmlFor} className="text-xs font-bold uppercase tracking-[0.16em] text-slate-500">
    {children}
  </label>
);

const Select = ({ value, onChange, children, id, name, ariaLabel }) => (
  <select
    id={id}
    name={name || id}
    aria-label={ariaLabel}
    className="nl-field"
    value={value}
    onChange={onChange}
  >
    {children}
  </select>
);

const normalizeSelectValue = (value) => String(value || "").trim();

const isValidPositionValue = (value) => {
  const normalized = normalizeSelectValue(value).toUpperCase();
  return Boolean(normalized) && !EXCLUDED_POSITION_VALUES.has(normalized);
};

const seasonSortKey = (value) => {
  const raw = String(value || "");
  const match = raw.match(/\d{4}/g);
  if (!match?.length) return 0;
  return Number(match[0]) || 0;
};

const sortSeasonsDesc = (values = []) =>
  uniqueOptions(values)
    .sort((a, b) => {
      const diff = seasonSortKey(b) - seasonSortKey(a);
      return diff || String(b).localeCompare(String(a));
    });

const uniqueOptions = (values = []) => {
  const seen = new Set();
  return values.filter((value) => {
    const normalized = normalizeSelectValue(value);
    if (!normalized || seen.has(normalized)) return false;
    seen.add(normalized);
    return true;
  });
};

const positionSortValue = (value) => {
  const normalized = normalizeSelectValue(value).toUpperCase();
  const index = POSITION_ORDER.indexOf(normalized);
  if (index >= 0) return index;
  if (normalized.includes("GK")) return 0;
  if (normalized.includes("CB")) return 3;
  if (normalized.includes("RB")) return 1;
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

const sortPositions = (values = []) =>
  uniqueOptions(values)
    .filter(isValidPositionValue)
    .sort((a, b) => {
      const diff = positionSortValue(a) - positionSortValue(b);
      return diff || String(a).localeCompare(String(b));
    });

const roleSortValue = (value) => {
  const normalized = normalizeSelectValue(value).toLowerCase();
  const index = ROLE_ORDER_HINTS.findIndex((hints) =>
    hints.some((hint) => (hint.length <= 3 ? normalized === hint : normalized.includes(hint)))
  );
  return index >= 0 ? index : 99;
};

const sortRoles = (values = []) =>
  uniqueOptions(values).sort((a, b) => {
    const diff = roleSortValue(a) - roleSortValue(b);
    return diff || String(a).localeCompare(String(b));
  });

const parseIntegerInput = (value, fallback = 0) => {
  const raw = String(value ?? "").replace(/[^\d]/g, "");
  if (!raw) return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const formatFilterValue = (value, fallback) => normalizeSelectValue(value) || fallback;

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

export default function RankingPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [prospectIds, setProspectIds] = useState(new Set());

  const [competitions, setCompetitions] = useState([]);
  const [seasons, setSeasons] = useState([]);
  const [roles, setRoles] = useState([]);
  const [positions, setPositions] = useState([]);
  const [teams, setTeams] = useState([]);

  const [filters, setFilters] = useState(DEFAULT_FILTERS);

  useEffect(() => {
    const loadMeta = async () => {
      try {
        const [comps, seasonsData, rolesData] = await Promise.all([
          fetchJsonCached("/meta/competitions"),
          fetchJsonCached("/meta/seasons"),
          fetchJsonCached("/meta/roles"),
        ]);
        setCompetitions(comps);
        setSeasons(seasonsData);
        setRoles(rolesData);
      } catch (err) {
        console.error(err);
      }
    };
    loadMeta();
  }, []);

  useEffect(() => {
    fetchJson("/prospects/ids")
      .then((res) => setProspectIds(new Set(res?.player_ids || [])))
      .catch(() => setProspectIds(new Set()));
  }, []);

  useEffect(() => {
    const loadDependent = async () => {
      try {
        const [positionsData, teamsData] = await Promise.all([
          fetchJson("/meta/positions", {
            competition: filters.competition,
            season: filters.season,
          }),
          fetchJson("/meta/teams", {
            competition: filters.competition,
            season: filters.season,
          }),
        ]);
        setPositions(positionsData);
        setTeams(teamsData);
      } catch (err) {
        console.error(err);
      }
    };
    loadDependent();
  }, [filters.competition, filters.season]);

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      setError("");
      try {
        const params = {
          competition: filters.competition,
          season: filters.season,
          role: filters.role,
          position: filters.position,
          team: filters.team,
          min_minutes: filters.min_minutes,
          age_min: filters.age_min || undefined,
          age_max: filters.age_max || undefined,
          limit: filters.limit,
          offset: page * filters.limit,
        };
        const res = await fetchJson("/ranking/page", params);
        setItems(res.items || []);
        setTotal(res.total || 0);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, [filters, page]);

  const competitionOptions = useMemo(() => {
    const aggregateNames = ["Big 5 Leagues", "Big 8 Leagues", "First Divisions Only", "Lower Divisions Only"];
    const names = uniqueOptions([DEFAULT_COMPETITION, ...competitions.map((c) => c.name)]);
    const aggregates = aggregateNames.filter((name) => names.includes(name));
    const regular = names.filter((name) => !aggregateNames.includes(name)).sort((a, b) => String(a).localeCompare(String(b)));
    return ["", ...aggregates, ...regular];
  }, [competitions]);

  const seasonOptions = useMemo(() => {
    const allSeasons = sortSeasonsDesc([DEFAULT_SEASON, filters.season, ...seasons]);
    if (!filters.competition) {
      return ["", ...allSeasons];
    }
    const found = competitions.find((c) => c.name === filters.competition);
    if (!found || !found.seasons) {
      return ["", ...allSeasons];
    }
    return ["", ...sortSeasonsDesc([DEFAULT_SEASON, filters.season, ...found.seasons, ...seasons])];
  }, [competitions, seasons, filters.competition, filters.season]);

  const roleOptions = useMemo(() => ["", ...sortRoles([filters.role, ...roles])], [roles, filters.role]);
  const positionOptions = useMemo(() => ["", ...sortPositions([filters.position, ...positions])], [positions, filters.position]);
  const teamOptions = useMemo(() => ["", ...uniqueOptions([filters.team, ...teams]).sort((a, b) => String(a).localeCompare(String(b)))], [teams, filters.team]);

  const totalPages = Math.max(1, Math.ceil(total / filters.limit));
  const pageLabel = `Page ${page + 1} / ${totalPages}`;

  const updateFilter = (patch) => {
    setFilters((prev) => ({ ...prev, ...patch }));
    setPage(0);
  };

  const updateNumericFilter = (key, value, fallback = 0) => {
    const parsed = parseIntegerInput(value, fallback);
    setFilters((prev) => {
      const next = { ...prev, [key]: parsed };
      if (key === "age_min" && parsed > Number(prev.age_max || parsed)) {
        next.age_max = parsed;
      }
      if (key === "age_max" && parsed < Number(prev.age_min || parsed)) {
        next.age_min = parsed;
      }
      return next;
    });
    setPage(0);
  };

  const resetFilters = () => {
    setFilters(DEFAULT_FILTERS);
    setPage(0);
  };

  const activeFilterCount = [
    filters.competition && filters.competition !== DEFAULT_COMPETITION,
    filters.season && filters.season !== DEFAULT_SEASON,
    filters.role,
    filters.position,
    filters.team,
    filters.min_minutes !== DEFAULT_MIN_MINUTES,
    filters.age_min !== DEFAULT_AGE_MIN,
    filters.age_max !== DEFAULT_AGE_MAX,
  ].filter(Boolean).length;

  return (
    <main className="nl-page px-4 py-8">
      <div className="mx-auto max-w-[1500px] space-y-6">
        <header className="nl-page-header">
          <p className="nl-kicker">
            Talent ranking
          </p>
          <h1 className="mt-2 text-3xl font-semibold tracking-normal text-slate-950 md:text-4xl">
            Market-ranked player database
          </h1>
          <p className="mt-2 max-w-3xl text-slate-600">
            Build the right cohort by league, role, position and age, then prioritize players with adjusted scores and percentile context.
          </p>
        </header>

        <Card className="nl-filter-bar p-0">
          <div className="flex flex-col gap-4 border-b border-white/10 px-4 py-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="nl-kicker">Cohort builder</p>
              <h2 className="mt-1 text-lg font-semibold text-white">Filter the player universe</h2>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-md border border-[#3A8967]/30 bg-[#2F7D5C]/15 px-3 py-2 text-xs font-semibold text-[#8CC7A7]">
                {total} players
              </span>
              <span className="rounded-md border border-white/10 bg-white/[0.04] px-3 py-2 text-xs font-semibold text-slate-500">
                {activeFilterCount} custom filters
              </span>
              <button type="button" className="nl-button-secondary px-3 py-2 text-xs" onClick={resetFilters}>
                Reset
              </button>
            </div>
          </div>

          <div className="space-y-4 p-4">
            <div className="grid grid-cols-1 gap-3 lg:grid-cols-[minmax(260px,1.2fr)_minmax(180px,0.8fr)_minmax(220px,1fr)]">
              <div className="flex flex-col gap-2">
              <Label htmlFor="ranking-competition">Competition</Label>
              <Select
                id="ranking-competition"
                value={filters.competition}
                onChange={(e) =>
                  updateFilter({ competition: e.target.value })
                }
              >
                {competitionOptions.map((c) => (
                  <option key={c} value={c}>
                    {c || "All competitions"}
                  </option>
                ))}
              </Select>
            </div>
              <div className="flex flex-col gap-2">
              <Label htmlFor="ranking-season">Season</Label>
              <Select
                id="ranking-season"
                value={filters.season}
                onChange={(e) => updateFilter({ season: e.target.value })}
              >
                {seasonOptions.map((s) => (
                  <option key={s} value={s}>
                    {s || "All seasons"}
                  </option>
                ))}
              </Select>
            </div>
              <div className="flex flex-col gap-2">
              <Label htmlFor="ranking-team">Team</Label>
              <Select
                id="ranking-team"
                value={filters.team}
                onChange={(e) => updateFilter({ team: e.target.value })}
              >
                {teamOptions.map((t) => (
                  <option key={t} value={t}>
                    {t || "All teams"}
                  </option>
                ))}
              </Select>
            </div>
            </div>

            <div className="grid grid-cols-1 gap-3 lg:grid-cols-[minmax(220px,1fr)_minmax(180px,0.8fr)_repeat(3,minmax(120px,0.55fr))]">
              <div className="flex flex-col gap-2">
                <Label htmlFor="ranking-role">Position group</Label>
                <Select
                  id="ranking-role"
                  value={filters.role}
                  onChange={(e) => updateFilter({ role: e.target.value })}
                >
                  {roleOptions.map((r) => (
                    <option key={r} value={r}>
                      {r || "All position groups"}
                    </option>
                  ))}
                </Select>
              </div>
              <div className="flex flex-col gap-2">
                <Label htmlFor="ranking-position">Position</Label>
                <Select
                  id="ranking-position"
                  value={filters.position}
                  onChange={(e) => updateFilter({ position: e.target.value })}
                >
                  {positionOptions.map((p) => (
                    <option key={p} value={p}>
                      {p || "All positions"}
                    </option>
                  ))}
                </Select>
              </div>
              <div className="flex flex-col gap-2">
                <Label htmlFor="ranking-min-minutes">Min minutes</Label>
                <input
                  id="ranking-min-minutes"
                  name="min_minutes"
                  aria-label="Minimum minutes"
                  type="number"
                  inputMode="numeric"
                  min={0}
                  max={10000}
                  step={1}
                  className="nl-field tabular-nums"
                  value={filters.min_minutes}
                  onChange={(e) => updateNumericFilter("min_minutes", e.target.value, 0)}
                />
              </div>
              <div className="flex flex-col gap-2">
                <Label htmlFor="ranking-age-min">Age min</Label>
                <input
                  id="ranking-age-min"
                  name="age_min"
                  aria-label="Minimum age"
                  type="number"
                  inputMode="numeric"
                  min={0}
                  max={filters.age_max || undefined}
                  step={1}
                  className="nl-field tabular-nums"
                  value={filters.age_min}
                  onChange={(e) => updateNumericFilter("age_min", e.target.value, DEFAULT_AGE_MIN)}
                />
              </div>
              <div className="flex flex-col gap-2">
                <Label htmlFor="ranking-age-max">Age max</Label>
                <input
                  id="ranking-age-max"
                  name="age_max"
                  aria-label="Maximum age"
                  type="number"
                  inputMode="numeric"
                  min={filters.age_min || 0}
                  max={80}
                  step={1}
                  className="nl-field tabular-nums"
                  value={filters.age_max}
                  onChange={(e) => updateNumericFilter("age_max", e.target.value, DEFAULT_AGE_MAX)}
                />
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2 pt-1">
              {[
                ["Competition", formatFilterValue(filters.competition, "All competitions")],
                ["Season", formatFilterValue(filters.season, "All seasons")],
                ["Group", formatFilterValue(filters.role, "All groups")],
                ["Position", formatFilterValue(filters.position, "All positions")],
                ["Team", formatFilterValue(filters.team, "All teams")],
                ["Minutes", `${filters.min_minutes}+`],
                ["Age", `${filters.age_min}-${filters.age_max}`],
              ].map(([label, value]) => (
                <span key={label} className="rounded-md border border-white/10 bg-white/[0.035] px-2.5 py-1.5 text-[11px] font-semibold text-slate-500">
                  <span className="text-white/45">{label}</span> <span className="text-white/80">{value}</span>
                </span>
              ))}
            </div>
          </div>
        </Card>

        <Card className="overflow-hidden p-0">
          <div className="flex flex-col gap-3 border-b border-white/10 px-4 py-3 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="text-sm font-semibold text-white">Ranking results</p>
              <p className="mt-0.5 text-xs font-semibold text-slate-500">{pageLabel} · {total} players</p>
            </div>
            <div className="rounded-md border border-white/10 bg-white/[0.035] px-3 py-2 text-xs font-semibold text-slate-500">
              Sorted by Score
            </div>
          </div>

        {error && (
          <Card>
            <p className="text-danger">Error: {error}</p>
          </Card>
        )}

          {loading ? (
            <div className="space-y-3 p-4">
              {Array.from({ length: 8 }).map((_, index) => (
                <div key={index} className="grid grid-cols-[48px_minmax(0,1fr)_110px_110px] gap-3 rounded-md border border-white/10 bg-white/[0.025] p-3">
                  <div className="nl-skeleton h-10 w-10 rounded-full" />
                  <div className="space-y-2">
                    <div className="nl-skeleton h-3 w-1/3" />
                    <div className="nl-skeleton h-3 w-2/3" />
                  </div>
                  <div className="nl-skeleton h-8" />
                  <div className="nl-skeleton h-8" />
                </div>
              ))}
            </div>
          ) : items.length === 0 ? (
            <div className="p-6">
              <div className="nl-empty-state">
                <p className="text-sm font-semibold text-slate-950">No players match these filters.</p>
                <p className="mt-1 text-xs text-slate-500">Broaden the cohort or lower the minutes threshold.</p>
              </div>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-[980px] w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-white/10 text-[11px] uppercase tracking-[0.14em] text-slate-500">
                    <th className="w-14 px-4 py-3">Rank</th>
                    <th className="px-4 py-3">Player</th>
                    <th className="px-4 py-3">Context</th>
                    <th className="px-4 py-3">Profile</th>
                    <th className="px-4 py-3 text-right">Score</th>
                    <th className="px-4 py-3 text-right">Percentiles</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/10">
                  {items.map((row, idx) => {
                    const tmFields = row.tm_fields || {};
                    const tmPhotoUrl = toAbsoluteUrl(tmFields.tm_profile_image_url || tmFields.profile_image_url);
                    const tmProfileUrl = toAbsoluteUrl(tmFields.tm_profile_url || row.tm_profile_url);
                    const tmMarketValue = formatCompactNumber(tmFields.tm_market_value);
                    const tmAgentName = tmFields.tm_agent_name;
                    const openReport = () =>
                      window.open(
                        `/report?player_id=${row.player_id}&player_season_id=${row.player_season_id}`,
                        "_blank",
                        "noopener,noreferrer"
                      );

                    return (
                      <tr
                        key={row.player_season_id}
                        className="cursor-pointer align-middle"
                        tabIndex={0}
                        onClick={openReport}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" || event.key === " ") {
                            event.preventDefault();
                            openReport();
                          }
                        }}
                      >
                        <td className="px-4 py-3">
                          <span className="flex h-8 w-8 items-center justify-center rounded-md border border-[#3A8967]/30 bg-[#2F7D5C]/15 text-xs font-semibold text-[#8CC7A7]">
                            {page * filters.limit + idx + 1}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex min-w-0 items-center gap-3">
                            {tmPhotoUrl ? (
                              <img src={tmPhotoUrl} alt={row.name} className="h-10 w-10 rounded-md border border-white/10 object-cover" />
                            ) : (
                              <div className="flex h-10 w-10 items-center justify-center rounded-md border border-white/10 bg-white/[0.04] text-xs font-semibold text-slate-500">
                                {getInitials(row.name)}
                              </div>
                            )}
                            <div className="min-w-0">
                              <div className="flex items-center gap-2">
                                <p className="truncate font-semibold text-slate-950">{row.name}</p>
                                {prospectIds.has(row.player_id) ? (
                                  <span className="rounded-md border border-amber-300/30 bg-amber-400/10 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-amber-200">
                                    Prospect
                                  </span>
                                ) : null}
                              </div>
                              {tmProfileUrl ? (
                                <a href={tmProfileUrl} target="_blank" rel="noreferrer" className="text-xs text-[#8CC7A7] hover:text-white" onClick={(event) => event.stopPropagation()}>
                                  Transfermarkt
                                </a>
                              ) : null}
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2">
                            <ClubLogo name={row.team} className="h-6 w-6 rounded" />
                            <div className="min-w-0">
                              <p className="truncate text-sm font-medium text-slate-700">{row.team || "—"}</p>
                              <p className="truncate text-xs text-slate-500">{row.competition_name} · {row.calendar || "–"}</p>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex max-w-[280px] flex-wrap gap-1.5">
                            {row.assigned_role ? <Badge>{row.assigned_role}</Badge> : null}
                            {row.position ? <Badge>{row.position}</Badge> : null}
                            {row.age ? <Badge>{row.age} yrs</Badge> : null}
                            <Badge>{Math.round(row.minutes_played || 0)} mins</Badge>
                            {tmMarketValue ? <Badge>{tmMarketValue}</Badge> : null}
                            {tmAgentName ? <Badge>{tmAgentName}</Badge> : null}
                          </div>
                        </td>
                        <td className="px-4 py-3 text-right">
                          <p className="text-xl font-semibold text-[#8CC7A7]">{row.global_score_adjusted?.toFixed(1) ?? "—"}</p>
                          <p className="text-[11px] text-slate-500">Score</p>
                        </td>
                        <td className="px-4 py-3 text-right">
                          <p className="font-semibold text-slate-950">
                            {row.assigned_role_pct_league?.toFixed(0) ?? "—"} / {row.assigned_role_pct_global?.toFixed(0) ?? "—"}
                          </p>
                          <p className="text-[11px] text-slate-500">League / global</p>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
          <div className="flex flex-col gap-3 border-t border-white/10 px-4 py-3 md:flex-row md:items-center md:justify-between">
            <p className="text-xs font-semibold text-slate-500">
              Showing {items.length ? page * filters.limit + 1 : 0}-{Math.min(total, (page + 1) * filters.limit)} of {total}
            </p>
            <div className="flex flex-wrap items-center gap-2">
              <Label htmlFor="ranking-limit">Rows per page</Label>
              <div className="w-24">
                <Select
                  id="ranking-limit"
                  value={filters.limit}
                  onChange={(e) => updateFilter({ limit: Number(e.target.value) })}
                >
                  {[20, 30, 50, 100].map((val) => (
                    <option key={val} value={val}>
                      {val}
                    </option>
                  ))}
                </Select>
              </div>
              <button
                type="button"
                className="nl-button-secondary px-3"
                disabled={page === 0 || loading}
                onClick={() => setPage((p) => Math.max(0, p - 1))}
              >
                Prev
              </button>
              <span className="rounded-md border border-white/10 bg-white/[0.035] px-3 py-2 text-xs font-semibold text-slate-500">{page + 1} / {totalPages}</span>
              <button
                type="button"
                className="nl-button-secondary px-3"
                disabled={page + 1 >= totalPages || loading}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </button>
            </div>
          </div>
        </Card>
      </div>
    </main>
  );
}
