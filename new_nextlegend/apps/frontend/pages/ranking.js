import { useEffect, useMemo, useState } from "react";
import { fetchJson, fetchJsonCached } from "@/lib/api";

const TM_BASE_URL = "https://www.transfermarkt.com";

const Card = ({ children, className = "", ...props }) => (
  <div
    className={`glass-panel rounded-xl p-4 border border-white/5 ${className}`}
    {...props}
  >
    {children}
  </div>
);

const Badge = ({ children }) => (
  <span className="px-2 py-1 rounded-full bg-slate-800 text-xs text-slate-200 border border-white/5">
    {children}
  </span>
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

  const [filters, setFilters] = useState({
    competition: "",
    season: "",
    role: "",
    position: "",
    team: "",
    min_minutes: 270,
    age_min: "",
    age_max: "",
    limit: 30,
  });

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

  const competitionOptions = useMemo(
    () => ["", ...competitions.map((c) => c.name)],
    [competitions]
  );

  const seasonOptions = useMemo(() => {
    if (!filters.competition) {
      return ["", ...seasons];
    }
    const found = competitions.find((c) => c.name === filters.competition);
    if (!found || !found.seasons) {
      return ["", ...seasons];
    }
    return ["", ...found.seasons];
  }, [competitions, seasons, filters.competition]);

  const roleOptions = useMemo(() => ["", ...roles], [roles]);
  const positionOptions = useMemo(() => ["", ...positions], [positions]);
  const teamOptions = useMemo(() => ["", ...teams], [teams]);

  const totalPages = Math.max(1, Math.ceil(total / filters.limit));
  const pageLabel = `Page ${page + 1} / ${totalPages}`;

  const updateFilter = (patch) => {
    setFilters((prev) => ({ ...prev, ...patch }));
    setPage(0);
  };

  return (
    <main className="min-h-screen bg-hero-pattern text-slate-100 py-10 px-4">
      <div className="max-w-6xl mx-auto space-y-6">
        <header className="flex flex-col gap-2">
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
            NextLegend v2
          </p>
          <h1 className="text-4xl font-bold text-white tracking-tight">
            Ranking
          </h1>
          <p className="text-slate-300 max-w-3xl">
            Filter by role, league, position, and age. Rankings are served by
            the v2 API with adjusted global scores and percentile context.
          </p>
        </header>

        <Card>
          <div className="grid grid-cols-1 lg:grid-cols-6 gap-4">
            <div className="flex flex-col gap-2">
              <Label>Competition</Label>
              <Select
                value={filters.competition}
                onChange={(e) =>
                  updateFilter({ competition: e.target.value, season: "", team: "" })
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
              <Label>Season</Label>
              <Select
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
              <Label>Role</Label>
              <Select
                value={filters.role}
                onChange={(e) => updateFilter({ role: e.target.value })}
              >
                {roleOptions.map((r) => (
                  <option key={r} value={r}>
                    {r || "All roles"}
                  </option>
                ))}
              </Select>
            </div>
            <div className="flex flex-col gap-2">
              <Label>Position</Label>
              <Select
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
              <Label>Team</Label>
              <Select
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
            <div className="flex flex-col gap-2">
              <Label>Min minutes</Label>
              <input
                type="number"
                min={0}
                className="bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100"
                value={filters.min_minutes}
                onChange={(e) =>
                  updateFilter({
                    min_minutes: Number(e.target.value || 0),
                  })
                }
              />
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-4">
            <div className="flex flex-col gap-2">
              <Label>Age min</Label>
              <input
                type="number"
                min={0}
                className="bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100"
                value={filters.age_min}
                onChange={(e) => updateFilter({ age_min: e.target.value })}
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label>Age max</Label>
              <input
                type="number"
                min={0}
                className="bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100"
                value={filters.age_max}
                onChange={(e) => updateFilter({ age_max: e.target.value })}
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label>Rows per page</Label>
              <Select
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
            <div className="flex flex-col gap-2">
              <Label>Pagination</Label>
              <div className="flex items-center gap-2">
                <button
                  className="px-3 py-2 rounded-md border border-slate-700 bg-slate-900/60 disabled:opacity-50"
                  disabled={page === 0 || loading}
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                >
                  Prev
                </button>
                <span className="text-xs text-slate-300">{pageLabel}</span>
                <button
                  className="px-3 py-2 rounded-md border border-slate-700 bg-slate-900/60 disabled:opacity-50"
                  disabled={page + 1 >= totalPages || loading}
                  onClick={() => setPage((p) => p + 1)}
                >
                  Next
                </button>
              </div>
              <p className="text-xs text-slate-500">{total} players</p>
            </div>
          </div>
        </Card>

        {error && (
          <Card>
            <p className="text-danger">Error: {error}</p>
          </Card>
        )}

        <section className="grid grid-cols-1 gap-4">
          {loading ? (
            <Card>
              <p className="text-slate-400">Loading ranking…</p>
            </Card>
          ) : (
            items.map((row, idx) => (
              <Card
                key={row.player_season_id}
                role="button"
                tabIndex={0}
                onClick={() =>
                  window.open(`/report?player_id=${row.player_id}`, "_blank", "noopener,noreferrer")
                }
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    window.open(
                      `/report?player_id=${row.player_id}`,
                      "_blank",
                      "noopener,noreferrer"
                    );
                  }
                }}
                className="cursor-pointer focus:outline-none focus:ring-2 focus:ring-emerald-400/60"
              >
                <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
                  <div className="flex items-center gap-3">
                    <div className="text-2xl font-semibold text-primary">
                      {page * filters.limit + idx + 1}
                    </div>
                    <div className="flex items-center gap-3">
                      {(() => {
                        const tmFields = row.tm_fields || {};
                        const tmPhotoUrl = toAbsoluteUrl(
                          tmFields.tm_profile_image_url || tmFields.profile_image_url
                        );
                        const tmProfileUrl = toAbsoluteUrl(
                          tmFields.tm_profile_url || row.tm_profile_url
                        );
                        const tmMarketValue = formatCompactNumber(tmFields.tm_market_value);
                        const tmAgentName = tmFields.tm_agent_name;
                        return (
                          <>
                            {tmPhotoUrl ? (
                              <img
                                src={tmPhotoUrl}
                                alt={row.name}
                                className="h-12 w-12 rounded-full object-cover border border-white/10"
                              />
                            ) : (
                              <div className="h-12 w-12 rounded-full bg-slate-800 border border-white/10 flex items-center justify-center text-slate-200 font-semibold">
                                {getInitials(row.name)}
                              </div>
                            )}
                            <div>
                              <div className="text-lg font-semibold text-white flex items-center gap-2">
                                {row.name}
                                {prospectIds.has(row.player_id) ? (
                                  <span className="text-yellow-400" aria-label="Prospect">
                                    ★
                                  </span>
                                ) : null}
                              </div>
                              <div className="text-slate-400 text-sm">
                                {row.team || "—"} • {row.competition_name} •{" "}
                                {row.calendar || "–"}
                              </div>
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
                              {tmMarketValue || tmAgentName ? (
                                <div className="text-xs text-slate-400 mt-1">
                                  {tmMarketValue ? `Market value: ${tmMarketValue}` : null}
                                  {tmMarketValue && tmAgentName ? " • " : null}
                                  {tmAgentName ? `Agent: ${tmAgentName}` : null}
                                </div>
                              ) : null}
                              <div className="flex flex-wrap gap-2 mt-2">
                                {row.assigned_role ? (
                                  <Badge>{row.assigned_role}</Badge>
                                ) : null}
                                {row.position ? <Badge>{row.position}</Badge> : null}
                                {row.age ? <Badge>{row.age} yrs</Badge> : null}
                                <Badge>{Math.round(row.minutes_played || 0)} mins</Badge>
                              </div>
                            </div>
                          </>
                        );
                      })()}
                    </div>
                  </div>
                  <div className="flex flex-wrap items-center gap-6">
                    <div className="text-right">
                      <p className="text-xs uppercase text-slate-400">
                        Global score (adj.)
                      </p>
                      <p className="text-2xl font-bold text-primary">
                        {row.global_score_adjusted?.toFixed(1) ?? "—"}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-xs uppercase text-slate-400">
                        Role pct (league/global)
                      </p>
                      <p className="text-lg font-semibold">
                        {row.assigned_role_pct_league?.toFixed(0) ?? "—"} /{" "}
                        {row.assigned_role_pct_global?.toFixed(0) ?? "—"}
                      </p>
                    </div>
                  </div>
                </div>
              </Card>
            ))
          )}
        </section>
      </div>
    </main>
  );
}
