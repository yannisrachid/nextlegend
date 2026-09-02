import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { deleteJson, fetchJson, postJson } from "@/lib/api";

const DEFAULT_FILTERS = {
  season: 2027,
  championship: "All",
  age_category: "All",
  birth_year: "",
  position_group: "All",
  position: "All",
  club: "",
  min_minutes: 0,
  limit: 30,
};

const POSITION_GROUP_ORDER = [
  "Goalkeepers",
  "Centre Backs",
  "Fullbacks",
  "Defensive Midfielders",
  "Central Midfielders",
  "Attacking Midfielders",
  "Wingers",
  "Forwards",
];

const POSITION_ORDER = ["G", "GB", "GK", "DD", "DC", "DG", "MDC", "MC", "MOC", "MD", "MG", "AD", "AG", "ATT", "BU"];

const Card = ({ children, className = "" }) => (
  <div className={`surface-panel rounded-lg ${className}`}>{children}</div>
);

const Label = ({ children, htmlFor }) => (
  <label htmlFor={htmlFor} className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[#6F7772]">
    {children}
  </label>
);

const Select = ({ id, value, onChange, children }) => (
  <select id={id} className="nl-field" value={value} onChange={onChange}>
    {children}
  </select>
);

const formatNumber = (value, digits = 0) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";
  return new Intl.NumberFormat("en", { maximumFractionDigits: digits }).format(numeric);
};

const formatScore = (value) => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(1) : "-";
};

const formatYouthSeason = (season) => {
  const numeric = Number(season);
  return Number.isFinite(numeric) ? `${numeric - 1}/${numeric}` : season || "-";
};

const parseIntFilter = (value, fallback = 0) => {
  const parsed = Number.parseInt(String(value || "").replace(/[^\d]/g, ""), 10);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const formatFilterValue = (value, fallback) => String(value || "").trim() || fallback;

const initials = (name) => {
  const parts = String(name || "").trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "YP";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase();
};

const selectedLabel = (item) =>
  [item?.display_name, item?.club_name, item?.championship, item?.calendar || formatYouthSeason(item?.season)]
    .filter(Boolean)
    .join(" - ");

const sortPositionGroups = (values = []) =>
  [...values].sort((a, b) => {
    const ai = POSITION_GROUP_ORDER.indexOf(a);
    const bi = POSITION_GROUP_ORDER.indexOf(b);
    return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi) || String(a).localeCompare(String(b));
  });

const sortPositions = (values = []) =>
  [...values].sort((a, b) => {
    const ai = POSITION_ORDER.indexOf(String(a).toUpperCase());
    const bi = POSITION_ORDER.indexOf(String(b).toUpperCase());
    return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi) || String(a).localeCompare(String(b));
  });

export default function YouthProspectsPage() {
  const [filters, setFilters] = useState(DEFAULT_FILTERS);
  const [page, setPage] = useState(0);
  const [meta, setMeta] = useState(null);
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [showResults, setShowResults] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetchJson("/youth/meta")
      .then(setMeta)
      .catch((err) => setError(err.message || "Unable to load Youth metadata."));
  }, []);

  const loadProspects = async (nextFilters = filters, nextPage = page) => {
    setLoading(true);
    setError("");
    try {
      const res = await fetchJson("/youth/prospects/page", {
        ...nextFilters,
        championship: nextFilters.championship === "All" ? undefined : nextFilters.championship,
        age_category: nextFilters.age_category === "All" ? undefined : nextFilters.age_category,
        birth_year: nextFilters.birth_year || undefined,
        position_group: nextFilters.position_group === "All" ? undefined : nextFilters.position_group,
        position: nextFilters.position === "All" ? undefined : nextFilters.position,
        offset: nextPage * nextFilters.limit,
      });
      setItems(res.items || []);
      setTotal(res.total || 0);
    } catch (err) {
      setError(err.message || "Unable to load Youth prospects.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError("");
      try {
        const res = await fetchJson("/youth/prospects/page", {
          ...filters,
          championship: filters.championship === "All" ? undefined : filters.championship,
          age_category: filters.age_category === "All" ? undefined : filters.age_category,
          birth_year: filters.birth_year || undefined,
          position_group: filters.position_group === "All" ? undefined : filters.position_group,
          position: filters.position === "All" ? undefined : filters.position,
          offset: page * filters.limit,
        });
        if (!cancelled) {
          setItems(res.items || []);
          setTotal(res.total || 0);
        }
      } catch (err) {
        if (!cancelled) setError(err.message || "Unable to load Youth prospects.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [filters, page]);

  useEffect(() => {
    if (query.trim().length < 2) {
      setResults([]);
      setShowResults(false);
      return;
    }
    const handle = setTimeout(async () => {
      try {
        const rows = await fetchJson("/youth/players", {
          q: query.trim(),
          season: filters.season,
          limit: 12,
        });
        setResults(rows || []);
        setShowResults(true);
      } catch (err) {
        console.error(err);
      }
    }, 180);
    return () => clearTimeout(handle);
  }, [filters.season, query]);

  const updateFilter = (patch) => {
    setFilters((current) => ({ ...current, ...patch }));
    setPage(0);
  };

  const addProspect = async (player) => {
    if (!player?.id || busy) return;
    setBusy(true);
    setMessage("");
    try {
      const res = await postJson("/youth/prospects", { youth_id: Number(player.id) });
      setMessage(res?.added ? "Youth prospect added." : "Youth prospect already in list.");
      setQuery("");
      setResults([]);
      setShowResults(false);
      await loadProspects(filters, 0);
      setPage(0);
    } catch (err) {
      setMessage(err.message || "Unable to add youth prospect.");
    } finally {
      setBusy(false);
    }
  };

  const removeProspect = async (youthId) => {
    if (!youthId || busy) return;
    const confirmed = window.confirm("Remove this youth player from prospects?");
    if (!confirmed) return;
    setBusy(true);
    setMessage("");
    try {
      await deleteJson(`/youth/prospects/${youthId}`);
      setMessage("Youth prospect removed.");
      await loadProspects(filters, page);
    } catch (err) {
      setMessage(err.message || "Unable to remove youth prospect.");
    } finally {
      setBusy(false);
    }
  };

  const seasons = meta?.seasons || [2027];
  const championships = useMemo(() => ["All", ...(meta?.championships || [])], [meta]);
  const ageCategories = useMemo(() => ["All", ...(meta?.age_categories || [])], [meta]);
  const birthYears = meta?.birth_years || [];
  const positionGroups = useMemo(() => ["All", ...sortPositionGroups(meta?.position_groups || [])], [meta]);
  const positions = useMemo(() => ["All", ...sortPositions(meta?.positions || [])], [meta]);
  const totalPages = Math.max(1, Math.ceil(total / filters.limit));
  const activeFilterCount = [
    filters.season !== DEFAULT_FILTERS.season,
    filters.championship !== DEFAULT_FILTERS.championship,
    filters.age_category !== DEFAULT_FILTERS.age_category,
    filters.birth_year,
    filters.position_group !== DEFAULT_FILTERS.position_group,
    filters.position !== DEFAULT_FILTERS.position,
    filters.club,
    filters.min_minutes !== DEFAULT_FILTERS.min_minutes,
  ].filter(Boolean).length;

  return (
    <main className="nl-page px-4 py-8">
      <div className="mx-auto max-w-[1500px] space-y-6">
        <header className="nl-page-header">
          <p className="nl-kicker">Youth Scouting</p>
          <div className="mt-2 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <h1 className="text-3xl font-semibold tracking-normal text-white md:text-4xl">Youth prospects</h1>
              <p className="mt-2 max-w-3xl text-sm text-[#A0A8A3]">
                Track selected Eyeball profiles from youth reports and monitor them by season, competition and position group.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2 lg:justify-end">
              <Link href="/youth-scouting/ranking" className="nl-button-secondary px-4 py-2 text-xs uppercase tracking-[0.14em]">
                Youth ranking
              </Link>
              <Link href="/youth-scouting/reports" className="nl-button-primary px-4 py-2 text-xs uppercase tracking-[0.14em]">
                Open report
              </Link>
            </div>
          </div>
        </header>

        <Card className="p-4">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div className="min-w-0 flex-1">
              <p className="nl-kicker">Add prospect</p>
              <h2 className="mt-1 text-lg font-semibold text-white">Search the current youth dataset</h2>
              <div className="relative mt-3 max-w-2xl">
                <input
                  className="nl-field"
                  value={query}
                  onChange={(event) => {
                    setQuery(event.target.value);
                    setMessage("");
                    setShowResults(true);
                  }}
                  onFocus={() => setShowResults(query.trim().length >= 2)}
                  onBlur={() => setTimeout(() => setShowResults(false), 150)}
                  placeholder="Search by player, club or competition"
                />
                {showResults && query.trim().length >= 2 ? (
                  <div className="absolute left-0 right-0 top-[calc(100%+8px)] z-[1000] max-h-80 overflow-auto rounded-lg border border-white/10 bg-[#080B0A] p-2 shadow-[0_22px_70px_rgba(0,0,0,0.45)]">
                    {results.length === 0 ? (
                      <div className="rounded-md border border-dashed border-white/10 bg-white/[0.03] px-4 py-6 text-center text-sm font-semibold text-[#A0A8A3]">
                        No youth player found.
                      </div>
                    ) : (
                      results.map((player) => (
                        <button
                          key={player.id}
                          type="button"
                          className="flex w-full items-center justify-between gap-3 rounded-md px-3 py-2 text-left transition hover:bg-white/[0.055]"
                          onMouseDown={(event) => event.preventDefault()}
                          onClick={() => addProspect(player)}
                          disabled={busy}
                        >
                          <span className="min-w-0">
                            <span className="block truncate text-sm font-semibold text-white">{player.display_name}</span>
                            <span className="block truncate text-xs text-[#6F7772]">{selectedLabel(player)}</span>
                          </span>
                          <span className="rounded-md border border-[#3A8967]/30 bg-[#2F7D5C]/15 px-2 py-1 text-[11px] font-semibold text-[#8CC7A7]">
                            {formatScore(player.score)}
                          </span>
                        </button>
                      ))
                    )}
                  </div>
                ) : null}
              </div>
            </div>
            <div className="rounded-md border border-[#3A8967]/30 bg-[#2F7D5C]/15 px-4 py-3 text-right">
              <p className="text-2xl font-semibold text-[#8CC7A7]">{formatNumber(total)}</p>
              <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#8CC7A7]">Tracked prospects</p>
            </div>
          </div>
          {message ? <p className="mt-3 text-xs font-semibold text-[#A0A8A3]">{message}</p> : null}
        </Card>

        <Card className="nl-filter-bar p-0">
          <div className="flex flex-col gap-4 border-b border-white/10 px-4 py-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="nl-kicker">Prospect filters</p>
              <h2 className="mt-1 text-lg font-semibold text-white">Refine the tracked list</h2>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-md border border-white/10 bg-white/[0.04] px-3 py-2 text-xs font-semibold text-[#A0A8A3]">
                {activeFilterCount} custom filters
              </span>
              <button type="button" className="nl-button-secondary px-3 py-2 text-xs" onClick={() => updateFilter(DEFAULT_FILTERS)}>
                Reset
              </button>
            </div>
          </div>

          <div className="space-y-4 p-4">
            <div className="grid grid-cols-1 gap-3 lg:grid-cols-[120px_minmax(260px,1.2fr)_minmax(220px,1fr)_minmax(150px,0.7fr)]">
              <div className="flex flex-col gap-2">
                <Label htmlFor="youth-prospect-season">Season</Label>
                <Select id="youth-prospect-season" value={filters.season} onChange={(event) => updateFilter({ season: Number(event.target.value) })}>
                  {seasons.map((season) => <option key={season} value={season}>{formatYouthSeason(season)}</option>)}
                </Select>
              </div>
              <div className="flex flex-col gap-2">
                <Label htmlFor="youth-prospect-championship">Competition / level</Label>
                <Select id="youth-prospect-championship" value={filters.championship} onChange={(event) => updateFilter({ championship: event.target.value })}>
                  {championships.map((item) => <option key={item} value={item}>{item}</option>)}
                </Select>
              </div>
              <div className="flex flex-col gap-2">
                <Label htmlFor="youth-prospect-club">Club</Label>
                <input
                  id="youth-prospect-club"
                  className="nl-field"
                  value={filters.club}
                  onChange={(event) => updateFilter({ club: event.target.value })}
                  placeholder="All clubs"
                />
              </div>
              <div className="flex flex-col gap-2">
                <Label htmlFor="youth-prospect-age-category">Age category</Label>
                <Select id="youth-prospect-age-category" value={filters.age_category} onChange={(event) => updateFilter({ age_category: event.target.value })}>
                  {ageCategories.map((item) => <option key={item} value={item}>{item}</option>)}
                </Select>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-3 lg:grid-cols-[minmax(220px,1fr)_minmax(160px,0.7fr)_minmax(220px,1fr)_minmax(140px,0.65fr)]">
              <div className="flex flex-col gap-2">
                <Label htmlFor="youth-prospect-position-group">Position group</Label>
                <Select id="youth-prospect-position-group" value={filters.position_group} onChange={(event) => updateFilter({ position_group: event.target.value })}>
                  {positionGroups.map((item) => <option key={item} value={item}>{item}</option>)}
                </Select>
              </div>
              <div className="flex flex-col gap-2">
                <Label htmlFor="youth-prospect-position">Position</Label>
                <Select id="youth-prospect-position" value={filters.position} onChange={(event) => updateFilter({ position: event.target.value })}>
                  {positions.map((item) => <option key={item} value={item}>{item}</option>)}
                </Select>
              </div>
              <div className="flex flex-col gap-2">
                <Label htmlFor="youth-prospect-birth-year">Birth year</Label>
                <Select id="youth-prospect-birth-year" value={filters.birth_year} onChange={(event) => updateFilter({ birth_year: event.target.value })}>
                  <option value="">All birth years</option>
                  {birthYears.map((item) => <option key={item} value={item}>{item}</option>)}
                </Select>
              </div>
              <div className="flex flex-col gap-2">
                <Label htmlFor="youth-prospect-min-minutes">Min minutes</Label>
                <input
                  id="youth-prospect-min-minutes"
                  type="number"
                  inputMode="numeric"
                  min={0}
                  step={1}
                  className="nl-field tabular-nums"
                  value={filters.min_minutes}
                  onChange={(event) => updateFilter({ min_minutes: parseIntFilter(event.target.value, 0) })}
                />
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2 pt-1">
              {[
                ["Competition", formatFilterValue(filters.championship, "All competitions")],
                ["Season", formatYouthSeason(filters.season)],
                ["Age category", formatFilterValue(filters.age_category, "All categories")],
                ["Birth year", formatFilterValue(filters.birth_year, "All birth years")],
                ["Group", formatFilterValue(filters.position_group, "All groups")],
                ["Position", formatFilterValue(filters.position, "All positions")],
                ["Club", formatFilterValue(filters.club, "All clubs")],
                ["Minutes", `${filters.min_minutes}+`],
              ].map(([label, value]) => (
                <span key={label} className="rounded-md border border-white/10 bg-white/[0.035] px-2.5 py-1.5 text-[11px] font-semibold text-[#A0A8A3]">
                  <span className="text-white/45">{label}</span> <span className="text-white/80">{value}</span>
                </span>
              ))}
            </div>
          </div>
        </Card>

        <Card className="overflow-hidden p-0">
          <div className="flex flex-col gap-3 border-b border-white/10 px-4 py-3 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="text-sm font-semibold text-white">Tracked youth players</p>
              <p className="mt-0.5 text-xs font-semibold text-[#6F7772]">
                Page {page + 1} / {totalPages} - {formatNumber(total)} prospects
              </p>
            </div>
            <span className="rounded-md border border-white/10 bg-white/[0.035] px-3 py-2 text-xs font-semibold text-[#A0A8A3]">
              Sorted by last added
            </span>
          </div>

          {error ? (
            <div className="p-4 text-sm font-semibold text-rose-200">{error}</div>
          ) : loading ? (
            <div className="space-y-2 p-4">
              {Array.from({ length: 6 }).map((_, index) => <div key={index} className="nl-skeleton h-14 rounded-md" />)}
            </div>
          ) : items.length === 0 ? (
            <div className="p-6">
              <div className="nl-empty-state">
                <p className="text-sm font-semibold text-white">No youth prospects match these filters.</p>
                <p className="mt-1 text-xs text-[#6F7772]">Add one from a Youth Report or use the search box above.</p>
              </div>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-[1120px] w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-white/10 text-[11px] uppercase tracking-[0.14em] text-[#6F7772]">
                    <th className="px-4 py-3">Player</th>
                    <th className="px-4 py-3">Club / level</th>
                    <th className="px-4 py-3">Profile</th>
                    <th className="px-4 py-3 text-right">Score</th>
                    <th className="px-4 py-3 text-right">Minutes</th>
                    <th className="px-4 py-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/10">
                  {items.map((row) => (
                    <tr key={row.id} className="align-middle">
                      <td className="px-4 py-3">
                        <Link href={`/youth-scouting/reports?youth_id=${row.id}`} className="flex min-w-0 items-center gap-3">
                          <span className="flex h-10 w-10 items-center justify-center rounded-md border border-[#3A8967]/30 bg-[#2F7D5C]/15 text-xs font-semibold text-[#8CC7A7]">
                            {initials(row.display_name)}
                          </span>
                          <span className="min-w-0">
                            <span className="block truncate font-semibold text-white">{row.display_name}</span>
                            <span className="block truncate text-xs text-[#A0A8A3]">
                              {[row.nationality_label || row.nationality_code, row.birth_year ? `Born ${row.birth_year}` : null].filter(Boolean).join(" - ") || "Youth profile"}
                            </span>
                          </span>
                        </Link>
                      </td>
                      <td className="px-4 py-3">
                        <p className="truncate font-medium text-white">{row.club_name || "-"}</p>
                        <p className="truncate text-xs text-[#6F7772]">{[row.championship, row.calendar || formatYouthSeason(row.season)].filter(Boolean).join(" - ") || "-"}</p>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex max-w-[320px] flex-wrap gap-1.5">
                          {[row.position_group, row.position, row.age_category].filter(Boolean).map((item) => (
                            <span key={item} className="rounded-md border border-white/10 bg-white/[0.035] px-2 py-1 text-[11px] font-semibold text-[#A0A8A3]">{item}</span>
                          ))}
                        </div>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <p className="text-xl font-semibold text-[#8CC7A7]">{formatScore(row.score)}</p>
                        <p className="text-[11px] text-[#6F7772]">Score</p>
                      </td>
                      <td className="px-4 py-3 text-right text-white">{formatNumber(row.minutes_played)}</td>
                      <td className="px-4 py-3 text-right">
                        <div className="flex justify-end gap-2">
                          <Link href={`/youth-scouting/reports?youth_id=${row.id}`} className="nl-button-secondary px-3 py-2 text-xs">
                            Report
                          </Link>
                          <button type="button" className="nl-button-secondary px-3 py-2 text-xs text-rose-200" disabled={busy} onClick={() => removeProspect(row.id)}>
                            Remove
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="flex flex-col gap-3 border-t border-white/10 px-4 py-3 md:flex-row md:items-center md:justify-between">
            <p className="text-xs font-semibold text-[#6F7772]">
              Showing {items.length ? page * filters.limit + 1 : 0}-{Math.min(total, (page + 1) * filters.limit)} of {formatNumber(total)}
            </p>
            <div className="flex items-center gap-2">
              <Label htmlFor="youth-prospect-limit">Rows per page</Label>
              <div className="w-24">
                <Select id="youth-prospect-limit" value={filters.limit} onChange={(event) => updateFilter({ limit: Number(event.target.value) })}>
                  {[20, 30, 50, 100].map((value) => <option key={value} value={value}>{value}</option>)}
                </Select>
              </div>
              <button type="button" className="nl-button-secondary px-3" disabled={page === 0 || loading} onClick={() => setPage((current) => Math.max(0, current - 1))}>
                Prev
              </button>
              <span className="rounded-md border border-white/10 bg-white/[0.035] px-3 py-2 text-xs font-semibold text-[#A0A8A3]">{page + 1} / {totalPages}</span>
              <button type="button" className="nl-button-secondary px-3" disabled={page + 1 >= totalPages || loading} onClick={() => setPage((current) => current + 1)}>
                Next
              </button>
            </div>
          </div>
        </Card>
      </div>
    </main>
  );
}
