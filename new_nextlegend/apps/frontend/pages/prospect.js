import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { deleteJson, fetchJson, fetchJsonCached, patchJson, postJson } from "@/lib/api";
import {
  DEFAULT_AGE_MAX,
  DEFAULT_AGE_MIN,
  DEFAULT_LIMIT,
  DEFAULT_SCOUTING_SEASON,
  formatFilterValue,
  parseIntegerInput,
  sortCompetitionNames,
  sortPositions,
  sortRoles,
  withDefaultSeason,
  uniqueOptions,
} from "@/lib/scoutingFilters";

const STAGES = ["Priority 1", "Priority 2", "Priority 3", "Completed"];
const TM_BASE_URL = "https://www.transfermarkt.com";
const PROSPECT_DEFAULT_COMPETITION = "";
const PROSPECT_DEFAULT_MIN_MINUTES = 0;

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

const groupNeeds = (needs, stage) =>
  needs
    .filter((need) => need.priority_stage === stage)
    .slice()
    .sort((a, b) => (a.sort_order ?? 0) - (b.sort_order ?? 0));

const buildNeedOrderPayload = (needs) =>
  needs.map((need) => ({
    id: need.id,
    priority_stage: need.priority_stage,
    sort_order: need.sort_order ?? 0,
  }));

const normalizeNeedOrders = (needs) => {
  const updated = needs.map((need) => ({ ...need }));
  STAGES.forEach((stage) => {
    const stageNeeds = groupNeeds(updated, stage);
    stageNeeds.forEach((need, idx) => {
      need.sort_order = idx;
    });
  });
  return updated;
};

const reorderNeeds = (needs, movedId, targetStage, targetNeedId = null) => {
  const cloned = needs.map((need) => ({ ...need }));
  const dragged = cloned.find((need) => need.id === movedId);
  if (!dragged) return needs;
  const originStage = dragged.priority_stage;
  dragged.priority_stage = targetStage;

  const stageBuckets = {};
  STAGES.forEach((stage) => {
    stageBuckets[stage] = groupNeeds(cloned, stage).filter((need) => need.id !== movedId);
  });

  const targetList = stageBuckets[targetStage] || [];
  if (targetNeedId) {
    const insertIndex = targetList.findIndex((need) => need.id === targetNeedId);
    if (insertIndex >= 0) {
      targetList.splice(insertIndex, 0, dragged);
    } else {
      targetList.push(dragged);
    }
  } else {
    targetList.push(dragged);
  }
  stageBuckets[targetStage] = targetList;

  const merged = [];
  STAGES.forEach((stage) => {
    stageBuckets[stage].forEach((need, idx) => {
      merged.push({ ...need, sort_order: idx });
    });
  });
  return merged;
};

export default function ProspectPage() {
  const [activeTab, setActiveTab] = useState("players");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);

  const [competitions, setCompetitions] = useState([]);
  const [seasons, setSeasons] = useState([]);
  const [roles, setRoles] = useState([]);
  const [positions, setPositions] = useState([]);
  const [teams, setTeams] = useState([]);

  const [filters, setFilters] = useState({
    competition: PROSPECT_DEFAULT_COMPETITION,
    season: DEFAULT_SCOUTING_SEASON,
    role: "",
    position: "",
    team: "",
    min_minutes: PROSPECT_DEFAULT_MIN_MINUTES,
    age_min: DEFAULT_AGE_MIN,
    age_max: DEFAULT_AGE_MAX,
    limit: DEFAULT_LIMIT,
  });

  const [clubs, setClubs] = useState([]);
  const [needs, setNeeds] = useState([]);
  const [needsLoading, setNeedsLoading] = useState(false);
  const [needsError, setNeedsError] = useState("");
  const [showNeedForm, setShowNeedForm] = useState(false);
  const [newNeed, setNewNeed] = useState({
    club_id: "",
    need_label: "",
    contact_name: "",
    contact_phone: "",
    assigned_user: "admin",
    priority_stage: "Priority 1",
  });
  const [dragNeedId, setDragNeedId] = useState(null);
  const [dragPlayer, setDragPlayer] = useState(null);

  const [addPlayerNeedId, setAddPlayerNeedId] = useState(null);
  const [playerQuery, setPlayerQuery] = useState("");
  const [playerResults, setPlayerResults] = useState([]);
  const [showPlayerResults, setShowPlayerResults] = useState(false);

  const [addProspectQuery, setAddProspectQuery] = useState("");
  const [addProspectResults, setAddProspectResults] = useState([]);
  const [showAddProspectResults, setShowAddProspectResults] = useState(false);
  const [addProspectBusy, setAddProspectBusy] = useState(false);
  const [addProspectMessage, setAddProspectMessage] = useState("");
  const addProspectAnchorRef = useRef(null);
  const [addProspectPos, setAddProspectPos] = useState(null);
  const addPlayerAnchorRef = useRef(null);
  const [addPlayerPos, setAddPlayerPos] = useState(null);

  useEffect(() => {
    const loadMeta = async () => {
      try {
        const [comps, seasonsData, rolesData] = await Promise.all([
          fetchJsonCached("/meta/competitions"),
          fetchJsonCached("/meta/seasons"),
          fetchJsonCached("/meta/roles"),
        ]);
        setCompetitions(comps || []);
        setSeasons(withDefaultSeason(seasonsData || []));
        setRoles(sortRoles(rolesData || []));
      } catch (err) {
        console.error(err);
      }
    };
    loadMeta();
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
        setPositions(sortPositions(positionsData || []));
        setTeams(uniqueOptions(teamsData || []).sort((a, b) => String(a).localeCompare(String(b), undefined, { sensitivity: "base" })));
      } catch (err) {
        console.error(err);
      }
    };
    loadDependent();
  }, [filters.competition, filters.season]);

  useEffect(() => {
    if (activeTab !== "players") return;
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
        const res = await fetchJson("/prospects/page", params);
        setItems(res.items || []);
        setTotal(res.total || 0);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, [activeTab, filters, page]);

  useEffect(() => {
    if (activeTab !== "clubs") return;
    const loadNeeds = async () => {
      setNeedsLoading(true);
      setNeedsError("");
      try {
        const [clubsData, needsData] = await Promise.all([
          fetchJson("/meta/clubs"),
          fetchJson("/prospect/club-needs"),
        ]);
        setClubs(clubsData || []);
        setNeeds(needsData?.needs || []);
      } catch (err) {
        setNeedsError(err.message);
      } finally {
        setNeedsLoading(false);
      }
    };
    loadNeeds();
  }, [activeTab]);

  useEffect(() => {
    if (!addPlayerNeedId || playerQuery.trim().length < 2) {
      setPlayerResults([]);
      return;
    }
    const handle = setTimeout(async () => {
      try {
        const res = await fetchJson("/players", {
          q: playerQuery.trim(),
          season: filters.season || undefined,
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
  }, [addPlayerNeedId, playerQuery, filters.season]);

  useEffect(() => {
    if (!showAddProspectResults || !addProspectAnchorRef.current) {
      setAddProspectPos(null);
      return;
    }
    const updatePos = () => {
      const rect = addProspectAnchorRef.current.getBoundingClientRect();
      setAddProspectPos({
        left: rect.left,
        top: rect.bottom + 8,
        width: rect.width,
      });
    };
    updatePos();
    window.addEventListener("resize", updatePos);
    window.addEventListener("scroll", updatePos, true);
    return () => {
      window.removeEventListener("resize", updatePos);
      window.removeEventListener("scroll", updatePos, true);
    };
  }, [showAddProspectResults, addProspectQuery]);

  useEffect(() => {
    if (!showPlayerResults || !addPlayerAnchorRef.current) {
      setAddPlayerPos(null);
      return;
    }
    const updatePos = () => {
      const rect = addPlayerAnchorRef.current.getBoundingClientRect();
      setAddPlayerPos({
        left: rect.left,
        top: rect.bottom + 8,
        width: rect.width,
      });
    };
    updatePos();
    window.addEventListener("resize", updatePos);
    window.addEventListener("scroll", updatePos, true);
    return () => {
      window.removeEventListener("resize", updatePos);
      window.removeEventListener("scroll", updatePos, true);
    };
  }, [showPlayerResults, playerQuery, addPlayerNeedId]);

  useEffect(() => {
    if (activeTab !== "players") return;
    if (!addProspectQuery || addProspectQuery.trim().length < 2) {
      setAddProspectResults([]);
      return;
    }
    const handle = setTimeout(async () => {
      try {
        const res = await fetchJson("/players", {
          q: addProspectQuery.trim(),
          season: filters.season || undefined,
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
        setAddProspectResults(Array.from(unique.values()));
      } catch (err) {
        console.error(err);
      }
    }, 200);
    return () => clearTimeout(handle);
  }, [activeTab, addProspectQuery, filters.season]);

  const competitionOptions = useMemo(
    () => ["", ...sortCompetitionNames(competitions.map((c) => c.name))],
    [competitions]
  );

  const seasonOptions = useMemo(() => {
    const allSeasons = withDefaultSeason([filters.season, ...seasons]);
    if (!filters.competition) {
      return ["", ...allSeasons];
    }
    const found = competitions.find((c) => c.name === filters.competition);
    if (!found || !found.seasons) {
      return ["", ...allSeasons];
    }
    return ["", ...withDefaultSeason([filters.season, ...found.seasons, ...seasons])];
  }, [competitions, seasons, filters.competition, filters.season]);

  const roleOptions = useMemo(() => ["", ...sortRoles([filters.role, ...roles])], [filters.role, roles]);
  const positionOptions = useMemo(() => ["", ...sortPositions([filters.position, ...positions])], [filters.position, positions]);
  const teamOptions = useMemo(() => ["", ...uniqueOptions([filters.team, ...teams]).sort((a, b) => String(a).localeCompare(String(b), undefined, { sensitivity: "base" }))], [filters.team, teams]);

  const totalPages = Math.max(1, Math.ceil(total / filters.limit));
  const pageLabel = `${items.length ? page * filters.limit + 1 : 0}-${Math.min(
    total,
    (page + 1) * filters.limit
  )}`;

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
    setFilters({
      competition: PROSPECT_DEFAULT_COMPETITION,
      season: DEFAULT_SCOUTING_SEASON,
      role: "",
      position: "",
      team: "",
      min_minutes: PROSPECT_DEFAULT_MIN_MINUTES,
      age_min: DEFAULT_AGE_MIN,
      age_max: DEFAULT_AGE_MAX,
      limit: filters.limit,
    });
    setPage(0);
  };

  const activeFilterCount = [
    filters.competition !== PROSPECT_DEFAULT_COMPETITION,
    filters.season !== DEFAULT_SCOUTING_SEASON,
    filters.role,
    filters.position,
    filters.team,
    filters.min_minutes !== PROSPECT_DEFAULT_MIN_MINUTES,
    filters.age_min !== DEFAULT_AGE_MIN,
    filters.age_max !== DEFAULT_AGE_MAX,
  ].filter(Boolean).length;

  const clubOptions = useMemo(() => {
    const items = clubs.map((club) => {
      const suffix = club.competition_name ? ` • ${club.competition_name}` : "";
      return { value: club.id, label: `${club.name}${suffix}` };
    });
    return [{ value: "", label: "Select club" }, ...items];
  }, [clubs]);

  const handleRemoveProspect = async (playerId) => {
    try {
      await deleteJson(`/prospects/${playerId}`);
      const res = await fetchJson("/prospects/page", {
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
      });
      setItems(res.items || []);
      setTotal(res.total || 0);
    } catch (err) {
      console.error(err);
    }
  };

  const handleAddProspect = async (player) => {
    if (!player?.id || addProspectBusy) return;
    setAddProspectBusy(true);
    setAddProspectMessage("");
    try {
      const res = await postJson("/prospects", {
        player_id: Number(player.id),
        player_season_id: player.player_season_id ? Number(player.player_season_id) : undefined,
      });
      setAddProspectMessage(res?.added ? "Prospect added." : "Prospect already in list.");
      const refreshed = await fetchJson("/prospects/page", {
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
      });
      setItems(refreshed.items || []);
      setTotal(refreshed.total || 0);
    } catch (err) {
      console.error(err);
      setAddProspectMessage("Unable to add prospect.");
    } finally {
      setAddProspectBusy(false);
      setAddProspectQuery("");
      setAddProspectResults([]);
      setShowAddProspectResults(false);
    }
  };

  const handleCreateNeed = async () => {
    if (!newNeed.need_label.trim()) return;
    try {
      await postJson("/prospect/club-needs", {
        club_id: newNeed.club_id ? Number(newNeed.club_id) : null,
        need_label: newNeed.need_label.trim(),
        contact_name: newNeed.contact_name.trim() || null,
        contact_phone: newNeed.contact_phone.trim() || null,
        assigned_user: newNeed.assigned_user.trim() || "admin",
        priority_stage: newNeed.priority_stage,
      });
      setShowNeedForm(false);
      setNewNeed({
        club_id: "",
        need_label: "",
        contact_name: "",
        contact_phone: "",
        assigned_user: "admin",
        priority_stage: "Priority 1",
      });
      const refreshed = await fetchJson("/prospect/club-needs");
      setNeeds(refreshed?.needs || []);
    } catch (err) {
      console.error(err);
    }
  };

  const handleNeedDrop = async (stage, targetNeedId = null) => {
    if (!dragNeedId) return;
    const updated = reorderNeeds(needs, dragNeedId, stage, targetNeedId);
    const normalized = normalizeNeedOrders(updated);
    setNeeds(normalized);
    setDragNeedId(null);
    try {
      await patchJson("/prospect/club-needs/reorder", {
        needs: buildNeedOrderPayload(normalized),
      });
    } catch (err) {
      console.error(err);
    }
  };

  const handlePlayerDrop = async (needId, targetPlayerId = null) => {
    if (!dragPlayer || dragPlayer.needId !== needId) return;
    const { playerId } = dragPlayer;
    let newOrder = [];
    setNeeds((prev) =>
      prev.map((need) => {
        if (need.id !== needId) return need;
        const players = (need.players || []).slice();
        const draggedIndex = players.findIndex((player) => player.player_id === playerId);
        if (draggedIndex === -1) return need;
        const [dragged] = players.splice(draggedIndex, 1);
        if (targetPlayerId) {
          const insertIndex = players.findIndex((player) => player.player_id === targetPlayerId);
          if (insertIndex >= 0) {
            players.splice(insertIndex, 0, dragged);
          } else {
            players.push(dragged);
          }
        } else {
          players.push(dragged);
        }
        newOrder = players.map((player) => player.player_id);
        return { ...need, players };
      })
    );
    setDragPlayer(null);
    if (newOrder.length > 0) {
      try {
        await patchJson(`/prospect/club-needs/${needId}/players/reorder`, {
          player_ids: newOrder,
        });
      } catch (err) {
        console.error(err);
      }
    }
  };

  const handleAddPlayer = async (needId, player) => {
    try {
      const response = await postJson(`/prospect/club-needs/${needId}/players`, {
        player_id: Number(player.id),
      });
      if (!response?.added) return;
      setNeeds((prev) =>
        prev.map((need) => {
          if (need.id !== needId) return need;
          const players = need.players || [];
          if (players.some((item) => item.player_id === player.id)) {
            return need;
          }
          return {
            ...need,
            players: [
              ...players,
              {
                player_id: player.id,
                name: player.name,
                competition_name: player.competition_name,
                calendar: player.calendar,
                team: player.team,
              },
            ],
          };
        })
      );
      setPlayerQuery("");
      setPlayerResults([]);
      setShowPlayerResults(false);
    } catch (err) {
      console.error(err);
    }
  };

  const handleRemoveNeedPlayer = async (needId, playerId) => {
    try {
      await deleteJson(`/prospect/club-needs/${needId}/players/${playerId}`);
      setNeeds((prev) =>
        prev.map((need) => {
          if (need.id !== needId) return need;
          return {
            ...need,
            players: (need.players || []).filter((player) => player.player_id !== playerId),
          };
        })
      );
    } catch (err) {
      console.error(err);
    }
  };

  const playerOptions = useMemo(() => {
    return playerResults.map((p) => ({
      id: String(p.id),
      label: `${p.name} - ${p.calendar || "—"} - ${p.competition_name || "—"}`,
      raw: p,
    }));
  }, [playerResults]);

  const addProspectOptions = useMemo(() => {
    return addProspectResults.map((p) => ({
      id: String(p.id),
      label: `${p.name} - ${p.team || "—"} - ${p.competition_name || "—"} - ${p.calendar || "—"}`,
      raw: p,
    }));
  }, [addProspectResults]);

  return (
    <main className="nl-page">
      <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
        <div>
          <h1 className="text-3xl font-semibold text-white">Prospect pipeline</h1>
          <p className="text-slate-400 mt-2">
            Track watched players, active club needs and the next recruitment conversations to pursue.
          </p>
        </div>

        <div className="flex items-center gap-4">
          {["players", "clubs"].map((tab) => (
            <button
              key={tab}
              type="button"
              className={`rounded-full px-4 py-2 text-xs uppercase tracking-[0.2em] border ${
                activeTab === tab
                  ? "border-primary text-primary"
                  : "border-slate-700 text-slate-300"
              }`}
              onClick={() => setActiveTab(tab)}
            >
              {tab === "players" ? "Players" : "Clubs"}
            </button>
          ))}
        </div>

        {activeTab === "players" ? (
          <>
            <Card className="nl-filter-bar p-0">
              <div className="flex flex-col gap-4 border-b border-white/10 px-4 py-4 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <p className="nl-kicker">Prospect filters</p>
                  <h2 className="mt-1 text-lg font-semibold text-white">Control the watched-player cohort</h2>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-md border border-[#3A8967]/30 bg-[#2F7D5C]/15 px-3 py-2 text-xs font-semibold text-[#8CC7A7]">
                    {total} prospects
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
                  <Label>Competition</Label>
                  <Select
                    id="prospect-competition"
                    value={filters.competition}
                    onChange={(e) =>
                      updateFilter({
                        competition: e.target.value,
                      })
                    }
                  >
                    {competitionOptions.map((opt) => (
                      <option key={opt} value={opt}>
                        {opt || "All"}
                      </option>
                    ))}
                  </Select>
                </div>
                <div className="flex flex-col gap-2">
                  <Label>Season</Label>
                  <Select
                    id="prospect-season"
                    value={filters.season}
                    onChange={(e) =>
                      updateFilter({ season: e.target.value })
                    }
                  >
                    {seasonOptions.map((opt) => (
                      <option key={opt} value={opt}>
                        {opt || "All"}
                      </option>
                    ))}
                  </Select>
                </div>
                <div className="flex flex-col gap-2">
                  <Label>Team</Label>
                  <Select
                    id="prospect-team"
                    value={filters.team}
                    onChange={(e) =>
                      updateFilter({ team: e.target.value })
                    }
                  >
                    {teamOptions.map((team) => (
                      <option key={team} value={team}>
                        {team || "All teams"}
                      </option>
                    ))}
                  </Select>
                </div>
              </div>
              <div className="grid grid-cols-1 gap-3 lg:grid-cols-[minmax(220px,1fr)_minmax(180px,0.8fr)_repeat(3,minmax(120px,0.55fr))]">
                <div className="flex flex-col gap-2">
                  <Label>Position group</Label>
                  <Select
                    id="prospect-role"
                    value={filters.role}
                    onChange={(e) =>
                      updateFilter({ role: e.target.value })
                    }
                  >
                    {roleOptions.map((role) => (
                      <option key={role} value={role}>
                        {role || "All position groups"}
                      </option>
                    ))}
                  </Select>
                </div>
                <div className="flex flex-col gap-2">
                  <Label>Position</Label>
                  <Select
                    id="prospect-position"
                    value={filters.position}
                    onChange={(e) =>
                      updateFilter({ position: e.target.value })
                    }
                  >
                    {positionOptions.map((pos) => (
                      <option key={pos} value={pos}>
                        {pos || "All positions"}
                      </option>
                    ))}
                  </Select>
                </div>
                <div className="flex flex-col gap-2">
                  <Label>Min minutes</Label>
                  <input
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
                  <Label>Age min</Label>
                  <input
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
                  <Label>Age max</Label>
                  <input
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
            <Card>
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h3 className="text-sm uppercase tracking-[0.2em] text-slate-400">
                    Add a prospect
                  </h3>
                  <p className="text-xs text-slate-500 mt-1">
                    Start typing a player name to add them to your prospect list.
                  </p>
                </div>
                {addProspectMessage ? (
                  <span className="text-xs text-slate-300">{addProspectMessage}</span>
                ) : null}
              </div>
              <div className="relative max-w-xl" ref={addProspectAnchorRef}>
                <input
                  className="nl-field"
                  value={addProspectQuery}
                  onChange={(e) => {
                    setAddProspectQuery(e.target.value);
                    setShowAddProspectResults(true);
                    setAddProspectMessage("");
                  }}
                  onBlur={() => setTimeout(() => setShowAddProspectResults(false), 150)}
                  placeholder="Player - Team - Competition - Season"
                />
                {showAddProspectResults && addProspectQuery.trim().length >= 2 && addProspectPos
                  ? createPortal(
                      <div
                        className="fixed z-[9999] max-h-56 overflow-auto rounded-lg border border-slate-700 bg-slate-900/95 shadow-xl"
                        style={{
                          left: `${addProspectPos.left}px`,
                          top: `${addProspectPos.top}px`,
                          width: `${addProspectPos.width}px`,
                        }}
                      >
                        {addProspectOptions.length === 0 ? (
                          <div className="px-3 py-2 text-sm text-slate-400">
                            No matches found.
                          </div>
                        ) : (
                          addProspectOptions.map((player) => (
                            <button
                              key={player.id}
                              type="button"
                              className="w-full text-left px-3 py-2 text-sm text-slate-200 hover:bg-slate-800/80 disabled:opacity-60"
                              onMouseDown={(e) => e.preventDefault()}
                              onClick={() => handleAddProspect(player.raw)}
                              disabled={addProspectBusy}
                            >
                              {player.label}
                            </button>
                          ))
                        )}
                      </div>,
                      document.body
                    )
                  : null}
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
                  <p className="text-slate-400">Loading prospects…</p>
                </Card>
              ) : items.length === 0 ? (
                <Card>
                  <p className="text-slate-400">No prospects match these filters.</p>
                </Card>
              ) : (
                items.map((row) => (
                  <Card key={row.player_season_id}>
                    <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
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
                                  className="h-14 w-14 rounded-full object-cover border border-white/10"
                                />
                              ) : (
                                <div className="h-14 w-14 rounded-full bg-slate-800 border border-white/10 flex items-center justify-center text-slate-200 font-semibold">
                                  {getInitials(row.name)}
                                </div>
                              )}
                              <div>
                                <div className="text-lg font-semibold text-white flex items-center gap-2">
                                  {row.name}
                                  <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-bold text-amber-800" aria-label="Prospect">
                                    Prospect
                                  </span>
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
                                  {row.assigned_role ? <Badge>{row.assigned_role}</Badge> : null}
                                  {row.position ? <Badge>{row.position}</Badge> : null}
                                  {row.age ? <Badge>{row.age} yrs</Badge> : null}
                                  <Badge>{Math.round(row.minutes_played || 0)} mins</Badge>
                                </div>
                              </div>
                            </>
                          );
                        })()}
                      </div>
                      <div className="flex items-center gap-4">
                        <button
                          type="button"
                          className="text-xs uppercase tracking-[0.2em] px-4 py-2 border border-yellow-400/70 text-yellow-200 rounded-full"
                          onClick={() => handleRemoveProspect(row.player_id)}
                        >
                          Remove to Prospect
                        </button>
                        <button
                          type="button"
                          className="text-xs uppercase tracking-[0.2em] px-4 py-2 border border-slate-700 text-slate-200 rounded-full"
                          onClick={() =>
                            window.open(
                              `/report?player_id=${row.player_id}`,
                              "_blank",
                              "noopener,noreferrer"
                            )
                          }
                        >
                          Open Report
                        </button>
                      </div>
                    </div>
                  </Card>
                ))
              )}
            </section>
            <Card className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
              <p className="text-xs font-semibold text-slate-500">
                Showing {pageLabel} of {total}
              </p>
              <div className="flex flex-wrap items-center gap-2">
                <Label>Rows per page</Label>
                <div className="w-24">
                  <Select
                    id="prospect-limit"
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
                <span className="rounded-md border border-white/10 bg-white/[0.035] px-3 py-2 text-xs font-semibold text-slate-500">
                  {page + 1} / {totalPages}
                </span>
                <button
                  type="button"
                  className="nl-button-secondary px-3"
                  disabled={page + 1 >= totalPages || loading}
                  onClick={() => setPage((p) => p + 1)}
                >
                  Next
                </button>
              </div>
            </Card>
          </>
        ) : (
          <>
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">Club Pipeline</h2>
              <button
                type="button"
                className="text-xs uppercase tracking-[0.2em] px-4 py-2 border border-primary text-primary rounded-full"
                onClick={() => setShowNeedForm((prev) => !prev)}
              >
                {showNeedForm ? "Close" : "New Need"}
              </button>
            </div>

            {showNeedForm ? (
              <Card>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="flex flex-col gap-2">
                    <Label>Club</Label>
                    <Select
                      value={newNeed.club_id}
                      onChange={(e) =>
                        setNewNeed((prev) => ({ ...prev, club_id: e.target.value }))
                      }
                    >
                      {clubOptions.map((club) => (
                        <option key={club.value} value={club.value}>
                          {club.label}
                        </option>
                      ))}
                    </Select>
                  </div>
                  <div className="flex flex-col gap-2">
                    <Label>Need</Label>
                    <input
                      className="nl-field"
                      value={newNeed.need_label}
                      onChange={(e) =>
                        setNewNeed((prev) => ({ ...prev, need_label: e.target.value }))
                      }
                      placeholder="e.g. Left-footed CB"
                    />
                  </div>
                  <div className="flex flex-col gap-2">
                    <Label>Priority</Label>
                    <Select
                      value={newNeed.priority_stage}
                      onChange={(e) =>
                        setNewNeed((prev) => ({ ...prev, priority_stage: e.target.value }))
                      }
                    >
                      {STAGES.map((stage) => (
                        <option key={stage} value={stage}>
                          {stage}
                        </option>
                      ))}
                    </Select>
                  </div>
                  <div className="flex flex-col gap-2">
                    <Label>Contact name</Label>
                    <input
                      className="nl-field"
                      value={newNeed.contact_name}
                      onChange={(e) =>
                        setNewNeed((prev) => ({ ...prev, contact_name: e.target.value }))
                      }
                      placeholder="Name"
                    />
                  </div>
                  <div className="flex flex-col gap-2">
                    <Label>Contact phone</Label>
                    <input
                      className="nl-field"
                      value={newNeed.contact_phone}
                      onChange={(e) =>
                        setNewNeed((prev) => ({ ...prev, contact_phone: e.target.value }))
                      }
                      placeholder="+33..."
                    />
                  </div>
                  <div className="flex flex-col gap-2">
                    <Label>Assigned user</Label>
                    <input
                      className="nl-field"
                      value={newNeed.assigned_user}
                      onChange={(e) =>
                        setNewNeed((prev) => ({ ...prev, assigned_user: e.target.value }))
                      }
                    />
                  </div>
                </div>
                <div className="mt-4">
                  <button
                    type="button"
                    className="text-xs uppercase tracking-[0.2em] px-4 py-2 border border-primary text-primary rounded-full"
                    onClick={handleCreateNeed}
                  >
                    Create need
                  </button>
                </div>
              </Card>
            ) : null}

            {needsError && (
              <Card>
                <p className="text-danger">Error: {needsError}</p>
              </Card>
            )}

            {needsLoading ? (
              <Card>
                <p className="text-slate-400">Loading club needs…</p>
              </Card>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {STAGES.map((stage) => {
                  const stageNeeds = groupNeeds(needs, stage);
                  return (
                    <div
                      key={stage}
                      className="rounded-xl border border-white/10 bg-slate-900/40 p-3 min-h-[300px]"
                      onDragOver={(event) => event.preventDefault()}
                      onDrop={() => handleNeedDrop(stage)}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <h3 className="text-sm uppercase tracking-[0.2em] text-slate-300">
                          {stage}
                        </h3>
                        <span className="text-xs text-slate-400">
                          {stageNeeds.length}
                        </span>
                      </div>
                      <div className="space-y-3">
                        {stageNeeds.map((need) => (
                          <div
                            key={need.id}
                            className="rounded-lg border border-white/10 bg-slate-950/80 p-3 space-y-3"
                            draggable
                            onDragStart={() => setDragNeedId(need.id)}
                            onDragEnd={() => setDragNeedId(null)}
                            onDragOver={(event) => event.preventDefault()}
                            onDrop={(event) => {
                              event.stopPropagation();
                              handleNeedDrop(stage, need.id);
                            }}
                          >
                            <div className="flex items-center justify-between">
                              <div>
                                <p className="text-sm font-semibold text-white">
                                  {need.need_label}
                                </p>
                                <p className="text-xs text-slate-400">
                                  {need.club_name || "Club"}{" "}
                                  {need.competition_name ? `• ${need.competition_name}` : ""}
                                </p>
                              </div>
                              <Badge>{need.assigned_user || "admin"}</Badge>
                            </div>
                            <div className="text-xs text-slate-400">
                              {need.contact_name || "No contact"}{" "}
                              {need.contact_phone ? `• ${need.contact_phone}` : ""}
                            </div>
                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                                  Players
                                </p>
                                <button
                                  type="button"
                                  className="text-xs text-primary hover:text-primary/80"
                                  onClick={() => {
                                    setAddPlayerNeedId(
                                      addPlayerNeedId === need.id ? null : need.id
                                    );
                                    setPlayerQuery("");
                                    setPlayerResults([]);
                                    setShowPlayerResults(false);
                                  }}
                                >
                                  {addPlayerNeedId === need.id
                                    ? "Close"
                                    : "Add player"}
                                </button>
                              </div>
                              {addPlayerNeedId === need.id ? (
                                <div className="relative" ref={addPlayerAnchorRef}>
                                  <input
                                    className="nl-field"
                                    value={playerQuery}
                                    onChange={(e) => {
                                      setPlayerQuery(e.target.value);
                                      setShowPlayerResults(true);
                                    }}
                                    onBlur={() =>
                                      setTimeout(() => setShowPlayerResults(false), 150)
                                    }
                                    placeholder="Start typing a player..."
                                  />
                                  {showPlayerResults &&
                                  playerQuery.trim().length >= 2 &&
                                  addPlayerPos
                                    ? createPortal(
                                        <div
                                          className="fixed z-[9999] max-h-56 overflow-auto rounded-lg border border-slate-700 bg-slate-900/95 shadow-xl"
                                          style={{
                                            left: `${addPlayerPos.left}px`,
                                            top: `${addPlayerPos.top}px`,
                                            width: `${addPlayerPos.width}px`,
                                          }}
                                        >
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
                                                onClick={() =>
                                                  handleAddPlayer(need.id, player.raw)
                                                }
                                              >
                                                {player.label}
                                              </button>
                                            ))
                                          )}
                                        </div>,
                                        document.body
                                      )
                                    : null}
                                </div>
                              ) : null}
                              <div
                                className="space-y-2"
                                onDragOver={(event) => event.preventDefault()}
                                onDrop={() => handlePlayerDrop(need.id)}
                              >
                                {(need.players || []).length === 0 ? (
                                  <p className="text-xs text-slate-500">
                                    No players have been added to this pipeline yet.
                                  </p>
                                ) : (
                                  need.players.map((player) => (
                                    <div
                                      key={`${need.id}-${player.player_id}`}
                                      role="button"
                                      tabIndex={0}
                                      className="w-full rounded-md border border-white/10 bg-slate-900/60 px-3 py-2 text-sm text-slate-200 hover:border-primary"
                                      draggable
                                      onDragStart={() =>
                                        setDragPlayer({
                                          playerId: player.player_id,
                                          needId: need.id,
                                        })
                                      }
                                      onDragEnd={() => setDragPlayer(null)}
                                      onDragOver={(event) => event.preventDefault()}
                                      onDrop={(event) => {
                                        event.stopPropagation();
                                        handlePlayerDrop(need.id, player.player_id);
                                      }}
                                      onClick={() =>
                                        window.open(
                                          `/report?player_id=${player.player_id}`,
                                          "_blank",
                                          "noopener,noreferrer"
                                        )
                                      }
                                      onKeyDown={(event) => {
                                        if (event.key === "Enter" || event.key === " ") {
                                          event.preventDefault();
                                          window.open(
                                            `/report?player_id=${player.player_id}`,
                                            "_blank",
                                            "noopener,noreferrer"
                                          );
                                        }
                                      }}
                                    >
                                      <div className="flex items-start justify-between gap-2">
                                        <div>
                                          <div className="font-semibold text-white">
                                            {player.name}
                                          </div>
                                          <div className="text-xs text-slate-400">
                                            {player.calendar || "—"} •{" "}
                                            {player.competition_name || "—"}
                                          </div>
                                        </div>
                                        <button
                                          type="button"
                                          className="shrink-0 rounded-full border border-red-500/60 p-1 text-red-400 hover:border-red-400 hover:text-red-300"
                                          onClick={(event) => {
                                            event.stopPropagation();
                                            handleRemoveNeedPlayer(need.id, player.player_id);
                                          }}
                                          aria-label="Remove player"
                                          title="Remove player"
                                        >
                                          <svg
                                            viewBox="0 0 24 24"
                                            className="h-3.5 w-3.5"
                                            fill="none"
                                            stroke="currentColor"
                                            strokeWidth="2"
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                          >
                                            <path d="M3 6h18" />
                                            <path d="M8 6V4h8v2" />
                                            <path d="M6 6l1 14h10l1-14" />
                                            <path d="M10 11v6" />
                                            <path d="M14 11v6" />
                                          </svg>
                                        </button>
                                      </div>
                                    </div>
                                  ))
                                )}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </>
        )}
      </div>
    </main>
  );
}
