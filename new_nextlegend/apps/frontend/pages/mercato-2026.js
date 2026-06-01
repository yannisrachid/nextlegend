import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/router";
import { fetchJson, fetchJsonCached, patchJson, postJson } from "@/lib/api";
import { useAuth } from "@/lib/auth";

const PRIORITIES = ["low", "medium", "high", "urgent"];
const STATUSES = ["new", "searching", "shortlist_ready", "proposed", "discussion", "closed"];
const DEAL_TYPES = ["any", "transfer", "loan", "free"];

const emptyForm = {
  club_id: "",
  club_query: "",
  title: "",
  priority: "medium",
  status: "new",
  budget_min: "",
  budget_max: "",
  salary_max: "",
  deal_type: "any",
  extra_info: "",
  assigned_agent_id: "",
  position: "",
  age_min: "",
  age_max: "",
  preferred_foot: "",
  height_min: "",
  target_league_level: "",
  required_player_level: "",
  nationality_preferences: "",
  notes: "",
};

const Card = ({ children, className = "" }) => (
  <div className={`glass-panel rounded-lg border border-white/5 p-4 ${className}`}>{children}</div>
);

const Label = ({ children }) => (
  <label className="text-xs uppercase tracking-[0.18em] text-slate-400">{children}</label>
);

const TextInput = (props) => (
  <input
    {...props}
    className={`w-full rounded-md border border-slate-700 bg-slate-950/70 px-3 py-2 text-sm text-slate-100 outline-none focus:border-primary ${props.className || ""}`}
  />
);

const Select = ({ children, ...props }) => (
  <select
    {...props}
    className={`w-full rounded-md border border-slate-700 bg-slate-950/70 px-3 py-2 text-sm text-slate-100 outline-none focus:border-primary ${props.className || ""}`}
  >
    {children}
  </select>
);

const TextArea = (props) => (
  <textarea
    {...props}
    className={`w-full rounded-md border border-slate-700 bg-slate-950/70 px-3 py-2 text-sm text-slate-100 outline-none focus:border-primary ${props.className || ""}`}
  />
);

const Badge = ({ children, tone = "default" }) => {
  const tones = {
    default: "border-slate-700 bg-slate-900 text-slate-200",
    green: "border-primary/50 bg-primary/10 text-primary",
    amber: "border-amber-400/40 bg-amber-400/10 text-amber-200",
    red: "border-red-400/40 bg-red-400/10 text-red-200",
  };
  return (
    <span className={`inline-flex rounded-full border px-2 py-1 text-xs ${tones[tone] || tones.default}`}>
      {children}
    </span>
  );
};

const priorityTone = (priority) => {
  if (priority === "urgent") return "red";
  if (priority === "high") return "amber";
  if (priority === "low") return "default";
  return "green";
};

const statusTone = (status) => {
  if (status === "closed" || status === "rejected") return "red";
  if (status === "shortlist_ready" || status === "approved" || status === "signed") return "green";
  if (status === "discussion" || status === "contacted" || status === "proposed") return "amber";
  return "default";
};

const formatMoney = (value) => {
  if (value === null || value === undefined || value === "") return "-";
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  if (Math.abs(num) >= 1e6) return `${Math.round((num / 1e6) * 10) / 10}M`;
  if (Math.abs(num) >= 1e3) return `${Math.round(num / 1e3)}K`;
  return `${Math.round(num)}`;
};

const firstNeed = (request) => (request?.needs || [])[0] || null;

export default function MercatoPage() {
  const router = useRouter();
  const { me } = useAuth();
  const [items, setItems] = useState([]);
  const [kpis, setKpis] = useState({});
  const [clubs, setClubs] = useState([]);
  const [competitions, setCompetitions] = useState([]);
  const [positions, setPositions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [filters, setFilters] = useState({
    club: "",
    position: "",
    status: "",
    priority: "",
    agent: "",
    competition: "",
    deal_type: "",
  });
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState(emptyForm);
  const [selectedId, setSelectedId] = useState(null);
  const [saving, setSaving] = useState(false);
  const [generatingNeedId, setGeneratingNeedId] = useState(null);
  const [playerQuery, setPlayerQuery] = useState("");
  const [playerResults, setPlayerResults] = useState([]);
  const [candidateNote, setCandidateNote] = useState("");
  const [candidateBusy, setCandidateBusy] = useState(false);
  const [searchCompetitions, setSearchCompetitions] = useState([]);

  const selected = useMemo(
    () => items.find((item) => item.id === selectedId) || items[0] || null,
    [items, selectedId]
  );
  const selectedNeed = firstNeed(selected);

  const loadMeta = async () => {
    const [clubData, competitionData, positionData] = await Promise.all([
      fetchJsonCached("/meta/clubs"),
      fetchJsonCached("/meta/competitions"),
      fetchJsonCached("/meta/positions"),
    ]);
    setClubs(clubData || []);
    setCompetitions((competitionData || []).map((item) => item.name).filter(Boolean));
    setPositions((positionData || []).filter((position) => {
      const normalized = String(position || "").trim().toLowerCase();
      return normalized && normalized !== "<na>" && normalized !== "na" && normalized !== "nan";
    }));
  };

  const loadRequests = async () => {
    setLoading(true);
    setError("");
    try {
      const data = await fetchJson("/mercato/requests", filters);
      setItems(data.items || []);
      setKpis(data.kpis || {});
      if (selectedId && !(data.items || []).some((item) => item.id === selectedId)) {
        setSelectedId(null);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadMeta().catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    loadRequests();
  }, [filters]);

  useEffect(() => {
    if (playerQuery.trim().length < 2) {
      setPlayerResults([]);
      return;
    }
    const handle = setTimeout(async () => {
      try {
        const res = await fetchJson("/players", { q: playerQuery.trim(), limit: 12 });
        setPlayerResults(res || []);
      } catch (err) {
        console.error(err);
      }
    }, 200);
    return () => clearTimeout(handle);
  }, [playerQuery]);

  const clubOptions = useMemo(() => {
    return clubs.map((club) => ({
      ...club,
      label: `${club.name}${club.competition_name ? ` - ${club.competition_name}` : ""}`,
    }));
  }, [clubs]);

  const filteredClubOptions = useMemo(() => {
    const query = String(form.club_query || "").trim().toLowerCase();
    if (!query) return clubOptions.slice(0, 12);
    return clubOptions
      .filter((club) => club.label.toLowerCase().includes(query))
      .slice(0, 12);
  }, [clubOptions, form.club_query]);

  const updateForm = (key, value) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const resetForm = () => {
    setForm({ ...emptyForm, assigned_agent_id: me?.username || "" });
    setShowForm(true);
    setMessage("");
  };

  const createRequest = async () => {
    if (!form.club_id) {
      setMessage("Select a club.");
      return;
    }
    if (!form.position.trim()) {
      setMessage("Select a position.");
      return;
    }
    setSaving(true);
    setMessage("");
    try {
      const payload = {
        club_id: form.club_id ? Number(form.club_id) : null,
        assigned_agent_id: form.assigned_agent_id || me?.username || null,
        season: "2026",
        title: form.title || form.position || "Mercato need",
        priority: form.priority,
        status: form.status,
        budget_min: form.budget_min ? Number(form.budget_min) : null,
        budget_max: form.budget_max ? Number(form.budget_max) : null,
        salary_max: form.salary_max ? Number(form.salary_max) : null,
        deal_type: form.deal_type,
        extra_info: form.extra_info || null,
        need: {
          position: form.position || null,
          role: null,
          age_min: form.age_min ? Number(form.age_min) : null,
          age_max: form.age_max ? Number(form.age_max) : null,
          preferred_foot: form.preferred_foot || null,
          height_min: form.height_min ? Number(form.height_min) : null,
          target_league_level: form.target_league_level || null,
          required_player_level: form.required_player_level ? Number(form.required_player_level) : null,
          nationality_preferences: form.nationality_preferences || null,
          contract_preferences: null,
          notes: form.notes || null,
        },
      };
      const created = await postJson("/mercato/requests", payload);
      setShowForm(false);
      setSelectedId(created?.id || null);
      await loadRequests();
      if (created?.id) {
        router.push(`/mercato-2026/${created.id}`);
      }
    } catch (err) {
      setMessage(err.message);
    } finally {
      setSaving(false);
    }
  };

  const generateShortlist = async (needId) => {
    setGeneratingNeedId(needId);
    setMessage("");
    try {
      const res = await postJson(`/mercato/needs/${needId}/generate-shortlist`, {
        competitions: searchCompetitions,
      });
      setMessage(res.generated ? `${res.generated} players shortlisted.` : "No new player found or all matching players are already assigned.");
      await loadRequests();
    } catch (err) {
      setMessage(err.message);
    } finally {
      setGeneratingNeedId(null);
    }
  };

  const addCandidate = async (player) => {
    if (!selectedNeed || candidateBusy) return;
    setCandidateBusy(true);
    setMessage("");
    try {
      const res = await postJson(`/mercato/needs/${selectedNeed.id}/candidates`, {
        player_id: Number(player.id),
        source: "manual",
        status: "suggested",
        agent_note: candidateNote || null,
      });
      setMessage(res.added ? "Player added to the need." : "This player is already assigned to this need.");
      setPlayerQuery("");
      setPlayerResults([]);
      setCandidateNote("");
      await loadRequests();
    } catch (err) {
      setMessage(err.message);
    } finally {
      setCandidateBusy(false);
    }
  };

  const updateCandidateStatus = async (candidateId, status) => {
    try {
      await postJson(`/mercato/candidates/${candidateId}/status`, { status });
      await loadRequests();
    } catch (err) {
      setMessage(err.message);
    }
  };

  const updateCandidateNote = async (candidateId, agentNote) => {
    try {
      await patchJson(`/mercato/candidates/${candidateId}`, { agent_note: agentNote });
      await loadRequests();
    } catch (err) {
      setMessage(err.message);
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-7xl px-4 py-8 space-y-6">
        <section className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-primary">Agency workspace</p>
            <h1 className="mt-2 text-4xl font-semibold text-white">MERCATO 2026</h1>
            <p className="mt-2 max-w-2xl text-sm text-slate-400">
              Centralize club needs, agent notes and calibrated shortlist recommendations.
            </p>
          </div>
          <button
            type="button"
            className="rounded-full border border-primary bg-primary/10 px-5 py-2 text-sm font-semibold text-primary"
            onClick={resetForm}
          >
            New need
          </button>
        </section>

        <section className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Card>
            <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Active needs</p>
            <p className="mt-2 text-3xl font-semibold text-white">{kpis.active_requests || 0}</p>
          </Card>
          <Card>
            <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Clubs covered</p>
            <p className="mt-2 text-3xl font-semibold text-white">{kpis.clubs_count || 0}</p>
          </Card>
          <Card>
            <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Shortlisted players</p>
            <p className="mt-2 text-3xl font-semibold text-white">{kpis.shortlisted_players || 0}</p>
          </Card>
          <Card>
            <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Urgent needs</p>
            <p className="mt-2 text-3xl font-semibold text-white">{kpis.urgent_requests || 0}</p>
          </Card>
        </section>

        <Card>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-4 lg:grid-cols-7">
            <div className="space-y-2">
              <Label>Club</Label>
              <TextInput value={filters.club} onChange={(e) => setFilters((p) => ({ ...p, club: e.target.value }))} />
            </div>
            <div className="space-y-2">
              <Label>Position</Label>
              <TextInput value={filters.position} onChange={(e) => setFilters((p) => ({ ...p, position: e.target.value }))} />
            </div>
            <div className="space-y-2">
              <Label>Status</Label>
              <Select value={filters.status} onChange={(e) => setFilters((p) => ({ ...p, status: e.target.value }))}>
                <option value="">All</option>
                {STATUSES.map((status) => <option key={status} value={status}>{status}</option>)}
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Priority</Label>
              <Select value={filters.priority} onChange={(e) => setFilters((p) => ({ ...p, priority: e.target.value }))}>
                <option value="">All</option>
                {PRIORITIES.map((priority) => <option key={priority} value={priority}>{priority}</option>)}
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Agent</Label>
              <TextInput value={filters.agent} onChange={(e) => setFilters((p) => ({ ...p, agent: e.target.value }))} />
            </div>
            <div className="space-y-2">
              <Label>League</Label>
              <TextInput value={filters.competition} onChange={(e) => setFilters((p) => ({ ...p, competition: e.target.value }))} />
            </div>
            <div className="space-y-2">
              <Label>Deal</Label>
              <Select value={filters.deal_type} onChange={(e) => setFilters((p) => ({ ...p, deal_type: e.target.value }))}>
                <option value="">All</option>
                {DEAL_TYPES.map((deal) => <option key={deal} value={deal}>{deal}</option>)}
              </Select>
            </div>
          </div>
        </Card>

        {error ? <Card className="border-red-400/30 text-red-200">{error}</Card> : null}
        {message ? <Card className="border-primary/30 text-slate-200">{message}</Card> : null}

        <section className="space-y-3">
            {loading ? <Card>Loading needs...</Card> : null}
            {!loading && items.length === 0 ? (
              <Card>
                <p className="font-semibold text-white">No Mercato needs yet.</p>
                <p className="mt-1 text-sm text-slate-400">Create a first need or adjust the filters.</p>
              </Card>
            ) : null}
            {items.map((requestItem) => {
              const need = firstNeed(requestItem);
              const candidates = need?.candidates || [];
              return (
                <Card
                  key={requestItem.id}
                  className="cursor-pointer transition hover:border-primary/50 hover:bg-white/[0.03]"
                >
                  <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                    <button
                      type="button"
                      className="min-w-0 flex-1 text-left"
                      onClick={() => router.push(`/mercato-2026/${requestItem.id}`)}
                    >
                      <div className="flex flex-wrap items-center gap-2">
                        <h2 className="truncate text-xl font-semibold text-white">{requestItem.club_name || "Undefined club"}</h2>
                        <Badge tone={priorityTone(requestItem.priority)}>{requestItem.priority}</Badge>
                        <Badge tone={statusTone(requestItem.status)}>{requestItem.status}</Badge>
                      </div>
                      <p className="mt-1 text-sm text-slate-400">{requestItem.competition_name || "League not set"}</p>
                      <div className="mt-3 flex flex-wrap gap-2">
                        <Badge>{need?.position || "Position -"}</Badge>
                        <Badge>{requestItem.deal_type || "any"}</Badge>
                        <Badge>{formatMoney(requestItem.budget_max)} budget max</Badge>
                      </div>
                      <p className="mt-3 text-sm text-slate-300">{requestItem.title}</p>
                      <p className="mt-2 text-xs text-slate-500">
                        Agent {requestItem.assigned_agent_name || requestItem.assigned_agent_id || "-"} - {candidates.length} players - {requestItem.created_at ? new Date(requestItem.created_at).toLocaleDateString() : "-"}
                      </p>
                    </button>
                    <div className="flex shrink-0 flex-wrap gap-2">
                      <button
                        type="button"
                        className="rounded-full border border-primary px-3 py-2 text-xs font-semibold text-primary"
                        onClick={() => router.push(`/mercato-2026/${requestItem.id}`)}
                      >
                        Open need
                      </button>
                    </div>
                  </div>
                </Card>
              );
            })}
        </section>
      </div>

      {showForm ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
          <div className="max-h-[92vh] w-full max-w-5xl overflow-auto rounded-lg border border-slate-700 bg-slate-950 p-5 shadow-2xl">
            <div className="flex items-center justify-between gap-4">
              <div>
                <h2 className="text-xl font-semibold text-white">New Mercato need</h2>
                <p className="text-sm text-slate-400">Enter the club need and matching constraints.</p>
              </div>
              <button type="button" className="text-sm text-slate-300" onClick={() => setShowForm(false)}>
                Close
              </button>
            </div>

            <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-3">
              <div className="space-y-2">
                <Label>Club *</Label>
                <TextInput
                  value={form.club_query}
                  onChange={(e) =>
                    setForm((prev) => ({
                      ...prev,
                      club_query: e.target.value,
                      club_id: "",
                    }))
                  }
                  placeholder="Type a club name"
                />
                {form.club_query && !form.club_id ? (
                  <div className="max-h-48 overflow-auto rounded-md border border-slate-800 bg-slate-950">
                    {filteredClubOptions.length === 0 ? (
                      <p className="px-3 py-2 text-sm text-slate-500">No matching club.</p>
                    ) : (
                      filteredClubOptions.map((club) => (
                        <button
                          key={club.id}
                          type="button"
                          className="block w-full border-b border-slate-800 px-3 py-2 text-left text-sm hover:bg-slate-900"
                          onClick={() =>
                            setForm((prev) => ({
                              ...prev,
                              club_id: String(club.id),
                              club_query: club.label,
                            }))
                          }
                        >
                          {club.label}
                        </button>
                      ))
                    )}
                  </div>
                ) : null}
              </div>
              <div className="space-y-2">
                <Label>Title</Label>
                <TextInput value={form.title} onChange={(e) => updateForm("title", e.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Agent</Label>
                <TextInput value={form.assigned_agent_id} onChange={(e) => updateForm("assigned_agent_id", e.target.value)} placeholder={me?.username || "agent"} />
              </div>
              <div className="space-y-2">
                <Label>Position *</Label>
                <Select value={form.position} onChange={(e) => updateForm("position", e.target.value)}>
                  <option value="">Select position</option>
                  {positions.map((position) => <option key={position} value={position}>{position}</option>)}
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Required level</Label>
                <TextInput type="number" value={form.required_player_level} onChange={(e) => updateForm("required_player_level", e.target.value)} placeholder="75" />
              </div>
              <div className="space-y-2">
                <Label>Priority</Label>
                <Select value={form.priority} onChange={(e) => updateForm("priority", e.target.value)}>
                  {PRIORITIES.map((priority) => <option key={priority} value={priority}>{priority}</option>)}
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Status</Label>
                <Select value={form.status} onChange={(e) => updateForm("status", e.target.value)}>
                  {STATUSES.map((status) => <option key={status} value={status}>{status}</option>)}
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Deal type</Label>
                <Select value={form.deal_type} onChange={(e) => updateForm("deal_type", e.target.value)}>
                  {DEAL_TYPES.map((deal) => <option key={deal} value={deal}>{deal}</option>)}
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Age min</Label>
                <TextInput type="number" value={form.age_min} onChange={(e) => updateForm("age_min", e.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Age max</Label>
                <TextInput type="number" value={form.age_max} onChange={(e) => updateForm("age_max", e.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Foot</Label>
                <TextInput value={form.preferred_foot} onChange={(e) => updateForm("preferred_foot", e.target.value)} placeholder="left / right / both" />
              </div>
              <div className="space-y-2">
                <Label>Min height</Label>
                <TextInput type="number" value={form.height_min} onChange={(e) => updateForm("height_min", e.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Budget min</Label>
                <TextInput type="number" value={form.budget_min} onChange={(e) => updateForm("budget_min", e.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Budget max</Label>
                <TextInput type="number" value={form.budget_max} onChange={(e) => updateForm("budget_max", e.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Max salary</Label>
                <TextInput type="number" value={form.salary_max} onChange={(e) => updateForm("salary_max", e.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Target league</Label>
                <TextInput value={form.target_league_level} onChange={(e) => updateForm("target_league_level", e.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Nationalities</Label>
                <TextInput value={form.nationality_preferences} onChange={(e) => updateForm("nationality_preferences", e.target.value)} />
              </div>
              <div className="space-y-2 md:col-span-3">
                <Label>Additional information</Label>
                <TextArea rows={4} value={form.extra_info} onChange={(e) => updateForm("extra_info", e.target.value)} />
              </div>
              <div className="space-y-2 md:col-span-3">
                <Label>Need notes</Label>
                <TextArea rows={3} value={form.notes} onChange={(e) => updateForm("notes", e.target.value)} />
              </div>
            </div>

            <div className="mt-5 flex justify-end gap-3">
              <button type="button" className="rounded-full border border-slate-700 px-4 py-2 text-sm text-slate-200" onClick={() => setShowForm(false)}>
                Cancel
              </button>
              <button
                type="button"
                className="rounded-full border border-primary bg-primary/10 px-4 py-2 text-sm font-semibold text-primary disabled:opacity-50"
                onClick={createRequest}
                disabled={saving}
              >
                {saving ? "Creating..." : "Create need"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}
