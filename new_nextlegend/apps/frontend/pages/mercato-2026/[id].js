import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/router";
import { deleteJson, fetchJson, fetchJsonCached, patchJson, postJson } from "@/lib/api";

const TM_BASE_URL = "https://www.transfermarkt.com";
const STATUS_ACTIONS = ["approved", "rejected", "follow_up", "contacted", "proposed"];
const PRIORITIES = ["low", "medium", "high", "urgent"];
const STATUSES = ["new", "searching", "shortlist_ready", "proposed", "discussion", "closed"];
const DEAL_TYPES = ["any", "transfer", "loan", "free"];

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
    blue: "border-sky-400/40 bg-sky-400/10 text-sky-200",
  };
  return (
    <span className={`inline-flex rounded-full border px-2 py-1 text-xs ${tones[tone] || tones.default}`}>
      {children}
    </span>
  );
};

const statusTone = (status) => {
  if (status === "closed" || status === "rejected") return "red";
  if (status === "shortlist_ready" || status === "approved" || status === "signed") return "green";
  if (status === "discussion" || status === "contacted" || status === "proposed") return "amber";
  return "default";
};

const priorityTone = (priority) => {
  if (priority === "urgent") return "red";
  if (priority === "high") return "amber";
  if (priority === "low") return "default";
  return "green";
};

const firstNeed = (request) => (request?.needs || [])[0] || null;

const toAbsoluteUrl = (value) => {
  if (!value) return "";
  const url = String(value).trim();
  if (!url) return "";
  if (url.startsWith("http://") || url.startsWith("https://")) return url;
  if (url.startsWith("/")) return `${TM_BASE_URL}${url}`;
  return url;
};

const getInitials = (name) =>
  String(name || "")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("") || "NL";

const formatMoney = (value) => {
  if (value === null || value === undefined || value === "") return "-";
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  if (Math.abs(num) >= 1e6) return `${Math.round((num / 1e6) * 10) / 10}M`;
  if (Math.abs(num) >= 1e3) return `${Math.round(num / 1e3)}K`;
  return `${Math.round(num)}`;
};

const reportUrl = (candidate) => {
  const params = new URLSearchParams();
  params.set("player_id", candidate.player_id);
  if (candidate.player_season_id) params.set("player_season_id", candidate.player_season_id);
  return `/report?${params.toString()}`;
};

const buildEditForm = (requestItem) => {
  const need = firstNeed(requestItem) || {};
  return {
    club_id: requestItem?.club_id ? String(requestItem.club_id) : "",
    club_query: `${requestItem?.club_name || ""}${requestItem?.competition_name ? ` - ${requestItem.competition_name}` : ""}`,
    title: requestItem?.title || "",
    assigned_agent_id: requestItem?.assigned_agent_id || "",
    priority: requestItem?.priority || "medium",
    status: requestItem?.status || "new",
    deal_type: requestItem?.deal_type || "any",
    budget_min: requestItem?.budget_min ?? "",
    budget_max: requestItem?.budget_max ?? "",
    salary_max: requestItem?.salary_max ?? "",
    extra_info: requestItem?.extra_info || "",
    need_id: need.id || "",
    position: need.position || "",
    age_min: need.age_min ?? "",
    age_max: need.age_max ?? "",
    preferred_foot: need.preferred_foot || "",
    height_min: need.height_min ?? "",
    target_league_level: need.target_league_level || "",
    required_player_level: need.required_player_level ?? "",
    nationality_preferences: need.nationality_preferences || "",
    notes: need.notes || "",
  };
};

const MatchingCinematic = ({ selectedCount }) => (
  <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/90 px-4 backdrop-blur-md">
    <div className="w-full max-w-2xl rounded-xl border border-primary/30 bg-slate-950 p-6 shadow-2xl">
      <div className="relative overflow-hidden rounded-lg border border-slate-800 bg-slate-900/60 p-5">
        <div className="mercato-pitch" aria-hidden="true">
          <span className="mercato-orbit orbit-a" />
          <span className="mercato-orbit orbit-b" />
          <span className="mercato-node node-a" />
          <span className="mercato-node node-b" />
          <span className="mercato-node node-c" />
        </div>
        <div className="relative z-10">
          <p className="text-xs uppercase tracking-[0.3em] text-primary">Mercato matching</p>
          <h2 className="mt-2 text-2xl font-semibold text-white">Building the shortlist</h2>
          <p className="mt-2 max-w-lg text-sm text-slate-300">
            Scanning profiles, recalibrating league context and checking squad-fit constraints.
          </p>
          <div className="mt-5 grid grid-cols-1 gap-2 sm:grid-cols-3">
            {[
              "League calibration",
              selectedCount > 0 ? `${selectedCount} selected leagues` : "All leagues",
              "Top 5 recommendation",
            ].map((label) => (
              <div key={label} className="rounded-md border border-slate-800 bg-slate-950/70 px-3 py-2 text-xs text-slate-200">
                {label}
              </div>
            ))}
          </div>
          <div className="mt-5 h-1.5 overflow-hidden rounded-full bg-slate-800">
            <div className="mercato-progress h-full rounded-full bg-primary" />
          </div>
        </div>
      </div>
      <style jsx>{`
        .mercato-pitch {
          position: absolute;
          inset: 0;
          background:
            linear-gradient(90deg, transparent 49%, rgba(45, 212, 191, 0.18) 50%, transparent 51%),
            linear-gradient(0deg, transparent 49%, rgba(45, 212, 191, 0.12) 50%, transparent 51%),
            radial-gradient(circle at center, rgba(45, 212, 191, 0.12), transparent 32%);
          opacity: 0.75;
        }
        .mercato-orbit,
        .mercato-node {
          position: absolute;
          display: block;
          border-radius: 999px;
        }
        .mercato-orbit {
          border: 1px solid rgba(45, 212, 191, 0.25);
          animation: pulse-ring 1.5s ease-in-out infinite;
        }
        .orbit-a {
          height: 170px;
          width: 170px;
          left: 8%;
          top: 16%;
        }
        .orbit-b {
          height: 220px;
          width: 220px;
          right: 6%;
          bottom: -20%;
          animation-delay: 0.35s;
        }
        .mercato-node {
          height: 10px;
          width: 10px;
          background: #2dd4bf;
          box-shadow: 0 0 24px rgba(45, 212, 191, 0.8);
          animation: float-node 1.2s ease-in-out infinite;
        }
        .node-a {
          left: 18%;
          top: 38%;
        }
        .node-b {
          left: 52%;
          top: 26%;
          animation-delay: 0.2s;
        }
        .node-c {
          right: 20%;
          bottom: 26%;
          animation-delay: 0.4s;
        }
        .mercato-progress {
          width: 45%;
          animation: scan-progress 1.2s ease-in-out infinite;
        }
        @keyframes pulse-ring {
          0%, 100% { transform: scale(0.92); opacity: 0.35; }
          50% { transform: scale(1.06); opacity: 0.8; }
        }
        @keyframes float-node {
          0%, 100% { transform: translateY(0); opacity: 0.65; }
          50% { transform: translateY(-10px); opacity: 1; }
        }
        @keyframes scan-progress {
          0% { transform: translateX(-120%); }
          100% { transform: translateX(240%); }
        }
      `}</style>
    </div>
  </div>
);

const CandidateCard = ({ candidate, onStatus, onNote }) => {
  const explanation = candidate.explanation_json || {};
  const strengths = explanation.strengths || [];
  const risks = explanation.risks || [];
  const tmFields = candidate.tm_fields || {};
  const photoUrl = toAbsoluteUrl(tmFields.tm_profile_image_url || tmFields.profile_image_url);
  const url = reportUrl(candidate);

  return (
    <Card className="overflow-hidden p-0 transition hover:border-primary/40">
      <div className="flex flex-col gap-4 p-4 lg:flex-row lg:items-start">
        <button
          type="button"
          className="flex min-w-0 flex-1 gap-4 text-left"
          onClick={() => window.open(url, "_blank", "noopener,noreferrer")}
        >
          {photoUrl ? (
            <img
              src={photoUrl}
              alt={candidate.name}
              className="h-20 w-20 shrink-0 rounded-lg border border-white/10 object-cover"
            />
          ) : (
            <div className="flex h-20 w-20 shrink-0 items-center justify-center rounded-lg border border-white/10 bg-slate-800 text-lg font-semibold text-slate-100">
              {getInitials(candidate.name)}
            </div>
          )}
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="truncate text-lg font-semibold text-white">{candidate.name}</h3>
              <Badge tone={statusTone(candidate.status)}>{candidate.status}</Badge>
              <Badge tone="blue">{candidate.source}</Badge>
            </div>
            <p className="mt-1 text-sm text-slate-400">
              {candidate.age ? `${Number(candidate.age).toFixed(0)} yrs - ` : ""}
              {[candidate.position, candidate.second_position].filter(Boolean).join(" / ") || "-"}
              {" - "}
              {candidate.team || "-"}
            </p>
            <p className="mt-1 text-xs text-slate-500">
              {candidate.competition_name || "-"} - {candidate.calendar || "-"}
            </p>
            {explanation.recommendation_reason ? (
              <p className="mt-3 text-sm text-slate-300">{explanation.recommendation_reason}</p>
            ) : null}
          </div>
        </button>

        <div className="grid shrink-0 grid-cols-3 gap-2 lg:w-56">
          <div className="rounded-md border border-slate-800 bg-slate-950/70 p-2">
            <p className="text-xs text-slate-500">Raw</p>
            <p className="text-lg font-semibold text-white">{candidate.raw_player_level != null ? Number(candidate.raw_player_level).toFixed(0) : "-"}</p>
          </div>
          <div className="rounded-md border border-slate-800 bg-slate-950/70 p-2">
            <p className="text-xs text-slate-500">Adjusted</p>
            <p className="text-lg font-semibold text-white">{candidate.calibrated_player_level != null ? Number(candidate.calibrated_player_level).toFixed(0) : "-"}</p>
          </div>
          <div className="rounded-md border border-slate-800 bg-slate-950/70 p-2">
            <p className="text-xs text-slate-500">Match</p>
            <p className="text-lg font-semibold text-primary">{candidate.match_score != null ? Number(candidate.match_score).toFixed(0) : "-"}</p>
          </div>
        </div>
      </div>

      <div className="grid gap-3 border-t border-slate-800 p-4 lg:grid-cols-[minmax(0,1fr)_280px]">
        <div className="space-y-2">
          {strengths.slice(0, 2).map((item) => (
            <p key={item} className="text-xs text-primary">{item}</p>
          ))}
          {risks.slice(0, 2).map((item) => (
            <p key={item} className="text-xs text-amber-200">{item}</p>
          ))}
        </div>
        <div className="space-y-3">
          <div className="flex flex-wrap gap-2">
            {STATUS_ACTIONS.map((status) => (
              <button
                key={status}
                type="button"
                className="rounded-full border border-slate-700 px-2 py-1 text-xs text-slate-200 hover:border-primary hover:text-primary"
                onClick={() => onStatus(candidate.id, status)}
              >
                {status}
              </button>
            ))}
          </div>
          <TextArea
            rows={2}
            placeholder="Agent note"
            defaultValue={candidate.agent_note || ""}
            onBlur={(event) => onNote(candidate.id, event.target.value)}
          />
          <button
            type="button"
            className="w-full rounded-full border border-primary px-3 py-2 text-xs font-semibold text-primary"
            onClick={() => window.open(url, "_blank", "noopener,noreferrer")}
          >
            Open report
          </button>
        </div>
      </div>
    </Card>
  );
};

const LeagueSelector = ({ competitions, selected, onChange }) => {
  const [query, setQuery] = useState("");
  const selectedSet = useMemo(() => new Set(selected), [selected]);
  const filteredCompetitions = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    const filtered = normalizedQuery
      ? competitions.filter((competition) => competition.toLowerCase().includes(normalizedQuery))
      : competitions;
    return filtered.slice(0, 80);
  }, [competitions, query]);

  const toggleCompetition = (competition) => {
    if (selectedSet.has(competition)) {
      onChange(selected.filter((item) => item !== competition));
      return;
    }
    onChange([...selected, competition]);
  };

  const selectVisible = () => {
    const next = new Set(selected);
    filteredCompetitions.forEach((competition) => next.add(competition));
    onChange(Array.from(next));
  };

  return (
    <div className="space-y-3">
      <div className="space-y-2">
        <Label>Search leagues</Label>
        <TextInput
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Type a league name"
        />
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={selected.length > 0 ? "green" : "default"}>
          {selected.length > 0 ? `${selected.length} selected` : "All leagues"}
        </Badge>
        <button
          type="button"
          className="text-xs text-slate-300 hover:text-white"
          onClick={selectVisible}
          disabled={filteredCompetitions.length === 0}
        >
          Select visible
        </button>
        <button
          type="button"
          className="text-xs text-slate-300 hover:text-white disabled:opacity-40"
          onClick={() => onChange([])}
          disabled={selected.length === 0}
        >
          Clear all
        </button>
      </div>

      {selected.length > 0 ? (
        <div className="flex max-h-24 flex-wrap gap-2 overflow-auto rounded-md border border-slate-800 bg-slate-950/50 p-2">
          {selected.map((competition) => (
            <button
              key={competition}
              type="button"
              className="rounded-full border border-primary/40 bg-primary/10 px-2 py-1 text-xs text-primary hover:border-primary"
              onClick={() => toggleCompetition(competition)}
              title="Remove league"
            >
              {competition} x
            </button>
          ))}
        </div>
      ) : (
        <p className="rounded-md border border-slate-800 bg-slate-950/50 px-3 py-2 text-xs text-slate-500">
          No league selected. Matching will search every league.
        </p>
      )}

      <div className="max-h-80 overflow-auto rounded-md border border-slate-800 bg-slate-950/70">
        {filteredCompetitions.length === 0 ? (
          <p className="px-3 py-3 text-sm text-slate-500">No matching league.</p>
        ) : (
          filteredCompetitions.map((competition) => {
            const checked = selectedSet.has(competition);
            return (
              <button
                key={competition}
                type="button"
                className={`flex w-full items-center gap-3 border-b border-slate-800 px-3 py-2 text-left text-sm hover:bg-slate-900 ${
                  checked ? "bg-primary/10 text-primary" : "text-slate-200"
                }`}
                onClick={() => toggleCompetition(competition)}
              >
                <span
                  className={`flex h-4 w-4 shrink-0 items-center justify-center rounded border text-[10px] ${
                    checked ? "border-primary bg-primary text-slate-950" : "border-slate-600"
                  }`}
                >
                  {checked ? "✓" : ""}
                </span>
                <span className="min-w-0 truncate">{competition}</span>
              </button>
            );
          })
        )}
      </div>
    </div>
  );
};

export default function MercatoNeedDetailPage() {
  const router = useRouter();
  const requestId = router.query.id;
  const [requestItem, setRequestItem] = useState(null);
  const [clubs, setClubs] = useState([]);
  const [competitions, setCompetitions] = useState([]);
  const [positions, setPositions] = useState([]);
  const [searchCompetitions, setSearchCompetitions] = useState([]);
  const [playerQuery, setPlayerQuery] = useState("");
  const [playerResults, setPlayerResults] = useState([]);
  const [candidateNote, setCandidateNote] = useState("");
  const [showEdit, setShowEdit] = useState(false);
  const [editForm, setEditForm] = useState(null);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [savingEdit, setSavingEdit] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [candidateBusy, setCandidateBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const need = firstNeed(requestItem);
  const candidates = need?.candidates || [];
  const clubOptions = useMemo(() => {
    return clubs.map((club) => ({
      ...club,
      label: `${club.name}${club.competition_name ? ` - ${club.competition_name}` : ""}`,
    }));
  }, [clubs]);
  const filteredClubOptions = useMemo(() => {
    const query = String(editForm?.club_query || "").trim().toLowerCase();
    if (!query) return clubOptions.slice(0, 12);
    return clubOptions.filter((club) => club.label.toLowerCase().includes(query)).slice(0, 12);
  }, [clubOptions, editForm?.club_query]);

  const loadRequest = async () => {
    if (!requestId) return;
    setLoading(true);
    setError("");
    try {
      const data = await fetchJson(`/mercato/requests/${requestId}`);
      setRequestItem(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    Promise.all([
      fetchJsonCached("/meta/clubs"),
      fetchJsonCached("/meta/competitions"),
      fetchJsonCached("/meta/positions"),
    ])
      .then(([clubItems, competitionItems, positionItems]) => {
        setClubs(clubItems || []);
        setCompetitions((competitionItems || []).map((item) => item.name).filter(Boolean));
        setPositions((positionItems || []).filter((position) => {
          const normalized = String(position || "").trim().toLowerCase();
          return normalized && normalized !== "<na>" && normalized !== "na" && normalized !== "nan";
        }));
      })
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    loadRequest();
  }, [requestId]);

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
        setMessage(err.message);
      }
    }, 200);
    return () => clearTimeout(handle);
  }, [playerQuery]);

  const generateShortlist = async () => {
    if (!need || generating) return;
    setGenerating(true);
    setMessage("");
    try {
      const res = await postJson(`/mercato/needs/${need.id}/generate-shortlist`, {
        competitions: searchCompetitions,
      });
      setMessage(res.generated ? `${res.generated} players shortlisted.` : "No new player found or all matching players are already assigned.");
      await loadRequest();
    } catch (err) {
      setMessage(err.message);
    } finally {
      setGenerating(false);
    }
  };

  const openEdit = () => {
    setEditForm(buildEditForm(requestItem));
    setShowEdit(true);
    setMessage("");
  };

  const updateEditForm = (key, value) => {
    setEditForm((prev) => ({ ...(prev || {}), [key]: value }));
  };

  const saveNeed = async () => {
    if (!editForm?.club_id) {
      setMessage("Select a club.");
      return;
    }
    if (!String(editForm.position || "").trim()) {
      setMessage("Select a position.");
      return;
    }
    setSavingEdit(true);
    setMessage("");
    try {
      await patchJson(`/mercato/requests/${requestId}`, {
        club_id: Number(editForm.club_id),
        assigned_agent_id: editForm.assigned_agent_id || null,
        title: editForm.title || editForm.position || "Mercato need",
        priority: editForm.priority,
        status: editForm.status,
        budget_min: editForm.budget_min !== "" ? Number(editForm.budget_min) : null,
        budget_max: editForm.budget_max !== "" ? Number(editForm.budget_max) : null,
        salary_max: editForm.salary_max !== "" ? Number(editForm.salary_max) : null,
        deal_type: editForm.deal_type || "any",
        extra_info: editForm.extra_info || null,
        need: {
          id: editForm.need_id,
          position: editForm.position || null,
          age_min: editForm.age_min !== "" ? Number(editForm.age_min) : null,
          age_max: editForm.age_max !== "" ? Number(editForm.age_max) : null,
          preferred_foot: editForm.preferred_foot || null,
          height_min: editForm.height_min !== "" ? Number(editForm.height_min) : null,
          target_league_level: editForm.target_league_level || null,
          required_player_level: editForm.required_player_level !== "" ? Number(editForm.required_player_level) : null,
          nationality_preferences: editForm.nationality_preferences || null,
          notes: editForm.notes || null,
        },
      });
      setShowEdit(false);
      setEditForm(null);
      setMessage("Need updated.");
      await loadRequest();
    } catch (err) {
      setMessage(err.message);
    } finally {
      setSavingEdit(false);
    }
  };

  const deleteNeed = async () => {
    if (!window.confirm("Delete this Mercato need? It will be closed and removed from the active workspace.")) {
      return;
    }
    setDeleting(true);
    setMessage("");
    try {
      await deleteJson(`/mercato/requests/${requestId}`);
      router.push("/mercato-2026");
    } catch (err) {
      setMessage(err.message);
    } finally {
      setDeleting(false);
    }
  };

  const addCandidate = async (player) => {
    if (!need || candidateBusy) return;
    setCandidateBusy(true);
    setMessage("");
    try {
      const res = await postJson(`/mercato/needs/${need.id}/candidates`, {
        player_id: Number(player.id),
        player_season_id: player.player_season_id ? Number(player.player_season_id) : null,
        source: "manual",
        status: "suggested",
        agent_note: candidateNote || null,
      });
      setMessage(res.added ? "Player added to the need." : "This player is already assigned to this need.");
      setPlayerQuery("");
      setPlayerResults([]);
      setCandidateNote("");
      await loadRequest();
    } catch (err) {
      setMessage(err.message);
    } finally {
      setCandidateBusy(false);
    }
  };

  const updateCandidateStatus = async (candidateId, status) => {
    try {
      await postJson(`/mercato/candidates/${candidateId}/status`, { status });
      await loadRequest();
    } catch (err) {
      setMessage(err.message);
    }
  };

  const updateCandidateNote = async (candidateId, agentNote) => {
    try {
      await patchJson(`/mercato/candidates/${candidateId}`, { agent_note: agentNote });
      await loadRequest();
    } catch (err) {
      setMessage(err.message);
    }
  };

  const statItems = useMemo(() => {
    if (!requestItem || !need) return [];
    return [
      ["Position", need.position || "-"],
      ["Age range", `${need.age_min || "-"} - ${need.age_max || "-"}`],
      ["Required level", need.required_player_level || "-"],
      ["Budget max", formatMoney(requestItem.budget_max)],
      ["Deal", requestItem.deal_type || "any"],
      ["Shortlist", candidates.length],
    ];
  }, [requestItem, need, candidates.length]);

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      {generating ? <MatchingCinematic selectedCount={searchCompetitions.length} /> : null}

      <div className="mx-auto max-w-7xl px-4 py-8 space-y-6">
        <button
          type="button"
          className="text-sm text-slate-300 hover:text-white"
          onClick={() => router.push("/mercato-2026")}
        >
          Back to Mercato
        </button>

        {loading ? <Card>Loading need...</Card> : null}
        {error ? <Card className="border-red-400/30 text-red-200">{error}</Card> : null}

        {!loading && requestItem && need ? (
          <>
            <section className="rounded-xl border border-white/5 bg-slate-900/50 p-5">
              <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge tone={priorityTone(requestItem.priority)}>{requestItem.priority}</Badge>
                    <Badge tone={statusTone(requestItem.status)}>{requestItem.status}</Badge>
                    <Badge>{requestItem.season}</Badge>
                  </div>
                  <h1 className="mt-3 text-4xl font-semibold text-white">
                    {requestItem.club_name || "Undefined club"} - {need.position || "Need"}
                  </h1>
                  <p className="mt-2 text-sm text-slate-400">
                    {requestItem.competition_name || "League not set"} - {requestItem.title}
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    className="rounded-full border border-slate-700 px-4 py-3 text-sm font-semibold text-slate-200 hover:border-primary hover:text-primary"
                    onClick={openEdit}
                  >
                    Edit need
                  </button>
                  <button
                    type="button"
                    className="rounded-full border border-red-400/40 px-4 py-3 text-sm font-semibold text-red-200 hover:border-red-300 disabled:opacity-50"
                    disabled={deleting}
                    onClick={deleteNeed}
                  >
                    {deleting ? "Deleting..." : "Delete need"}
                  </button>
                  <button
                    type="button"
                    className="rounded-full border border-primary bg-primary/10 px-5 py-3 text-sm font-semibold text-primary disabled:opacity-50"
                    disabled={requestItem.status === "closed" || generating}
                    onClick={generateShortlist}
                  >
                    Start matching
                  </button>
                </div>
              </div>

              <div className="mt-6 grid grid-cols-2 gap-3 lg:grid-cols-6">
                {statItems.map(([label, value]) => (
                  <div key={label} className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
                    <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{label}</p>
                    <p className="mt-2 text-lg font-semibold text-white">{value}</p>
                  </div>
                ))}
              </div>
            </section>

            {message ? <Card className="border-primary/30 text-slate-200">{message}</Card> : null}

            <section className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
              <div className="space-y-4">
                <Card>
                  <h2 className="text-lg font-semibold text-white">Need brief</h2>
                  <div className="mt-4 grid grid-cols-1 gap-3 text-sm md:grid-cols-2">
                    <p><span className="text-slate-500">Club:</span> {requestItem.club_name || "-"}</p>
                    <p><span className="text-slate-500">League:</span> {requestItem.competition_name || "-"}</p>
                    <p><span className="text-slate-500">Agent:</span> {requestItem.assigned_agent_name || requestItem.assigned_agent_id || "-"}</p>
                    <p><span className="text-slate-500">Created:</span> {requestItem.created_at ? new Date(requestItem.created_at).toLocaleDateString() : "-"}</p>
                    <p><span className="text-slate-500">Foot:</span> {need.preferred_foot || "Any"}</p>
                    <p><span className="text-slate-500">Min height:</span> {need.height_min || "-"}</p>
                    <p><span className="text-slate-500">Budget:</span> {formatMoney(requestItem.budget_min)} - {formatMoney(requestItem.budget_max)}</p>
                    <p><span className="text-slate-500">Max salary:</span> {formatMoney(requestItem.salary_max)}</p>
                  </div>
                  <div className="mt-4 grid gap-3 md:grid-cols-2">
                    <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
                      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Additional information</p>
                      <p className="mt-2 text-sm text-slate-300">{requestItem.extra_info || "-"}</p>
                    </div>
                    <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
                      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Need notes</p>
                      <p className="mt-2 text-sm text-slate-300">{need.notes || "-"}</p>
                    </div>
                  </div>
                </Card>

                <div className="space-y-3">
                  <div className="flex items-end justify-between gap-3">
                    <div>
                      <h2 className="text-xl font-semibold text-white">Shortlist</h2>
                      <p className="text-sm text-slate-400">Saved candidates are shared with all authenticated users.</p>
                    </div>
                    <Badge tone="green">{candidates.length} players</Badge>
                  </div>
                  {candidates.length === 0 ? (
                    <Card>
                      <p className="font-semibold text-white">No player assigned yet.</p>
                      <p className="mt-1 text-sm text-slate-400">Run matching or add a player manually.</p>
                    </Card>
                  ) : null}
                  {candidates.map((candidate) => (
                    <CandidateCard
                      key={candidate.id}
                      candidate={candidate}
                      onStatus={updateCandidateStatus}
                      onNote={updateCandidateNote}
                    />
                  ))}
                </div>
              </div>

              <aside className="space-y-4">
                <Card>
                  <h2 className="text-lg font-semibold text-white">Matching setup</h2>
                  <div className="mt-4">
                    <LeagueSelector
                      competitions={competitions}
                      selected={searchCompetitions}
                      onChange={setSearchCompetitions}
                    />
                  </div>
                  <button
                    type="button"
                    className="mt-4 w-full rounded-full border border-primary bg-primary/10 px-4 py-3 text-sm font-semibold text-primary disabled:opacity-50"
                    disabled={requestItem.status === "closed" || generating}
                    onClick={generateShortlist}
                  >
                    Start matching
                  </button>
                </Card>

                <Card>
                  <h2 className="text-lg font-semibold text-white">Add player manually</h2>
                  <div className="mt-4 space-y-3">
                    <TextInput
                      placeholder="Search player"
                      value={playerQuery}
                      onChange={(event) => setPlayerQuery(event.target.value)}
                      disabled={requestItem.status === "closed"}
                    />
                    <TextInput
                      placeholder="Optional agent note"
                      value={candidateNote}
                      onChange={(event) => setCandidateNote(event.target.value)}
                      disabled={requestItem.status === "closed"}
                    />
                    {playerResults.length > 0 ? (
                      <div className="max-h-72 overflow-auto rounded-md border border-slate-800 bg-slate-950">
                        {playerResults.map((player) => {
                          const tmFields = player.tm_fields || {};
                          const photoUrl = toAbsoluteUrl(tmFields.tm_profile_image_url || tmFields.profile_image_url);
                          return (
                            <button
                              key={`${player.id}-${player.player_season_id}`}
                              type="button"
                              className="flex w-full gap-3 border-b border-slate-800 px-3 py-2 text-left text-sm hover:bg-slate-900"
                              onClick={() => addCandidate(player)}
                              disabled={candidateBusy || requestItem.status === "closed"}
                            >
                              {photoUrl ? (
                                <img src={photoUrl} alt={player.name} className="h-10 w-10 rounded-md object-cover" />
                              ) : (
                                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-slate-800 text-xs font-semibold text-slate-200">
                                  {getInitials(player.name)}
                                </span>
                              )}
                              <span className="min-w-0">
                                <span className="block truncate font-medium text-white">{player.name}</span>
                                <span className="block truncate text-xs text-slate-400">
                                  {player.team || "-"} - {player.competition_name || "-"} - {player.calendar || "-"}
                                </span>
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    ) : null}
                  </div>
                </Card>
              </aside>
            </section>
          </>
        ) : null}
      </div>

      {showEdit && editForm ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
          <div className="max-h-[92vh] w-full max-w-5xl overflow-auto rounded-lg border border-slate-700 bg-slate-950 p-5 shadow-2xl">
            <div className="flex items-center justify-between gap-4">
              <div>
                <h2 className="text-xl font-semibold text-white">Edit Mercato need</h2>
                <p className="text-sm text-slate-400">Update the club brief, constraints and ownership.</p>
              </div>
              <button type="button" className="text-sm text-slate-300" onClick={() => setShowEdit(false)}>
                Close
              </button>
            </div>

            <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-3">
              <div className="space-y-2">
                <Label>Club *</Label>
                <TextInput
                  value={editForm.club_query}
                  onChange={(event) =>
                    setEditForm((prev) => ({
                      ...(prev || {}),
                      club_query: event.target.value,
                      club_id: "",
                    }))
                  }
                  placeholder="Type a club name"
                />
                {editForm.club_query && !editForm.club_id ? (
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
                            setEditForm((prev) => ({
                              ...(prev || {}),
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
                <Label>Position *</Label>
                <Select value={editForm.position} onChange={(event) => updateEditForm("position", event.target.value)}>
                  <option value="">Select position</option>
                  {positions.map((position) => <option key={position} value={position}>{position}</option>)}
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Title</Label>
                <TextInput value={editForm.title} onChange={(event) => updateEditForm("title", event.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Agent</Label>
                <TextInput value={editForm.assigned_agent_id} onChange={(event) => updateEditForm("assigned_agent_id", event.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Priority</Label>
                <Select value={editForm.priority} onChange={(event) => updateEditForm("priority", event.target.value)}>
                  {PRIORITIES.map((priority) => <option key={priority} value={priority}>{priority}</option>)}
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Status</Label>
                <Select value={editForm.status} onChange={(event) => updateEditForm("status", event.target.value)}>
                  {STATUSES.map((status) => <option key={status} value={status}>{status}</option>)}
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Deal type</Label>
                <Select value={editForm.deal_type} onChange={(event) => updateEditForm("deal_type", event.target.value)}>
                  {DEAL_TYPES.map((deal) => <option key={deal} value={deal}>{deal}</option>)}
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Required level</Label>
                <TextInput type="number" value={editForm.required_player_level} onChange={(event) => updateEditForm("required_player_level", event.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Target league</Label>
                <TextInput value={editForm.target_league_level} onChange={(event) => updateEditForm("target_league_level", event.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Age min</Label>
                <TextInput type="number" value={editForm.age_min} onChange={(event) => updateEditForm("age_min", event.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Age max</Label>
                <TextInput type="number" value={editForm.age_max} onChange={(event) => updateEditForm("age_max", event.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Foot</Label>
                <TextInput value={editForm.preferred_foot} onChange={(event) => updateEditForm("preferred_foot", event.target.value)} placeholder="left / right / both" />
              </div>
              <div className="space-y-2">
                <Label>Min height</Label>
                <TextInput type="number" value={editForm.height_min} onChange={(event) => updateEditForm("height_min", event.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Budget min</Label>
                <TextInput type="number" value={editForm.budget_min} onChange={(event) => updateEditForm("budget_min", event.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Budget max</Label>
                <TextInput type="number" value={editForm.budget_max} onChange={(event) => updateEditForm("budget_max", event.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Max salary</Label>
                <TextInput type="number" value={editForm.salary_max} onChange={(event) => updateEditForm("salary_max", event.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Nationalities</Label>
                <TextInput value={editForm.nationality_preferences} onChange={(event) => updateEditForm("nationality_preferences", event.target.value)} />
              </div>
              <div className="space-y-2 md:col-span-3">
                <Label>Additional information</Label>
                <TextArea rows={4} value={editForm.extra_info} onChange={(event) => updateEditForm("extra_info", event.target.value)} />
              </div>
              <div className="space-y-2 md:col-span-3">
                <Label>Need notes</Label>
                <TextArea rows={3} value={editForm.notes} onChange={(event) => updateEditForm("notes", event.target.value)} />
              </div>
            </div>

            <div className="mt-5 flex justify-end gap-3">
              <button type="button" className="rounded-full border border-slate-700 px-4 py-2 text-sm text-slate-200" onClick={() => setShowEdit(false)}>
                Cancel
              </button>
              <button
                type="button"
                className="rounded-full border border-primary bg-primary/10 px-4 py-2 text-sm font-semibold text-primary disabled:opacity-50"
                onClick={saveNeed}
                disabled={savingEdit}
              >
                {savingEdit ? "Saving..." : "Save changes"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}
