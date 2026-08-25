import Link from "next/link";
import { useRouter } from "next/router";
import { useEffect, useMemo, useState } from "react";
import {
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import ClubLogo from "@/components/ClubLogo";
import { deleteJson, fetchJson, patchJson, postJson } from "@/lib/api";
import { METRIC_LABELS } from "@/lib/metricLabels";
import { englishRole, normalizeRoleForUse } from "@/lib/roles";

const AGENTS = ["Steven", "Don", "Yannis", "Lidahi"];
const PRIORITIES = ["A", "B", "C", "D"];
const DOC_TYPES = [
  { value: "contract", label: "Contract" },
  { value: "mandate", label: "Mandate" },
  { value: "medical", label: "Medical" },
  { value: "passport", label: "Passport" },
  { value: "report", label: "Report" },
  { value: "other", label: "Other" },
];
const EXCLUDED_DOMINANT_PREFIXES = [
  "calendar",
  "page_number",
  "row_number",
  "player",
  "player_id",
  "birth_year",
  "age",
  "matches_played",
  "minutes_played",
  "team",
  "competition",
  "assigned_role",
  "summary_",
  "tm_",
];

const ALWAYS_EXCLUDED_DOMINANT_METRICS = [
  "height",
  "height_cm",
  "weight",
  "birth",
  "market_value",
  "contract",
  "yellow_card",
  "red_card",
  "foul",
  "goals_conceded",
  "xg_against",
];

const WIDE_SERVICE_METRICS = ["cross", "deep_cross", "goal_area_cross", "passes_to_penalty_area", "xa"];
const BALL_CARRY_METRICS = ["dribble", "progressive_run", "acceleration"];

const isRelevantDominantMetric = (key, playerContext = {}) => {
  const normalizedKey = String(key || "").toLowerCase();
  if (ALWAYS_EXCLUDED_DOMINANT_METRICS.some((item) => normalizedKey.includes(item))) return false;
  const position = String(playerContext.position || "").toUpperCase();
  const role = normalizeRoleForUse(playerContext.role || "");
  const isWideRole =
    ["LB", "RB", "LWB", "RWB", "LW", "RW", "LM", "RM"].some((item) => position.includes(item)) ||
    role.includes("wing") ||
    role.includes("wide") ||
    role.includes("full back") ||
    role.includes("left back") ||
    role.includes("right back");
  const isCentralDefender =
    ["CB", "LCB", "RCB"].some((item) => position === item || position.includes(item)) ||
    role.includes("centre back") ||
    role.includes("center back");
  if (isCentralDefender && WIDE_SERVICE_METRICS.some((item) => normalizedKey.includes(item))) return false;
  if (isCentralDefender && !role.includes("wide") && BALL_CARRY_METRICS.some((item) => normalizedKey.includes(item))) return false;
  if (!isWideRole && normalizedKey.includes("accurate_crosses_percent")) return false;
  return true;
};

const emptyDoc = {
  document_type: "contract",
  title: "",
  file_name: "",
  file_key: "",
  storage_url: "",
  content_type: "",
  size_bytes: null,
  notes: "",
};

const STATIC_PLAYER_PROSPECTS = {
  Kevin: [
    ["Monaco", "Interest", "", "", ""],
    ["Marseille", "Pending", "", "", ""],
    ["Everton", "Pending", "", "", ""],
    ["Nottingham", "Pending", "", "", ""],
    ["Brentford", "No interest", "", "", ""],
    ["Newcaslte", "No interest", "", "", ""],
    ["Atletico Madrid", "No interest", "", "", ""],
  ],
  Lilian: [["Bologna", "Interest", "", "", ""]],
  Mario: [["Al Shabab", "Offer", "3.5 net", "", ""]],
  Simon: [["Cruzeiro Esporte Clube", "No interest", "", "Damon intermediary", ""]],
};

const money = (value) => {
  if (value === null || value === undefined || value === "") return "-";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";
  if (Math.abs(numeric) >= 1000000) return `${Math.round((numeric / 1000000) * 10) / 10}M`;
  if (Math.abs(numeric) >= 1000) return `${Math.round(numeric / 1000)}K`;
  return `${Math.round(numeric)}`;
};

const initials = (name) =>
  String(name || "")
    .split(" ")
    .map((part) => part[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

const value = (input) => (input === null || input === undefined || input === "" ? "-" : input);

const externalHref = (url) => {
  const clean = String(url || "").trim();
  if (!clean) return "";
  if (/^https?:\/\//i.test(clean)) return clean;
  return `https://${clean}`;
};

const contactHref = (type, value) => {
  const clean = String(value || "").trim();
  if (!clean) return "";
  if (type === "email") return `mailto:${clean}`;
  return `tel:${clean.replace(/[^\d+]/g, "")}`;
};

const docTypeLabel = (type) =>
  DOC_TYPES.find((item) => item.value === type)?.label || "Other";

const formatMetricLabel = (metric) =>
  METRIC_LABELS[metric] ||
  String(metric || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());

const formatBytes = (size) => {
  const numeric = Number(size);
  if (!Number.isFinite(numeric) || numeric <= 0) return "-";
  if (numeric >= 1024 * 1024) return `${(numeric / (1024 * 1024)).toFixed(1)} MB`;
  if (numeric >= 1024) return `${Math.round(numeric / 1024)} KB`;
  return `${numeric} B`;
};

const formatTransferDate = (input) => {
  if (!input) return "Date to confirm";
  const parsed = new Date(input);
  if (Number.isNaN(parsed.getTime())) return String(input);
  return new Intl.DateTimeFormat("en", { day: "2-digit", month: "short", year: "numeric" }).format(parsed);
};

const transferFeeLabel = (input) => {
  const clean = String(input || "").trim();
  if (!clean) return "Undisclosed";
  const numeric = Number(clean.replace(/[^\d.-]/g, ""));
  if (Number.isFinite(numeric) && numeric > 0 && /^[\d\s.,€£$-]+$/.test(clean)) {
    return money(numeric);
  }
  return clean;
};

const readFileAsDataUrl = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error("Unable to read file."));
    reader.readAsDataURL(file);
  });

const canInlinePreview = (doc) =>
  String(doc?.content_type || "").startsWith("image/") ||
  String(doc?.content_type || "").includes("pdf") ||
  String(doc?.storage_url || "").startsWith("data:image/") ||
  String(doc?.storage_url || "").startsWith("data:application/pdf");

const DocumentPreview = ({ doc }) => {
  if (!doc?.storage_url) {
    return (
      <div className="rounded-md border border-dashed border-slate-300 bg-white p-4 text-sm font-semibold text-slate-500">
        No file attached.
      </div>
    );
  }
  if (!canInlinePreview(doc)) {
    return (
      <div className="rounded-md border border-slate-200 bg-white p-4 text-sm text-slate-600">
        Preview is not available for this file type. Use Download to open it locally.
      </div>
    );
  }
  if (String(doc.content_type || "").startsWith("image/") || String(doc.storage_url || "").startsWith("data:image/")) {
    return <img src={doc.storage_url} alt="" className="max-h-80 w-full rounded-md border border-slate-200 object-contain bg-white" />;
  }
  return (
    <iframe
      title={`${doc.title || "Document"} preview`}
      src={doc.storage_url}
      className="h-80 w-full rounded-md border border-slate-200 bg-white"
    />
  );
};

const DocumentTilePreview = ({ doc }) => {
  const isImage = String(doc?.content_type || "").startsWith("image/") || String(doc?.storage_url || "").startsWith("data:image/");
  const isPdf = String(doc?.content_type || "").includes("pdf") || String(doc?.storage_url || "").startsWith("data:application/pdf");
  if (isImage && doc?.storage_url) {
    return <img src={doc.storage_url} alt="" className="h-full w-full object-cover" />;
  }
  return (
    <div className="flex h-full w-full flex-col items-center justify-center bg-white">
      <div className="flex h-11 w-9 items-center justify-center rounded border border-slate-300 bg-slate-50 text-[10px] font-black uppercase text-slate-600 shadow-sm">
        {isPdf ? "PDF" : "DOC"}
      </div>
      <div className="mt-2 h-1 w-10 rounded-full bg-slate-200" />
      <div className="mt-1 h-1 w-7 rounded-full bg-slate-200" />
    </div>
  );
};

const RadarTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const item = payload[0]?.payload || {};
  return (
    <div className="rounded-md border border-slate-200 bg-white p-3 text-xs shadow-xl">
      <p className="font-extrabold text-slate-950">{item.metric}</p>
      <p className="mt-1 text-slate-600">Percentile: {Math.round(item.value || 0)}</p>
      <p className="text-slate-500">Raw: {item.raw != null ? Number(item.raw).toFixed(2) : "-"}</p>
    </div>
  );
};

export default function HdPlayerDetailPage() {
  const router = useRouter();
  const { id } = router.query;
  const [player, setPlayer] = useState(null);
  const [report, setReport] = useState(null);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [docForm, setDocForm] = useState(emptyDoc);
  const [editingDocId, setEditingDocId] = useState(null);
  const [editingDoc, setEditingDoc] = useState(emptyDoc);
  const [message, setMessage] = useState("");
  const [isDocDragging, setIsDocDragging] = useState(false);
  const [selectedDocId, setSelectedDocId] = useState(null);

  const load = async () => {
    if (!id) return;
    const data = await fetchJson(`/hd-players/${id}`);
    setPlayer(data);
    if (data.player_id) {
      try {
        setReport(await fetchJson(`/players/${data.player_id}/report`));
      } catch (err) {
        setReport(null);
      }
    } else {
      setReport(null);
    }
  };

  useEffect(() => {
    load().catch((err) => setMessage(err.message));
  }, [id]);

  useEffect(() => {
    if (query.trim().length < 2) {
      setResults([]);
      return;
    }
    const handle = setTimeout(async () => {
      try {
        setResults(await fetchJson("/players", { q: query.trim(), limit: 10 }));
      } catch (err) {
        setResults([]);
      }
    }, 200);
    return () => clearTimeout(handle);
  }, [query]);

  const staticProspects = useMemo(() => {
    if (!player?.display_name) return [];
    const firstName = String(player.display_name).split(" ")[0];
    return STATIC_PLAYER_PROSPECTS[firstName] || [];
  }, [player?.display_name]);

  const selectedDoc = useMemo(() => {
    if (!selectedDocId) return null;
    return (player?.documents || []).find((doc) => String(doc.id) === String(selectedDocId)) || null;
  }, [player?.documents, selectedDocId]);

  const dataProfile = useMemo(() => {
    const fullMetrics = report?.metrics || {};
    const rawMetrics = report?.raw_metrics || {};
    const radarKeys = Array.isArray(report?.radar_metrics) && report.radar_metrics.length > 0
      ? report.radar_metrics
      : Object.keys(rawMetrics).slice(0, 8);
    const radarData = radarKeys
      .map((key) => {
        const league = fullMetrics[`${key}_pct_league`];
        const global = fullMetrics[`${key}_pct_global`];
        const pct = global ?? league;
        if (pct === null || pct === undefined) return null;
        return {
          key,
          metric: formatMetricLabel(key),
          value: Math.max(0, Math.min(100, Number(pct) || 0)),
          raw: rawMetrics[key] ?? fullMetrics[key],
        };
      })
      .filter(Boolean);
    const dominantMetrics = Object.keys(fullMetrics)
      .filter((key) => {
        if (key.endsWith("_pct_league") || key.endsWith("_pct_global")) return false;
        if (key.includes(" - ")) return false;
        if (!isRelevantDominantMetric(key, { position: report?.player?.position || player?.position, role: report?.player?.assigned_role })) return false;
        return !EXCLUDED_DOMINANT_PREFIXES.some((prefix) => key.startsWith(prefix));
      })
      .map((key) => {
        const global = fullMetrics[`${key}_pct_global`];
        const league = fullMetrics[`${key}_pct_league`];
        const pct = global ?? league;
        if (pct === null || pct === undefined) return null;
        return {
          key,
          label: formatMetricLabel(key),
          raw: fullMetrics[key],
          percentile: Number(pct) || 0,
        };
      })
      .filter(Boolean)
      .sort((a, b) => b.percentile - a.percentile)
      .slice(0, 5);
    return { radarData, dominantMetrics };
  }, [player?.position, report]);

  const updatePlayer = async (patch) => {
    try {
      const updated = await patchJson(`/hd-players/${id}`, patch);
      setPlayer(updated);
      if (patch.player_id) {
        setReport(await fetchJson(`/players/${patch.player_id}/report`));
      }
    } catch (err) {
      setMessage(err.message);
    }
  };

  const linkScoutingPlayer = async (result) => {
    setQuery("");
    setResults([]);
    await updatePlayer({
      player_id: result.id,
      display_name: player.display_name || result.name,
      current_club: player.current_club || result.team,
      position: player.position || result.position,
    });
  };

  const addDocument = async (override = {}) => {
    const payload = { ...docForm, ...override };
    if (!String(payload.title || "").trim()) return;
    try {
      const created = await postJson(`/hd-players/${id}/documents`, payload);
      setDocForm(emptyDoc);
      await load();
      setSelectedDocId(created?.id || null);
    } catch (err) {
      setMessage(err.message);
    }
  };

  const addDocumentFile = async (file) => {
    if (!file) return;
    try {
      const storageUrl = await readFileAsDataUrl(file);
      const documentName = docForm.file_name.trim() || file.name;
      await addDocument({
        title: docForm.title.trim() || documentName,
        file_name: documentName,
        file_key: "",
        storage_url: storageUrl,
        content_type: file.type || "application/octet-stream",
        size_bytes: file.size,
      });
    } catch (err) {
      setMessage(err.message);
    } finally {
      setIsDocDragging(false);
    }
  };

  const startDocEdit = (doc) => {
    setSelectedDocId(doc.id);
    setEditingDocId(doc.id);
    setEditingDoc({
      document_type: doc.document_type || "other",
      title: doc.title || "",
      file_name: doc.file_name || "",
      file_key: doc.file_key || "",
      storage_url: doc.storage_url || "",
      content_type: doc.content_type || "",
      size_bytes: doc.size_bytes || null,
      notes: doc.notes || "",
    });
  };

  const updateDocumentFile = async (documentId, file) => {
    if (!documentId || !file) return;
    try {
      const storageUrl = await readFileAsDataUrl(file);
      const documentName = editingDoc.file_name?.trim() || file.name;
      await patchJson(`/hd-players/documents/${documentId}`, {
        document_type: editingDoc.document_type || "other",
        title: editingDoc.title || documentName,
        file_name: documentName,
        file_key: "",
        storage_url: storageUrl,
        content_type: file.type || "application/octet-stream",
        size_bytes: file.size,
        notes: editingDoc.notes || "",
      });
      setEditingDocId(null);
      setEditingDoc(emptyDoc);
      await load();
      setSelectedDocId(documentId);
    } catch (err) {
      setMessage(err.message);
    }
  };

  const saveDocument = async () => {
    if (!editingDocId) return;
    try {
      await patchJson(`/hd-players/documents/${editingDocId}`, editingDoc);
      setEditingDocId(null);
      setEditingDoc(emptyDoc);
      await load();
      setSelectedDocId(editingDocId);
    } catch (err) {
      setMessage(err.message);
    }
  };

  const removeDocument = async (documentId) => {
    try {
      await deleteJson(`/hd-players/documents/${documentId}`);
      if (String(selectedDocId) === String(documentId)) {
        setSelectedDocId(null);
      }
      await load();
    } catch (err) {
      setMessage(err.message);
    }
  };

  if (!player) {
    return (
      <main className="nl-page px-4 py-8">
        <div className="mx-auto max-w-[1500px]">
          <div className="surface-panel rounded-lg p-6 text-sm font-semibold text-slate-600">
            {message ? (
              <div className="space-y-3">
                <p className="text-rose-700">Unable to load this player room.</p>
                <p className="text-xs text-slate-500">{message}</p>
                <Link href="/hd-players" className="inline-flex rounded-md bg-slate-950 px-4 py-2 text-xs font-black uppercase tracking-[0.2em] text-white">
                  Back to HD Players
                </Link>
              </div>
            ) : (
              "Loading player room..."
            )}
          </div>
        </div>
      </main>
    );
  }

  const reportPlayer = report?.player || {};
  const summary = report?.summary || {};
  const metrics = report?.raw_metrics || {};

  return (
    <main className="nl-page px-4 py-8">
      <div className="mx-auto max-w-[1500px] space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Link href="/hd-players" className="nl-button-secondary">Back to HD Players</Link>
          {player.player_id ? <Link href={`/report?player_id=${player.player_id}`} className="nl-button-primary">Open player report</Link> : null}
        </div>

        <section className="surface-panel overflow-hidden rounded-lg">
          <div className="grid lg:grid-cols-[420px_minmax(0,1fr)]">
            <div className="relative min-h-[420px] bg-[radial-gradient(circle_at_top_left,rgba(20,184,166,0.18),transparent_34%),linear-gradient(135deg,#f8fafc,#e2e8f0)]">
              {player.photo_url ? (
                <img src={player.photo_url} alt="" className="h-full min-h-[420px] w-full object-cover" />
              ) : (
                <div className="flex h-full min-h-[420px] items-center justify-center text-7xl font-black text-slate-300">
                  {initials(player.display_name)}
                </div>
              )}
            </div>
            <div className="space-y-5 p-6">
              <div>
                <p className="nl-kicker">HD Sports player room</p>
                <h1 className="mt-2 text-4xl font-extrabold text-slate-950">{player.display_name}</h1>
                <div className="mt-3 flex items-center gap-2 text-sm font-semibold text-slate-500">
                  <ClubLogo name={player.current_club} className="h-8 w-8" />
                  <span className="min-w-0 truncate">
                    {value(player.current_club)} {player.position ? `• ${player.position}` : ""}
                  </span>
                </div>
                {player.eyeball_url || player.transfermarkt_url ? (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {player.eyeball_url ? (
                      <a
                        href={externalHref(player.eyeball_url)}
                        target="_blank"
                        rel="noreferrer"
                        className="rounded-full border border-sky-200 bg-sky-50 px-3 py-1.5 text-xs font-black uppercase tracking-[0.08em] text-sky-800 transition hover:border-sky-300 hover:bg-sky-100"
                      >
                        Eyeball
                      </a>
                    ) : null}
                    {player.transfermarkt_url ? (
                      <a
                        href={externalHref(player.transfermarkt_url)}
                        target="_blank"
                        rel="noreferrer"
                        className="rounded-full border border-teal-200 bg-teal-50 px-3 py-1.5 text-xs font-black uppercase tracking-[0.08em] text-teal-800 transition hover:border-teal-300 hover:bg-teal-100"
                      >
                        Transfermarkt
                      </a>
                    ) : null}
                  </div>
                ) : null}
              </div>

              <div className="grid gap-3 md:grid-cols-4">
                <input className="nl-field md:col-span-2" name="photo_url" aria-label="Photo URL" value={player.photo_url || ""} onChange={(e) => setPlayer((p) => ({ ...p, photo_url: e.target.value }))} onBlur={(e) => updatePlayer({ photo_url: e.target.value })} placeholder="Photo URL" />
                <select className="nl-field" name="priority" aria-label="Priority" value={player.priority || "B"} onChange={(e) => updatePlayer({ priority: e.target.value })}>
                  {PRIORITIES.map((priority) => <option key={priority}>{priority}</option>)}
                </select>
                <select className="nl-field" name="assigned_agent" aria-label="Assigned agent" value={player.assigned_agent || ""} onChange={(e) => updatePlayer({ assigned_agent: e.target.value })}>
                  <option value="">Unassigned</option>
                  {AGENTS.map((agent) => <option key={agent}>{agent}</option>)}
                </select>
                <input className="nl-field" name="current_club" aria-label="Current club" value={player.current_club || ""} onChange={(e) => setPlayer((p) => ({ ...p, current_club: e.target.value }))} onBlur={(e) => updatePlayer({ current_club: e.target.value })} placeholder="Current club" />
                <input className="nl-field" name="position" aria-label="Position" value={player.position || ""} onChange={(e) => setPlayer((p) => ({ ...p, position: e.target.value }))} onBlur={(e) => updatePlayer({ position: e.target.value })} placeholder="Position" />
                <input className="nl-field" name="demanded_transfer_fee" aria-label="Demanded transfer fee" value={player.demanded_transfer_fee || ""} onChange={(e) => setPlayer((p) => ({ ...p, demanded_transfer_fee: e.target.value }))} onBlur={(e) => updatePlayer({ demanded_transfer_fee: e.target.value ? Number(e.target.value) : null })} placeholder="Demanded transfer fee" />
                <input className="nl-field" name="contract_expiry" aria-label="Contract expiry" type="date" value={player.contract_expiry || ""} onChange={(e) => updatePlayer({ contract_expiry: e.target.value })} />
                <input className="nl-field md:col-span-2" name="plan" aria-label="Market plan" value={player.plan || ""} onChange={(e) => setPlayer((p) => ({ ...p, plan: e.target.value }))} onBlur={(e) => updatePlayer({ plan: e.target.value })} placeholder="Market plan" />
                <input className="nl-field md:col-span-2" name="next_step" aria-label="Next step" value={player.next_step || ""} onChange={(e) => setPlayer((p) => ({ ...p, next_step: e.target.value }))} onBlur={(e) => updatePlayer({ next_step: e.target.value })} placeholder="Next step" />
                <input className="nl-field md:col-span-2" name="eyeball_url" aria-label="Eyeball link" value={player.eyeball_url || ""} onChange={(e) => setPlayer((p) => ({ ...p, eyeball_url: e.target.value }))} onBlur={(e) => updatePlayer({ eyeball_url: e.target.value })} placeholder="Eyeball link" />
                <input className="nl-field md:col-span-2" name="transfermarkt_url" aria-label="Transfermarkt link" value={player.transfermarkt_url || ""} onChange={(e) => setPlayer((p) => ({ ...p, transfermarkt_url: e.target.value }))} onBlur={(e) => updatePlayer({ transfermarkt_url: e.target.value })} placeholder="Transfermarkt link" />
              </div>
            </div>
          </div>
        </section>

        <section className="space-y-5">
            <div className="surface-panel relative z-40 overflow-visible rounded-lg p-5">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="nl-kicker">Performance data link</p>
                  <h2 className="mt-2 text-2xl font-extrabold text-slate-950">{player.linked_player_name || "No linked player yet"}</h2>
                  <p className="mt-1 text-sm text-slate-600">Connect this room to the right Scouting Lab profile to unlock season data, reports and market evidence.</p>
                </div>
                {player.player_id ? <span className="rounded-full border border-teal-200 bg-teal-50 px-3 py-1 text-xs font-black text-teal-800">Linked</span> : null}
              </div>
              <div className="relative z-50 mt-4">
                <input className="nl-field" name="scouting_search" aria-label="Search Scouting Lab player" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Type a player name, e.g. J. Diaz" />
                {results.length > 0 ? (
                  <div className="absolute z-[9999] mt-2 max-h-80 w-full overflow-auto rounded-md border border-slate-200 bg-white shadow-2xl">
                    {results.map((result) => (
                      <button
                        key={`${result.id}-${result.player_season_id || ""}`}
                        type="button"
                        className="grid w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-3 border-b border-slate-100 px-4 py-3 text-left hover:bg-teal-50"
                        onMouseDown={(event) => event.preventDefault()}
                        onClick={() => linkScoutingPlayer(result)}
                      >
                        <span className="flex min-w-0 items-center gap-3">
                          <ClubLogo name={result.team} className="h-8 w-8" />
                          <span className="min-w-0">
                          <span className="block font-extrabold text-slate-950">{result.name}</span>
                          <span className="text-xs font-semibold text-slate-500">{result.team || "-"} • {result.competition_name || "-"} • {result.calendar || "-"}</span>
                          </span>
                        </span>
                        <span className="rounded bg-teal-700 px-3 py-1 text-xs font-bold text-white">Link</span>
                      </button>
                    ))}
                  </div>
                ) : null}
              </div>
            </div>

            <div className="surface-panel rounded-lg p-5">
              <p className="nl-kicker">Season data summary</p>
              <h2 className="mt-2 text-2xl font-extrabold text-slate-950">{report ? `${reportPlayer.name} • ${reportPlayer.calendar}` : "Link a Scouting Lab player to unlock data"}</h2>
              <div className="mt-4 grid gap-3 md:grid-cols-4">
                {[
                  ["Team", reportPlayer.team],
                  ["Competition", reportPlayer.competition_name],
                  ["Position group", englishRole(reportPlayer.assigned_role)],
                  ["Position", reportPlayer.position],
                  ["Minutes", reportPlayer.minutes_played],
                  ["Global score", reportPlayer.global_score_adjusted != null ? Number(reportPlayer.global_score_adjusted).toFixed(1) : null],
                  ["League percentile", reportPlayer.assigned_role_pct_league != null ? Math.round(Number(reportPlayer.assigned_role_pct_league)) : null],
                  ["Age", reportPlayer.age],
                ].map(([label, item]) => (
                  <div key={label} className="rounded-md border border-slate-200 bg-slate-50 p-3">
                    <p className="text-[11px] font-black uppercase tracking-[0.12em] text-slate-500">{label}</p>
                    {label === "Team" ? (
                      <div className="mt-2 flex items-center gap-2">
                        <ClubLogo name={item} className="h-8 w-8" />
                        <p className="min-w-0 truncate text-lg font-extrabold text-slate-950">{value(item)}</p>
                      </div>
                    ) : (
                      <p className="mt-1 text-lg font-extrabold text-slate-950">{value(item)}</p>
                    )}
                  </div>
                ))}
              </div>
              {report ? (
                <div className="mt-5 grid gap-5 xl:grid-cols-[minmax(0,1fr)_440px]">
                  <div className="rounded-lg border border-slate-200 bg-white p-4">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div>
                        <p className="text-[11px] font-black uppercase tracking-[0.12em] text-slate-500">Performance radar</p>
                        <h3 className="mt-1 text-xl font-extrabold text-slate-950">Position group percentile shape</h3>
                      </div>
                      <span className="rounded-full border border-teal-200 bg-teal-50 px-3 py-1 text-xs font-black text-teal-800">
                        Global percentile
                      </span>
                    </div>
                    <div className="mt-4 h-[360px]">
                      {dataProfile.radarData.length > 0 ? (
                        <ResponsiveContainer width="100%" height="100%">
                          <RadarChart data={dataProfile.radarData} outerRadius="78%">
                            <PolarGrid stroke="#cbd5e1" />
                            <PolarAngleAxis dataKey="metric" tick={{ fill: "#334155", fontSize: 11, fontWeight: 700 }} />
                            <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: "#64748b", fontSize: 10 }} />
                            <Tooltip content={<RadarTooltip />} />
                            <Radar dataKey="value" name="Percentile" stroke="#0f766e" fill="#14b8a6" fillOpacity={0.28} strokeWidth={2} />
                          </RadarChart>
                        </ResponsiveContainer>
                      ) : (
                        <div className="flex h-full items-center justify-center rounded-md border border-dashed border-slate-300 text-sm font-semibold text-slate-500">
                          No radar data available.
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                    <p className="text-[11px] font-black uppercase tracking-[0.12em] text-slate-500">Dominant metrics</p>
                    <h3 className="mt-1 text-xl font-extrabold text-slate-950">Top 5 strengths</h3>
                    <div className="mt-4 space-y-3">
                      {dataProfile.dominantMetrics.length > 0 ? dataProfile.dominantMetrics.map((item) => (
                        <div key={item.key} className="rounded-md border border-slate-200 bg-white p-3">
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <p className="font-extrabold text-slate-950">{item.label}</p>
                              <p className="mt-1 text-xs font-semibold text-slate-500">Raw value: {item.raw != null ? Number(item.raw).toFixed(2) : "-"}</p>
                            </div>
                            <span className="rounded-full border border-teal-200 bg-teal-50 px-3 py-1 text-sm font-black text-teal-800">
                              {Math.round(item.percentile)}
                            </span>
                          </div>
                          <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-200">
                            <div className="h-full rounded-full bg-teal-600" style={{ width: `${Math.max(0, Math.min(100, item.percentile))}%` }} />
                          </div>
                        </div>
                      )) : (
                        <p className="rounded-md border border-dashed border-slate-300 bg-white p-4 text-sm font-semibold text-slate-500">
                          No dominant metric available.
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ) : null}
            </div>

            <div className="surface-panel rounded-lg p-5">
              <p className="nl-kicker">Prospect clubs</p>
              <h2 className="mt-2 text-2xl font-extrabold text-slate-950">Mercato 2026 pipeline</h2>
              <div className="mt-4 overflow-x-auto">
                <table className="min-w-[820px] w-full text-left text-sm">
                  <thead className="bg-slate-50 text-xs uppercase tracking-[0.12em] text-slate-500">
                    <tr>
                      <th className="px-3 py-2">Club</th>
                      <th className="px-3 py-2">Status</th>
                      <th className="px-3 py-2">Offer</th>
                      <th className="px-3 py-2">Contact</th>
                      <th className="px-3 py-2">Notes</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200">
                    {staticProspects.map((row, index) => (
                      <tr key={`static-${index}`} className="bg-white">
                        {row.map((cell, cellIndex) => (
                          <td key={cellIndex} className="px-3 py-2">
                            {cellIndex === 0 ? (
                              <span className="flex items-center gap-2 font-semibold text-slate-950">
                                <ClubLogo name={cell} className="h-7 w-7" />
                                <span>{cell || "-"}</span>
                              </span>
                            ) : (
                              cell || "-"
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                    {(player.mercato_prospects || []).map((row) => (
                      <tr key={`match-${row.request_id}-${row.club_name}`} className="bg-white">
                        <td className="px-3 py-2 font-semibold text-slate-950">
                          <span className="flex items-center gap-2">
                            <ClubLogo name={row.club_name} className="h-7 w-7" />
                            <span>{row.club_name || "-"}</span>
                          </span>
                        </td>
                        <td className="px-3 py-2">{row.candidate_status || row.request_status || "-"}</td>
                        <td className="px-3 py-2">{row.match_score != null ? `${Math.round(row.match_score)} match` : "-"}</td>
                        <td className="px-3 py-2">{row.assigned_agent_name || row.assigned_agent_id || "-"}</td>
                        <td className="px-3 py-2">{row.agent_note || row.title || "-"}</td>
                      </tr>
                    ))}
                    {staticProspects.length === 0 && (player.mercato_prospects || []).length === 0 ? (
                      <tr><td className="px-3 py-4 text-slate-500" colSpan={5}>No prospect club has been attached to this player yet.</td></tr>
                    ) : null}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="surface-panel rounded-lg p-5">
              <p className="nl-kicker">Internal notes</p>
              <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <p className="text-[11px] font-black uppercase tracking-[0.12em] text-slate-500">Contacts</p>
                      <h3 className="mt-1 text-lg font-extrabold text-slate-950">Player & entourage</h3>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {player.player_phone ? <a className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-black text-slate-700 hover:border-teal-300" href={contactHref("phone", player.player_phone)}>Call player</a> : null}
                      {player.player_email ? <a className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-black text-slate-700 hover:border-teal-300" href={contactHref("email", player.player_email)}>Mail player</a> : null}
                      {player.entourage_phone ? <a className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-black text-slate-700 hover:border-teal-300" href={contactHref("phone", player.entourage_phone)}>Call entourage</a> : null}
                      {player.entourage_email ? <a className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-black text-slate-700 hover:border-teal-300" href={contactHref("email", player.entourage_email)}>Mail entourage</a> : null}
                    </div>
                  </div>
                  <div className="mt-4 grid gap-3 md:grid-cols-2">
                    <input className="nl-field" name="player_phone" aria-label="Player phone" value={player.player_phone || ""} onChange={(e) => setPlayer((p) => ({ ...p, player_phone: e.target.value }))} onBlur={(e) => updatePlayer({ player_phone: e.target.value })} placeholder="Player phone" />
                    <input className="nl-field" name="player_email" aria-label="Player email" type="email" value={player.player_email || ""} onChange={(e) => setPlayer((p) => ({ ...p, player_email: e.target.value }))} onBlur={(e) => updatePlayer({ player_email: e.target.value })} placeholder="Player email" />
                    <input className="nl-field" name="entourage_phone" aria-label="Entourage phone" value={player.entourage_phone || ""} onChange={(e) => setPlayer((p) => ({ ...p, entourage_phone: e.target.value }))} onBlur={(e) => updatePlayer({ entourage_phone: e.target.value })} placeholder="Entourage phone" />
                    <input className="nl-field" name="entourage_email" aria-label="Entourage email" type="email" value={player.entourage_email || ""} onChange={(e) => setPlayer((p) => ({ ...p, entourage_email: e.target.value }))} onBlur={(e) => updatePlayer({ entourage_email: e.target.value })} placeholder="Entourage email" />
                  </div>
                </div>
                <div className="rounded-lg border border-emerald-200 bg-emerald-50/70 p-4">
                  <p className="text-[11px] font-black uppercase tracking-[0.12em] text-emerald-700">Season objectives</p>
                  <h3 className="mt-1 text-lg font-extrabold text-slate-950">Player objectives</h3>
                  <textarea
                    className="nl-field mt-4 min-h-[154px] bg-white"
                    name="season_objectives"
                    aria-label="Season objectives"
                    value={player.season_objectives || ""}
                    onChange={(e) => setPlayer((p) => ({ ...p, season_objectives: e.target.value }))}
                    onBlur={(e) => updatePlayer({ season_objectives: e.target.value })}
                    placeholder="Season targets, role development, minutes target, market objectives..."
                  />
                </div>
              </div>
              <div className="mt-4 space-y-3">
                {["contract_status", "mandate_status", "medical_status", "current_club_situation"].map((field) => (
                  <input key={field} className="nl-field" name={field} aria-label={field.replaceAll("_", " ")} value={player[field] || ""} onChange={(e) => setPlayer((p) => ({ ...p, [field]: e.target.value }))} onBlur={(e) => updatePlayer({ [field]: e.target.value })} placeholder={field.replaceAll("_", " ")} />
                ))}
                <textarea className="nl-field min-h-[110px]" name="market_notes" aria-label="Market notes" value={player.market_notes || ""} onChange={(e) => setPlayer((p) => ({ ...p, market_notes: e.target.value }))} onBlur={(e) => updatePlayer({ market_notes: e.target.value })} placeholder="Market notes" />
                <textarea className="nl-field min-h-[110px]" name="scouting_notes" aria-label="Scouting notes" value={player.scouting_notes || ""} onChange={(e) => setPlayer((p) => ({ ...p, scouting_notes: e.target.value }))} onBlur={(e) => updatePlayer({ scouting_notes: e.target.value })} placeholder="Scouting notes" />
              </div>
            </div>

            <div className="surface-panel rounded-lg p-5">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="nl-kicker">Documents</p>
                  <h2 className="mt-2 text-2xl font-extrabold text-slate-950">Player files</h2>
                  <p className="mt-1 text-sm text-slate-600">Choose the type, add a title, name the document, then drop the file.</p>
                </div>
                <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-black text-slate-700">
                  {(player.documents || []).length} files
                </span>
              </div>
              <div className="mt-4 grid gap-3 lg:grid-cols-[190px_minmax(0,1fr)_minmax(0,1fr)]">
                <label className="block">
                  <span className="mb-1 block text-[11px] font-black uppercase tracking-[0.12em] text-slate-500">Document type</span>
                  <select className="nl-field" name="document_type" aria-label="Document type" value={docForm.document_type} onChange={(e) => setDocForm((p) => ({ ...p, document_type: e.target.value }))}>
                    {DOC_TYPES.map((type) => <option key={type.value} value={type.value}>{type.label}</option>)}
                  </select>
                </label>
                <label className="block">
                  <span className="mb-1 block text-[11px] font-black uppercase tracking-[0.12em] text-slate-500">Title</span>
                  <input className="nl-field" name="document_title" aria-label="Document title" value={docForm.title} onChange={(e) => setDocForm((p) => ({ ...p, title: e.target.value }))} placeholder="Medical check, signed mandate..." />
                </label>
                <label className="block">
                  <span className="mb-1 block text-[11px] font-black uppercase tracking-[0.12em] text-slate-500">Document name</span>
                  <input className="nl-field" name="document_name" aria-label="Document name" value={docForm.file_name} onChange={(e) => setDocForm((p) => ({ ...p, file_name: e.target.value }))} placeholder="Auto-filled from uploaded file" />
                </label>
              </div>
              <label
                className={`mt-4 flex min-h-[150px] cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-6 text-center transition ${
                  isDocDragging ? "border-teal-500 bg-teal-50" : "border-slate-300 bg-slate-50 hover:border-teal-400 hover:bg-teal-50/60"
                }`}
                onDragOver={(event) => {
                  event.preventDefault();
                  setIsDocDragging(true);
                }}
                onDragLeave={() => setIsDocDragging(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  addDocumentFile(event.dataTransfer.files?.[0]);
                }}
              >
                <span className="text-lg font-extrabold text-slate-950">Drop the file here</span>
                <span className="mt-1 text-sm font-semibold text-slate-500">
                  {docForm.file_name ? `${docForm.file_name} will be attached immediately.` : "or click to select it. The document name is auto-filled if empty."}
                </span>
                <input
                  type="file"
                  className="sr-only"
                  aria-label="Upload player document"
                  onChange={(event) => addDocumentFile(event.target.files?.[0])}
                />
              </label>

              <div className="mt-5 grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-6">
                {(player.documents || []).map((doc) => (
                  <button
                    key={doc.id}
                    type="button"
                    className={`group rounded-lg border bg-white p-2 text-left transition hover:-translate-y-0.5 hover:border-teal-400 hover:shadow-sm ${
                      String(selectedDocId) === String(doc.id) ? "border-teal-500 ring-2 ring-teal-100" : "border-slate-200"
                    }`}
                    onClick={() => {
                      setSelectedDocId(doc.id);
                      setEditingDocId(null);
                    }}
                  >
                    <div className="aspect-[4/3] overflow-hidden rounded-md border border-slate-200 bg-slate-50">
                      <DocumentTilePreview doc={doc} />
                    </div>
                    <p className="mt-2 truncate text-sm font-extrabold text-slate-950">{doc.title}</p>
                    <p className="truncate text-xs font-semibold text-slate-500">{doc.file_name || "No document name"}</p>
                    <p className="mt-1 text-[11px] font-black text-teal-700">{docTypeLabel(doc.document_type)}</p>
                  </button>
                ))}
                {(player.documents || []).length === 0 ? (
                  <p className="col-span-full rounded-md border border-dashed border-slate-300 bg-slate-50 p-4 text-sm font-semibold text-slate-500">
                    No document has been added yet.
                  </p>
                ) : null}
              </div>
            </div>

            <div className="surface-panel rounded-lg p-5">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="nl-kicker">Transfer history</p>
                  <h2 className="mt-2 text-2xl font-extrabold text-slate-950">Career movement timeline</h2>
                  <p className="mt-1 text-sm text-slate-600">
                    Review the player’s club movements with verified source context for faster market conversations.
                  </p>
                </div>
                <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-black text-slate-700">
                  {(player.transfer_history || []).length} moves
                </span>
              </div>

              {(player.transfer_history || []).length ? (
                <div className="mt-5 space-y-3">
                  {(player.transfer_history || []).map((transfer) => (
                    <div
                      key={`${transfer.id}-${transfer.transfer_date || "date"}-${transfer.team_in_name || "in"}-${transfer.team_out_name || "out"}`}
                      className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm"
                    >
                      <div className="flex flex-wrap items-start justify-between gap-3">
                        <div className="min-w-0">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="rounded-full bg-slate-950 px-2.5 py-1 text-[11px] font-black uppercase tracking-[0.08em] text-white">
                              {formatTransferDate(transfer.transfer_date)}
                            </span>
                            <span className="rounded-full border border-teal-200 bg-teal-50 px-2.5 py-1 text-[11px] font-black uppercase tracking-[0.08em] text-teal-700">
                              {transfer.transfer_type || "Transfer"}
                            </span>
                            {transfer.match_type === "name_club" ? (
                              <span className="rounded-full border border-amber-200 bg-amber-50 px-2.5 py-1 text-[11px] font-black uppercase tracking-[0.08em] text-amber-700">
                                Name + club match
                              </span>
                            ) : null}
                          </div>
                          <div className="mt-4 grid gap-3 md:grid-cols-[minmax(0,1fr)_44px_minmax(0,1fr)] md:items-center">
                            <div className="flex min-w-0 items-center gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
                              <ClubLogo name={transfer.team_out_name} className="h-10 w-10" />
                              <div className="min-w-0">
                                <p className="text-[11px] font-black uppercase tracking-[0.12em] text-slate-500">From</p>
                                <p className="truncate text-sm font-extrabold text-slate-950">{transfer.team_out_name || "Free agent"}</p>
                              </div>
                            </div>
                            <div className="hidden h-10 items-center justify-center rounded-full border border-slate-200 bg-white text-lg font-black text-slate-400 md:flex">
                              →
                            </div>
                            <div className="flex min-w-0 items-center gap-3 rounded-lg border border-emerald-200 bg-emerald-50 p-3">
                              <ClubLogo name={transfer.team_in_name} className="h-10 w-10" />
                              <div className="min-w-0">
                                <p className="text-[11px] font-black uppercase tracking-[0.12em] text-emerald-700">To</p>
                                <p className="truncate text-sm font-extrabold text-slate-950">{transfer.team_in_name || "Free agent"}</p>
                              </div>
                            </div>
                          </div>
                        </div>
                        <div className="min-w-[150px] rounded-lg border border-slate-200 bg-slate-50 p-3 text-right">
                          <p className="text-[11px] font-black uppercase tracking-[0.12em] text-slate-500">Fee</p>
                          <p className="mt-1 text-lg font-extrabold text-slate-950">{transferFeeLabel(transfer.transfer_fee)}</p>
                          <p className="mt-1 text-xs font-semibold text-slate-500">{transfer.league_name || transfer.team_name_context || "League to confirm"}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="mt-5 rounded-lg border border-dashed border-slate-300 bg-slate-50 p-5">
                  <p className="text-sm font-extrabold text-slate-950">No verified transfer history yet.</p>
                  <p className="mt-1 text-sm font-semibold text-slate-500">
                    Connect the player to Scouting Lab and confirm the current club to enrich this room with Wyscout movement history.
                  </p>
                </div>
              )}
            </div>
        </section>

        {selectedDoc ? (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/50 p-4 backdrop-blur-sm" role="dialog" aria-modal="true">
            <div className="max-h-[92vh] w-full max-w-5xl overflow-auto rounded-lg border border-slate-200 bg-white shadow-2xl">
              <div className="sticky top-0 z-10 flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 bg-white px-5 py-4">
                <div>
                  <p className="text-[11px] font-black tracking-[0.08em] text-teal-700">{docTypeLabel(selectedDoc.document_type)}</p>
                  <h3 className="mt-1 text-xl font-extrabold text-slate-950">{selectedDoc.title}</h3>
                  <p className="text-xs font-semibold text-slate-500">
                    {selectedDoc.file_name || "No document name"} • {formatBytes(selectedDoc.size_bytes)}
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  {selectedDoc.storage_url ? (
                    <a href={selectedDoc.storage_url} download={selectedDoc.file_name || selectedDoc.title} className="nl-button-secondary">Download</a>
                  ) : null}
                  <button type="button" className="nl-button-secondary" onClick={() => startDocEdit(selectedDoc)}>Edit</button>
                  <button type="button" className="rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-sm font-extrabold text-rose-700 hover:border-rose-300" onClick={() => removeDocument(selectedDoc.id)}>Delete</button>
                  <button type="button" className="nl-button-primary" onClick={() => {
                    setSelectedDocId(null);
                    setEditingDocId(null);
                  }}>Close</button>
                </div>
              </div>
              <div className="grid gap-5 p-5 lg:grid-cols-[minmax(0,1fr)_360px]">
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <DocumentPreview doc={selectedDoc} />
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                  {editingDocId === selectedDoc.id ? (
                    <div className="space-y-3">
                      <label className="block">
                        <span className="mb-1 block text-[11px] font-black uppercase tracking-[0.12em] text-slate-500">Document type</span>
                        <select className="nl-field" value={editingDoc.document_type} onChange={(e) => setEditingDoc((p) => ({ ...p, document_type: e.target.value }))}>
                          {DOC_TYPES.map((type) => <option key={type.value} value={type.value}>{type.label}</option>)}
                        </select>
                      </label>
                      <label className="block">
                        <span className="mb-1 block text-[11px] font-black uppercase tracking-[0.12em] text-slate-500">Title</span>
                        <input className="nl-field" aria-label="Document title" value={editingDoc.title} onChange={(e) => setEditingDoc((p) => ({ ...p, title: e.target.value }))} />
                      </label>
                      <label className="block">
                        <span className="mb-1 block text-[11px] font-black uppercase tracking-[0.12em] text-slate-500">Document name</span>
                        <input className="nl-field" aria-label="Document name" value={editingDoc.file_name} onChange={(e) => setEditingDoc((p) => ({ ...p, file_name: e.target.value }))} />
                      </label>
                      <label className="flex cursor-pointer items-center justify-center rounded-md border border-dashed border-slate-300 bg-white px-3 py-3 text-sm font-bold text-slate-600 hover:border-teal-400 hover:text-teal-700">
                        Replace file
                        <input type="file" className="sr-only" onChange={(event) => updateDocumentFile(selectedDoc.id, event.target.files?.[0])} />
                      </label>
                      <div className="flex gap-2">
                        <button type="button" className="nl-button-primary" onClick={saveDocument}>Save</button>
                        <button type="button" className="nl-button-secondary" onClick={() => setEditingDocId(null)}>Cancel</button>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-3 text-sm">
                      <div>
                        <p className="text-[11px] font-black uppercase tracking-[0.12em] text-slate-500">Document name</p>
                        <p className="mt-1 font-extrabold text-slate-950">{selectedDoc.file_name || "No document name"}</p>
                      </div>
                      <div>
                        <p className="text-[11px] font-black uppercase tracking-[0.12em] text-slate-500">Type</p>
                        <p className="mt-1 font-extrabold text-slate-950">{docTypeLabel(selectedDoc.document_type)}</p>
                      </div>
                      <div>
                        <p className="text-[11px] font-black uppercase tracking-[0.12em] text-slate-500">Size</p>
                        <p className="mt-1 font-extrabold text-slate-950">{formatBytes(selectedDoc.size_bytes)}</p>
                      </div>
                      <p className="rounded-md border border-slate-200 bg-white p-3 text-slate-600">
                        Click Edit to update metadata or replace the file.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        ) : null}

        {message ? <p className="text-sm font-semibold text-rose-700">{message}</p> : null}
      </div>
    </main>
  );
}
