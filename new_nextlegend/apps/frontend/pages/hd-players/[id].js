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
import { deleteJson, fetchJson, patchJson, postForm, postJson } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { METRIC_LABELS } from "@/lib/metricLabels";
import { englishRole, normalizeRoleForUse } from "@/lib/roles";

const AGENTS = ["Steven", "Don", "Yannis", "Lidahi"];
const POSITION_OPTIONS = [
  "Goalkeeper",
  "Centre Back",
  "Left Centre Back",
  "Right Centre Back",
  "Left Back",
  "Right Back",
  "Left Wing Back",
  "Right Wing Back",
  "Defensive Midfielder",
  "Central Midfielder",
  "Attacking Midfielder",
  "Left Winger",
  "Right Winger",
  "Striker",
];
const DOC_TYPES = [
  { value: "all", label: "All" },
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

const emptyProspectClub = {
  club_id: "",
  club_name: "",
  competition_name: "",
  status: "Watching",
  offer: "",
  contact: "",
  notes: "",
};

const emptyManualPerformance = {
  calendar: "",
  team: "",
  competition: "",
  position: "",
  minutes_played: "",
  matches_played: "",
  goals: "",
  assists: "",
  notes: "",
};

const emptyManualTransfer = {
  transfer_date: "",
  transfer_type: "Transfer",
  team_out_name: "",
  team_in_name: "",
  transfer_fee: "",
  league_name: "",
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

const formatLargeNumber = (value) => {
  if (value === null || value === undefined || value === "") return "";
  const clean = String(value).replace(/[^\d.-]/g, "");
  const numeric = Number(clean);
  if (!Number.isFinite(numeric)) return String(value);
  return new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 }).format(numeric).replace(/\u202f/g, " ");
};

const parseLargeNumber = (value) => {
  const clean = String(value || "").replace(/[^\d.-]/g, "");
  if (!clean) return null;
  const numeric = Number(clean);
  return Number.isFinite(numeric) ? numeric : null;
};

const initials = (name) =>
  String(name || "")
    .split(" ")
    .map((part) => part[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

const value = (input) => (input === null || input === undefined || input === "" ? "-" : input);

const displayValue = (input) => {
  const formatted = value(input);
  if (formatted === "-") return formatted;
  return String(formatted)
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
};

const externalHref = (url) => {
  const clean = String(url || "").trim();
  if (!clean) return "";
  if (/^https?:\/\//i.test(clean)) return clean;
  return `https://${clean}`;
};

const storageHref = (url) => {
  const clean = String(url || "").trim();
  if (!clean) return "";
  try {
    const parsed = new URL(clean);
    if (parsed.hostname === "api" && parsed.port === "8000") {
      if (typeof window !== "undefined") {
        const host = window.location.hostname === "0.0.0.0" ? "localhost" : window.location.hostname;
        return `${window.location.protocol}//${host}:8000${parsed.pathname}${parsed.search}`;
      }
      return `http://localhost:8000${parsed.pathname}${parsed.search}`;
    }
  } catch {
    return clean;
  }
  return clean;
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

const canInlinePreview = (doc) =>
  String(doc?.content_type || "").startsWith("image/") ||
  String(doc?.content_type || "").includes("pdf") ||
  String(doc?.storage_url || "").startsWith("data:image/") ||
  String(doc?.storage_url || "").startsWith("data:application/pdf");

const DocumentPreview = ({ doc }) => {
  const href = storageHref(doc?.storage_url);
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
    return <img src={href} alt="" className="max-h-80 w-full rounded-md border border-slate-200 object-contain bg-white" />;
  }
  return (
    <iframe
      title={`${doc.title || "Document"} preview`}
      src={href}
      className="h-80 w-full rounded-md border border-slate-200 bg-white"
    />
  );
};

const DocumentTilePreview = ({ doc }) => {
  const isImage = String(doc?.content_type || "").startsWith("image/") || String(doc?.storage_url || "").startsWith("data:image/");
  const isPdf = String(doc?.content_type || "").includes("pdf") || String(doc?.storage_url || "").startsWith("data:application/pdf");
  if (isImage && doc?.storage_url) {
    return <img src={storageHref(doc.storage_url)} alt="" className="h-full w-full object-cover" />;
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

const FormRow = ({ label, children, action }) => (
  <div className="grid gap-2 rounded-md border border-white/10 bg-white/[0.025] p-3 md:grid-cols-[180px_minmax(0,1fr)] md:items-center">
    <span className="block text-[11px] font-bold uppercase tracking-[0.12em] text-slate-500">{label}</span>
    <span className="flex min-w-0 items-center gap-2">
      <span className="min-w-0 flex-1">{children}</span>
      {action}
    </span>
  </div>
);

const InfoPill = ({ label, children }) => (
  <div className="rounded-md border border-white/10 bg-white/[0.03] p-3">
    <p className="text-[11px] font-bold uppercase tracking-[0.12em] text-slate-500">{label}</p>
    <p className="mt-1 min-w-0 truncate text-sm font-semibold text-slate-950">{children}</p>
  </div>
);

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
  const { me } = useAuth();
  const [player, setPlayer] = useState(null);
  const [report, setReport] = useState(null);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [docForm, setDocForm] = useState(emptyDoc);
  const [editingDocId, setEditingDocId] = useState(null);
  const [editingDoc, setEditingDoc] = useState(emptyDoc);
  const [message, setMessage] = useState("");
  const [isDocDragging, setIsDocDragging] = useState(false);
  const [isPhotoDragging, setIsPhotoDragging] = useState(false);
  const [selectedDocId, setSelectedDocId] = useState(null);
  const [documentModalOpen, setDocumentModalOpen] = useState(false);
  const [documentFilter, setDocumentFilter] = useState("all");
  const [clubs, setClubs] = useState([]);
  const [currentClubOpen, setCurrentClubOpen] = useState(false);
  const [prospectModalOpen, setProspectModalOpen] = useState(false);
  const [prospectForm, setProspectForm] = useState(emptyProspectClub);
  const [transferModalOpen, setTransferModalOpen] = useState(false);
  const [transferForm, setTransferForm] = useState(emptyManualTransfer);
  const [archiveModalOpen, setArchiveModalOpen] = useState(false);
  const [archiveConfirmName, setArchiveConfirmName] = useState("");
  const [archiving, setArchiving] = useState(false);

  const canArchivePlayer = me?.role === "admin" && me?.username === "yrachid";

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
    fetchJson("/meta/clubs")
      .then((data) => setClubs(Array.isArray(data) ? data : []))
      .catch(() => setClubs([]));
  }, []);

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

  const filteredDocuments = useMemo(() => {
    const docs = player?.documents || [];
    if (documentFilter === "all") return docs;
    return docs.filter((doc) => doc.document_type === documentFilter);
  }, [documentFilter, player?.documents]);

  const clubMatches = useMemo(() => {
    const queryValue = String(prospectForm.club_name || "").trim().toLowerCase();
    if (!queryValue) return clubs.slice(0, 8);
    return clubs
      .filter((club) => `${club.name || ""} ${club.competition_name || ""} ${club.country || ""}`.toLowerCase().includes(queryValue))
      .slice(0, 8);
  }, [clubs, prospectForm.club_name]);

  const currentClubMatches = useMemo(() => {
    const queryValue = String(player?.current_club || "").trim().toLowerCase();
    if (!queryValue) return clubs.slice(0, 8);
    return clubs
      .filter((club) => `${club.name || ""} ${club.competition_name || ""} ${club.country || ""}`.toLowerCase().includes(queryValue))
      .slice(0, 8);
  }, [clubs, player?.current_club]);

  const manualPerformance = {
    ...emptyManualPerformance,
    ...(player?.manual_performance || {}),
  };

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
      is_young_player: false,
      display_name: player.display_name || result.name,
      current_club: player.current_club || result.team,
      position: player.position || result.position,
    });
  };

  const uploadPlayerFile = async (file, purpose) => {
    const formData = new FormData();
    formData.append("file", file);
    return postForm(`/hd-players/${id}/upload`, formData, { purpose });
  };

  const uploadPhotoFile = async (file) => {
    if (!file) return;
    try {
      const uploaded = await uploadPlayerFile(file, "photo");
      await updatePlayer({ photo_url: uploaded.storage_url });
    } catch (err) {
      setMessage(err.message);
    } finally {
      setIsPhotoDragging(false);
    }
  };

  const markYoungPlayer = async (checked) => {
    await updatePlayer({
      is_young_player: checked,
      player_id: checked ? null : player.player_id,
    });
    if (checked) setReport(null);
  };

  const updateManualPerformanceField = (field, fieldValue) => {
    const next = {
      ...emptyManualPerformance,
      ...(player.manual_performance || {}),
      [field]: fieldValue,
    };
    setPlayer((prev) => ({ ...prev, manual_performance: next }));
  };

  const saveManualPerformance = async () => {
    await updatePlayer({ manual_performance: player.manual_performance || {} });
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
      const uploaded = await uploadPlayerFile(file, "document");
      const documentName = uploaded.file_name || file.name;
      await addDocument({
        title: docForm.title.trim() || documentName,
        file_name: documentName,
        file_key: uploaded.file_key || "",
        storage_url: uploaded.storage_url,
        content_type: uploaded.content_type || file.type || "application/octet-stream",
        size_bytes: uploaded.size_bytes || file.size,
      });
      setDocumentModalOpen(false);
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
      const uploaded = await uploadPlayerFile(file, "document");
      const documentName = uploaded.file_name || editingDoc.file_name?.trim() || file.name;
      await patchJson(`/hd-players/documents/${documentId}`, {
        document_type: editingDoc.document_type || "other",
        title: editingDoc.title || documentName,
        file_name: documentName,
        file_key: uploaded.file_key || "",
        storage_url: uploaded.storage_url,
        content_type: uploaded.content_type || file.type || "application/octet-stream",
        size_bytes: uploaded.size_bytes || file.size,
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

  const selectProspectClub = (club) => {
    setProspectForm((prev) => ({
      ...prev,
      club_id: club.id ? String(club.id) : "",
      club_name: club.name || "",
      competition_name: club.competition_name || "",
    }));
  };

  const selectCurrentClub = async (club) => {
    setCurrentClubOpen(false);
    setPlayer((prev) => ({ ...prev, current_club: club.name || "" }));
    await updatePlayer({ current_club: club.name || "" });
  };

  const addProspectClub = async () => {
    try {
      await postJson(`/hd-players/${id}/prospect-clubs`, {
        ...prospectForm,
        club_id: prospectForm.club_id ? Number(prospectForm.club_id) : null,
      });
      setProspectForm(emptyProspectClub);
      setProspectModalOpen(false);
      await load();
    } catch (err) {
      setMessage(err.message);
    }
  };

  const removeProspectClub = async (prospectId) => {
    try {
      await deleteJson(`/hd-players/prospect-clubs/${prospectId}`);
      await load();
    } catch (err) {
      setMessage(err.message);
    }
  };

  const addManualTransfer = async () => {
    try {
      await postJson(`/hd-players/${id}/transfers`, transferForm);
      setTransferForm(emptyManualTransfer);
      setTransferModalOpen(false);
      await load();
    } catch (err) {
      setMessage(err.message);
    }
  };

  const removeManualTransfer = async (transferId) => {
    try {
      await deleteJson(`/hd-players/transfers/${transferId}`);
      await load();
    } catch (err) {
      setMessage(err.message);
    }
  };

  const archivePlayer = async () => {
    if (!player || archiveConfirmName.trim() !== player.display_name) return;
    setArchiving(true);
    try {
      await deleteJson(`/hd-players/${id}`);
      router.push("/hd-players");
    } catch (err) {
      setMessage(err.message);
      setArchiving(false);
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
  const isYoungPlayer = Boolean(player.is_young_player);
  const seasonSummaryItems = report
    ? [
        ["Team", reportPlayer.team],
        ["Competition", reportPlayer.competition_name],
        ["Position group", englishRole(reportPlayer.assigned_role)],
        ["Position", reportPlayer.position],
        ["Minutes", reportPlayer.minutes_played],
        ["Global score", reportPlayer.global_score_adjusted != null ? Number(reportPlayer.global_score_adjusted).toFixed(1) : null],
        ["League percentile", reportPlayer.assigned_role_pct_league != null ? Math.round(Number(reportPlayer.assigned_role_pct_league)) : null],
        ["Age", reportPlayer.age],
      ]
    : isYoungPlayer
      ? [
          ["Team", manualPerformance.team],
          ["Competition", manualPerformance.competition],
          ["Calendar", manualPerformance.calendar],
          ["Position", manualPerformance.position || player.position],
          ["Minutes", manualPerformance.minutes_played],
          ["Matches", manualPerformance.matches_played],
          ["Goals", manualPerformance.goals],
          ["Assists", manualPerformance.assists],
        ]
      : [];

  return (
    <main className="nl-page px-4 py-8">
      <div className="mx-auto max-w-[1500px] space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Link href="/hd-players" className="nl-button-secondary">Back to HD Players</Link>
          {player.player_id ? <Link href={`/report?player_id=${player.player_id}`} className="nl-button-primary">Open player report</Link> : null}
        </div>

        <section className="surface-panel overflow-hidden rounded-lg">
          <div className="grid lg:grid-cols-[340px_minmax(0,1fr)]">
            <div className="border-r border-white/10 bg-black/20 p-4">
              <div
                className={`relative aspect-[4/5] overflow-hidden rounded-lg bg-[radial-gradient(circle_at_top_left,rgba(58,137,103,0.24),transparent_36%),linear-gradient(135deg,#090B0A,#131714)] ${
                  isPhotoDragging ? "ring-2 ring-[#559A78]" : ""
                }`}
                onDragOver={(event) => {
                  event.preventDefault();
                  setIsPhotoDragging(true);
                }}
                onDragLeave={() => setIsPhotoDragging(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  uploadPhotoFile(event.dataTransfer.files?.[0]);
                }}
              >
                {player.photo_url ? (
                  <img src={storageHref(player.photo_url)} alt="" className="h-full w-full object-cover" />
                ) : (
                  <div className="flex h-full w-full items-center justify-center text-6xl font-black text-white/18">
                    {initials(player.display_name)}
                  </div>
                )}
              </div>
              <div className="mt-3 rounded-lg border border-white/10 bg-black/45 p-3">
                <p className="text-[11px] font-bold uppercase tracking-[0.14em] text-slate-500">PLAYER IMAGE</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  <label className="nl-button-primary cursor-pointer">
                    Import image
                    <input type="file" accept="image/*" className="sr-only" onChange={(event) => uploadPhotoFile(event.target.files?.[0])} />
                  </label>
                  {player.photo_url ? (
                    <a href={storageHref(player.photo_url)} target="_blank" rel="noreferrer" className="nl-button-secondary">
                      Open image
                    </a>
                  ) : null}
                </div>
                <p className="mt-2 text-xs font-semibold text-slate-500">Drop an image here or use a hosted URL in the profile fields.</p>
              </div>
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

              <div className="grid gap-3 xl:grid-cols-2">
                <FormRow label="Assigned agent" value={player.assigned_agent}>
                  <select className="nl-field" name="assigned_agent" aria-label="Assigned agent" value={player.assigned_agent || ""} onChange={(e) => updatePlayer({ assigned_agent: e.target.value })}>
                    <option value="">Unassigned</option>
                    {AGENTS.map((agent) => <option key={agent}>{agent}</option>)}
                  </select>
                </FormRow>
                <FormRow label="Birth date" value={player.birth_date}>
                  <input
                    className="nl-field"
                    name="birth_date"
                    aria-label="Birth date"
                    type="date"
                    value={player.birth_date || ""}
                    onChange={(e) => updatePlayer({ birth_date: e.target.value })}
                  />
                </FormRow>
                <FormRow label="Current club" value={player.current_club}>
                  <div className="relative">
                    <input
                      className="nl-field"
                      name="current_club"
                      aria-label="Current club"
                      value={player.current_club || ""}
                      onFocus={() => setCurrentClubOpen(true)}
                      onChange={(e) => {
                        setCurrentClubOpen(true);
                        setPlayer((p) => ({ ...p, current_club: e.target.value }));
                      }}
                      onBlur={(e) => {
                        setTimeout(() => setCurrentClubOpen(false), 160);
                        updatePlayer({ current_club: e.target.value });
                      }}
                      placeholder="Type a club name"
                    />
                    {currentClubOpen ? (
                      <div className="absolute z-[9999] mt-2 max-h-64 w-full overflow-auto rounded-md border border-white/10 bg-[#0D100E] shadow-2xl">
                        {currentClubMatches.length ? currentClubMatches.map((club) => (
                          <button
                            key={club.id}
                            type="button"
                            className="flex w-full items-center gap-3 border-b border-white/10 px-3 py-2.5 text-left text-sm text-slate-300 transition hover:bg-[#2F7D5C]/15 hover:text-white"
                            onMouseDown={(event) => event.preventDefault()}
                            onClick={() => selectCurrentClub(club)}
                          >
                            <ClubLogo name={club.name} className="h-7 w-7" />
                            <span className="min-w-0">
                              <span className="block truncate font-semibold">{club.name}</span>
                              <span className="block truncate text-xs text-slate-500">{club.competition_name || club.country || "-"}</span>
                            </span>
                          </button>
                        )) : (
                          <p className="px-3 py-3 text-sm text-slate-500">No matching club. The typed value will be saved.</p>
                        )}
                      </div>
                    ) : null}
                  </div>
                </FormRow>
                <FormRow label="Position" value={player.position}>
                  <select
                    className="nl-field"
                    name="position"
                    aria-label="Position"
                    value={player.position || ""}
                    onChange={(e) => updatePlayer({ position: e.target.value })}
                  >
                    <option value="">Select position</option>
                    {player.position && !POSITION_OPTIONS.includes(player.position) ? (
                      <option value={player.position}>{player.position}</option>
                    ) : null}
                    {POSITION_OPTIONS.map((position) => (
                      <option key={position} value={position}>{position}</option>
                    ))}
                  </select>
                </FormRow>
                <FormRow label="Demanded transfer fee" value={money(player.demanded_transfer_fee)}>
                  <input
                    className="nl-field"
                    name="demanded_transfer_fee"
                    aria-label="Demanded transfer fee"
                    inputMode="numeric"
                    value={formatLargeNumber(player.demanded_transfer_fee)}
                    onChange={(e) => setPlayer((p) => ({ ...p, demanded_transfer_fee: parseLargeNumber(e.target.value) ?? "" }))}
                    onBlur={(e) => updatePlayer({ demanded_transfer_fee: parseLargeNumber(e.target.value) })}
                    placeholder="8 000 000"
                  />
                </FormRow>
                <FormRow label="Contract expiry" value={player.contract_expiry}>
                  <input className="nl-field" name="contract_expiry" aria-label="Contract expiry" type="date" value={player.contract_expiry || ""} onChange={(e) => updatePlayer({ contract_expiry: e.target.value })} />
                </FormRow>
                <FormRow label="Market plan" value={player.plan}>
                  <input className="nl-field" name="plan" aria-label="Market plan" value={player.plan || ""} onChange={(e) => setPlayer((p) => ({ ...p, plan: e.target.value }))} onBlur={(e) => updatePlayer({ plan: e.target.value })} placeholder="Plan" />
                </FormRow>
                <FormRow label="Next step" value={player.next_step}>
                  <input className="nl-field" name="next_step" aria-label="Next step" value={player.next_step || ""} onChange={(e) => setPlayer((p) => ({ ...p, next_step: e.target.value }))} onBlur={(e) => updatePlayer({ next_step: e.target.value })} placeholder="Next step" />
                </FormRow>
                <FormRow label="Eyeball link" value={player.eyeball_url} action={player.eyeball_url ? <a href={externalHref(player.eyeball_url)} target="_blank" rel="noreferrer" className="nl-button-secondary shrink-0">Open</a> : null}>
                  <input className="nl-field" name="eyeball_url" aria-label="Eyeball link" value={player.eyeball_url || ""} onChange={(e) => setPlayer((p) => ({ ...p, eyeball_url: e.target.value }))} onBlur={(e) => updatePlayer({ eyeball_url: e.target.value })} placeholder="https://..." />
                </FormRow>
                <FormRow label="Transfermarkt link" value={player.transfermarkt_url} action={player.transfermarkt_url ? <a href={externalHref(player.transfermarkt_url)} target="_blank" rel="noreferrer" className="nl-button-secondary shrink-0">Open</a> : null}>
                  <input className="nl-field" name="transfermarkt_url" aria-label="Transfermarkt link" value={player.transfermarkt_url || ""} onChange={(e) => setPlayer((p) => ({ ...p, transfermarkt_url: e.target.value }))} onBlur={(e) => updatePlayer({ transfermarkt_url: e.target.value })} placeholder="https://..." />
                </FormRow>
              </div>
            </div>
          </div>
        </section>

        <section className="space-y-5">
            <div className="surface-panel relative z-40 overflow-visible rounded-lg p-5">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="nl-kicker">Performance data link</p>
                  <h2 className="mt-2 text-2xl font-extrabold text-slate-950">
                    {isYoungPlayer ? "Young Player profile" : player.linked_player_name || "No linked player yet"}
                  </h2>
                  <p className="mt-1 text-sm text-slate-600">
                    Link a professional player to Scouting Lab data, or activate Young Player to enter season performance manually.
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  {player.player_id ? <span className="rounded-full border border-teal-200 bg-teal-50 px-3 py-1 text-xs font-black text-teal-800">Linked</span> : null}
                  <button
                    type="button"
                    className={`rounded-md border px-3 py-2 text-xs font-bold uppercase tracking-[0.12em] transition ${
                      isYoungPlayer ? "border-[#3A8967]/40 bg-[#2F7D5C]/20 text-[#DDF3E8]" : "border-white/10 bg-white/[0.035] text-white/65 hover:border-white/20 hover:text-white"
                    }`}
                    onClick={() => markYoungPlayer(!isYoungPlayer)}
                  >
                    Young Player
                  </button>
                </div>
              </div>
              <div className="relative z-50 mt-4">
                <input
                  className="nl-field"
                  name="scouting_search"
                  aria-label="Search Scouting Lab player"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  disabled={isYoungPlayer}
                  placeholder={isYoungPlayer ? "Manual performance mode is active" : "Type a player name, e.g. J. Diaz"}
                />
                {results.length > 0 && !isYoungPlayer ? (
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
              <h2 className="mt-2 text-2xl font-extrabold text-slate-950">
                {report ? `${reportPlayer.name} • ${reportPlayer.calendar}` : isYoungPlayer ? "Manual season performance" : "Link a Scouting Lab player to unlock data"}
              </h2>
              {seasonSummaryItems.length ? (
              <div className="mt-4 grid gap-3 md:grid-cols-4">
                {seasonSummaryItems.map(([label, item]) => (
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
              ) : null}
              {isYoungPlayer ? (
                <div className="mt-5 rounded-lg border border-white/10 bg-white/[0.025] p-4">
                  <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                    {[
                      ["calendar", "Calendar", "2026/2027"],
                      ["team", "Team", "U19 / Reserve team"],
                      ["competition", "Competition", "Youth league"],
                      ["position", "Position", "Striker"],
                      ["minutes_played", "Minutes played", "900"],
                      ["matches_played", "Matches played", "12"],
                      ["goals", "Goals", "7"],
                      ["assists", "Assists", "3"],
                    ].map(([field, label, placeholder]) => (
                      <FormRow key={field} label={label} value={manualPerformance[field]}>
                        <input
                          className="nl-field"
                          value={manualPerformance[field] || ""}
                          onChange={(event) => updateManualPerformanceField(field, event.target.value)}
                          onBlur={saveManualPerformance}
                          placeholder={placeholder}
                        />
                      </FormRow>
                    ))}
                  </div>
                  <FormRow label="Performance notes" value={manualPerformance.notes}>
                    <textarea
                      className="nl-field min-h-[96px]"
                      value={manualPerformance.notes || ""}
                      onChange={(event) => updateManualPerformanceField("notes", event.target.value)}
                      onBlur={saveManualPerformance}
                      placeholder="Context, level, minutes target, coach feedback..."
                    />
                  </FormRow>
                </div>
              ) : null}
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
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="nl-kicker">Prospect clubs</p>
                  <h2 className="mt-2 text-2xl font-extrabold text-slate-950">Prospect club pipeline</h2>
                </div>
                <button type="button" className="nl-button-primary" onClick={() => setProspectModalOpen(true)}>
                  + Add prospect
                </button>
              </div>
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
                    {(player.manual_prospect_clubs || []).map((row) => (
                      <tr key={`manual-${row.id}`} className="bg-white">
                        <td className="px-3 py-2 font-semibold text-slate-950">
                          <span className="flex items-center gap-2">
                            <ClubLogo name={row.club_name} className="h-7 w-7" />
                            <span>
                              <span className="block">{displayValue(row.club_name)}</span>
                              {row.competition_name ? <span className="block text-xs text-slate-500">{displayValue(row.competition_name)}</span> : null}
                            </span>
                          </span>
                        </td>
                        <td className="px-3 py-2">{displayValue(row.status)}</td>
                        <td className="px-3 py-2">{value(row.offer)}</td>
                        <td className="px-3 py-2">{value(row.contact)}</td>
                        <td className="px-3 py-2">
                          <div className="flex items-center justify-between gap-3">
                            <span>{value(row.notes)}</span>
                            <button type="button" className="text-xs font-bold text-rose-300 hover:text-rose-200" onClick={() => removeProspectClub(row.id)}>
                              Remove
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                    {staticProspects.map((row, index) => (
                      <tr key={`static-${index}`} className="bg-white">
                        {row.map((cell, cellIndex) => (
                          <td key={cellIndex} className="px-3 py-2">
                            {cellIndex === 0 ? (
                              <span className="flex items-center gap-2 font-semibold text-slate-950">
                                <ClubLogo name={cell} className="h-7 w-7" />
                                <span>{displayValue(cell)}</span>
                              </span>
                            ) : (
                              displayValue(cell)
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
                            <span>{displayValue(row.club_name)}</span>
                          </span>
                        </td>
                        <td className="px-3 py-2">{displayValue(row.candidate_status || row.request_status)}</td>
                        <td className="px-3 py-2">{row.match_score != null ? `${Math.round(row.match_score)} match` : "-"}</td>
                        <td className="px-3 py-2">{displayValue(row.assigned_agent_name || row.assigned_agent_id)}</td>
                        <td className="px-3 py-2">{row.agent_note || row.title || "-"}</td>
                      </tr>
                    ))}
                    {staticProspects.length === 0 && (player.mercato_prospects || []).length === 0 && (player.manual_prospect_clubs || []).length === 0 ? (
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
                  <div className="mt-4 grid gap-3">
                    <FormRow label="Player phone" action={player.player_phone ? <a className="nl-button-secondary shrink-0" href={contactHref("phone", player.player_phone)}>Call</a> : null}>
                      <input className="nl-field" name="player_phone" aria-label="Player phone" value={player.player_phone || ""} onChange={(e) => setPlayer((p) => ({ ...p, player_phone: e.target.value }))} onBlur={(e) => updatePlayer({ player_phone: e.target.value })} placeholder="+33..." />
                    </FormRow>
                    <FormRow label="Player email" action={player.player_email ? <a className="nl-button-secondary shrink-0" href={contactHref("email", player.player_email)}>Mail</a> : null}>
                      <input className="nl-field" name="player_email" aria-label="Player email" type="email" value={player.player_email || ""} onChange={(e) => setPlayer((p) => ({ ...p, player_email: e.target.value }))} onBlur={(e) => updatePlayer({ player_email: e.target.value })} placeholder="player@email.com" />
                    </FormRow>
                    <FormRow label="Entourage phone" action={player.entourage_phone ? <a className="nl-button-secondary shrink-0" href={contactHref("phone", player.entourage_phone)}>Call</a> : null}>
                      <input className="nl-field" name="entourage_phone" aria-label="Entourage phone" value={player.entourage_phone || ""} onChange={(e) => setPlayer((p) => ({ ...p, entourage_phone: e.target.value }))} onBlur={(e) => updatePlayer({ entourage_phone: e.target.value })} placeholder="+33..." />
                    </FormRow>
                    <FormRow label="Entourage email" action={player.entourage_email ? <a className="nl-button-secondary shrink-0" href={contactHref("email", player.entourage_email)}>Mail</a> : null}>
                      <input className="nl-field" name="entourage_email" aria-label="Entourage email" type="email" value={player.entourage_email || ""} onChange={(e) => setPlayer((p) => ({ ...p, entourage_email: e.target.value }))} onBlur={(e) => updatePlayer({ entourage_email: e.target.value })} placeholder="entourage@email.com" />
                    </FormRow>
                  </div>
                </div>
                <div className="rounded-lg border border-white/10 bg-white/[0.025] p-4">
                  <p className="text-[11px] font-black uppercase tracking-[0.12em] text-[#8CC7A7]">Season objectives</p>
                  <h3 className="mt-1 text-lg font-extrabold text-slate-950">Player objectives</h3>
                  <textarea
                    className="nl-field mt-4 min-h-[154px]"
                    name="season_objectives"
                    aria-label="Season objectives"
                    value={player.season_objectives || ""}
                    onChange={(e) => setPlayer((p) => ({ ...p, season_objectives: e.target.value }))}
                    onBlur={(e) => updatePlayer({ season_objectives: e.target.value })}
                    placeholder="Season targets, role development, minutes target, market objectives..."
                  />
                </div>
              </div>
              <div className="mt-4 grid gap-3 xl:grid-cols-2">
                {[
                  ["contract_status", "Contract status"],
                  ["mandate_status", "Mandate status"],
                  ["medical_status", "Medical status"],
                  ["current_club_situation", "Current club situation"],
                ].map(([field, label]) => (
                  <FormRow key={field} label={label} value={displayValue(player[field])}>
                    <input
                      className="nl-field"
                      name={field}
                      aria-label={label}
                      value={player[field] || ""}
                      onChange={(e) => setPlayer((p) => ({ ...p, [field]: e.target.value }))}
                      onBlur={(e) => updatePlayer({ [field]: e.target.value })}
                      placeholder={label}
                    />
                  </FormRow>
                ))}
                <FormRow label="Market notes" value={player.market_notes}>
                  <textarea className="nl-field min-h-[110px]" name="market_notes" aria-label="Market notes" value={player.market_notes || ""} onChange={(e) => setPlayer((p) => ({ ...p, market_notes: e.target.value }))} onBlur={(e) => updatePlayer({ market_notes: e.target.value })} placeholder="Market notes" />
                </FormRow>
                <FormRow label="Scouting notes" value={player.scouting_notes}>
                  <textarea className="nl-field min-h-[110px]" name="scouting_notes" aria-label="Scouting notes" value={player.scouting_notes || ""} onChange={(e) => setPlayer((p) => ({ ...p, scouting_notes: e.target.value }))} onBlur={(e) => updatePlayer({ scouting_notes: e.target.value })} placeholder="Scouting notes" />
                </FormRow>
              </div>
            </div>

            <div className="surface-panel rounded-lg p-5">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="nl-kicker">Documents</p>
                  <h2 className="mt-2 text-2xl font-extrabold text-slate-950">Player files</h2>
                  <p className="mt-1 text-sm text-slate-600">Choose the domain, add a title, then drop a file. Click any document to preview it.</p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-black text-slate-700">
                    {(player.documents || []).length} files
                  </span>
                  <button type="button" className="nl-button-primary" onClick={() => setDocumentModalOpen(true)}>
                    + Add document
                  </button>
                </div>
              </div>
              <div className="mt-4 flex flex-wrap gap-2">
                {DOC_TYPES.map((type) => (
                  <button
                    key={type.value}
                    type="button"
                    className={`rounded-md border px-3 py-1.5 text-xs font-semibold transition ${
                      documentFilter === type.value
                        ? "border-[#3A8967]/40 bg-[#2F7D5C]/20 text-[#DDF3E8]"
                        : "border-white/10 bg-white/[0.035] text-white/60 hover:border-white/20 hover:text-white"
                    }`}
                    onClick={() => setDocumentFilter(type.value)}
                  >
                    {type.label}
                  </button>
                ))}
              </div>
              <div className="mt-5 grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-6">
                {filteredDocuments.map((doc) => (
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
                {filteredDocuments.length === 0 ? (
                  <p className="col-span-full rounded-md border border-dashed border-slate-300 bg-slate-50 p-4 text-sm font-semibold text-slate-500">
                    No document matches this filter.
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
                <button type="button" className="nl-button-primary" onClick={() => setTransferModalOpen(true)}>
                  + Add transfer
                </button>
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
                            {transfer.match_type === "manual" ? (
                              <span className="rounded-full border border-[#3A8967]/40 bg-[#2F7D5C]/20 px-2.5 py-1 text-[11px] font-black uppercase tracking-[0.08em] text-[#DDF3E8]">
                                Manual
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
                          {transfer.match_type === "manual" ? (
                            <button type="button" className="mt-3 text-xs font-bold text-rose-300 hover:text-rose-200" onClick={() => removeManualTransfer(transfer.id)}>
                              Remove manual move
                            </button>
                          ) : null}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="mt-5 rounded-lg border border-dashed border-slate-300 bg-slate-50 p-5">
                  <p className="text-sm font-extrabold text-slate-950">No verified transfer history yet.</p>
                  <p className="mt-1 text-sm font-semibold text-slate-500">
                    Connect the player to Scouting Lab and confirm the current club, or add a manual movement for academy and young players.
                  </p>
                  <button type="button" className="nl-button-primary mt-4" onClick={() => setTransferModalOpen(true)}>
                    Add transfer manually
                  </button>
                </div>
              )}
            </div>
        </section>

        {canArchivePlayer ? (
          <section className="surface-panel rounded-lg border-rose-500/25 bg-rose-950/10 p-5">
            <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
              <div>
                <p className="text-xs font-bold uppercase tracking-[0.16em] text-rose-300">Admin controls</p>
                <h2 className="mt-2 text-xl font-semibold text-white">Archive this player room</h2>
                <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-400">
                  This hides the player from the application. The database record and related data remain recoverable.
                </p>
              </div>
              <button type="button" className="rounded-md border border-rose-400/35 bg-rose-500/12 px-4 py-2 text-sm font-semibold text-rose-100 transition hover:border-rose-300/60 hover:bg-rose-500/20" onClick={() => setArchiveModalOpen(true)}>
                Delete player
              </button>
            </div>
          </section>
        ) : null}

        {documentModalOpen ? (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4 backdrop-blur-sm" role="dialog" aria-modal="true">
            <div className="w-full max-w-2xl overflow-hidden rounded-lg border border-white/10 bg-[#080B0A] shadow-2xl">
              <div className="flex items-start justify-between gap-4 border-b border-white/10 p-5">
                <div>
                  <p className="nl-kicker">Document upload</p>
                  <h3 className="mt-2 text-2xl font-semibold text-slate-950">Add player document</h3>
                </div>
                <button
                  type="button"
                  className="nl-icon-button"
                  onClick={() => {
                    setDocumentModalOpen(false);
                    setIsDocDragging(false);
                  }}
                  aria-label="Close document form"
                >
                  x
                </button>
              </div>
              <div className="space-y-4 p-5">
                <FormRow label="Document domain">
                  <select className="nl-field" name="document_type" aria-label="Document type" value={docForm.document_type} onChange={(e) => setDocForm((p) => ({ ...p, document_type: e.target.value }))}>
                    {DOC_TYPES.filter((type) => type.value !== "all").map((type) => <option key={type.value} value={type.value}>{type.label}</option>)}
                  </select>
                </FormRow>
                <FormRow label="Title">
                  <input className="nl-field" name="document_title" aria-label="Document title" value={docForm.title} onChange={(e) => setDocForm((p) => ({ ...p, title: e.target.value }))} placeholder="Medical check, signed mandate..." />
                </FormRow>
                <label
                  className={`flex min-h-[180px] cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-6 text-center transition ${
                    isDocDragging ? "border-[#559A78] bg-[#2F7D5C]/14" : "border-white/15 bg-white/[0.025] hover:border-[#3A8967]/50 hover:bg-[#2F7D5C]/10"
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
                  <span className="flex h-10 w-10 items-center justify-center rounded-md border border-[#3A8967]/40 bg-[#2F7D5C]/20 text-xl font-semibold text-[#DDF3E8]">+</span>
                  <span className="mt-3 text-lg font-extrabold text-slate-950">Drop the file here</span>
                  <span className="mt-1 text-sm font-semibold text-slate-500">or click to select it. The file name is captured automatically.</span>
                  <input
                    type="file"
                    className="sr-only"
                    aria-label="Upload player document"
                    onChange={(event) => addDocumentFile(event.target.files?.[0])}
                  />
                </label>
              </div>
            </div>
          </div>
        ) : null}

        {prospectModalOpen ? (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4 backdrop-blur-sm" role="dialog" aria-modal="true">
            <div className="w-full max-w-2xl overflow-hidden rounded-lg border border-white/10 bg-[#080B0A] shadow-2xl">
              <div className="flex items-start justify-between gap-4 border-b border-white/10 p-5">
                <div>
                  <p className="nl-kicker">Prospect club</p>
                  <h3 className="mt-2 text-2xl font-semibold text-slate-950">Add a club target</h3>
                </div>
                <button type="button" className="nl-icon-button" onClick={() => setProspectModalOpen(false)} aria-label="Close prospect form">x</button>
              </div>
              <div className="space-y-4 p-5">
                <FormRow label="Club" value={prospectForm.club_name}>
                  <div className="relative">
                    <input
                      className="nl-field"
                      value={prospectForm.club_name}
                      onChange={(event) => setProspectForm((prev) => ({ ...prev, club_name: event.target.value, club_id: "", competition_name: "" }))}
                      placeholder="Type a club name"
                    />
                    {prospectForm.club_name && !prospectForm.club_id ? (
                      <div className="absolute z-[9999] mt-2 max-h-64 w-full overflow-auto rounded-md border border-white/10 bg-[#0D100E] shadow-2xl">
                        {clubMatches.length ? clubMatches.map((club) => (
                          <button
                            key={club.id}
                            type="button"
                            className="flex w-full items-center gap-3 border-b border-white/10 px-3 py-2.5 text-left text-sm text-slate-300 transition hover:bg-[#2F7D5C]/15 hover:text-white"
                            onMouseDown={(event) => event.preventDefault()}
                            onClick={() => selectProspectClub(club)}
                          >
                            <ClubLogo name={club.name} className="h-7 w-7" />
                            <span className="min-w-0">
                              <span className="block truncate font-semibold">{club.name}</span>
                              <span className="block truncate text-xs text-slate-500">{club.competition_name || club.country || "-"}</span>
                            </span>
                          </button>
                        )) : (
                          <p className="px-3 py-3 text-sm text-slate-500">No matching club. You can still save this typed club name.</p>
                        )}
                      </div>
                    ) : null}
                  </div>
                </FormRow>
                <FormRow label="Competition" value={prospectForm.competition_name}>
                  <input className="nl-field" value={prospectForm.competition_name} onChange={(event) => setProspectForm((prev) => ({ ...prev, competition_name: event.target.value }))} placeholder="Competition" />
                </FormRow>
                <FormRow label="Status" value={prospectForm.status}>
                  <input className="nl-field" value={prospectForm.status} onChange={(event) => setProspectForm((prev) => ({ ...prev, status: event.target.value }))} placeholder="Interest, watching, offer..." />
                </FormRow>
                <FormRow label="Offer" value={prospectForm.offer}>
                  <input className="nl-field" value={prospectForm.offer} onChange={(event) => setProspectForm((prev) => ({ ...prev, offer: event.target.value }))} placeholder="Fee, salary or terms" />
                </FormRow>
                <FormRow label="Contact" value={prospectForm.contact}>
                  <input className="nl-field" value={prospectForm.contact} onChange={(event) => setProspectForm((prev) => ({ ...prev, contact: event.target.value }))} placeholder="Contact person" />
                </FormRow>
                <FormRow label="Notes" value={prospectForm.notes}>
                  <textarea className="nl-field min-h-[90px]" value={prospectForm.notes} onChange={(event) => setProspectForm((prev) => ({ ...prev, notes: event.target.value }))} placeholder="Context, timing, next step..." />
                </FormRow>
              </div>
              <div className="flex justify-end gap-2 border-t border-white/10 p-5">
                <button type="button" className="nl-button-secondary" onClick={() => setProspectModalOpen(false)}>Cancel</button>
                <button type="button" className="nl-button-primary" onClick={addProspectClub}>Add prospect</button>
              </div>
            </div>
          </div>
        ) : null}

        {transferModalOpen ? (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4 backdrop-blur-sm" role="dialog" aria-modal="true">
            <div className="w-full max-w-2xl overflow-hidden rounded-lg border border-white/10 bg-[#080B0A] shadow-2xl">
              <div className="flex items-start justify-between gap-4 border-b border-white/10 p-5">
                <div>
                  <p className="nl-kicker">Manual transfer</p>
                  <h3 className="mt-2 text-2xl font-semibold text-slate-950">Add career movement</h3>
                </div>
                <button type="button" className="nl-icon-button" onClick={() => setTransferModalOpen(false)} aria-label="Close transfer form">x</button>
              </div>
              <div className="grid gap-4 p-5 md:grid-cols-2">
                <FormRow label="Date" value={transferForm.transfer_date}>
                  <input className="nl-field" type="date" value={transferForm.transfer_date} onChange={(event) => setTransferForm((prev) => ({ ...prev, transfer_date: event.target.value }))} />
                </FormRow>
                <FormRow label="Type" value={transferForm.transfer_type}>
                  <input className="nl-field" value={transferForm.transfer_type} onChange={(event) => setTransferForm((prev) => ({ ...prev, transfer_type: event.target.value }))} placeholder="Transfer, loan, free..." />
                </FormRow>
                <FormRow label="From club" value={transferForm.team_out_name}>
                  <input className="nl-field" value={transferForm.team_out_name} onChange={(event) => setTransferForm((prev) => ({ ...prev, team_out_name: event.target.value }))} placeholder="Previous club" />
                </FormRow>
                <FormRow label="To club" value={transferForm.team_in_name}>
                  <input className="nl-field" value={transferForm.team_in_name} onChange={(event) => setTransferForm((prev) => ({ ...prev, team_in_name: event.target.value }))} placeholder="New club" />
                </FormRow>
                <FormRow label="Fee" value={transferForm.transfer_fee}>
                  <input className="nl-field" value={transferForm.transfer_fee} onChange={(event) => setTransferForm((prev) => ({ ...prev, transfer_fee: event.target.value }))} placeholder="Undisclosed, 250K..." />
                </FormRow>
                <FormRow label="League" value={transferForm.league_name}>
                  <input className="nl-field" value={transferForm.league_name} onChange={(event) => setTransferForm((prev) => ({ ...prev, league_name: event.target.value }))} placeholder="League" />
                </FormRow>
                <div className="md:col-span-2">
                  <FormRow label="Notes" value={transferForm.notes}>
                    <textarea className="nl-field min-h-[90px]" value={transferForm.notes} onChange={(event) => setTransferForm((prev) => ({ ...prev, notes: event.target.value }))} placeholder="Source, context, confirmation status..." />
                  </FormRow>
                </div>
              </div>
              <div className="flex justify-end gap-2 border-t border-white/10 p-5">
                <button type="button" className="nl-button-secondary" onClick={() => setTransferModalOpen(false)}>Cancel</button>
                <button type="button" className="nl-button-primary" onClick={addManualTransfer}>Add transfer</button>
              </div>
            </div>
          </div>
        ) : null}

        {archiveModalOpen ? (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-md" role="dialog" aria-modal="true">
            <div className="w-full max-w-xl overflow-hidden rounded-lg border border-rose-400/25 bg-[#080B0A] shadow-2xl">
              <div className="border-b border-white/10 p-5">
                <p className="text-xs font-bold uppercase tracking-[0.16em] text-rose-300">Confirm deletion</p>
                <h3 className="mt-2 text-2xl font-semibold text-white">Delete player room?</h3>
                <p className="mt-2 text-sm leading-6 text-slate-400">
                  This action will hide <span className="font-semibold text-white">{player.display_name}</span> from the application. The data will remain in the database and can be recovered by an administrator.
                </p>
              </div>
              <div className="space-y-4 p-5">
                <label className="block">
                  <span className="mb-2 block text-xs font-bold uppercase tracking-[0.12em] text-slate-400">
                    Type the player name to confirm
                  </span>
                  <input
                    className="nl-field"
                    value={archiveConfirmName}
                    onChange={(event) => setArchiveConfirmName(event.target.value)}
                    placeholder={player.display_name}
                    autoFocus
                  />
                </label>
              </div>
              <div className="flex flex-col-reverse gap-2 border-t border-white/10 p-5 sm:flex-row sm:justify-end">
                <button
                  type="button"
                  className="nl-button-secondary"
                  onClick={() => {
                    setArchiveModalOpen(false);
                    setArchiveConfirmName("");
                  }}
                  disabled={archiving}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="rounded-md border border-rose-400/35 bg-rose-500/16 px-4 py-2 text-sm font-semibold text-rose-100 transition hover:border-rose-300/60 hover:bg-rose-500/25 disabled:cursor-not-allowed disabled:opacity-45"
                  onClick={archivePlayer}
                  disabled={archiving || archiveConfirmName.trim() !== player.display_name}
                >
                  {archiving ? "Deleting..." : "Delete player"}
                </button>
              </div>
            </div>
          </div>
        ) : null}

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
                    <a href={storageHref(selectedDoc.storage_url)} download={selectedDoc.file_name || selectedDoc.title} className="nl-button-secondary">Download</a>
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
                          {DOC_TYPES.filter((type) => type.value !== "all").map((type) => <option key={type.value} value={type.value}>{type.label}</option>)}
                        </select>
                      </label>
                      <label className="block">
                        <span className="mb-1 block text-[11px] font-black uppercase tracking-[0.12em] text-slate-500">Title</span>
                        <input className="nl-field" aria-label="Document title" value={editingDoc.title} onChange={(e) => setEditingDoc((p) => ({ ...p, title: e.target.value }))} />
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
                        <p className="text-[11px] font-black uppercase tracking-[0.12em] text-slate-500">File name</p>
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
