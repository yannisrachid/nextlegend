import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/router";
import ClubLogo from "@/components/ClubLogo";
import { apiUrl, deleteJson, fetchJson, fetchJsonCached, patchJson, postJson } from "@/lib/api";
import { useAuth } from "@/lib/auth";

const PRIORITIES = ["low", "medium", "high", "urgent"];
const STATUSES = ["new", "searching", "shortlist_ready", "proposed", "discussion", "closed"];
const DEAL_TYPES = ["any", "transfer", "loan", "free"];
const FALLBACK_POSITIONS = [
  "GK",
  "CB",
  "LCB",
  "RCB",
  "LB",
  "RB",
  "LWB",
  "RWB",
  "DMF",
  "LDMF",
  "RDMF",
  "CM",
  "LCMF",
  "RCMF",
  "AMF",
  "LAMF",
  "RAMF",
  "LW",
  "RW",
  "CF",
  "ST",
];
const EXCEL_PLAYER_SHEETS = {
  Kevin: {
    columns: ["Club", "Status", "Offer", "Contact", "Notes"],
    rows: [
      ["Monaco", "Interest", "", "", ""],
      ["Marseille", "Pending", "", "", ""],
      ["Everton", "Pending", "", "", ""],
      ["Nottingham", "Pending", "", "", ""],
      ["Brentford", "No interest", "", "", ""],
      ["Newcaslte", "No interest", "", "", ""],
      ["Atletico Madrid", "No interest", "", "", ""],
    ],
  },
  Lilian: {
    columns: ["Club", "Status", "Offer", "Contact", "Meet", "Notes"],
    rows: [["Bologna", "Interest", "", "", "", ""]],
  },
  Mario: {
    columns: ["Club", "Status", "Offer", "Contact", "Meet", "Notes"],
    rows: [["Al Shabab", "Offer", "3.5 net", "", "", ""]],
  },
  Simon: {
    columns: ["Club", "Status", "Offer", "Contact", "Meet", "Notes"],
    rows: [["Cruzeiro Esporte Clube", "No interest", "", "Damon intermediary", "", ""]],
  },
};

const WORKBOOK_OVERVIEW_COLUMNS = [
  "Name",
  "Current club",
  "Contract expiry",
  "Current club situation",
  "Plan",
  "Priority",
  "Demanded TF",
  "Next step",
];

const CLUB_REQUIREMENT_COLUMNS = ["Clubs", "Category", "Profile", "Fee", "Net wages", "Suggestion", "Status"];
const WORKBOOK_STORAGE_KEY = "nextlegend_mercato_workbook_v1";
const REQUIREMENTS_SHEET_ID = "requirements";

const makeRow = (id, cells, meta = {}) => ({
  id,
  cells: cells.map((cell) => (cell === null || cell === undefined ? "" : String(cell))),
  meta,
});

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
  <p className="text-xs uppercase tracking-[0.18em] text-slate-400">{children}</p>
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

const EditableWorkbookSheet = ({
  sheet,
  sortState,
  onSort,
  onCellChange,
  onCellBlur,
  getCellSuggestions,
  onAddRow,
  onDeleteRow,
}) => {
  const sortedRows = useMemo(() => {
    if (!sortState || sortState.columnIndex === null || sortState.columnIndex === undefined) {
      return sheet.rows;
    }
    const rows = [...sheet.rows];
    rows.sort((a, b) => {
      const left = a.cells[sortState.columnIndex] || "";
      const right = b.cells[sortState.columnIndex] || "";
      const leftNum = Number(left);
      const rightNum = Number(right);
      let order = 0;
      if (Number.isFinite(leftNum) && Number.isFinite(rightNum) && left !== "" && right !== "") {
        order = leftNum - rightNum;
      } else {
        order = String(left).localeCompare(String(right), undefined, { numeric: true, sensitivity: "base" });
      }
      return sortState.direction === "desc" ? -order : order;
    });
    return rows;
  }, [sheet.rows, sortState]);

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 bg-white px-4 py-3">
        <div>
          <p className="text-xs font-black uppercase tracking-[0.14em] text-teal-700">Editable sheet</p>
          <h2 className="mt-1 text-xl font-extrabold text-slate-950">{sheet.label}</h2>
        </div>
        <button
          type="button"
          className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-teal-200 bg-teal-50 text-xl font-black leading-none text-teal-800 transition hover:border-teal-500 hover:bg-teal-100 focus:outline-none focus:ring-4 focus:ring-teal-700/10"
          onClick={onAddRow}
          aria-label={`Add a row to ${sheet.label}`}
          title="Add row"
        >
          +
        </button>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-[1040px] w-full border-collapse text-left text-sm">
          <thead className={`${sheet.headerClass || "bg-[#1f4e78]"} text-xs uppercase tracking-[0.08em] text-white`}>
            <tr>
              <th className="w-12 border border-slate-300 px-3 py-2 text-center">#</th>
              {sheet.columns.map((column, columnIndex) => (
                <th key={`${sheet.id}-${column}`} className="border border-slate-300 p-0">
                  <button
                    type="button"
                    className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left font-black"
                    onClick={() => onSort(columnIndex)}
                  >
                    <span>{column}</span>
                    <span className="text-[10px] opacity-80">
                      {sortState?.columnIndex === columnIndex ? (sortState.direction === "asc" ? "ASC" : "DESC") : "SORT"}
                    </span>
                  </button>
                </th>
              ))}
              <th className="w-14 border border-slate-300 px-3 py-2 text-center">+</th>
            </tr>
          </thead>
          <tbody>
            {sortedRows.map((row, visualIndex) => (
              <tr key={row.id} className="bg-white hover:bg-[#ddebf7]">
                <td className="border border-slate-300 bg-slate-50 px-3 py-2 text-center text-xs font-bold text-slate-500">
                  {visualIndex + 1}
                </td>
                {sheet.columns.map((column, columnIndex) => (
                  <td key={`${row.id}-${column}`} className="border border-slate-300 p-0">
                    {row.meta?.reportUrl && column === "Player" ? (
                      <a
                        href={row.meta.reportUrl}
                        target="_blank"
                        rel="noreferrer"
                        className="flex h-10 w-full min-w-[140px] items-center justify-between gap-3 bg-transparent px-3 py-2 text-left text-sm font-black text-teal-800 outline-none transition hover:bg-teal-50 focus:bg-teal-50 focus:ring-2 focus:ring-inset focus:ring-teal-500"
                        aria-label={`Open report for ${row.cells[columnIndex] || "player"}`}
                        title="Open player report in a new tab"
                      >
                        <span className="truncate">{row.cells[columnIndex] || ""}</span>
                        <span className="text-[10px] uppercase tracking-[0.12em] text-teal-600">Report</span>
                      </a>
                    ) : (
                      <>
                        {(() => {
                          const suggestions = getCellSuggestions?.(sheet.id, columnIndex, row.cells[columnIndex] || "") || [];
                          const listId = suggestions.length ? `${sheet.id}-${row.id}-${columnIndex}-suggestions` : undefined;
                          const showClubLogo = (sheet.id === "requirements" || sheet.id === "matching") && columnIndex === 0;
                          return (
                            <>
                              <div className={`flex h-10 min-w-[160px] items-center ${showClubLogo ? "gap-2 px-2" : ""}`}>
                                {showClubLogo ? <ClubLogo name={row.cells[columnIndex]} className="h-7 w-7" /> : null}
                                <input
                                  className="h-10 w-full min-w-0 bg-transparent px-3 py-2 text-sm text-slate-950 outline-none focus:bg-teal-50 focus:ring-2 focus:ring-inset focus:ring-teal-500"
                                  aria-label={`${column} row ${visualIndex + 1}`}
                                  value={row.cells[columnIndex] || ""}
                                  list={listId}
                                  onChange={(event) => onCellChange(row.id, columnIndex, event.target.value)}
                                  onBlur={(event) => onCellBlur?.(row.id, columnIndex, event.target.value)}
                                />
                              </div>
                              {listId ? (
                                <datalist id={listId}>
                                  {suggestions.map((suggestion, suggestionIndex) => (
                                    <option key={`${suggestion.value}-${suggestion.label}-${suggestionIndex}`} value={suggestion.value}>
                                      {suggestion.label}
                                    </option>
                                  ))}
                                </datalist>
                              ) : null}
                            </>
                          );
                        })()}
                      </>
                    )}
                  </td>
                ))}
                <td className="border border-slate-300 bg-slate-50 px-2 py-1 text-center">
                  <button
                    type="button"
                    className="inline-flex h-7 w-7 items-center justify-center rounded border border-red-200 bg-white text-lg font-black leading-none text-red-600 transition hover:border-red-400 hover:bg-red-50 focus:outline-none focus:ring-4 focus:ring-red-500/10"
                    onClick={() => onDeleteRow(row.id)}
                    aria-label={`Delete row ${visualIndex + 1} from ${sheet.label}`}
                    title="Delete row"
                  >
                    -
                  </button>
                </td>
              </tr>
            ))}
            {sortedRows.length === 0 ? (
              <tr>
                <td className="border border-slate-300 px-3 py-5 text-slate-500" colSpan={sheet.columns.length + 2}>
                  This sheet is ready for new market intelligence. Add a row to start building it.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const formatMoney = (value) => {
  if (value === null || value === undefined || value === "") return "-";
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  if (Math.abs(num) >= 1e6) return `${Math.round((num / 1e6) * 10) / 10}M`;
  if (Math.abs(num) >= 1e3) return `${Math.round(num / 1e3)}K`;
  return `${Math.round(num)}`;
};

const formatMetric = (value) => {
  if (value === null || value === undefined || value === "") return "-";
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  if (Math.abs(num) >= 1000) return formatMoney(num);
  return String(Math.round(num * 10) / 10);
};

const absoluteUrl = (value) => {
  if (!value) return "";
  const raw = String(value);
  if (raw.startsWith("http://") || raw.startsWith("https://") || raw.startsWith("data:")) return raw;
  if (raw.startsWith("//")) return `https:${raw}`;
  return raw;
};

const initials = (name) =>
  String(name || "")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("") || "NL";

const recommendationKey = (candidate) => `${candidate?.player_id || candidate?.id || ""}-${candidate?.player_season_id || ""}`;

const MatchingPlayerCard = ({ candidate, selected, onToggle }) => {
  const tm = candidate.tm_fields || {};
  const photoUrl = absoluteUrl(tm.tm_profile_image_url || tm.profile_image_url || candidate.profile_image_url);
  const reason = candidate.explanation_json?.recommendation_reason || "Strong data fit for the selected need.";
  const strengths = Array.isArray(candidate.explanation_json?.strengths) ? candidate.explanation_json.strengths.slice(0, 2) : [];
  const keyMetrics = (candidate.key_metrics || [])
    .filter((metric) => metric.value !== null && metric.value !== undefined && metric.value !== "")
    .slice(0, 5);

  return (
    <button
      type="button"
      className={`group flex h-full flex-col rounded-lg border bg-white p-4 text-left shadow-sm transition hover:-translate-y-0.5 hover:border-teal-500 hover:shadow-xl focus:outline-none focus:ring-4 focus:ring-teal-700/10 ${
        selected ? "border-teal-600 ring-4 ring-teal-700/10" : "border-slate-200"
      }`}
      onClick={onToggle}
      aria-pressed={selected}
    >
      <div className="flex items-start gap-3">
        <div className="h-16 w-16 overflow-hidden rounded-lg border border-slate-200 bg-slate-100">
          {photoUrl ? (
            <img src={photoUrl} alt="" className="h-full w-full object-cover" />
          ) : (
            <div className="flex h-full w-full items-center justify-center bg-slate-900 text-sm font-black text-white">
              {initials(candidate.name)}
            </div>
          )}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0">
              <p className="truncate text-base font-black text-slate-950">{candidate.name || "Unnamed player"}</p>
              <p className="mt-1 text-xs font-bold text-slate-500">
                {[candidate.position, candidate.second_position, candidate.age ? `${candidate.age} yrs` : ""].filter(Boolean).join(" / ") || "Profile"}
              </p>
            </div>
            <span className={`rounded-full px-2 py-1 text-[11px] font-black ${selected ? "bg-teal-700 text-white" : "bg-slate-100 text-slate-600"}`}>
              {selected ? "Selected" : "Select"}
            </span>
          </div>
          <div className="mt-2 flex items-center gap-2 text-xs text-slate-500">
            <ClubLogo name={candidate.team} className="h-6 w-6 rounded" />
            <span className="min-w-0 truncate">
              {[candidate.team, candidate.competition_name, candidate.calendar].filter(Boolean).join(" - ")}
            </span>
          </div>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-2">
        <div className="rounded-md bg-slate-950 p-2 text-white">
          <p className="text-[10px] font-black uppercase tracking-[0.12em] text-slate-400">Match</p>
          <p className="mt-1 text-lg font-black">{formatMetric(candidate.match_score)}</p>
        </div>
        <div className="rounded-md bg-teal-50 p-2 text-teal-950">
          <p className="text-[10px] font-black uppercase tracking-[0.12em] text-teal-700">Global</p>
          <p className="mt-1 text-lg font-black">{formatMetric(candidate.raw_player_level)}</p>
        </div>
        <div className="rounded-md bg-lime-50 p-2 text-lime-950">
          <p className="text-[10px] font-black uppercase tracking-[0.12em] text-lime-700">Adjusted</p>
          <p className="mt-1 text-lg font-black">{formatMetric(candidate.calibrated_player_level)}</p>
        </div>
      </div>

      <p className="mt-4 line-clamp-2 text-sm leading-5 text-slate-600">{reason}</p>

      {keyMetrics.length ? (
        <div className="mt-4 grid grid-cols-2 gap-2">
          {keyMetrics.map((metric) => (
            <div key={metric.label} className="rounded-md border border-slate-200 px-2 py-1.5">
              <p className="truncate text-[10px] font-black uppercase tracking-[0.1em] text-slate-400">{metric.label}</p>
              <p className="mt-1 text-sm font-black text-slate-900">{formatMetric(metric.value)}</p>
            </div>
          ))}
        </div>
      ) : null}

      {strengths.length ? (
        <div className="mt-4 flex flex-wrap gap-1.5">
          {strengths.map((strength) => (
            <span key={strength} className="rounded-full bg-teal-50 px-2 py-1 text-[11px] font-bold text-teal-800">
              {strength}
            </span>
          ))}
        </div>
      ) : null}
    </button>
  );
};

const firstNeed = (request) => (request?.needs || [])[0] || null;

const parseSheetNumber = (value) => {
  const clean = String(value || "").replace(/[^\d.,-]/g, "").replace(",", ".");
  const numeric = Number(clean);
  return Number.isFinite(numeric) ? numeric : null;
};

const normalizeClubSearch = (value) => {
  const normalized = String(value || "")
    .trim()
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, "");
  const aliases = {
    villareal: "villarreal",
  };
  return aliases[normalized] || normalized;
};

const requirementSemanticKey = (row) => {
  const cells = row?.cells || [];
  return [
    normalizeClubSearch(cells[0]),
    normalizeClubSearch(cells[1]),
    normalizeClubSearch(cells[2]),
  ].join("|");
};

const dedupeWorkbookRows = (sheetId, rows) => {
  const seenIds = new Set();
  const seenRequirements = new Set();
  return (rows || []).filter((row) => {
    if (!row?.id || seenIds.has(row.id)) return false;
    seenIds.add(row.id);
    if (sheetId === REQUIREMENTS_SHEET_ID) {
      const key = requirementSemanticKey(row);
      if (key !== "||") {
        if (seenRequirements.has(key)) return false;
        seenRequirements.add(key);
      }
    }
    return true;
  });
};

const levenshteinDistance = (left, right) => {
  if (left === right) return 0;
  if (!left) return right.length;
  if (!right) return left.length;
  const previous = Array.from({ length: right.length + 1 }, (_, index) => index);
  const current = Array(right.length + 1).fill(0);
  for (let i = 1; i <= left.length; i += 1) {
    current[0] = i;
    for (let j = 1; j <= right.length; j += 1) {
      const cost = left[i - 1] === right[j - 1] ? 0 : 1;
      current[j] = Math.min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + cost);
    }
    for (let j = 0; j <= right.length; j += 1) previous[j] = current[j];
  }
  return previous[right.length];
};

export default function MercatoPage() {
  const router = useRouter();
  const { me } = useAuth();
  const [items, setItems] = useState([]);
  const [matchingItems, setMatchingItems] = useState([]);
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
  const [activeSheet, setActiveSheet] = useState("overview");
  const [hdPlayers, setHdPlayers] = useState([]);
  const [sheetEdits, setSheetEdits] = useState({});
  const [customSheets, setCustomSheets] = useState([]);
  const [sortBySheet, setSortBySheet] = useState({});
  const [newSheetName, setNewSheetName] = useState("");
  const [workbookHydrated, setWorkbookHydrated] = useState(false);
  const [matchingNeedId, setMatchingNeedId] = useState("");
  const [matchingLeagueQuery, setMatchingLeagueQuery] = useState("");
  const [matchingCompetitions, setMatchingCompetitions] = useState([]);
  const [matchingFilters, setMatchingFilters] = useState({
    age_min: "",
    age_max: "",
    min_minutes: "270",
    min_match_score: "",
  });
  const [matchingBusy, setMatchingBusy] = useState(false);
  const [matchingResults, setMatchingResults] = useState([]);
  const [selectedRecommendations, setSelectedRecommendations] = useState([]);
  const [addingRecommendations, setAddingRecommendations] = useState(false);
  const [savingDraftRows, setSavingDraftRows] = useState([]);

  const selected = useMemo(
    () => items.find((item) => item.id === selectedId) || items[0] || null,
    [items, selectedId]
  );
  const selectedNeed = firstNeed(selected);

  const needOptions = useMemo(() => {
    return matchingItems.flatMap((item) =>
      (item.needs || []).map((need) => ({
        request: item,
        need,
        needId: need.id,
        label: `${item.club_name || "Unknown club"} - ${need.position || item.title || "Need"} - ${item.priority || "medium"}`,
      }))
    );
  }, [matchingItems]);

  const matchingNeed = useMemo(
    () => needOptions.find((option) => String(option.needId) === String(matchingNeedId)) || null,
    [matchingNeedId, needOptions]
  );

  const filteredMatchingCompetitions = useMemo(() => {
    const query = matchingLeagueQuery.trim().toLowerCase();
    const source = query
      ? competitions.filter((competition) => competition.toLowerCase().includes(query))
      : competitions;
    return source.filter((competition) => !matchingCompetitions.includes(competition)).slice(0, 18);
  }, [competitions, matchingCompetitions, matchingLeagueQuery]);

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

  const loadMatchingRequests = async () => {
    try {
      const data = await fetchJson("/mercato/requests");
      setMatchingItems(data.items || []);
    } catch (err) {
      setMessage(err.message);
    }
  };

  const loadHdPlayers = async () => {
    try {
      const data = await fetchJson("/hd-players");
      setHdPlayers(data.items || []);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    loadMeta().catch((err) => setError(err.message));
    loadHdPlayers();
    loadMatchingRequests();
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.localStorage.getItem(WORKBOOK_STORAGE_KEY);
      if (raw) {
        const saved = JSON.parse(raw);
        setSheetEdits(saved.sheetEdits || {});
        setCustomSheets(Array.isArray(saved.customSheets) ? saved.customSheets : []);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setWorkbookHydrated(true);
    }
  }, []);

  useEffect(() => {
    if (!workbookHydrated || typeof window === "undefined") return;
    window.localStorage.setItem(WORKBOOK_STORAGE_KEY, JSON.stringify({ sheetEdits, customSheets }));
  }, [customSheets, sheetEdits, workbookHydrated]);

  useEffect(() => {
    loadRequests();
  }, [filters]);

  useEffect(() => {
    if (!needOptions.length) {
      if (matchingNeedId) setMatchingNeedId("");
      return;
    }
    if (!matchingNeedId || !needOptions.some((option) => String(option.needId) === String(matchingNeedId))) {
      setMatchingNeedId(String(needOptions[0].needId));
    }
  }, [matchingNeedId, needOptions]);

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

  const positionOptions = useMemo(() => {
    return Array.from(new Set([...(positions || []), ...FALLBACK_POSITIONS]))
      .filter(Boolean)
      .sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true }));
  }, [positions]);

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
      const createdNeedId = firstNeed(created)?.id;
      if (created) {
        setItems((prev) => [created, ...prev.filter((item) => item.id !== created.id)]);
        setMatchingItems((prev) => [created, ...prev.filter((item) => item.id !== created.id)]);
      }
      if (createdNeedId) {
        setMatchingNeedId(String(createdNeedId));
      }
      await Promise.all([loadRequests(), loadMatchingRequests()]);
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
      await Promise.all([loadRequests(), loadMatchingRequests()]);
    } catch (err) {
      setMessage(err.message);
    } finally {
      setGeneratingNeedId(null);
    }
  };

  const updateMatchingFilter = (key, value) => {
    setMatchingFilters((prev) => ({ ...prev, [key]: value }));
  };

  const toggleMatchingCompetition = (competition) => {
    setMatchingCompetitions((prev) =>
      prev.includes(competition) ? prev.filter((item) => item !== competition) : [...prev, competition]
    );
  };

  const runMatchingPreview = async () => {
    if (!matchingNeedId) {
      setMessage("Select a need before running the matching algorithm.");
      return;
    }
    setMatchingBusy(true);
    setMessage("");
    setMatchingResults([]);
    setSelectedRecommendations([]);
    try {
      const payload = {
        competitions: matchingCompetitions,
        age_min: matchingFilters.age_min ? Number(matchingFilters.age_min) : null,
        age_max: matchingFilters.age_max ? Number(matchingFilters.age_max) : null,
        min_minutes: matchingFilters.min_minutes ? Number(matchingFilters.min_minutes) : null,
        min_match_score: matchingFilters.min_match_score ? Number(matchingFilters.min_match_score) : null,
      };
      const res = await postJson(`/mercato/needs/${matchingNeedId}/preview-shortlist`, payload);
      const candidates = res.candidates || [];
      setMatchingResults(candidates);
      setMessage(candidates.length ? `${candidates.length} matching players ready for review.` : "No matching player found with the current filters.");
    } catch (err) {
      setMessage(err.message);
    } finally {
      setMatchingBusy(false);
    }
  };

  const toggleRecommendation = (candidate) => {
    const key = recommendationKey(candidate);
    setSelectedRecommendations((prev) =>
      prev.includes(key) ? prev.filter((item) => item !== key) : [...prev, key]
    );
  };

  const addSelectedRecommendations = async () => {
    if (!matchingNeedId || selectedRecommendations.length === 0 || addingRecommendations) return;
    const selectedPlayers = matchingResults.filter((candidate) => selectedRecommendations.includes(recommendationKey(candidate)));
    setAddingRecommendations(true);
    setMessage("");
    try {
      let added = 0;
      for (const candidate of selectedPlayers) {
        const res = await postJson(`/mercato/needs/${matchingNeedId}/candidates`, {
          player_id: Number(candidate.player_id),
          player_season_id: candidate.player_season_id ? Number(candidate.player_season_id) : null,
          source: "algorithm",
          status: "suggested",
          agent_note: candidate.explanation_json?.recommendation_reason || null,
        });
        if (res.added) added += 1;
      }
      setMessage(added ? `${added} player${added > 1 ? "s" : ""} added to prospects.` : "Selected players were already in prospects.");
      setSelectedRecommendations([]);
      setMatchingResults((prev) => prev.filter((candidate) => !selectedRecommendations.includes(recommendationKey(candidate))));
      setActiveSheet("matching");
      await Promise.all([loadRequests(), loadMatchingRequests()]);
    } catch (err) {
      setMessage(err.message);
    } finally {
      setAddingRecommendations(false);
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

  const updateRequestAgent = async (requestId, assignedAgentId) => {
    try {
      await patchJson(`/mercato/requests/${requestId}`, { assigned_agent_id: assignedAgentId || null });
      await loadRequests();
    } catch (err) {
      setMessage(err.message);
    }
  };

  const exportExcel = () => {
    window.location.href = apiUrl("/mercato/requests/export.xlsx");
  };

  const baseSheets = useMemo(() => {
    const sheets = {
      overview: {
        id: "overview",
        label: "Overview",
        columns: WORKBOOK_OVERVIEW_COLUMNS,
        headerClass: "bg-[#1f4e78]",
        rows: hdPlayers.map((player) => makeRow(`hd-player-${player.id}`, [
          player.display_name,
          player.current_club,
          player.contract_expiry,
          player.current_club_situation,
          player.plan,
          player.priority,
          player.demanded_transfer_fee,
          player.next_step,
        ])),
      },
      requirements: {
        id: "requirements",
        label: "Clubs requirements",
        columns: CLUB_REQUIREMENT_COLUMNS,
        headerClass: "bg-[#70ad47]",
        rows: items.flatMap((requestItem) => (requestItem.needs || []).map((need) => makeRow(`need-${requestItem.id}-${need.id}`, [
          requestItem.club_name,
          need.position,
          need.notes || requestItem.extra_info,
          requestItem.budget_max,
          requestItem.salary_max,
          (need.candidates || [])[0]?.player_name || "",
          (need.candidates || []).length ? "Player suggested" : "No suggestion",
        ], {
          requestId: requestItem.id,
          needId: need.id,
          clubId: requestItem.club_id,
        }))),
      },
      matching: {
        id: "matching",
        label: "Matching shortlist",
        columns: ["Club", "Need", "Player", "Score", "Status", "Agent note"],
        headerClass: "bg-slate-800",
        rows: items.flatMap((requestItem) => (requestItem.needs || []).flatMap((need) => (need.candidates || []).map((candidate) => makeRow(`candidate-${candidate.id}`, [
          requestItem.club_name,
          need.position,
          candidate.player_name || candidate.name,
          candidate.match_score ? Math.round(candidate.match_score) : "",
          candidate.status,
          candidate.agent_note,
        ], {
          playerId: candidate.player_id,
          playerSeasonId: candidate.player_season_id,
          reportUrl: candidate.player_id ? `/report?player_id=${candidate.player_id}` : "",
        })))),
      },
    };
    Object.entries(EXCEL_PLAYER_SHEETS).forEach(([id, sheet]) => {
      sheets[id] = {
        id,
        label: id,
        columns: sheet.columns,
        headerClass: "bg-[#5b9bd5]",
        rows: sheet.rows.map((row, rowIndex) => makeRow(`${id}-${rowIndex}`, row)),
      };
    });
    customSheets.forEach((sheet) => {
      sheets[sheet.id] = sheet;
    });
    return sheets;
  }, [customSheets, hdPlayers, items]);

  const getSheet = (sheetId) => {
    const edited = sheetEdits[sheetId];
    const base = baseSheets[sheetId] || baseSheets.overview;
    if (!edited) return { ...base, rows: dedupeWorkbookRows(sheetId, base.rows || []) };
    const baseRows = new Map((base?.rows || []).map((row) => [row.id, row]));
    const editedRows = new Map((edited.rows || []).map((row) => [row.id, row]));
    const mergedBaseRows = (base?.rows || []).map((row) => ({
      ...row,
      ...(editedRows.get(row.id) || {}),
      meta: editedRows.get(row.id)?.meta || row.meta || {},
    }));
    const extraRows = (edited.rows || []).filter((row) => !baseRows.has(row.id));
    return {
      ...base,
      ...edited,
      columns: edited.columns || base.columns,
      headerClass: edited.headerClass || base.headerClass,
      rows: dedupeWorkbookRows(sheetId, [...mergedBaseRows, ...extraRows]),
    };
  };

  const activeWorkbookSheet = getSheet(activeSheet);

  const workbookTabs = Object.values(baseSheets).map((sheet) => ({
    id: sheet.id,
    label: sheet.label,
    count: getSheet(sheet.id).rows.length,
    custom: Boolean(sheet.custom),
  }));

  const updateSheetCell = (sheetId, rowId, columnIndex, value) => {
    const source = getSheet(sheetId);
    setSheetEdits((prev) => {
      if (!source) return prev;
      return {
        ...prev,
        [sheetId]: {
          ...source,
          rows: source.rows.map((row) => {
            if (row.id !== rowId) return row;
            const cells = [...row.cells];
            cells[columnIndex] = value;
            return { ...row, cells };
          }),
        },
      };
    });
  };

  const findClubFromWorkbookCell = (clubName) => {
    const raw = String(clubName || "").trim();
    const normalized = normalizeClubSearch(raw);
    if (!normalized) return null;
    const exact =
      clubOptions.find((club) => normalizeClubSearch(club.name) === normalized) ||
      clubOptions.find((club) => normalizeClubSearch(club.label) === normalized);
    if (exact) return exact;
    const contains =
      clubOptions.find((club) => normalizeClubSearch(club.name).includes(normalized)) ||
      clubOptions.find((club) => normalizeClubSearch(club.label).includes(normalized)) ||
      clubOptions.find((club) => normalized.includes(normalizeClubSearch(club.name)));
    if (contains) return contains;
    if (normalized.length >= 5) {
      const fuzzy = clubOptions
        .map((club) => ({
          club,
          score: Math.min(
            levenshteinDistance(normalized, normalizeClubSearch(club.name)),
            levenshteinDistance(normalized, normalizeClubSearch(club.label))
          ),
        }))
        .filter((item) => item.score <= 2)
        .sort((a, b) => a.score - b.score || String(a.club.name || "").length - String(b.club.name || "").length)[0];
      if (fuzzy) return fuzzy.club;
    }
    return null;
  };

  const getWorkbookCellSuggestions = (sheetId, columnIndex, inputValue) => {
    if (sheetId !== REQUIREMENTS_SHEET_ID) return [];
    const query = normalizeClubSearch(inputValue);
    if (columnIndex === 0) {
      const ranked = clubOptions
        .map((club) => {
          const name = normalizeClubSearch(club.name);
          const label = normalizeClubSearch(club.label);
          const score =
            name === query || label === query
              ? 0
              : name.startsWith(query) || label.startsWith(query)
                ? 1
                : name.includes(query) || label.includes(query)
                  ? 2
                  : query && levenshteinDistance(query, name) <= 2
                    ? 3
                    : 9;
          return { club, score };
        })
        .filter((item) => !query || item.score < 9)
        .sort((a, b) => a.score - b.score || String(a.club.name || "").localeCompare(String(b.club.name || "")))
        .slice(0, 24);
      return ranked.map(({ club }) => ({
        value: club.name,
        label: club.competition_name ? `${club.name} - ${club.competition_name}` : club.name,
      }));
    }
    if (columnIndex === 1) {
      const normalizedQuery = String(inputValue || "").trim().toLowerCase();
      return positionOptions
        .filter((position) => {
          if (!normalizedQuery) return true;
          return String(position).toLowerCase().includes(normalizedQuery);
        })
        .slice(0, 24)
        .map((position) => ({ value: position, label: position }));
    }
    return [];
  };

  const buildRequirementPayloadFromCells = (cells, clubOverride = null) => {
    const [clubCell, positionCell, profileCell, feeCell, salaryCell, , statusCell] = cells;
    const club = clubOverride || findClubFromWorkbookCell(clubCell);
    const position = String(positionCell || "").trim();
    if (!club || !position) return null;
    const status = STATUSES.includes(String(statusCell || "").trim()) ? String(statusCell).trim() : "new";
    const notes = String(profileCell || "").trim() || null;
    return {
      club,
      payload: {
        club_id: Number(club.id),
        title: `${position} need`,
        priority: "medium",
        status,
        budget_max: parseSheetNumber(feeCell),
        salary_max: parseSheetNumber(salaryCell),
        extra_info: notes,
        need: {
          position,
          notes,
        },
      },
    };
  };

  const removeLocalSheetRow = (sheetId, rowId) => {
    setSheetEdits((prev) => {
      const source = prev[sheetId];
      if (!source) return prev;
      return {
        ...prev,
        [sheetId]: {
          ...source,
          rows: (source.rows || []).filter((row) => row.id !== rowId),
        },
      };
    });
  };

  const createRequirementFromDraftRow = async (rowId, cells) => {
    if (!String(rowId || "").startsWith(`${REQUIREMENTS_SHEET_ID}-new-`)) return;
    if (savingDraftRows.includes(rowId)) return;
    const [clubCell, positionCell, profileCell, feeCell, salaryCell, , statusCell] = cells;
    const club = findClubFromWorkbookCell(clubCell);
    const position = String(positionCell || "").trim();
    if (!String(clubCell || "").trim() || !position) return;
    if (!club) {
      setMessage(`Club "${clubCell}" not found. Check the spelling or use an existing club name before saving the new requirement row.`);
      return;
    }
    const existingNeed = matchingItems
      .flatMap((item) => (item.needs || []).map((need) => ({ request: item, need })))
      .find(({ request, need }) => {
        const sameClub =
          Number(request.club_id) === Number(club.id) ||
          normalizeClubSearch(request.club_name) === normalizeClubSearch(club.name);
        return sameClub && normalizeClubSearch(need.position) === normalizeClubSearch(position);
      });
    if (existingNeed) {
      setMatchingNeedId(String(existingNeed.need.id));
      setSelectedId(existingNeed.request.id);
      removeLocalSheetRow(REQUIREMENTS_SHEET_ID, rowId);
      setMessage("This club need already exists and is now selected in the matching launcher.");
      return;
    }
    setSavingDraftRows((prev) => [...prev, rowId]);
    setMessage("");
    try {
      const built = buildRequirementPayloadFromCells(cells, club);
      if (!built) return;
      const payload = {
        ...built.payload,
        assigned_agent_id: me?.username || null,
        season: "2026",
        budget_min: null,
        deal_type: "any",
        need: {
          ...built.payload.need,
          role: null,
          age_min: null,
          age_max: null,
          preferred_foot: null,
          height_min: null,
          target_league_level: null,
          required_player_level: null,
          nationality_preferences: null,
          contract_preferences: null,
        },
      };
      const created = await postJson("/mercato/requests", payload);
      const createdNeedId = firstNeed(created)?.id;
      if (created) {
        setItems((prev) => [created, ...prev.filter((item) => item.id !== created.id)]);
        setMatchingItems((prev) => [created, ...prev.filter((item) => item.id !== created.id)]);
        setSelectedId(created.id);
      }
      if (createdNeedId) {
        setMatchingNeedId(String(createdNeedId));
        setMatchingResults([]);
        setSelectedRecommendations([]);
      }
      removeLocalSheetRow(REQUIREMENTS_SHEET_ID, rowId);
      setMessage("New club need saved and available in the matching launcher.");
      await Promise.all([loadRequests(), loadMatchingRequests()]);
    } catch (err) {
      setMessage(err.message);
    } finally {
      setSavingDraftRows((prev) => prev.filter((item) => item !== rowId));
    }
  };

  const updateRequirementFromSheetRow = async (row, latestCells) => {
    if (!row?.meta?.requestId || !row?.meta?.needId) return;
    if (savingDraftRows.includes(row.id)) return;
    const built = buildRequirementPayloadFromCells(latestCells);
    if (!built) {
      setMessage("Select an existing club and a valid category before saving this requirement.");
      return;
    }
    setSavingDraftRows((prev) => [...prev, row.id]);
    setMessage("");
    try {
      const updated = await patchJson(`/mercato/requests/${row.meta.requestId}`, {
        ...built.payload,
        need: {
          id: row.meta.needId,
          ...built.payload.need,
        },
      });
      if (updated) {
        setItems((prev) => [updated, ...prev.filter((item) => item.id !== updated.id)]);
        setMatchingItems((prev) => [updated, ...prev.filter((item) => item.id !== updated.id)]);
      }
      setMessage("Club requirement updated.");
      await Promise.all([loadRequests(), loadMatchingRequests()]);
    } catch (err) {
      setMessage(err.message);
    } finally {
      setSavingDraftRows((prev) => prev.filter((item) => item !== row.id));
    }
  };

  const handleSheetCellBlur = (sheetId, rowId, columnIndex, latestValue) => {
    if (sheetId !== REQUIREMENTS_SHEET_ID) return;
    const sheet = getSheet(sheetId);
    const row = (sheet.rows || []).find((item) => item.id === rowId);
    if (!row) return;
    const cells = [...row.cells];
    cells[columnIndex] = latestValue;
    if (String(rowId || "").startsWith(`${REQUIREMENTS_SHEET_ID}-new-`)) {
      createRequirementFromDraftRow(rowId, cells);
      return;
    }
    updateRequirementFromSheetRow(row, cells);
  };

  const addSheetRow = (sheetId) => {
    setSheetEdits((prev) => {
      const source = prev[sheetId] || baseSheets[sheetId];
      if (!source) return prev;
      const rowId = `${sheetId}-new-${Date.now()}`;
      return {
        ...prev,
        [sheetId]: {
          ...source,
          rows: [
            ...source.rows,
            makeRow(rowId, source.columns.map(() => ""), { draft: sheetId === REQUIREMENTS_SHEET_ID }),
          ],
        },
      };
    });
    if (sheetId === REQUIREMENTS_SHEET_ID) {
      setMessage("New requirement row added. Fill Club and Category, then leave the cell to save it for matching.");
    }
  };

  const deleteSheetRow = async (sheetId, rowId) => {
    const source = getSheet(sheetId);
    if (!source) return;
    const row = (source.rows || []).find((item) => item.id === rowId);
    if (!row) return;
    if (sheetId === REQUIREMENTS_SHEET_ID && row.meta?.requestId) {
      setSavingDraftRows((prev) => [...prev, row.id]);
      setMessage("");
      try {
        await deleteJson(`/mercato/requests/${row.meta.requestId}`);
        setItems((prev) => prev.filter((item) => item.id !== row.meta.requestId));
        setMatchingItems((prev) => prev.filter((item) => item.id !== row.meta.requestId));
        setSheetEdits((prev) => {
          const edited = prev[sheetId];
          if (!edited) return prev;
          return {
            ...prev,
            [sheetId]: {
              ...edited,
              rows: (edited.rows || []).filter((item) => item.id !== rowId),
            },
          };
        });
        if (String(matchingNeedId) === String(row.meta.needId)) {
          setMatchingNeedId("");
          setMatchingResults([]);
          setSelectedRecommendations([]);
        }
        setMessage("Club requirement deleted.");
        await Promise.all([loadRequests(), loadMatchingRequests()]);
      } catch (err) {
        setMessage(err.message);
      } finally {
        setSavingDraftRows((prev) => prev.filter((item) => item !== row.id));
      }
      return;
    }
    setSheetEdits((prev) => {
      return {
        ...prev,
        [sheetId]: {
          ...source,
          rows: source.rows.filter((row) => row.id !== rowId),
        },
      };
    });
  };

  useEffect(() => {
    if (!workbookHydrated || !clubOptions.length) return;
    const requirementRows = sheetEdits[REQUIREMENTS_SHEET_ID]?.rows || [];
    const draftRows = requirementRows.filter((row) => {
      if (!String(row.id || "").startsWith(`${REQUIREMENTS_SHEET_ID}-new-`)) return false;
      if (savingDraftRows.includes(row.id)) return false;
      return String(row.cells?.[0] || "").trim() && String(row.cells?.[1] || "").trim();
    });
    if (!draftRows.length) return;
    draftRows.slice(0, 3).forEach((row) => createRequirementFromDraftRow(row.id, row.cells || []));
  }, [clubOptions.length, matchingItems.length, sheetEdits, savingDraftRows, workbookHydrated]);

  const toggleSheetSort = (sheetId, columnIndex) => {
    setSortBySheet((prev) => {
      const current = prev[sheetId];
      if (current?.columnIndex === columnIndex) {
        return {
          ...prev,
          [sheetId]: { columnIndex, direction: current.direction === "asc" ? "desc" : "asc" },
        };
      }
      return { ...prev, [sheetId]: { columnIndex, direction: "asc" } };
    });
  };

  const addCustomSheet = () => {
    const label = newSheetName.trim();
    if (!label) return;
    const id = `custom-${Date.now()}`;
    const sheet = {
      id,
      label,
      custom: true,
      columns: ["Column A", "Column B", "Column C", "Column D"],
      headerClass: "bg-teal-700",
      rows: [makeRow(`${id}-row-1`, ["", "", "", ""])],
    };
    setCustomSheets((prev) => [...prev, sheet]);
    setActiveSheet(id);
    setNewSheetName("");
  };

  const deleteActiveSheet = () => {
    const sheet = baseSheets[activeSheet];
    if (!sheet?.custom) {
      setMessage("Only custom sheets can be deleted.");
      return;
    }
    setCustomSheets((prev) => prev.filter((item) => item.id !== activeSheet));
    setSheetEdits((prev) => {
      const next = { ...prev };
      delete next[activeSheet];
      return next;
    });
    setSortBySheet((prev) => {
      const next = { ...prev };
      delete next[activeSheet];
      return next;
    });
    setActiveSheet("overview");
  };

  const agentAssignments = useMemo(() => {
    const map = new Map();
    items.forEach((item) => {
      const agentName = item.assigned_agent_name || item.assigned_agent_id || "Unassigned";
      const current = map.get(agentName) || { agent: agentName, needs: 0, candidates: 0, urgent: 0 };
      current.needs += 1;
      current.candidates += (item.needs || []).reduce((total, need) => total + (need.candidates || []).length, 0);
      if (item.priority === "urgent" || item.priority === "high") current.urgent += 1;
      map.set(agentName, current);
    });
    return Array.from(map.values()).sort((a, b) => b.needs - a.needs);
  }, [items]);

  return (
    <main className="nl-page">
      <div className="mx-auto max-w-[1500px] px-4 py-8 space-y-5">
        <section className="surface-panel rounded-lg p-6">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
              <p className="nl-kicker">HD Sports market desk</p>
              <h1 className="mt-2 text-4xl font-extrabold text-slate-950">MERCATO 2026</h1>
              <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
                Run club requirements, player shortlists and agent ownership from a live workbook built for the 2026 market.
            </p>
          </div>
            <div className="flex flex-wrap gap-2">
          <button
            type="button"
                className="nl-button-secondary"
                onClick={exportExcel}
              >
                Export Excel
              </button>
              <button
                type="button"
                className="nl-button-primary"
            onClick={resetForm}
          >
            New need
          </button>
            </div>
          </div>
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
              <TextInput name="mercato_filter_club" aria-label="Club filter" value={filters.club} onChange={(e) => setFilters((p) => ({ ...p, club: e.target.value }))} />
            </div>
            <div className="space-y-2">
              <Label>Position</Label>
              <TextInput name="mercato_filter_position" aria-label="Position filter" value={filters.position} onChange={(e) => setFilters((p) => ({ ...p, position: e.target.value }))} />
            </div>
            <div className="space-y-2">
              <Label>Status</Label>
              <Select name="mercato_filter_status" aria-label="Status filter" value={filters.status} onChange={(e) => setFilters((p) => ({ ...p, status: e.target.value }))}>
                <option value="">All</option>
                {STATUSES.map((status) => <option key={status} value={status}>{status}</option>)}
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Priority</Label>
              <Select name="mercato_filter_priority" aria-label="Priority filter" value={filters.priority} onChange={(e) => setFilters((p) => ({ ...p, priority: e.target.value }))}>
                <option value="">All</option>
                {PRIORITIES.map((priority) => <option key={priority} value={priority}>{priority}</option>)}
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Agent</Label>
              <TextInput name="mercato_filter_agent" aria-label="Agent filter" value={filters.agent} onChange={(e) => setFilters((p) => ({ ...p, agent: e.target.value }))} />
            </div>
            <div className="space-y-2">
              <Label>League</Label>
              <TextInput name="mercato_filter_league" aria-label="League filter" value={filters.competition} onChange={(e) => setFilters((p) => ({ ...p, competition: e.target.value }))} />
            </div>
            <div className="space-y-2">
              <Label>Deal</Label>
              <Select name="mercato_filter_deal" aria-label="Deal filter" value={filters.deal_type} onChange={(e) => setFilters((p) => ({ ...p, deal_type: e.target.value }))}>
                <option value="">All</option>
                {DEAL_TYPES.map((deal) => <option key={deal} value={deal}>{deal}</option>)}
              </Select>
            </div>
          </div>
        </Card>

        {error ? <Card className="border-red-400/30 text-red-200">{error}</Card> : null}
        {message ? <Card className="border-primary/30 text-slate-200">{message}</Card> : null}

        <section className="surface-panel overflow-hidden rounded-lg">
          <div className="border-b border-slate-200 bg-slate-50">
            <div className="flex gap-2 overflow-x-auto p-2">
              {workbookTabs.map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  className={`whitespace-nowrap rounded-md border px-4 py-2 text-sm font-extrabold transition ${
                    activeSheet === tab.id
                      ? "border-[#3A8967]/40 bg-[#2F7D5C]/20 text-[#DDF3E8] shadow-[inset_0_0_0_1px_rgba(85,154,120,0.12)]"
                      : "border-transparent text-slate-600 hover:border-white/10 hover:bg-white/[0.07] hover:text-white"
                  }`}
                  onClick={() => setActiveSheet(tab.id)}
                >
                  {tab.label} <span className="ml-2 text-xs text-slate-400">{tab.count}</span>
                </button>
              ))}
            </div>
            <div className="flex flex-wrap items-center gap-2 border-t border-slate-200 px-2 py-2">
              <input
                className="nl-field h-10 max-w-xs bg-white"
                name="new_sheet_name"
                aria-label="New sheet name"
                value={newSheetName}
                onChange={(event) => setNewSheetName(event.target.value)}
                placeholder="New sheet name"
              />
              <button type="button" className="nl-button-primary" onClick={addCustomSheet}>
                Add sheet
              </button>
              <button
                type="button"
                className="nl-button-secondary"
                onClick={deleteActiveSheet}
                disabled={!baseSheets[activeSheet]?.custom}
              >
                Delete sheet
              </button>
              <p className="ml-auto text-xs font-bold text-slate-500">
                Click a header to sort. Edit any cell directly.
              </p>
            </div>
          </div>

          {loading ? <div className="p-5 text-sm font-semibold text-slate-600">Loading needs...</div> : null}
          {!loading && items.length === 0 && ["requirements", "matching"].includes(activeSheet) ? (
            <div className="p-5">
              <p className="font-semibold text-slate-950">No active club needs match this view.</p>
              <p className="mt-1 text-sm text-slate-500">Create a new requirement or adjust the filters to surface existing opportunities.</p>
            </div>
          ) : null}

          <EditableWorkbookSheet
            sheet={activeWorkbookSheet}
            sortState={sortBySheet[activeSheet]}
            onSort={(columnIndex) => toggleSheetSort(activeSheet, columnIndex)}
            onCellChange={(rowId, columnIndex, value) => updateSheetCell(activeSheet, rowId, columnIndex, value)}
            onCellBlur={(rowId, columnIndex, value) => handleSheetCellBlur(activeSheet, rowId, columnIndex, value)}
            getCellSuggestions={getWorkbookCellSuggestions}
            onAddRow={() => addSheetRow(activeSheet)}
            onDeleteRow={(rowId) => deleteSheetRow(activeSheet, rowId)}
          />
        </section>

        <section className="surface-panel overflow-hidden rounded-lg">
          <div className="border-b border-slate-200 bg-white px-5 py-5">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
              <div>
                <p className="nl-kicker">Matching engine</p>
                <h2 className="mt-2 text-3xl font-black text-slate-950">Generate a shortlist</h2>
                <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
                  Select a club need, refine the market filters and review five recommended players before adding the best fits to prospects.
                </p>
              </div>
              <div className="grid grid-cols-4 gap-2 text-center">
                {["Need", "Filters", "Run", "Select"].map((step, index) => (
                  <div key={step} className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2">
                    <p className="text-[10px] font-black uppercase tracking-[0.16em] text-teal-700">Step {index + 1}</p>
                    <p className="mt-1 text-xs font-black text-slate-800">{step}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="grid gap-0 lg:grid-cols-[420px_1fr]">
            <div className="border-b border-slate-200 bg-slate-50 p-5 lg:border-b-0 lg:border-r">
              <div className="space-y-5">
                <div className="space-y-2">
                  <Label>1. Select need</Label>
                  <select
                    className="nl-field bg-white"
                    value={matchingNeedId}
                    onChange={(event) => {
                      setMatchingNeedId(event.target.value);
                      setMatchingResults([]);
                      setSelectedRecommendations([]);
                    }}
                    aria-label="Select a Mercato need for matching"
                  >
                    {needOptions.length ? null : <option value="">No need available</option>}
                    {needOptions.map((option) => (
                      <option key={option.needId} value={option.needId}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  {matchingNeed ? (
                    <div className="rounded-lg border border-slate-200 bg-white p-4">
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex min-w-0 items-start gap-3">
                          <ClubLogo name={matchingNeed.request.club_name} className="h-10 w-10" />
                          <div className="min-w-0">
                          <p className="text-lg font-black text-slate-950">{matchingNeed.request.club_name || "Unknown club"}</p>
                          <p className="mt-1 text-sm font-bold text-slate-500">
                            {[matchingNeed.need.position, matchingNeed.request.competition_name, matchingNeed.request.deal_type].filter(Boolean).join(" - ")}
                          </p>
                          </div>
                        </div>
                        <Badge tone={priorityTone(matchingNeed.request.priority)}>{matchingNeed.request.priority || "medium"}</Badge>
                      </div>
                      <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
                        <div className="rounded-md bg-slate-50 p-2">
                          <p className="font-black uppercase tracking-[0.1em] text-slate-400">Age</p>
                          <p className="mt-1 font-black text-slate-900">
                            {matchingNeed.need.age_min || "-"} - {matchingNeed.need.age_max || "-"}
                          </p>
                        </div>
                        <div className="rounded-md bg-slate-50 p-2">
                          <p className="font-black uppercase tracking-[0.1em] text-slate-400">Budget</p>
                          <p className="mt-1 font-black text-slate-900">{formatMoney(matchingNeed.request.budget_max)}</p>
                        </div>
                        <div className="rounded-md bg-slate-50 p-2">
                          <p className="font-black uppercase tracking-[0.1em] text-slate-400">Level</p>
                          <p className="mt-1 font-black text-slate-900">{formatMetric(matchingNeed.need.required_player_level)}</p>
                        </div>
                      </div>
                    </div>
                  ) : null}
                </div>

                <div className="space-y-3">
                  <Label>2. Add filters</Label>
                  <div className="grid grid-cols-2 gap-2">
                    <input
                      className="nl-field bg-white"
                      type="number"
                      value={matchingFilters.age_min}
                      onChange={(event) => updateMatchingFilter("age_min", event.target.value)}
                      placeholder="Min age"
                      aria-label="Minimum age"
                    />
                    <input
                      className="nl-field bg-white"
                      type="number"
                      value={matchingFilters.age_max}
                      onChange={(event) => updateMatchingFilter("age_max", event.target.value)}
                      placeholder="Max age"
                      aria-label="Maximum age"
                    />
                    <input
                      className="nl-field bg-white"
                      type="number"
                      value={matchingFilters.min_minutes}
                      onChange={(event) => updateMatchingFilter("min_minutes", event.target.value)}
                      placeholder="Min minutes"
                      aria-label="Minimum minutes"
                    />
                    <input
                      className="nl-field bg-white"
                      type="number"
                      value={matchingFilters.min_match_score}
                      onChange={(event) => updateMatchingFilter("min_match_score", event.target.value)}
                      placeholder="Min score"
                      aria-label="Minimum match score"
                    />
                  </div>

                  <div className="rounded-lg border border-slate-200 bg-white p-3">
                    <input
                      className="nl-field h-10 bg-slate-50"
                      value={matchingLeagueQuery}
                      onChange={(event) => setMatchingLeagueQuery(event.target.value)}
                      placeholder="Search leagues"
                      aria-label="Search matching leagues"
                    />
                    {matchingCompetitions.length ? (
                      <div className="mt-3 flex flex-wrap gap-2">
                        {matchingCompetitions.map((competition) => (
                          <button
                            key={competition}
                            type="button"
                            className="rounded-full bg-teal-700 px-3 py-1 text-xs font-black text-white"
                            onClick={() => toggleMatchingCompetition(competition)}
                          >
                            {competition} x
                          </button>
                        ))}
                      </div>
                    ) : null}
                    <div className="mt-3 max-h-36 overflow-auto">
                      <div className="flex flex-wrap gap-2">
                        {filteredMatchingCompetitions.map((competition) => (
                          <button
                            key={competition}
                            type="button"
                            className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-bold text-slate-600 transition hover:border-teal-500 hover:text-teal-800"
                            onClick={() => toggleMatchingCompetition(competition)}
                          >
                            {competition}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>3. Run</Label>
                  <button
                    type="button"
                    className="nl-button-primary w-full justify-center disabled:opacity-50"
                    onClick={runMatchingPreview}
                    disabled={!matchingNeedId || matchingBusy}
                  >
                    {matchingBusy ? "Running matching..." : "Run matching"}
                  </button>
                </div>
              </div>
            </div>

            <div className="bg-white p-5">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div>
                  <Label>4. Review recommendations</Label>
                  <h3 className="mt-1 text-xl font-black text-slate-950">Top 5 proposed players</h3>
                </div>
                <button
                  type="button"
                  className="nl-button-secondary disabled:opacity-50"
                  onClick={addSelectedRecommendations}
                  disabled={!selectedRecommendations.length || addingRecommendations}
                >
                  {addingRecommendations ? "Adding..." : `Add selected to prospects (${selectedRecommendations.length})`}
                </button>
              </div>

              {matchingBusy ? (
                <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                  {[0, 1, 2].map((item) => (
                    <div key={item} className="h-72 animate-pulse rounded-lg border border-slate-200 bg-slate-100" />
                  ))}
                </div>
              ) : null}

              {!matchingBusy && !matchingResults.length ? (
                <div className="mt-5 rounded-lg border border-dashed border-slate-300 bg-slate-50 p-8 text-center">
                  <p className="text-lg font-black text-slate-950">No recommendation loaded</p>
                  <p className="mt-2 text-sm text-slate-500">Generate recommendations to review five selectable player cards for this club need.</p>
                </div>
              ) : null}

              {!matchingBusy && matchingResults.length ? (
                <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                  {matchingResults.map((candidate) => {
                    const key = recommendationKey(candidate);
                    return (
                      <MatchingPlayerCard
                        key={key}
                        candidate={candidate}
                        selected={selectedRecommendations.includes(key)}
                        onToggle={() => toggleRecommendation(candidate)}
                      />
                    );
                  })}
                </div>
              ) : null}
            </div>
          </div>
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
                  {positionOptions.map((position) => <option key={position} value={position}>{position}</option>)}
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
