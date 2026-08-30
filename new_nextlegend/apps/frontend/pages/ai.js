import { useEffect, useMemo, useState } from "react";
import { fetchJson, fetchJsonCached, postJson, patchJson } from "@/lib/api";
import {
  DEFAULT_SCOUTING_SEASON,
  withDefaultSeason,
} from "@/lib/scoutingFilters";

const TM_BASE_URL = "https://www.transfermarkt.com";

const Card = ({ children, className = "", ...rest }) => (
  <div
    className={`glass-panel min-w-0 rounded-xl border border-white/5 p-4 ${className}`}
    {...rest}
  >
    {children}
  </div>
);

const Label = ({ children }) => (
  <label className="text-xs uppercase tracking-[0.2em] text-slate-400">
    {children}
  </label>
);

const Input = ({ value, onChange, placeholder, ...rest }) => (
  <input
    className="nl-field"
    value={value}
    onChange={onChange}
    placeholder={placeholder}
    {...rest}
  />
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

const iconPaths = {
  spark: ["M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z", "M19 3v4", "M21 5h-4", "M5 17v4", "M7 19H3"],
  plus: ["M12 5v14", "M5 12h14"],
  edit: ["M12 20h9", "M16.5 3.5a2.1 2.1 0 0 1 3 3L8 18l-4 1 1-4z"],
  send: ["M22 2 11 13", "M22 2l-7 20-4-9-9-4z"],
  user: ["M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2", "M12 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8"],
  bot: ["M12 8V4", "M8 4h8", "M6 8h12v9a3 3 0 0 1-3 3H9a3 3 0 0 1-3-3z", "M9 13h.01", "M15 13h.01", "M10 17h4"],
  database: ["M4 6c0-2 16-2 16 0s-16 2-16 0", "M4 6v6c0 2 16 2 16 0V6", "M4 12v6c0 2 16 2 16 0v-6"],
  gauge: ["M4 14a8 8 0 0 1 16 0", "M12 14l4-4", "M8 18h8"],
  clock: ["M12 7v5l3 2", "M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0"],
  external: ["M14 3h7v7", "M10 14 21 3", "M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5"],
  filter: ["M4 5h16", "M7 12h10", "M10 19h4"],
};

const Icon = ({ name, className = "h-4 w-4" }) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.8"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
    aria-hidden="true"
  >
    {(iconPaths[name] || iconPaths.spark).map((path) => (
      <path key={path} d={path} />
    ))}
  </svg>
);

const MetricTile = ({ label, value, tone = "default" }) => (
  <div
    className={`min-w-0 rounded-lg border px-3 py-3 ${
      tone === "green"
        ? "border-[#3A8967]/35 bg-[#2F7D5C]/15"
        : "border-white/10 bg-white/[0.025]"
    }`}
  >
    <p className="break-words text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
      {label}
    </p>
    <p className={`mt-1 break-words text-sm font-semibold leading-5 ${tone === "green" ? "text-[#8CC7A7]" : "text-slate-100"}`}>
      {value}
    </p>
  </div>
);

const PromptChip = ({ children, onClick }) => (
  <button
    type="button"
    onClick={onClick}
    className="rounded-md border border-white/10 bg-white/[0.03] px-3 py-2 text-left text-xs font-medium text-slate-300 transition hover:border-[#3A8967]/35 hover:bg-[#2F7D5C]/12 hover:text-white"
  >
    {children}
  </button>
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

const extractTmFields = (row) => {
  const tmFields = {};
  Object.entries(row || {}).forEach(([key, value]) => {
    if (key.startsWith("tm_")) {
      tmFields[key] = value;
    }
  });
  return tmFields;
};

export default function AIPage() {
  const [mode, setMode] = useState("scout");
  const [prompt, setPrompt] = useState("");
  const [language, setLanguage] = useState("auto");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [userId, setUserId] = useState("");
  const [currentUser, setCurrentUser] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [editingConversationId, setEditingConversationId] = useState(null);
  const [editingTitle, setEditingTitle] = useState("");

  const [playerQuery, setPlayerQuery] = useState("");
  const [playerResults, setPlayerResults] = useState([]);
  const [showResults, setShowResults] = useState(false);
  const [selectedPlayerId, setSelectedPlayerId] = useState("");
  const [selectedPlayerSeasonId, setSelectedPlayerSeasonId] = useState("");
  const [seasons, setSeasons] = useState([]);
  const [selectedSeason, setSelectedSeason] = useState(DEFAULT_SCOUTING_SEASON);
  const [usageTotal, setUsageTotal] = useState(null);
  const [usageConversation, setUsageConversation] = useState(null);
  const [usageError, setUsageError] = useState("");
  const isUsageAdmin = currentUser?.role === "admin" && currentUser?.username === "yrachid";

  useEffect(() => {
    fetchJson("/auth/me")
      .then((res) => {
        if (res?.username) {
          setCurrentUser(res);
          setUserId(res.username);
        }
      })
      .catch(() => {});
  }, []);

  const loadConversations = async (targetId = null) => {
    if (!userId) return;
    const res = await fetchJson("/ai/conversations", { user_id: userId });
    const items = res.items || [];
    setConversations(items);
    if (targetId) {
      setActiveConversationId(targetId);
    } else if (!activeConversationId && items.length > 0) {
      setActiveConversationId(items[0].id);
    }
  };

  const createConversation = async (desiredMode = mode) => {
    if (!userId) return null;
    const res = await postJson("/ai/conversations", {
      user_id: userId,
      mode: desiredMode,
    });
    setConversations((prev) => [res, ...prev]);
    setActiveConversationId(res.id);
    setMessages([]);
    setMode(res.mode || desiredMode);
    return res;
  };

  const startEditConversation = (conversation) => {
    setEditingConversationId(conversation.id);
    setEditingTitle(conversation.title || "");
  };

  const cancelEditConversation = () => {
    setEditingConversationId(null);
    setEditingTitle("");
  };

  const saveConversationTitle = async (conversationId) => {
    if (!userId) return;
    const title = editingTitle.trim();
    try {
      const updated = await patchJson(`/ai/conversations/${conversationId}`, {
        user_id: userId,
        title: title || null,
      });
      setConversations((prev) =>
        prev.map((item) => (item.id === conversationId ? updated : item))
      );
      if (activeConversationId === conversationId) {
        setMessages((prev) => prev);
      }
      cancelEditConversation();
    } catch (err) {
      setError(err.message);
    }
  };

  const loadConversationMessages = async (conversationId) => {
    if (!userId || !conversationId) return;
    const res = await fetchJson(`/ai/conversations/${conversationId}`, {
      user_id: userId,
    });
    setMessages(res.messages || []);
    setMode(res.conversation?.mode || "scout");
  };

  useEffect(() => {
    if (!userId) return;
    loadConversations();
  }, [userId]);

  useEffect(() => {
    if (!activeConversationId) return;
    loadConversationMessages(activeConversationId);
  }, [activeConversationId]);

  useEffect(() => {
    if (!isUsageAdmin) {
      setUsageTotal(null);
      setUsageConversation(null);
      setUsageError("");
      return;
    }
    loadUsage(activeConversationId);
  }, [activeConversationId, messages.length, userId, isUsageAdmin]);

  useEffect(() => {
    fetchJsonCached("/meta/seasons")
      .then((data) => setSeasons(withDefaultSeason(data || [])))
      .catch(() => setSeasons([]));
  }, []);

  useEffect(() => {
    if (!playerQuery || playerQuery.trim().length < 2) {
      setPlayerResults([]);
      setSelectedPlayerId("");
      setSelectedPlayerSeasonId("");
      return;
    }
    const handle = setTimeout(async () => {
      try {
        const res = await fetchJson("/players", {
          q: playerQuery.trim(),
          season: selectedSeason || undefined,
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
  }, [playerQuery, selectedSeason]);

  const playerOptions = useMemo(() => {
    return playerResults.map((p) => ({
      id: String(p.id),
      seasonId: p.player_season_id ? String(p.player_season_id) : "",
      label: `${p.name} - ${p.team || "—"} - ${p.competition_name || "—"} - ${p.calendar || "—"}`,
    }));
  }, [playerResults]);

  const normalizeName = (value) => {
    return value
      ? value
          .normalize("NFD")
          .replace(/[\u0300-\u036f]/g, "")
          .replace(/[^a-z0-9]/gi, "")
          .toLowerCase()
      : "";
  };

  const mergeCandidates = (payload) => {
    const shortlist = payload?.shortlist || [];
    const candidates = payload?.candidates || [];
    const shortlistMap = new Map();
    const shortlistNameMap = new Map();
    shortlist.forEach((row) => {
      if (row.player_id != null) {
        shortlistMap.set(String(row.player_id), row);
      }
      if (row.player_name) {
        shortlistNameMap.set(normalizeName(row.player_name), row);
      }
    });
    return candidates.map((candidate) => {
      let extra = null;
      if (candidate.player_id) {
        extra = shortlistMap.get(String(candidate.player_id)) || null;
      } else if (candidate.player_name) {
        extra = shortlistNameMap.get(normalizeName(candidate.player_name)) || null;
      }
      return { ...candidate, ...(extra || {}) };
    });
  };

  const formatTokens = (value) => {
    if (value === null || value === undefined) return "--";
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return "--";
    return numeric.toLocaleString();
  };

  const formatUsd = (value) => {
    if (value === null || value === undefined) return "--";
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return "--";
    if (numeric === 0) return "$0.000";
    return `$${numeric.toFixed(3)}`;
  };

  const loadUsage = async (conversationId) => {
    if (!userId || !isUsageAdmin) return;
    setUsageError("");
    try {
      const [total, current] = await Promise.all([
        fetchJson("/ai/usage", { user_id: userId }),
        conversationId
          ? fetchJson("/ai/usage", { user_id: userId, conversation_id: conversationId })
          : Promise.resolve(null),
      ]);
      setUsageTotal(total || null);
      setUsageConversation(current || null);
    } catch (err) {
      console.error(err);
      setUsageError("Unable to load usage.");
    }
  };

  const handleSend = async () => {
    if (!prompt.trim() || loading) return;
    if (mode === "player" && !selectedPlayerId) {
      setError("Select a player first.");
      return;
    }
    setLoading(true);
    setError("");
    const content = prompt;
    setPrompt("");

    let conversationId = activeConversationId;
    if (!conversationId) {
      const newConv = await createConversation(mode);
      conversationId = newConv?.id;
    }
    if (!conversationId) {
      setLoading(false);
      setError("Unable to start a new conversation.");
      return;
    }

    const tempUserMessage = {
      id: `temp-${Date.now()}`,
      role: "user",
      content,
    };
    const pendingMessage = {
      id: `pending-${Date.now()}`,
      role: "assistant",
      content: "",
      pending: true,
    };
    setMessages((prev) => [...prev, tempUserMessage, pendingMessage]);
    try {
      const res = await postJson(`/ai/conversations/${conversationId}/messages`, {
        user_id: userId,
        prompt: content,
        mode,
        player_id: mode === "player" ? Number(selectedPlayerId) : null,
        player_season_id:
          mode === "player" && selectedPlayerSeasonId
            ? Number(selectedPlayerSeasonId)
            : null,
        season: mode === "scout" ? selectedSeason || null : null,
        language,
      });
      setMessages((prev) => [
        ...prev.filter(
          (msg) => msg.id !== tempUserMessage.id && msg.id !== pendingMessage.id
        ),
        res.user_message,
        res.assistant_message,
      ]);
      setConversations((prev) => {
        const updated = res.conversation;
        const next = prev.filter((item) => item.id !== updated.id);
        return [updated, ...next];
      });
    } catch (err) {
      setMessages((prev) =>
        prev.filter(
          (msg) => msg.id !== tempUserMessage.id && msg.id !== pendingMessage.id
        )
      );
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenReport = async (candidate) => {
    if (!candidate) return;
    const seasonQuery =
      candidate.player_season_id != null
        ? `?player_id=${candidate.player_id}&player_season_id=${candidate.player_season_id}`
        : `?player_id=${candidate.player_id}`;
    if (candidate.player_id) {
      window.open(
        `/report${seasonQuery}`,
        "_blank",
        "noopener,noreferrer"
      );
      return;
    }
    if (!candidate.player_name) return;
    const popup = window.open("", "_blank", "noopener,noreferrer");
    try {
      const res = await fetchJson("/players", {
        q: candidate.player_name,
        season: selectedSeason || undefined,
      });
      const first = (res || [])[0];
      if (first?.id) {
        const firstQuery = first.player_season_id
          ? `?player_id=${first.id}&player_season_id=${first.player_season_id}`
          : `?player_id=${first.id}`;
        if (popup) {
          popup.location = `/report${firstQuery}`;
        } else {
          window.open(
            `/report${firstQuery}`,
            "_blank",
            "noopener,noreferrer"
          );
        }
      } else if (popup) {
        popup.close();
      }
    } catch (err) {
      console.error(err);
      setError("Unable to open player report.");
      if (popup) {
        popup.close();
      }
    }
  };

  const activeConversation = conversations.find((item) => item.id === activeConversationId);
  const activeModeLabel = mode === "player" ? "Player report" : "Scout advisor";
  const promptSuggestions = [
    "Build a shortlist of U23 centre backs ready for a Big 5 move.",
    "Identify undervalued wide forwards with strong ball progression.",
    "Summarize the risk profile for a player before a recruitment meeting.",
  ];
  const contextPanels = (
    <>
      <Card className="space-y-2.5">
        <div className="flex items-center gap-2">
          <span className="flex h-8 w-8 items-center justify-center rounded-md border border-[#3A8967]/30 bg-[#2F7D5C]/15 text-[#8CC7A7]">
            <Icon name="database" className="h-4 w-4" />
          </span>
          <div className="min-w-0">
            <p className="text-sm font-semibold text-white">Context lock</p>
            <p className="text-xs text-slate-500">Current query scope</p>
          </div>
        </div>
        <div className="overflow-hidden rounded-md border border-white/10 bg-white/[0.025]">
          {[
            ["Mode", activeModeLabel],
            ["Language", language === "auto" ? "Auto" : language.toUpperCase()],
            ["Selected player", selectedPlayerId ? "Locked" : mode === "player" ? "Required" : "Not needed"],
          ].map(([label, value], index) => (
            <div
              key={label}
              className={`flex min-w-0 items-center justify-between gap-3 px-3 py-2 ${
                index === 0 ? "bg-[#2F7D5C]/10" : "border-t border-white/5"
              }`}
            >
              <span className="shrink-0 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                {label}
              </span>
              <span className={`min-w-0 truncate text-right text-xs font-semibold ${index === 0 ? "text-[#8CC7A7]" : "text-slate-200"}`}>
                {value}
              </span>
            </div>
          ))}
        </div>
      </Card>

      {isUsageAdmin ? (
        <Card className="space-y-3 text-xs text-slate-300">
          <div className="flex items-center justify-between gap-3">
            <span className="inline-flex items-center gap-2 font-semibold text-white">
              <Icon name="gauge" className="h-4 w-4 text-[#8CC7A7]" />
              Usage
            </span>
            <span className="truncate text-slate-500">{usageTotal?.model || "gpt-4o"}</span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <MetricTile label="This chat" value={`${formatTokens(usageConversation?.total_tokens)} tokens`} />
            <MetricTile label="Chat cost" value={formatUsd(usageConversation?.estimated_cost_usd)} />
            <MetricTile label="Total" value={`${formatTokens(usageTotal?.total_tokens)} tokens`} />
            <MetricTile label="Total cost" value={formatUsd(usageTotal?.estimated_cost_usd)} />
          </div>
          <div className="rounded-md border border-white/10 bg-white/[0.025] px-3 py-2 text-[11px] leading-5 text-slate-500">
            Remaining balance is not available via OpenAI API.
          </div>
          {usageError ? (
            <div className="rounded-md border border-amber-400/25 bg-amber-500/10 px-3 py-2 text-[11px] text-amber-200">
              {usageError}
            </div>
          ) : null}
        </Card>
      ) : null}

      <Card className="space-y-3">
        <div className="flex items-center gap-2">
          <span className="flex h-8 w-8 items-center justify-center rounded-md border border-white/10 bg-white/[0.035] text-slate-300">
            <Icon name="filter" className="h-4 w-4" />
          </span>
          <div className="min-w-0">
            <p className="text-sm font-semibold text-white">Good brief inputs</p>
            <p className="text-xs text-slate-500">Keep answers operational.</p>
          </div>
        </div>
        <div className="space-y-2 text-xs leading-5 text-slate-400">
          <p>Define role, league level, budget, age range and minutes threshold.</p>
          <p>For player reports, lock one player before asking for tactical or market analysis.</p>
          <p>Open candidate cards to jump directly to the full player report.</p>
        </div>
      </Card>
    </>
  );

  return (
    <main className="nl-page px-4 py-8">
      <div className="mx-auto max-w-[1760px] space-y-5">
        <div className="grid grid-cols-1 gap-5 xl:grid-cols-[320px_minmax(0,1fr)] 2xl:grid-cols-[340px_minmax(0,1fr)]">
          <aside className="min-w-0 space-y-3 xl:sticky xl:top-4 xl:max-h-[calc(100vh-2rem)] xl:self-start xl:overflow-auto xl:pr-1">
            <button
              type="button"
              className="nl-button-primary w-full justify-center"
              onClick={() => createConversation(mode)}
              disabled={!userId}
            >
              <Icon name="plus" className="h-4 w-4" />
              New conversation
            </button>

            <Card className="overflow-hidden p-0">
              <div className="border-b border-white/10 px-4 py-3">
                <p className="nl-kicker">Conversation memory</p>
              </div>
              <div className="max-h-[360px] space-y-1 overflow-auto p-2">
                {conversations.length === 0 ? (
                  <div className="rounded-md border border-dashed border-white/10 bg-white/[0.025] px-3 py-8 text-center">
                    <p className="text-sm font-semibold text-slate-200">No conversation yet</p>
                    <p className="mt-1 text-xs text-slate-500">Start with a brief and it will appear here.</p>
                  </div>
                ) : (
                  conversations.map((conv) => {
                    const isActive = conv.id === activeConversationId;
                    const isEditing = editingConversationId === conv.id;
                    return (
                      <div
                        key={conv.id}
                        className={`rounded-md border p-2 transition ${
                          isActive
                            ? "border-[#3A8967]/40 bg-[#2F7D5C]/16"
                            : "border-transparent bg-transparent hover:border-white/10 hover:bg-white/[0.035]"
                        }`}
                      >
                        {isEditing ? (
                          <div className="space-y-2">
                            <Input
                              value={editingTitle}
                              onChange={(event) => setEditingTitle(event.target.value)}
                              placeholder="Conversation title"
                            />
                            <div className="flex items-center gap-2">
                              <button
                                type="button"
                                className="nl-button-primary px-3 py-1.5 text-xs"
                                onClick={() => saveConversationTitle(conv.id)}
                              >
                                Save
                              </button>
                              <button
                                type="button"
                                className="nl-button-secondary px-3 py-1.5 text-xs"
                                onClick={cancelEditConversation}
                              >
                                Cancel
                              </button>
                            </div>
                          </div>
                        ) : (
                          <div className="flex items-start gap-2">
                            <button
                              type="button"
                              onClick={() => setActiveConversationId(conv.id)}
                              className="min-w-0 flex-1 text-left"
                            >
                              <div className="truncate text-sm font-semibold text-slate-100">
                                {conv.title || "New chat"}
                              </div>
                              <div className="mt-1 flex items-center gap-2 text-[11px] font-medium text-slate-500">
                                <span>{conv.mode === "player" ? "Player report" : "Scout advisor"}</span>
                                <span className="h-1 w-1 rounded-full bg-slate-700" />
                                <span>{conv.message_count ?? "Open"}</span>
                              </div>
                            </button>
                            <button
                              type="button"
                              className="rounded-md p-1.5 text-slate-500 transition hover:bg-white/[0.06] hover:text-white"
                              onClick={() => startEditConversation(conv)}
                              aria-label="Edit conversation title"
                            >
                              <Icon name="edit" className="h-3.5 w-3.5" />
                            </button>
                          </div>
                        )}
                      </div>
                    );
                  })
                )}
              </div>
            </Card>

            {contextPanels}
          </aside>

          <section className="min-w-0 space-y-4">
            <header className="overflow-hidden rounded-lg border border-white/10 bg-[#070807]">
              <div className="relative grid gap-5 p-5 lg:grid-cols-[1fr_auto] lg:items-end">
                <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-[#559A78]/70 to-transparent" />
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="inline-flex items-center gap-2 rounded-md border border-[#3A8967]/35 bg-[#2F7D5C]/15 px-3 py-1.5 text-xs font-semibold text-[#8CC7A7]">
                      <Icon name="spark" className="h-3.5 w-3.5" />
                      AI Assistant
                    </span>
                    <span className="rounded-md border border-white/10 bg-white/[0.03] px-3 py-1.5 text-xs font-semibold text-slate-400">
                      {activeModeLabel}
                    </span>
                  </div>
                  <h1 className="mt-4 text-3xl font-semibold tracking-tight text-white md:text-4xl">
                    Scouting brief assistant
                  </h1>
                  <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
                    Query the Next Legend database, structure recruitment thinking, and turn player context into usable scouting briefs.
                  </p>
                </div>
                <div className={`grid gap-2 ${isUsageAdmin ? "grid-cols-3 sm:min-w-[420px]" : "grid-cols-2 sm:min-w-[300px]"}`}>
                  <MetricTile label="Season" value={selectedSeason || "All"} tone="green" />
                  <MetricTile label="Messages" value={messages.length} />
                  {isUsageAdmin ? <MetricTile label="Model" value={usageTotal?.model || "gpt-4o"} /> : null}
                </div>
              </div>
            </header>

            <Card className="overflow-visible p-0">
              <div className="flex flex-col gap-4 border-b border-white/10 px-4 py-4 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <p className="nl-kicker">Assistant context</p>
                  <h2 className="mt-1 text-lg font-semibold text-white">
                    {activeConversation?.title || "Untitled scouting brief"}
                  </h2>
                </div>
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-3 lg:min-w-[520px]">
                  <div>
                    <Label>Mode</Label>
                    <Select
                      id="ai-mode"
                      value={mode}
                      onChange={(e) => setMode(e.target.value)}
                    >
                      <option value="scout">Scout Advisor</option>
                      <option value="player">Player Agent / Report</option>
                    </Select>
                  </div>
                  <div>
                    <Label>Language</Label>
                    <Select
                      id="ai-language"
                      value={language}
                      onChange={(e) => setLanguage(e.target.value)}
                    >
                      <option value="auto">Auto</option>
                      <option value="en">English</option>
                      <option value="fr">French</option>
                    </Select>
                  </div>
                  <div>
                    <Label>Season</Label>
                    <Select
                      id="ai-season"
                      value={selectedSeason}
                      onChange={(e) => {
                        setSelectedSeason(e.target.value);
                        setPlayerResults([]);
                        setSelectedPlayerId("");
                        setSelectedPlayerSeasonId("");
                        if (mode === "player") {
                          setPlayerQuery("");
                        }
                      }}
                    >
                      <option value="">All seasons</option>
                      {seasons.map((season) => (
                        <option key={season} value={season}>
                          {season}
                        </option>
                      ))}
                    </Select>
                  </div>
                </div>
              </div>
            </Card>

            <Card className="min-h-[520px] overflow-hidden p-0">
              <div className="border-b border-white/10 bg-white/[0.018] px-4 py-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-semibold text-white">Conversation</p>
                    <p className="mt-1 text-xs text-slate-500">Chat history, candidate cards and database context.</p>
                  </div>
                  <span className="rounded-md border border-white/10 bg-black/20 px-2.5 py-1 text-xs font-semibold text-slate-400">
                    {messages.length || 0} messages
                  </span>
                </div>
              </div>

              <div className="space-y-6 p-4 md:p-5">
                {messages.length === 0 ? (
                  <div className="grid min-h-[360px] place-items-center rounded-lg border border-dashed border-white/10 bg-black/[0.16] px-4 py-10 text-center">
                    <div className="max-w-xl">
                      <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-lg border border-[#3A8967]/35 bg-[#2F7D5C]/15 text-[#8CC7A7]">
                        <Icon name="spark" className="h-5 w-5" />
                      </div>
                      <h3 className="mt-4 text-lg font-semibold text-white">Start a structured scouting brief</h3>
                      <p className="mt-2 text-sm leading-6 text-slate-400">
                        Use a focused prompt, select the right context, and the assistant will answer from the scouting database.
                      </p>
                      <div className="mt-5 grid gap-2 text-left sm:grid-cols-3">
                        {promptSuggestions.map((suggestion) => (
                          <PromptChip key={suggestion} onClick={() => setPrompt(suggestion)}>
                            {suggestion}
                          </PromptChip>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  messages.map((message, idx) => {
                    const isUser = message.role === "user";
                    const isPending = message.pending;
                    const payload = message.payload || {};
                    const mergedCandidates = payload?.candidates
                      ? mergeCandidates(payload)
                      : [];
                    return (
                      <div
                        key={message.id || idx}
                        className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}
                      >
                        {!isUser ? (
                        <div
                          className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-white/10 bg-white/[0.04] text-slate-300"
                        >
                          <Icon name="bot" className="h-4 w-4" />
                        </div>
                        ) : null}
                        <div
                          className={`min-w-0 rounded-lg border px-4 py-3 ${
                            isUser
                              ? "max-w-[min(760px,82%)] border-[#3A8967]/35 bg-[#2F7D5C]/16 text-slate-100"
                              : "w-full max-w-[1180px] border-white/10 bg-[#0A0C0B]/80 text-slate-200"
                          }`}
                        >
                          {isPending ? (
                            <div className="flex items-center gap-3 text-sm text-slate-400" aria-label="Loading">
                              <div className="typing-dots">
                                <span />
                                <span />
                                <span />
                              </div>
                              <span>Composing answer</span>
                            </div>
                          ) : (
                            <p className="whitespace-pre-wrap text-sm leading-7">
                              {message.content}
                            </p>
                          )}

                          {payload?.filters && !isPending ? (
                            <details className="mt-4 rounded-md border border-white/10 bg-black/[0.22] px-3 py-2 text-xs text-slate-400">
                              <summary className="cursor-pointer font-semibold text-slate-300">
                                Applied filters
                              </summary>
                              <pre className="mt-3 max-h-56 overflow-auto whitespace-pre-wrap text-[11px] text-slate-500">
                                {JSON.stringify(payload.filters, null, 2)}
                              </pre>
                            </details>
                          ) : null}

                          {mergedCandidates.length > 0 && !isPending ? (
                            <div className="mt-4 space-y-3">
                              {mergedCandidates.map((candidate, cardIndex) => {
                                const tmFields = extractTmFields(candidate);
                                const tmPhotoUrl = toAbsoluteUrl(
                                  tmFields.tm_profile_image_url ||
                                    tmFields.profile_image_url
                                );
                                const tmProfileUrl = toAbsoluteUrl(
                                  tmFields.tm_profile_url || candidate.tm_profile_url
                                );
                                const tmMarketValue = formatCompactNumber(
                                  tmFields.tm_market_value
                                );
                                const tmAgentName = tmFields.tm_agent_name;
                                const reportHref =
                                  candidate.player_id !== null &&
                                  candidate.player_id !== undefined
                                    ? `/report?player_id=${candidate.player_id}${
                                        candidate.player_season_id != null
                                          ? `&player_season_id=${candidate.player_season_id}`
                                          : ""
                                      }`
                                    : null;
                                return (
                                  <div
                                    key={`${candidate.player_id || candidate.player_name}-${cardIndex}`}
                                    role="button"
                                    tabIndex={0}
                                    onClick={() => {
                                      if (reportHref) {
                                        window.open(
                                          reportHref,
                                          "_blank",
                                          "noopener,noreferrer"
                                        );
                                      } else {
                                        handleOpenReport(candidate);
                                      }
                                    }}
                                    onKeyDown={(event) => {
                                      if (
                                        (event.key === "Enter" || event.key === " ") &&
                                        (candidate.player_id || candidate.player_name)
                                      ) {
                                        event.preventDefault();
                                        if (reportHref) {
                                          window.open(
                                            reportHref,
                                            "_blank",
                                            "noopener,noreferrer"
                                          );
                                        } else {
                                          handleOpenReport(candidate);
                                        }
                                      }
                                    }}
                                    className="cursor-pointer rounded-lg border border-white/10 bg-white/[0.025] p-4 transition hover:border-[#3A8967]/35 hover:bg-[#2F7D5C]/10 focus:outline-none focus:ring-2 focus:ring-[#3A8967]/50"
                                  >
                                    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_260px] xl:items-start">
                                      <div className="flex min-w-0 items-start gap-3">
                                        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-[#3A8967]/35 bg-[#2F7D5C]/15 text-sm font-semibold text-[#8CC7A7]">
                                          {cardIndex + 1}
                                        </div>
                                        {tmPhotoUrl ? (
                                          <img
                                            src={tmPhotoUrl}
                                            alt={candidate.player_name}
                                            className="h-12 w-12 shrink-0 rounded-md border border-white/10 object-cover"
                                          />
                                        ) : (
                                          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-md border border-white/10 bg-white/[0.04] text-sm font-semibold text-slate-200">
                                            {getInitials(candidate.player_name)}
                                          </div>
                                        )}
                                        <div className="min-w-0 flex-1">
                                          <div className="min-w-0">
                                            <h4 className="break-words text-lg font-semibold leading-6 text-white">
                                              {candidate.player_name}
                                            </h4>
                                          </div>
                                          <p className="mt-1 text-sm text-slate-400">
                                            {candidate.team || "—"} · {candidate.competition_name || "—"} · {candidate.calendar || "—"}
                                          </p>
                                          {tmProfileUrl ? (
                                            <a
                                              href={tmProfileUrl}
                                              target="_blank"
                                              rel="noreferrer"
                                              className="mt-2 inline-flex items-center gap-1 text-xs font-semibold text-[#8CC7A7] hover:text-white"
                                              onClick={(event) => event.stopPropagation()}
                                            >
                                              Transfermarkt profile
                                              <Icon name="external" className="h-3 w-3" />
                                            </a>
                                          ) : null}
                                        </div>
                                      </div>
                                      <div className="grid grid-cols-2 gap-2 xl:min-w-[260px]">
                                        <MetricTile
                                          label="Score"
                                          value={candidate.global_score_adjusted?.toFixed(1) ?? "—"}
                                          tone="green"
                                        />
                                        <MetricTile
                                          label="Role pct"
                                          value={`${candidate.assigned_role_pct_league?.toFixed(0) ?? "—"} / ${candidate.assigned_role_pct_global?.toFixed(0) ?? "—"}`}
                                        />
                                      </div>
                                    </div>
                                    <div className="mt-3 flex flex-wrap gap-2">
                                      {candidate.assigned_role ? (
                                        <span className="rounded-md border border-white/10 bg-black/[0.2] px-2 py-1 text-xs font-medium text-slate-300">
                                          {candidate.assigned_role}
                                        </span>
                                      ) : null}
                                      {candidate.position ? (
                                        <span className="rounded-md border border-white/10 bg-black/[0.2] px-2 py-1 text-xs font-medium text-slate-300">
                                          {candidate.position}
                                        </span>
                                      ) : null}
                                      {candidate.age ? (
                                        <span className="rounded-md border border-white/10 bg-black/[0.2] px-2 py-1 text-xs font-medium text-slate-300">
                                          {candidate.age} yrs
                                        </span>
                                      ) : null}
                                      <span className="rounded-md border border-white/10 bg-black/[0.2] px-2 py-1 text-xs font-medium text-slate-300">
                                        {Math.round(candidate.minutes_played || 0)} mins
                                      </span>
                                      {tmMarketValue ? (
                                        <span className="rounded-md border border-white/10 bg-black/[0.2] px-2 py-1 text-xs font-medium text-slate-300">
                                          Market value: {tmMarketValue}
                                        </span>
                                      ) : null}
                                      {tmAgentName ? (
                                        <span className="rounded-md border border-white/10 bg-black/[0.2] px-2 py-1 text-xs font-medium text-slate-300">
                                          Agent: {tmAgentName}
                                        </span>
                                      ) : null}
                                    </div>
                                    <div className="mt-4 grid gap-3 lg:grid-cols-2">
                                      <div className="rounded-md border border-white/10 bg-black/[0.18] p-3">
                                        <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                                          Reason
                                        </p>
                                        <p className="mt-2 text-sm leading-6 text-slate-300">
                                          {candidate.reason || "—"}
                                        </p>
                                      </div>
                                      <div className="rounded-md border border-white/10 bg-black/[0.18] p-3">
                                        <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                                          Role summary
                                        </p>
                                        <p className="mt-2 text-sm leading-6 text-slate-300">
                                          {candidate.role_summary || "—"}
                                        </p>
                                      </div>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          ) : null}

                          {payload?.report ? (
                            <details className="mt-4 rounded-md border border-white/10 bg-black/[0.22] px-3 py-2 text-xs text-slate-400">
                              <summary className="cursor-pointer font-semibold text-slate-300">
                                Context used
                              </summary>
                              <pre className="mt-3 max-h-56 overflow-auto whitespace-pre-wrap text-[11px] text-slate-500">
                                {JSON.stringify(payload.context || {}, null, 2)}
                              </pre>
                            </details>
                          ) : null}
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </Card>

            {error ? (
              <Card className="border-rose-400/35 bg-rose-500/10">
                <p className="text-sm font-semibold text-rose-200">Error: {error}</p>
              </Card>
            ) : null}

            <Card className="overflow-visible p-0">
              <div className="border-b border-white/10 px-4 py-3">
                <p className="text-sm font-semibold text-white">Composer</p>
                <p className="mt-1 text-xs text-slate-500">Write the exact recruitment question or player report request.</p>
              </div>
              <div className="space-y-3 p-4">
                {mode === "player" ? (
                  <div className="relative z-30">
                    <Label>Player</Label>
                    <Input
                      value={playerQuery}
                      placeholder="Start typing a player name..."
                      autoComplete="off"
                      onChange={(e) => {
                        setPlayerQuery(e.target.value);
                        setSelectedPlayerId("");
                        setSelectedPlayerSeasonId("");
                        setShowResults(true);
                      }}
                      onFocus={() => setShowResults(true)}
                      onBlur={() => setTimeout(() => setShowResults(false), 150)}
                    />
                    {showResults && playerQuery.trim().length >= 2 ? (
                      <div className="absolute z-50 mt-2 max-h-72 w-full overflow-auto rounded-lg border border-white/10 bg-[#080B0A] shadow-2xl">
                        {playerOptions.length === 0 ? (
                          <div className="px-3 py-3 text-sm text-slate-400">
                            No matches found.
                          </div>
                        ) : (
                          playerOptions.map((player) => (
                            <button
                              key={`${player.id}-${player.seasonId || "latest"}`}
                              type="button"
                              className="w-full border-b border-white/5 px-3 py-2.5 text-left text-sm text-slate-300 transition last:border-b-0 hover:bg-[#2F7D5C]/12 hover:text-white"
                              onMouseDown={(e) => e.preventDefault()}
                              onClick={() => {
                                setSelectedPlayerId(player.id);
                                setSelectedPlayerSeasonId(player.seasonId || "");
                                setPlayerQuery(player.label);
                                setShowResults(false);
                              }}
                            >
                              {player.label}
                            </button>
                          ))
                        )}
                      </div>
                    ) : null}
                  </div>
                ) : null}
                <div className="space-y-2">
                  <Label>Message</Label>
                  <textarea
                    className="nl-field min-h-[132px] resize-y"
                    placeholder="Describe the scouting brief or the report you need..."
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    onKeyDown={(event) => {
                      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
                        event.preventDefault();
                        handleSend();
                      }
                    }}
                  />
                </div>
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <p className="text-xs text-slate-500">
                    Cmd/Ctrl + Enter to send. Answers keep the selected mode, language and season.
                  </p>
                  <button
                    type="button"
                    className="nl-button-primary justify-center disabled:cursor-not-allowed disabled:opacity-45"
                    onClick={handleSend}
                    disabled={loading || !prompt.trim()}
                  >
                    <Icon name="send" className="h-4 w-4" />
                    {loading ? "Sending" : "Send brief"}
                  </button>
                </div>
              </div>
            </Card>
          </section>

        </div>
      </div>
    </main>
  );
}
