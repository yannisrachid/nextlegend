import { useEffect, useMemo, useState } from "react";
import { fetchJson, fetchJsonCached, postJson, patchJson } from "@/lib/api";

const TM_BASE_URL = "https://www.transfermarkt.com";

const Card = ({ children, className = "", ...rest }) => (
  <div
    className={`glass-panel rounded-xl p-4 border border-white/5 ${className}`}
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
    className="bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100 w-full"
    value={value}
    onChange={onChange}
    placeholder={placeholder}
    {...rest}
  />
);

const Select = ({ value, onChange, children }) => (
  <select
    className="bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100 w-full"
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

const seasonSortKey = (value) => {
  const text = String(value || "").trim();
  if (!text) return 0;
  const matchRange = text.match(/(20\d{2})\s*[/-]\s*((?:20)?\d{2,4})/);
  if (matchRange) {
    const start = Number(matchRange[1]);
    const rawEnd = matchRange[2];
    const end =
      rawEnd.length === 2
        ? Math.floor(start / 100) * 100 + Number(rawEnd)
        : Number(rawEnd.slice(-4));
    return end * 10000 + start;
  }
  const matchYear = text.match(/20\d{2}/);
  return matchYear ? Number(matchYear[0]) * 10000 + Number(matchYear[0]) : 0;
};

const sortSeasonsDesc = (items) =>
  Array.from(new Set((items || []).filter(Boolean))).sort((a, b) => {
    const diff = seasonSortKey(b) - seasonSortKey(a);
    if (diff !== 0) return diff;
    return String(b).localeCompare(String(a), undefined, { sensitivity: "base" });
  });

const extractTmFields = (row) => {
  const tmFields = {};
  Object.entries(row || {}).forEach(([key, value]) => {
    if (key.startsWith("tm_")) {
      tmFields[key] = value;
    }
  });
  return tmFields;
};

const PriorityBadge = ({ value }) => {
  const styles = {
    1: "bg-emerald-500/20 text-emerald-200 border-emerald-400/40",
    2: "bg-amber-500/20 text-amber-200 border-amber-400/40",
    3: "bg-sky-500/20 text-sky-200 border-sky-400/40",
  };
  return (
    <span
      className={`px-2 py-1 rounded-full border text-xs font-semibold ${styles[value] || "border-slate-600 text-slate-200"}`}
    >
      Priority {value}
    </span>
  );
};

export default function AIPage() {
  const [mode, setMode] = useState("scout");
  const [prompt, setPrompt] = useState("");
  const [language, setLanguage] = useState("auto");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [userId, setUserId] = useState("");
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
  const [selectedSeason, setSelectedSeason] = useState("");
  const [usageTotal, setUsageTotal] = useState(null);
  const [usageConversation, setUsageConversation] = useState(null);
  const [usageError, setUsageError] = useState("");

  useEffect(() => {
    fetchJson("/auth/me")
      .then((res) => {
        if (res?.username) {
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
    loadUsage(activeConversationId);
  }, [activeConversationId, messages.length, userId]);

  useEffect(() => {
    fetchJsonCached("/meta/seasons")
      .then((data) => setSeasons(sortSeasonsDesc(data || [])))
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
    if (!userId) return;
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

  return (
    <main className="min-h-screen bg-hero-pattern text-slate-100 py-10 px-4">
      <div className="max-w-6xl mx-auto space-y-6">
        <header className="flex flex-col gap-2">
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
            AI Assistant
          </p>
          <h1 className="text-4xl font-bold text-white tracking-tight">
            Agentic Scouting Workspace
          </h1>
          <p className="text-slate-300 max-w-3xl">
            Minimalist chat experience powered by the Postgres scouting database.
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-[260px_1fr] gap-6">
          <aside className="space-y-3">
            <button
              className="w-full px-3 py-2 rounded-md border border-slate-700 bg-slate-900/60 text-sm text-slate-200 hover:bg-slate-800/80"
              onClick={() => createConversation(mode)}
            >
              + New conversation
            </button>
            <Card className="space-y-2 text-xs text-slate-300">
              <div className="flex items-center justify-between">
                <span className="uppercase tracking-[0.2em] text-slate-400">
                  Usage
                </span>
                <span className="text-slate-500">
                  {usageTotal?.model || "gpt-4o"}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span>This chat</span>
                <span>
                  {formatTokens(usageConversation?.total_tokens)} tokens
                </span>
              </div>
              <div className="flex items-center justify-between text-slate-400">
                <span>Cost</span>
                <span>{formatUsd(usageConversation?.estimated_cost_usd)}</span>
              </div>
              <div className="flex items-center justify-between mt-2">
                <span>Total</span>
                <span>{formatTokens(usageTotal?.total_tokens)} tokens</span>
              </div>
              <div className="flex items-center justify-between text-slate-400">
                <span>Cost</span>
                <span>{formatUsd(usageTotal?.estimated_cost_usd)}</span>
              </div>
              <div className="text-[11px] text-slate-500 mt-2">
                Remaining balance is not available via OpenAI API.
              </div>
              {usageError ? (
                <div className="text-[11px] text-amber-300">{usageError}</div>
              ) : null}
            </Card>
            <div className="space-y-2">
              {conversations.length === 0 ? (
                <Card className="text-xs text-slate-400">
                  No conversations yet.
                </Card>
              ) : (
                conversations.map((conv) => {
                  const isActive = conv.id === activeConversationId;
                  const isEditing = editingConversationId === conv.id;
                  return (
                    <div
                      key={conv.id}
                      className={`w-full rounded-md border ${
                        isActive
                          ? "border-primary/60 bg-primary/10 text-white"
                          : "border-slate-800 bg-slate-900/40 text-slate-300"
                      } px-3 py-2`}
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
                              className="px-2 py-1 rounded-md bg-primary text-slate-900 text-xs font-semibold"
                              onClick={() => saveConversationTitle(conv.id)}
                            >
                              Save
                            </button>
                            <button
                              className="px-2 py-1 rounded-md border border-slate-700 text-xs text-slate-200"
                              onClick={cancelEditConversation}
                            >
                              Cancel
                            </button>
                          </div>
                        </div>
                      ) : (
                        <div className="flex items-start justify-between gap-2">
                          <button
                            type="button"
                            onClick={() => setActiveConversationId(conv.id)}
                            className="text-left flex-1"
                          >
                            <div className="text-sm font-semibold">
                              {conv.title || "New chat"}
                            </div>
                            <div className="text-xs text-slate-500">
                              {conv.mode === "player"
                                ? "Player report"
                                : "Scout advisor"}
                            </div>
                          </button>
                          <button
                            type="button"
                            className="text-xs text-slate-400 hover:text-slate-200"
                            onClick={() => startEditConversation(conv)}
                          >
                            Edit
                          </button>
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </aside>

          <section className="space-y-4">
            <Card>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <div>
                  <Label>Mode</Label>
                  <Select
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
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                  >
                    <option value="auto">Auto</option>
                    <option value="en">English</option>
                    <option value="fr">Francais</option>
                  </Select>
                </div>
                <div>
                  <Label>Season Filter</Label>
                  <Select
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
            </Card>

            <div className="space-y-4">
              {messages.length === 0 ? (
                <Card className="text-sm text-slate-400">
                  Start the conversation by sending a scouting brief.
                </Card>
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
                      className={`flex ${isUser ? "justify-end" : "justify-start"}`}
                    >
                      <div
                        className={`max-w-[85%] rounded-2xl px-4 py-3 border ${
                          isUser
                            ? "bg-primary/15 border-primary/40 text-slate-100"
                            : "bg-slate-900/70 border-slate-800 text-slate-200"
                        }`}
                      >
                        {isPending ? (
                          <div className="typing-dots" aria-label="Loading">
                            <span />
                            <span />
                            <span />
                          </div>
                        ) : (
                          <p className="whitespace-pre-wrap text-sm leading-relaxed">
                            {message.content}
                          </p>
                        )}

                        {payload?.filters && !isPending ? (
                          <details className="mt-3 text-xs text-slate-400">
                            <summary className="cursor-pointer">
                              Applied filters
                            </summary>
                            <pre className="mt-2 whitespace-pre-wrap">
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
                              const cardContent = (
                                <Card
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
                                  className="cursor-pointer focus:outline-none focus:ring-2 focus:ring-emerald-400/60"
                                >
                                  <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
                                    <div className="flex items-center gap-3">
                                      <div className="text-2xl font-semibold text-primary">
                                        {cardIndex + 1}
                                      </div>
                                      <div className="flex items-center gap-3">
                                        {tmPhotoUrl ? (
                                          <img
                                            src={tmPhotoUrl}
                                            alt={candidate.player_name}
                                            className="h-12 w-12 rounded-full object-cover border border-white/10"
                                          />
                                        ) : (
                                          <div className="h-12 w-12 rounded-full bg-slate-800 border border-white/10 flex items-center justify-center text-slate-200 font-semibold">
                                            {getInitials(candidate.player_name)}
                                          </div>
                                        )}
                                        <div>
                                          <div className="text-lg font-semibold text-white flex items-center gap-2">
                                            {candidate.player_name}
                                            <PriorityBadge value={candidate.priority} />
                                          </div>
                                          <div className="text-slate-400 text-sm">
                                            {candidate.team || "—"} •{" "}
                                            {candidate.competition_name} •{" "}
                                            {candidate.calendar || "–"}
                                          </div>
                                          {tmProfileUrl ? (
                                            <a
                                              href={tmProfileUrl}
                                              target="_blank"
                                              rel="noreferrer"
                                              className="text-xs text-primary hover:text-primary/80"
                                              onClick={(event) =>
                                                event.stopPropagation()
                                              }
                                            >
                                              Transfermarkt profile
                                            </a>
                                          ) : null}
                                          {tmMarketValue || tmAgentName ? (
                                            <div className="text-xs text-slate-400 mt-1">
                                              {tmMarketValue
                                                ? `Market value: ${tmMarketValue}`
                                                : null}
                                              {tmMarketValue && tmAgentName
                                                ? " • "
                                                : null}
                                              {tmAgentName
                                                ? `Agent: ${tmAgentName}`
                                                : null}
                                            </div>
                                          ) : null}
                                          <div className="flex flex-wrap gap-2 mt-2">
                                            {candidate.assigned_role ? (
                                              <span className="px-2 py-1 rounded-full bg-slate-800 text-xs text-slate-200 border border-white/5">
                                                {candidate.assigned_role}
                                              </span>
                                            ) : null}
                                            {candidate.position ? (
                                              <span className="px-2 py-1 rounded-full bg-slate-800 text-xs text-slate-200 border border-white/5">
                                                {candidate.position}
                                              </span>
                                            ) : null}
                                            {candidate.age ? (
                                              <span className="px-2 py-1 rounded-full bg-slate-800 text-xs text-slate-200 border border-white/5">
                                                {candidate.age} yrs
                                              </span>
                                            ) : null}
                                            <span className="px-2 py-1 rounded-full bg-slate-800 text-xs text-slate-200 border border-white/5">
                                              {Math.round(
                                                candidate.minutes_played || 0
                                              )}{" "}
                                              mins
                                            </span>
                                          </div>
                                        </div>
                                      </div>
                                    </div>
                                    <div className="flex flex-wrap items-center gap-6">
                                      <div className="text-right">
                                        <p className="text-xs uppercase text-slate-400">
                                          Global score (adj.)
                                        </p>
                                        <p className="text-2xl font-bold text-primary">
                                          {candidate.global_score_adjusted?.toFixed(1) ??
                                            "—"}
                                        </p>
                                      </div>
                                      <div className="text-right">
                                        <p className="text-xs uppercase text-slate-400">
                                          Role pct (league/global)
                                        </p>
                                        <p className="text-lg font-semibold">
                                          {candidate.assigned_role_pct_league?.toFixed(0) ??
                                            "—"}{" "}
                                          /{" "}
                                          {candidate.assigned_role_pct_global?.toFixed(0) ??
                                            "—"}
                                        </p>
                                      </div>
                                    </div>
                                  </div>
                                  <div className="mt-3 grid grid-cols-1 lg:grid-cols-[1.1fr_1fr] gap-4">
                                    <div>
                                      <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                                        Reason
                                      </div>
                                      <p className="text-sm text-slate-300">
                                        {candidate.reason}
                                      </p>
                                    </div>
                                    <div>
                                      <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                                        Role summary
                                      </div>
                                      <p className="text-sm text-slate-300">
                                        {candidate.role_summary}
                                      </p>
                                    </div>
                                  </div>
                                </Card>
                              );
                              return cardContent;
                            })}
                          </div>
                        ) : null}

                        {payload?.report ? (
                          <details className="mt-3 text-xs text-slate-400">
                            <summary className="cursor-pointer">
                              Context used
                            </summary>
                            <pre className="mt-2 whitespace-pre-wrap">
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

            {error && (
              <Card>
                <p className="text-danger">Error: {error}</p>
              </Card>
            )}

            <Card className="space-y-3">
              {mode === "player" ? (
                <div className="relative z-30">
                  <Label>Player</Label>
                  <Input
                    value={playerQuery}
                    placeholder="Start typing a player name..."
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
                    <div className="absolute z-50 mt-2 w-full max-h-72 overflow-auto rounded-lg border border-slate-700 bg-slate-900/95 shadow-xl">
                      {playerOptions.length === 0 ? (
                        <div className="px-3 py-2 text-sm text-slate-400">
                          No matches found.
                        </div>
                      ) : (
                        playerOptions.map((player) => (
                          <button
                            key={`${player.id}-${player.seasonId || "latest"}`}
                            type="button"
                            className="w-full text-left px-3 py-2 text-sm text-slate-200 hover:bg-slate-800/80"
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
                  className="w-full min-h-[120px] bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100"
                  placeholder="Describe the scouting brief or the report you need..."
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                />
              </div>
              <div className="flex justify-end">
                <button
                  className="px-4 py-2 rounded-md bg-primary text-slate-900 font-semibold hover:bg-primary/90 transition"
                  onClick={handleSend}
                  disabled={loading || !prompt.trim()}
                >
                  Send
                </button>
              </div>
            </Card>
          </section>
        </div>
      </div>
    </main>
  );
}
