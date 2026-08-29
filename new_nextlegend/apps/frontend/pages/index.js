import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { deleteJson, fetchJson, patchJson, postJson } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { EmptyState, PageHeader, Panel } from "@/components/ui/product";

const AGENTS = ["Steven", "Don", "Yannis", "Lidahi"];

const AGENT_COLORS = {
  Yannis: "#3A8967",
  Lidahi: "#B7793D",
  Steven: "#4F7BA7",
  Don: "#A85151",
};

const EVENT_TYPES = [
  { value: "meeting", label: "Meeting" },
  { value: "match", label: "Match" },
  { value: "travel", label: "Travel" },
  { value: "deadline", label: "Deadline" },
  { value: "team", label: "Team" },
];

const PRIORITIES = [
  { value: "urgent", label: "Urgent", className: "bg-rose-50 text-rose-800 border-rose-200" },
  { value: "high", label: "High", className: "bg-amber-50 text-amber-800 border-amber-200" },
  { value: "medium", label: "Medium", className: "bg-teal-50 text-teal-800 border-teal-200" },
  { value: "low", label: "Low", className: "bg-slate-50 text-slate-700 border-slate-200" },
];

const KANBAN_COLUMNS = [
  { value: "backlog", label: "Backlog", helper: "Opportunities to qualify before action." },
  { value: "todo", label: "To do", helper: "Validated priorities ready for ownership." },
  { value: "in_progress", label: "In progress", helper: "Live workstreams owned by agents." },
  { value: "review", label: "Review", helper: "Actions waiting for a decision or follow-up." },
  { value: "done", label: "Done", helper: "Completed work archived for visibility." },
];

const MODULES = [
  {
    title: "HD Players",
    href: "/hd-players",
    metric: "Portfolio",
    desc: "Player rooms, documents, season data and market strategy.",
  },
  {
    title: "Network",
    href: "/crm",
    metric: "CRM",
    desc: "Clubs, players, contacts and relationship workflows.",
  },
  {
    title: "Scouting Lab",
    href: "/scouting-lab",
    metric: "Intelligence",
    desc: "Ranking, reports, projections and visual decision support.",
  },
];

const today = () => new Date().toISOString().slice(0, 10);

const emptyForm = {
  title: "",
  description: "",
  agent_name: "Yannis",
  priority: "medium",
  status: "todo",
  start_date: today(),
  end_date: "",
  related_page: "/",
};

const emptyEventForm = {
  title: "",
  description: "",
  event_type: "meeting",
  agent_names: [],
  start_date: today(),
  end_date: "",
  location: "",
  related_page: "",
};

const normalizeStatus = (status) => {
  if (status === "planned") return "todo";
  if (status === "completed") return "done";
  return KANBAN_COLUMNS.some((column) => column.value === status) ? status : "todo";
};

const priorityClass = (priority) =>
  PRIORITIES.find((item) => item.value === priority)?.className || PRIORITIES[2].className;

const issueKey = (item) => `HQ-${String(item.id || "NEW").padStart(3, "0")}`;

const agentInitials = (name = "HD") =>
  name
    .split(" ")
    .map((part) => part[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

const dueState = (dateValue) => {
  if (!dateValue) return { label: "No due date", tone: "neutral" };
  const due = new Date(`${dateValue}T00:00:00`);
  const now = new Date(`${today()}T00:00:00`);
  const diffDays = Math.round((due - now) / 86400000);
  if (diffDays < 0) return { label: `${Math.abs(diffDays)}d overdue`, tone: "danger" };
  if (diffDays === 0) return { label: "Due today", tone: "warning" };
  if (diffDays <= 3) return { label: `Due in ${diffDays}d`, tone: "warning" };
  return { label: dateValue, tone: "neutral" };
};

const toDateKey = (date) => {
  const value = new Date(date);
  value.setHours(12, 0, 0, 0);
  return value.toISOString().slice(0, 10);
};

const parseDateKey = (dateKey) => new Date(`${dateKey}T12:00:00`);

const startOfMonth = (date) => new Date(date.getFullYear(), date.getMonth(), 1);

const endOfMonth = (date) => new Date(date.getFullYear(), date.getMonth() + 1, 0);

const addMonths = (date, count) => new Date(date.getFullYear(), date.getMonth() + count, 1);

const monthTitle = (date) =>
  new Intl.DateTimeFormat("en", { month: "long", year: "numeric" }).format(date);

const shortDateLabel = (dateKey) =>
  new Intl.DateTimeFormat("en", { weekday: "short", month: "short", day: "numeric" }).format(parseDateKey(dateKey));

const isSameDay = (left, right) => toDateKey(left) === toDateKey(right);

const monthWeeks = (monthDate) => {
  const start = startOfMonth(monthDate);
  const end = endOfMonth(monthDate);
  const leading = (start.getDay() + 6) % 7;
  const firstVisibleDay = new Date(start);
  firstVisibleDay.setDate(start.getDate() - leading);
  const trailing = 6 - ((end.getDay() + 6) % 7);
  const totalDays = leading + end.getDate() + trailing;
  const days = Array.from({ length: totalDays }, (_, index) => {
    const date = new Date(firstVisibleDay);
    date.setDate(firstVisibleDay.getDate() + index);
    date.setHours(12, 0, 0, 0);
    return {
      date,
      dateKey: toDateKey(date),
      inMonth: date.getMonth() === monthDate.getMonth(),
      isToday: isSameDay(date, new Date()),
    };
  });
  const weeks = [];
  for (let index = 0; index < days.length; index += 7) {
    weeks.push(days.slice(index, index + 7));
  }
  return weeks;
};

const eventAgents = (event) => (Array.isArray(event.agent_names) ? event.agent_names : []).filter(Boolean);

const eventAccent = (event) => {
  const agents = eventAgents(event);
  if (agents.length === 0) {
    return event.color || "#8A938D";
  }
  if (agents.length === 1) {
    return AGENT_COLORS[agents[0]] || "#8A938D";
  }
  const segment = 100 / agents.length;
  return `linear-gradient(90deg, ${agents.map((agent, index) => {
    const color = AGENT_COLORS[agent] || "#8A938D";
    return `${color} ${index * segment}% ${(index + 1) * segment}%`;
  }).join(", ")})`;
};

const eventEndDate = (event) =>
  event.end_date && event.end_date >= event.start_date ? event.end_date : event.start_date;

const eventLabel = (event) => {
  if (event.source === "task") return `Task due: ${event.title}`;
  const description = String(event.description || "").trim();
  return description ? `${event.title}: ${description}` : event.title;
};

const weekEventSegments = (week, events, monthDate) => {
  const weekStart = week[0].dateKey;
  const weekEnd = week[6].dateKey;
  const monthKey = toDateKey(monthDate).slice(0, 7);
  return events
    .filter((event) => event.start_date && event.start_date.slice(0, 7) <= monthKey && eventEndDate(event).slice(0, 7) >= monthKey)
    .filter((event) => event.start_date <= weekEnd && eventEndDate(event) >= weekStart)
    .sort((a, b) => String(a.start_date || "").localeCompare(String(b.start_date || "")) || String(eventEndDate(a)).localeCompare(String(eventEndDate(b))))
    .map((event) => {
      const startIndex = week.findIndex((day) => day.dateKey >= event.start_date);
      const endIndex = [...week].reverse().findIndex((day) => day.dateKey <= eventEndDate(event));
      const safeStart = startIndex === -1 ? 0 : startIndex;
      const safeEnd = endIndex === -1 ? 6 : 6 - endIndex;
      return {
        event,
        startColumn: safeStart + 1,
        endColumn: safeEnd + 2,
      };
    });
};

export default function Home() {
  const { me } = useAuth();
  const [items, setItems] = useState([]);
  const [form, setForm] = useState(emptyForm);
  const [calendarEvents, setCalendarEvents] = useState([]);
  const [calendarOffset, setCalendarOffset] = useState(0);
  const [eventForm, setEventForm] = useState(emptyEventForm);
  const [editingEventId, setEditingEventId] = useState(null);
  const [eventModalOpen, setEventModalOpen] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [eventSaving, setEventSaving] = useState(false);
  const [agentFilter, setAgentFilter] = useState("all");
  const [priorityFilter, setPriorityFilter] = useState("all");
  const [draggedId, setDraggedId] = useState(null);
  const [dragOverStatus, setDragOverStatus] = useState("");
  const [issueModalOpen, setIssueModalOpen] = useState(false);

  const currentAgent = useMemo(() => {
    const identity = [me?.display_name, me?.username].filter(Boolean).map((item) => String(item).toLowerCase());
    return AGENTS.find((agent) => identity.some((item) => item.includes(agent.toLowerCase()))) || "Yannis";
  }, [me?.display_name, me?.username]);

  const visibleMonths = useMemo(() => {
    const first = addMonths(startOfMonth(new Date()), calendarOffset);
    return [0, 1, 2].map((offset) => addMonths(first, offset));
  }, [calendarOffset]);

  const calendarStart = toDateKey(visibleMonths[0]);
  const calendarEnd = toDateKey(endOfMonth(visibleMonths[2]));

  const loadItems = async () => {
    try {
      const priorityData = await fetchJson("/hq/priorities");
      setItems(priorityData.items || []);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  };

  useEffect(() => {
    loadItems();
  }, []);

  const loadCalendarEvents = async () => {
    try {
      const data = await fetchJson("/hq/calendar-events", { start: calendarStart, end: calendarEnd });
      setCalendarEvents(data.items || []);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  };

  useEffect(() => {
    loadCalendarEvents();
  }, [calendarStart, calendarEnd]);

  const visibleItems = useMemo(() => {
    return items.filter((item) => {
      const agentOk = agentFilter === "all" || (item.agent_name || "Yannis") === agentFilter;
      const priorityOk = priorityFilter === "all" || (item.priority || "medium") === priorityFilter;
      return agentOk && priorityOk;
    });
  }, [agentFilter, items, priorityFilter]);

  const byStatus = useMemo(() => {
    const groups = Object.fromEntries(KANBAN_COLUMNS.map((column) => [column.value, []]));
    visibleItems.forEach((item) => {
      groups[normalizeStatus(item.status)].push(item);
    });
    Object.values(groups).forEach((group) => {
      group.sort((a, b) => {
        const priorityWeight = { urgent: 0, high: 1, medium: 2, low: 3 };
        const priorityDelta = (priorityWeight[a.priority] ?? 2) - (priorityWeight[b.priority] ?? 2);
        if (priorityDelta !== 0) return priorityDelta;
        return String(a.end_date || "9999-12-31").localeCompare(String(b.end_date || "9999-12-31"));
      });
    });
    return groups;
  }, [visibleItems]);

  const upcomingEvents = useMemo(() => {
    const current = today();
    return [...calendarEvents]
      .filter((event) => (event.end_date && event.end_date >= event.start_date ? event.end_date : event.start_date) >= current)
      .sort((a, b) => String(a.start_date || "").localeCompare(String(b.start_date || "")))
      .slice(0, 8);
  }, [calendarEvents]);

  const openNewIssue = (status = "todo") => {
    setError("");
    setEditingId(null);
    setForm({ ...emptyForm, status, start_date: today() });
    setIssueModalOpen(true);
  };

  const openNewEvent = (dateValue = today()) => {
    setError("");
    setEditingEventId(null);
    setEventForm({
      ...emptyEventForm,
      agent_names: [currentAgent],
      start_date: dateValue,
      end_date: "",
    });
    setEventModalOpen(true);
  };

  const startEditEvent = (event) => {
    if (event.source === "task" && event.task_id) {
      const task = items.find((item) => String(item.id) === String(event.task_id));
      if (task) startEdit(task);
      return;
    }
    if (!event.can_edit) return;
    setError("");
    setEditingEventId(event.id);
    setEventForm({
      title: event.title || "",
      description: event.description || "",
      event_type: event.event_type || "meeting",
      agent_names: eventAgents(event),
      start_date: event.start_date || today(),
      end_date: event.end_date || "",
      location: event.location || "",
      related_page: event.related_page || "",
    });
    setEventModalOpen(true);
  };

  const closeEventModal = () => {
    setEventModalOpen(false);
    setEditingEventId(null);
    setEventForm({ ...emptyEventForm, agent_names: [currentAgent], start_date: today() });
  };

  const startEdit = (item) => {
    setError("");
    setEditingId(item.id);
    setForm({
      title: item.title || "",
      description: item.description || "",
      agent_name: item.agent_name || "Yannis",
      priority: item.priority || "medium",
      status: normalizeStatus(item.status),
      start_date: item.start_date || today(),
      end_date: item.end_date || "",
      related_page: item.related_page || "/",
    });
    setIssueModalOpen(true);
  };

  const resetForm = () => {
    setEditingId(null);
    setForm({ ...emptyForm, start_date: today() });
  };

  const closeIssueModal = () => {
    setIssueModalOpen(false);
    resetForm();
  };

  const saveItem = async () => {
    if (!form.title.trim()) {
      setError("Add a task title.");
      return;
    }
    setSaving(true);
    try {
      if (editingId) {
        await patchJson(`/hq/priorities/${editingId}`, form);
      } else {
        await postJson("/hq/priorities", form);
      }
      closeIssueModal();
      await loadItems();
      await loadCalendarEvents();
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const saveEvent = async () => {
    if (!eventForm.title.trim()) {
      setError("Add an event title.");
      return;
    }
    if (!eventForm.start_date) {
      setError("Choose a start date.");
      return;
    }
    if (eventForm.end_date && eventForm.end_date < eventForm.start_date) {
      setError("End date cannot be earlier than start date.");
      return;
    }
    setEventSaving(true);
    try {
      const payload = {
        ...eventForm,
        agent_names: eventForm.agent_names || [],
        end_date: eventForm.end_date || "",
      };
      if (editingEventId) {
        await patchJson(`/hq/calendar-events/${editingEventId}`, payload);
      } else {
        await postJson("/hq/calendar-events", payload);
      }
      closeEventModal();
      await loadCalendarEvents();
    } catch (err) {
      setError(err.message);
    } finally {
      setEventSaving(false);
    }
  };

  const moveItem = async (item, status) => {
    if (normalizeStatus(item.status) === status) return;
    try {
      await patchJson(`/hq/priorities/${item.id}`, { ...item, status });
      await loadItems();
      await loadCalendarEvents();
    } catch (err) {
      setError(err.message);
    }
  };

  const handleDrop = async (status) => {
    const item = items.find((entry) => String(entry.id) === String(draggedId));
    setDraggedId(null);
    setDragOverStatus("");
    if (!item) return;
    await moveItem(item, status);
  };

  const removeItem = async (id) => {
    try {
      await deleteJson(`/hq/priorities/${id}`);
      await loadItems();
      await loadCalendarEvents();
    } catch (err) {
      setError(err.message);
    }
  };

  const removeEvent = async () => {
    if (!editingEventId) return;
    setEventSaving(true);
    try {
      await deleteJson(`/hq/calendar-events/${editingEventId}`);
      closeEventModal();
      await loadCalendarEvents();
    } catch (err) {
      setError(err.message);
    } finally {
      setEventSaving(false);
    }
  };

  const toggleEventAgent = (agent) => {
    setEventForm((current) => {
      const selected = new Set(current.agent_names || []);
      if (selected.has(agent)) {
        selected.delete(agent);
      } else {
        selected.add(agent);
      }
      return { ...current, agent_names: Array.from(selected) };
    });
  };

  return (
    <main className="nl-page px-4 py-6 md:py-8">
      <div className="mx-auto max-w-[1500px] space-y-6">
        <PageHeader
          eyebrow="HQ"
          title="Agency operating room."
          description="Priority actions, ownership and decisions for the HD Sports team."
          actions={
            <button type="button" className="nl-button-primary" onClick={() => openNewIssue("todo")}>
              New task
            </button>
          }
        />

        <section className="surface-panel overflow-hidden rounded-lg">
          <div className="border-b border-white/10 p-5">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <p className="nl-kicker">Upcoming events</p>
                <h2 className="mt-2 text-2xl font-semibold tracking-tight text-slate-950">
                  Team agenda for {monthTitle(visibleMonths[0])} to {monthTitle(visibleMonths[2])}
                </h2>
                <p className="mt-1 text-sm leading-6 text-slate-600">
                  Plan meetings, matches, travel and task due dates in the same operating calendar.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <button type="button" className="nl-icon-button" onClick={() => setCalendarOffset((value) => value - 3)} aria-label="Previous three months">
                  ‹
                </button>
                <button type="button" className="nl-button-secondary" onClick={() => setCalendarOffset(0)}>
                  Today
                </button>
                <button type="button" className="nl-icon-button" onClick={() => setCalendarOffset((value) => value + 3)} aria-label="Next three months">
                  ›
                </button>
                <button type="button" className="nl-button-primary" onClick={() => openNewEvent(today())}>
                  New event
                </button>
              </div>
            </div>
          </div>

          <div className="grid gap-0 xl:grid-cols-[minmax(0,1fr)_340px]">
            <div className="grid gap-4 p-4 lg:grid-cols-3">
              {visibleMonths.map((month) => (
                <div key={month.toISOString()} className="rounded-lg border border-white/10 bg-black/20 p-3">
                  <div className="mb-3 flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-slate-950">{monthTitle(month)}</h3>
                    <span className="text-[11px] font-semibold text-slate-500">
                      {calendarEvents.filter((event) => event.start_date?.slice(0, 7) === toDateKey(month).slice(0, 7)).length} events
                    </span>
                  </div>
                  <div className="grid grid-cols-7 gap-1 text-center text-[10px] font-bold uppercase tracking-[0.08em] text-slate-500">
                    {["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map((day) => <span key={day}>{day}</span>)}
                  </div>
                  <div className="mt-2 space-y-1.5">
                    {monthWeeks(month).map((week, weekIndex) => {
                      const segments = weekEventSegments(week, calendarEvents, month);
                      const visibleSegments = segments.slice(0, 4);
                      return (
                        <div key={`${month.toISOString()}-week-${weekIndex}`} className="relative min-h-[124px]">
                          <div className="grid h-full min-h-[124px] grid-cols-7 gap-1">
                            {week.map((day) => (
                              <button
                                key={day.dateKey}
                                type="button"
                                onClick={() => openNewEvent(day.dateKey)}
                                className={`rounded-md border p-1.5 text-left transition ${
                                  day.inMonth
                                    ? day.isToday
                                      ? "border-[#3A8967]/60 bg-[#2F7D5C]/14"
                                      : "border-white/10 bg-white/[0.025] hover:border-[#3A8967]/35 hover:bg-white/[0.05]"
                                    : "border-white/[0.04] bg-white/[0.012] text-white/25 hover:border-white/10"
                                }`}
                              >
                                <span className={`block text-xs font-semibold ${day.inMonth ? "text-slate-950" : "text-slate-500"}`}>
                                  {day.date.getDate()}
                                </span>
                              </button>
                            ))}
                          </div>
                          <div className="pointer-events-none absolute inset-x-1 top-8 grid grid-cols-7 gap-1">
                            {visibleSegments.map((segment, index) => (
                              <button
                                key={`${segment.event.source}-${segment.event.id}-${index}`}
                                type="button"
                                onClick={(clickEvent) => {
                                  clickEvent.stopPropagation();
                                  startEditEvent(segment.event);
                                }}
                                className="pointer-events-auto h-[22px] overflow-hidden rounded-sm border border-white/15 px-2 text-left text-[10px] font-semibold leading-[20px] text-white shadow-[0_8px_18px_rgba(0,0,0,0.28)] transition hover:border-white/35 hover:brightness-110"
                                style={{
                                  gridColumn: `${segment.startColumn} / ${segment.endColumn}`,
                                  gridRow: `${index + 1}`,
                                  background: eventAccent(segment.event),
                                }}
                                title={eventLabel(segment.event)}
                              >
                                <span className="block truncate drop-shadow">{eventLabel(segment.event)}</span>
                              </button>
                            ))}
                            {segments.length > visibleSegments.length ? (
                              <span
                                className="pointer-events-none h-[20px] rounded-sm border border-white/10 bg-black/50 px-2 text-[10px] font-semibold leading-[18px] text-white/65"
                                style={{
                                  gridColumn: "1 / 8",
                                  gridRow: `${visibleSegments.length + 1}`,
                                }}
                              >
                                +{segments.length - visibleSegments.length} more events this week
                              </span>
                            ) : null}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>

            <aside className="border-t border-white/10 p-4 xl:border-l xl:border-t-0">
              <div className="rounded-lg border border-white/10 bg-black/20 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="nl-kicker">Next up</p>
                    <h3 className="mt-1 text-lg font-semibold text-slate-950">Operational agenda</h3>
                  </div>
                  <span className="rounded-md border border-white/10 bg-white/[0.04] px-2 py-1 text-xs font-semibold text-slate-500">
                    {calendarEvents.length}
                  </span>
                </div>
                <div className="mt-4 space-y-2">
                  {upcomingEvents.length ? upcomingEvents.map((event) => {
                    const agents = eventAgents(event);
                    return (
                      <button
                        key={`${event.source}-${event.id}`}
                        type="button"
                        onClick={() => startEditEvent(event)}
                        className="block w-full rounded-md border border-white/10 bg-white/[0.025] p-3 text-left transition hover:border-[#3A8967]/40 hover:bg-white/[0.055]"
                      >
                        <span className="flex items-start gap-3">
                          <span className="mt-1 h-10 w-1.5 shrink-0 rounded-full" style={{ background: eventAccent(event) }} />
                          <span className="min-w-0 flex-1">
                            <span className="flex flex-wrap items-center gap-2">
                              <span className="text-xs font-semibold text-[#8CC7A7]">{shortDateLabel(event.start_date)}</span>
                              <span className="rounded-full border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[10px] font-bold uppercase text-slate-500">
                                {event.source === "task" ? "Task due" : event.event_type || "event"}
                              </span>
                            </span>
                            <span className="mt-1 block truncate text-sm font-semibold text-slate-950">{event.title}</span>
                            <span className="mt-1 block truncate text-xs text-slate-500">
                              {agents.length ? agents.join(" + ") : event.location || "Neutral event"}
                            </span>
                          </span>
                        </span>
                      </button>
                    );
                  }) : (
                    <EmptyState
                      title="No upcoming event"
                      description="Add a meeting, match or travel item to build the team agenda."
                      action={<button type="button" className="nl-button-secondary" onClick={() => openNewEvent(today())}>Add event</button>}
                    />
                  )}
                </div>
                <div className="mt-5 border-t border-white/10 pt-4">
                  <p className="text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Agent colors</p>
                  <div className="mt-3 grid grid-cols-2 gap-2">
                    {AGENTS.map((agent) => (
                      <button key={agent} type="button" className="flex items-center gap-2 rounded-md border border-white/10 bg-white/[0.025] px-2 py-2 text-left text-xs font-semibold text-slate-500 hover:bg-white/[0.05]" onClick={() => setAgentFilter(agent)}>
                        <span className="h-2.5 w-2.5 rounded-full" style={{ background: AGENT_COLORS[agent] }} />
                        {agent}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </aside>
          </div>
        </section>

        <section className="grid gap-4 md:grid-cols-3">
          {MODULES.map((module) => (
            <Link key={module.title} href={module.href} className="surface-panel group rounded-lg p-5 transition hover:-translate-y-0.5 hover:border-[#3A8967]/40 hover:bg-white/[0.06]">
              <span className="text-xs font-extrabold uppercase tracking-[0.16em] text-[#8CC7A7]">{module.metric}</span>
              <h2 className="mt-3 text-xl font-semibold text-slate-950">{module.title}</h2>
              <p className="mt-2 text-sm leading-6 text-slate-600">{module.desc}</p>
              <span className="mt-5 inline-flex text-sm font-semibold text-[#8CC7A7]">Enter workspace</span>
            </Link>
          ))}
        </section>

        <section className="surface-panel overflow-hidden rounded-lg">
          <div className="border-b border-white/10 p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="nl-kicker">Agency board</p>
                <h2 className="mt-2 text-2xl font-semibold tracking-tight text-slate-950">Tasks</h2>
                <p className="mt-1 text-sm leading-6 text-slate-600">
                  Track priorities from first signal to final decision with clear ownership and delivery status.
                </p>
              </div>
              <button type="button" className="nl-button-primary" onClick={() => openNewIssue("todo")}>
                New task
              </button>
            </div>

            <div className="mt-4 flex flex-wrap items-center gap-2">
              {["all", ...AGENTS].map((agent) => {
                const active = agentFilter === agent;
                return (
                  <button
                    key={agent}
                    type="button"
                    className={`rounded-md border px-3 py-1.5 text-xs font-semibold transition ${
                      active
                        ? "border-[#3A8967]/40 bg-[#2F7D5C]/20 text-[#DDF3E8]"
                        : "border-white/10 bg-white/[0.035] text-white/60 hover:border-white/20 hover:text-white"
                    }`}
                    onClick={() => setAgentFilter(agent)}
                  >
                    {agent === "all" ? "All owners" : agent}
                  </button>
                );
              })}
              <span className="mx-1 h-5 w-px bg-white/10" />
              {["all", ...PRIORITIES.map((priority) => priority.value)].map((priority) => {
                const active = priorityFilter === priority;
                return (
                  <button
                    key={priority}
                    type="button"
                    className={`rounded-md border px-3 py-1.5 text-xs font-semibold capitalize transition ${
                      active
                        ? "border-[#3A8967]/40 bg-[#2F7D5C]/20 text-[#DDF3E8]"
                        : "border-white/10 bg-white/[0.035] text-white/60 hover:border-white/20 hover:text-white"
                    }`}
                    onClick={() => setPriorityFilter(priority)}
                  >
                    {priority === "all" ? "All priority" : priority}
                  </button>
                );
              })}
              <span className="ml-auto text-xs font-semibold text-slate-500">
                Showing {visibleItems.length} of {items.length}
              </span>
            </div>
            {error ? <p className="mt-3 rounded-md border border-rose-400/25 bg-rose-400/10 px-3 py-2 text-sm font-semibold text-rose-700">{error}</p> : null}
          </div>

          <div className="overflow-x-auto">
          <div className="grid min-w-[1180px] grid-cols-5 gap-3 p-4">
            {KANBAN_COLUMNS.map((column) => {
              const columnItems = byStatus[column.value] || [];
              return (
                <div
                  key={column.value}
                  className={`flex min-h-[520px] flex-col rounded-md border transition ${
                    dragOverStatus === column.value
                      ? "border-[#3A8967] bg-[#2F7D5C]/15 shadow-[inset_0_0_0_1px_rgba(85,154,120,0.25)]"
                      : "border-white/10 bg-black/20"
                  }`}
                  onDragOver={(event) => {
                    event.preventDefault();
                    setDragOverStatus(column.value);
                  }}
                  onDragLeave={(event) => {
                    if (!event.currentTarget.contains(event.relatedTarget)) {
                      setDragOverStatus("");
                    }
                  }}
                  onDrop={() => handleDrop(column.value)}
                >
                  <div className="shrink-0 border-b border-white/10 bg-[#070807]/95 p-3 backdrop-blur">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <h3 className="truncate text-xs font-bold uppercase tracking-[0.14em] text-slate-950">{column.label}</h3>
                          <span className="rounded-full border border-white/10 bg-white/[0.045] px-2 py-0.5 text-[11px] font-semibold text-slate-500">
                            {columnItems.length}
                          </span>
                        </div>
                        <p className="mt-1 truncate text-xs text-slate-500">{column.helper}</p>
                      </div>
                      <button type="button" className="nl-icon-button h-8 w-8" onClick={() => openNewIssue(column.value)} aria-label={`Create task in ${column.label}`}>
                        +
                      </button>
                    </div>
                  </div>

                  <div className="flex flex-1 flex-col gap-3 px-3 pb-3 pt-5">
                    {columnItems.length === 0 ? (
                      draggedId ? (
                        <EmptyState title="Drop here" description="Move the selected task into this lane." />
                      ) : (
                        <div className="rounded-md border border-dashed border-white/15 bg-white/[0.02] px-3 py-5 text-center text-xs font-semibold text-white/45">
                          No task in this lane
                        </div>
                      )
                    ) : null}
                    {columnItems.map((item) => (
                      <article
                        key={item.id}
                        className={`group cursor-grab rounded-md border border-white/10 bg-white/[0.045] p-3 shadow-[0_18px_36px_rgba(0,0,0,0.22)] transition active:cursor-grabbing ${
                          String(draggedId) === String(item.id)
                            ? "scale-[0.99] opacity-60 ring-2 ring-[#3A8967]/40"
                            : "hover:-translate-y-0.5 hover:border-[#3A8967]/40 hover:bg-white/[0.065]"
                        }`}
                        draggable
                        onDragStart={(event) => {
                          event.dataTransfer.effectAllowed = "move";
                          event.dataTransfer.setData("text/plain", String(item.id));
                          setDraggedId(item.id);
                        }}
                        onDragEnd={() => {
                          setDraggedId(null);
                          setDragOverStatus("");
                        }}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-[11px] font-semibold text-[#8CC7A7]">{issueKey(item)}</span>
                          <span className={`rounded-full border px-2 py-0.5 text-[10px] font-bold uppercase ${priorityClass(item.priority)}`}>
                            {item.priority || "medium"}
                          </span>
                        </div>
                        <button
                          type="button"
                          onClick={() => startEdit(item)}
                          className="mt-2 line-clamp-2 text-left text-sm font-semibold leading-5 text-slate-950 transition group-hover:text-white"
                        >
                          {item.title}
                        </button>
                        {item.description ? <p className="mt-2 line-clamp-3 text-xs leading-5 text-slate-500">{item.description}</p> : null}
                        <div className="mt-3 flex items-center justify-between gap-2">
                          <span className="flex items-center gap-2 text-xs font-semibold text-slate-500">
                            <span className="flex h-6 w-6 items-center justify-center rounded-full border border-white/10 bg-white/[0.06] text-[10px] text-slate-950">
                              {agentInitials(item.agent_name)}
                            </span>
                            {item.agent_name || "Yannis"}
                          </span>
                          <span className={`text-[11px] font-semibold ${dueState(item.end_date).tone === "danger" ? "text-rose-700" : dueState(item.end_date).tone === "warning" ? "text-amber-700" : "text-slate-500"}`}>
                            {dueState(item.end_date).label}
                          </span>
                        </div>
                        <div className="mt-3 flex items-center justify-between gap-2 border-t border-white/10 pt-3">
                          {item.related_page ? (
                            <Link href={item.related_page} className="text-xs font-semibold text-[#8CC7A7] hover:text-white">
                              Open
                            </Link>
                          ) : (
                            <span />
                          )}
                          <div className="flex gap-1 opacity-0 transition group-hover:opacity-100">
                            <button type="button" className="rounded border border-white/10 bg-white/[0.035] px-2 py-1 text-xs font-semibold text-white/70 hover:text-white" onClick={() => startEdit(item)}>
                              Edit
                            </button>
                            <button type="button" className="rounded border border-rose-400/25 bg-rose-400/10 px-2 py-1 text-xs font-semibold text-rose-700 hover:bg-rose-400/15" onClick={() => removeItem(item.id)}>
                              Delete
                            </button>
                          </div>
                        </div>
                      </article>
                    ))}
                    <button
                      type="button"
                      onClick={() => openNewIssue(column.value)}
                      className="mt-auto flex min-h-11 items-center justify-center rounded-md border border-dashed border-white/15 bg-white/[0.025] px-3 py-2.5 text-center text-sm font-semibold text-white/55 transition hover:border-[#3A8967]/40 hover:bg-[#2F7D5C]/10 hover:text-white"
                    >
                      Add task
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
          </div>
        </section>
      </div>

      {eventModalOpen ? (
        <div className="fixed inset-0 z-[8000] flex items-start justify-center overflow-auto bg-black/75 px-4 py-10 backdrop-blur-md" role="dialog" aria-modal="true">
          <div className="w-full max-w-2xl overflow-hidden rounded-lg border border-white/10 bg-[#080B0A] shadow-[0_42px_120px_rgba(0,0,0,0.62)]">
            <div className="flex items-start justify-between gap-4 border-b border-white/10 p-5">
              <div>
                <p className="nl-kicker">{editingEventId ? "Edit event" : "New event"}</p>
                <h2 className="mt-2 text-2xl font-semibold tracking-tight text-slate-950">
                  {editingEventId ? "Update calendar event" : "Create calendar event"}
                </h2>
              </div>
              <button type="button" className="nl-icon-button" onClick={closeEventModal} aria-label="Close event form">
                x
              </button>
            </div>

            <div className="space-y-5 p-5">
              <label className="block">
                <span className="mb-1 block text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Title</span>
                <input
                  className="nl-field h-11"
                  name="calendar_event_title"
                  aria-label="Event title"
                  value={eventForm.title}
                  onChange={(event) => setEventForm((current) => ({ ...current, title: event.target.value }))}
                  placeholder="Example: Junior match vs Lorient"
                />
              </label>

              <div className="grid gap-3 md:grid-cols-2">
                <label className="block">
                  <span className="mb-1 block text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Type</span>
                  <select className="nl-field" name="calendar_event_type" aria-label="Event type" value={eventForm.event_type} onChange={(event) => setEventForm((current) => ({ ...current, event_type: event.target.value }))}>
                    {EVENT_TYPES.map((type) => <option key={type.value} value={type.value}>{type.label}</option>)}
                  </select>
                </label>
                <label className="block">
                  <span className="mb-1 block text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Location</span>
                  <input
                    className="nl-field"
                    name="calendar_event_location"
                    aria-label="Event location"
                    value={eventForm.location}
                    onChange={(event) => setEventForm((current) => ({ ...current, location: event.target.value }))}
                    placeholder="Stadium, city, call link..."
                  />
                </label>
                <label className="block">
                  <span className="mb-1 block text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Start</span>
                  <input
                    className="nl-field"
                    name="calendar_event_start"
                    aria-label="Event start date"
                    type="date"
                    value={eventForm.start_date}
                    onChange={(event) => setEventForm((current) => ({
                      ...current,
                      start_date: event.target.value,
                      end_date: current.end_date && current.end_date < event.target.value ? "" : current.end_date,
                    }))}
                  />
                </label>
                <label className="block">
                  <span className="mb-1 block text-xs font-bold uppercase tracking-[0.14em] text-slate-500">End</span>
                  <input
                    className="nl-field"
                    name="calendar_event_end"
                    aria-label="Event end date"
                    type="date"
                    min={eventForm.start_date || undefined}
                    value={eventForm.end_date}
                    onChange={(event) => setEventForm((current) => ({ ...current, end_date: event.target.value }))}
                  />
                </label>
              </div>

              <div>
                <p className="mb-2 text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Agents involved</p>
                <div className="grid gap-2 sm:grid-cols-2">
                  {AGENTS.map((agent) => {
                    const active = (eventForm.agent_names || []).includes(agent);
                    return (
                      <button
                        key={agent}
                        type="button"
                        className={`flex items-center justify-between rounded-md border px-3 py-2 text-sm font-semibold transition ${
                          active ? "border-[#3A8967]/45 bg-[#2F7D5C]/18 text-white" : "border-white/10 bg-white/[0.025] text-white/60 hover:border-white/20 hover:text-white"
                        }`}
                        onClick={() => toggleEventAgent(agent)}
                      >
                        <span className="flex items-center gap-2">
                          <span className="h-2.5 w-2.5 rounded-full" style={{ background: AGENT_COLORS[agent] }} />
                          {agent}
                        </span>
                        <span className="text-xs">{active ? "Selected" : "Add"}</span>
                      </button>
                    );
                  })}
                </div>
                <p className="mt-2 text-xs text-slate-500">Leave all agents unselected for a neutral team or match event.</p>
              </div>

              <label className="block">
                <span className="mb-1 block text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Linked page</span>
                <input
                  className="nl-field"
                  name="calendar_event_related_page"
                  aria-label="Event linked page"
                  value={eventForm.related_page}
                  onChange={(event) => setEventForm((current) => ({ ...current, related_page: event.target.value }))}
                  placeholder="/hd-players/10"
                />
              </label>

              <label className="block">
                <span className="mb-1 block text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Description</span>
                <textarea
                  className="nl-field min-h-[112px]"
                  name="calendar_event_description"
                  aria-label="Event description"
                  value={eventForm.description}
                  onChange={(event) => setEventForm((current) => ({ ...current, description: event.target.value }))}
                  placeholder="Context, people involved, notes for the team..."
                />
              </label>
              {error ? <p className="rounded-md border border-rose-400/25 bg-rose-400/10 px-3 py-2 text-sm font-semibold text-rose-700">{error}</p> : null}
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-white/10 bg-white/[0.025] p-5">
              <div>
                {editingEventId ? (
                  <button type="button" className="rounded-md border border-rose-400/25 bg-rose-400/10 px-3 py-2 text-sm font-semibold text-rose-700 hover:bg-rose-400/15" onClick={removeEvent} disabled={eventSaving}>
                    Delete event
                  </button>
                ) : null}
              </div>
              <div className="flex gap-2">
                <button type="button" className="nl-button-secondary" onClick={closeEventModal}>
                  Cancel
                </button>
                <button type="button" className="nl-button-primary" onClick={saveEvent} disabled={eventSaving}>
                  {eventSaving ? "Saving..." : editingEventId ? "Update event" : "Create event"}
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {issueModalOpen ? (
        <div className="fixed inset-0 z-[8000] flex items-start justify-center overflow-auto bg-black/75 px-4 py-10 backdrop-blur-md" role="dialog" aria-modal="true">
          <div className="w-full max-w-2xl overflow-hidden rounded-lg border border-white/10 bg-[#080B0A] shadow-[0_42px_120px_rgba(0,0,0,0.62)]">
            <div className="flex items-start justify-between gap-4 border-b border-white/10 p-5">
              <div>
                <p className="nl-kicker">{editingId ? issueKey({ id: editingId }) : "New task"}</p>
                <h2 className="mt-2 text-2xl font-semibold tracking-tight text-slate-950">
                  {editingId ? "Edit priority task" : "Create priority task"}
                </h2>
              </div>
              <button type="button" className="nl-icon-button" onClick={closeIssueModal} aria-label="Close task form">
                x
              </button>
            </div>

            <div className="space-y-5 p-5">
              <label className="block">
                <span className="mb-1 block text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Summary</span>
                <input
                  className="nl-field h-11"
                  name="task_title"
                  aria-label="Task title"
                  value={form.title}
                  onChange={(event) => setForm((current) => ({ ...current, title: event.target.value }))}
                  placeholder="Example: Send shortlist to Sporting Director"
                />
              </label>

              <label className="block">
                <span className="mb-1 block text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Description</span>
                <textarea
                  className="nl-field min-h-[112px]"
                  name="task_description"
                  aria-label="Task notes"
                  value={form.description}
                  onChange={(event) => setForm((current) => ({ ...current, description: event.target.value }))}
                  placeholder="Decision context, club constraints, next action..."
                />
              </label>

              <div className="grid gap-3 md:grid-cols-2">
                <label className="block">
                  <span className="mb-1 block text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Owner</span>
                  <select className="nl-field" name="task_agent" aria-label="Agent" value={form.agent_name} onChange={(event) => setForm((current) => ({ ...current, agent_name: event.target.value }))}>
                    {AGENTS.map((agent) => <option key={agent}>{agent}</option>)}
                  </select>
                </label>
                <label className="block">
                  <span className="mb-1 block text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Priority</span>
                  <select className="nl-field" name="task_priority" aria-label="Priority" value={form.priority} onChange={(event) => setForm((current) => ({ ...current, priority: event.target.value }))}>
                    {PRIORITIES.map((priority) => <option key={priority.value} value={priority.value}>{priority.label}</option>)}
                  </select>
                </label>
                <label className="block">
                  <span className="mb-1 block text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Status</span>
                  <select className="nl-field" name="task_status" aria-label="Status" value={form.status} onChange={(event) => setForm((current) => ({ ...current, status: event.target.value }))}>
                    {KANBAN_COLUMNS.map((column) => <option key={column.value} value={column.value}>{column.label}</option>)}
                  </select>
                </label>
                <label className="block">
                  <span className="mb-1 block text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Linked page</span>
                  <input className="nl-field" name="task_related_page" aria-label="Linked page" value={form.related_page} onChange={(event) => setForm((current) => ({ ...current, related_page: event.target.value }))} placeholder="/scouting-lab" />
                </label>
                <label className="block">
                  <span className="mb-1 block text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Start</span>
                  <input className="nl-field" name="task_start_date" aria-label="Start date" type="date" value={form.start_date} onChange={(event) => setForm((current) => ({ ...current, start_date: event.target.value }))} />
                </label>
                <label className="block">
                  <span className="mb-1 block text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Due</span>
                  <input className="nl-field" name="task_due_date" aria-label="Due date" type="date" value={form.end_date} onChange={(event) => setForm((current) => ({ ...current, end_date: event.target.value }))} />
                </label>
              </div>
              {error ? <p className="rounded-md border border-rose-400/25 bg-rose-400/10 px-3 py-2 text-sm font-semibold text-rose-700">{error}</p> : null}
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-white/10 bg-white/[0.025] p-5">
              <button type="button" className="nl-button-secondary" onClick={closeIssueModal}>
                Cancel
              </button>
              <button type="button" className="nl-button-primary" onClick={saveItem} disabled={saving}>
                {saving ? "Saving..." : editingId ? "Update task" : "Create task"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}
