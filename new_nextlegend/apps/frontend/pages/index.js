import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { deleteJson, fetchJson, patchJson, postJson } from "@/lib/api";
import { EmptyState, PageHeader, Panel } from "@/components/ui/product";

const AGENTS = ["Steven", "Don", "Yannis", "Lidahi"];

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

const isOpenIssue = (item) => !["done", "completed"].includes(normalizeStatus(item.status));

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

export default function Home() {
  const [items, setItems] = useState([]);
  const [form, setForm] = useState(emptyForm);
  const [editingId, setEditingId] = useState(null);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [agentFilter, setAgentFilter] = useState("all");
  const [priorityFilter, setPriorityFilter] = useState("all");
  const [draggedId, setDraggedId] = useState(null);
  const [dragOverStatus, setDragOverStatus] = useState("");
  const [issueModalOpen, setIssueModalOpen] = useState(false);

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

  const statusSummary = useMemo(
    () => KANBAN_COLUMNS.map((column) => ({
      ...column,
      count: items.filter((item) => normalizeStatus(item.status) === column.value).length,
    })),
    [items]
  );

  const summary = useMemo(() => {
    const openItems = items.filter(isOpenIssue);
    const urgentItems = openItems.filter((item) => item.priority === "urgent");
    const highItems = openItems.filter((item) => item.priority === "high");
    const reviewItems = openItems.filter((item) => normalizeStatus(item.status) === "review");
    const overdueItems = openItems.filter((item) => dueState(item.end_date).tone === "danger");
    const nextActions = [...urgentItems, ...highItems, ...reviewItems]
      .filter((item, index, source) => source.findIndex((entry) => entry.id === item.id) === index)
      .slice(0, 4);
    const ownerLoad = AGENTS.map((agent) => ({
      agent,
      count: openItems.filter((item) => (item.agent_name || "Yannis") === agent).length,
    })).sort((a, b) => b.count - a.count);
    return { openItems, urgentItems, highItems, reviewItems, overdueItems, nextActions, ownerLoad };
  }, [items]);

  const openNewIssue = (status = "todo") => {
    setError("");
    setEditingId(null);
    setForm({ ...emptyForm, status, start_date: today() });
    setIssueModalOpen(true);
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
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const moveItem = async (item, status) => {
    if (normalizeStatus(item.status) === status) return;
    try {
      await patchJson(`/hq/priorities/${item.id}`, { ...item, status });
      await loadItems();
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
    } catch (err) {
      setError(err.message);
    }
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
              New issue
            </button>
          }
        />

        <section className="grid gap-4 xl:grid-cols-[minmax(0,1.55fr)_minmax(320px,0.85fr)]">
          <Panel className="overflow-hidden p-0">
            <div className="border-b border-white/10 p-5">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="nl-kicker">Priority briefing</p>
                  <h2 className="mt-2 text-2xl font-semibold tracking-tight text-slate-950">
                    {summary.nextActions.length ? `${summary.nextActions.length} actions need attention` : "No critical action pending"}
                  </h2>
                </div>
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div className="rounded-md border border-white/10 bg-white/[0.035] px-3 py-2">
                    <p className="text-lg font-semibold text-slate-950">{summary.urgentItems.length}</p>
                    <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-500">Urgent</p>
                  </div>
                  <div className="rounded-md border border-white/10 bg-white/[0.035] px-3 py-2">
                    <p className="text-lg font-semibold text-slate-950">{summary.reviewItems.length}</p>
                    <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-500">Review</p>
                  </div>
                  <div className="rounded-md border border-white/10 bg-white/[0.035] px-3 py-2">
                    <p className="text-lg font-semibold text-slate-950">{summary.overdueItems.length}</p>
                    <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-500">Late</p>
                  </div>
                </div>
              </div>
            </div>
            <div className="divide-y divide-white/10">
              {summary.nextActions.length ? (
                summary.nextActions.map((item) => {
                  const due = dueState(item.end_date);
                  return (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => startEdit(item)}
                      className="grid w-full gap-3 px-5 py-4 text-left transition hover:bg-white/[0.045] md:grid-cols-[minmax(0,1fr)_120px_96px]"
                    >
                      <span className="min-w-0">
                        <span className="mb-1 flex flex-wrap items-center gap-2">
                          <span className="text-xs font-semibold text-[#8CC7A7]">{issueKey(item)}</span>
                          <span className={`rounded-full border px-2 py-0.5 text-[10px] font-bold uppercase ${priorityClass(item.priority)}`}>
                            {item.priority || "medium"}
                          </span>
                        </span>
                        <span className="block truncate text-sm font-semibold text-slate-950">{item.title}</span>
                        {item.description ? <span className="mt-1 block truncate text-xs text-slate-500">{item.description}</span> : null}
                      </span>
                      <span className="flex items-center gap-2 text-xs font-semibold text-slate-500">
                        <span className="flex h-6 w-6 items-center justify-center rounded-full border border-white/10 bg-white/[0.06] text-[10px] text-slate-950">
                          {agentInitials(item.agent_name)}
                        </span>
                        {item.agent_name || "Yannis"}
                      </span>
                      <span className={`text-xs font-semibold ${due.tone === "danger" ? "text-rose-700" : due.tone === "warning" ? "text-amber-700" : "text-slate-500"}`}>
                        {due.label}
                      </span>
                    </button>
                  );
                })
              ) : (
                <div className="p-5">
                  <EmptyState
                    title="Board is under control"
                    description="No urgent, high-priority or review item is currently blocking the team."
                    action={
                      <button type="button" className="nl-button-secondary" onClick={() => openNewIssue("todo")}>
                        Create next action
                      </button>
                    }
                  />
                </div>
              )}
            </div>
          </Panel>

          <Panel className="p-5">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="nl-kicker">Ownership</p>
                <h2 className="mt-1 text-xl font-semibold text-slate-950">Team load</h2>
              </div>
              <span className="rounded-md border border-[#3A8967]/30 bg-[#2F7D5C]/15 px-2.5 py-1 text-xs font-semibold text-[#8CC7A7]">
                {summary.openItems.length} open
              </span>
            </div>
            <div className="mt-5 space-y-4">
              {summary.ownerLoad.map((owner) => {
                const width = summary.openItems.length ? Math.max(8, (owner.count / summary.openItems.length) * 100) : 0;
                return (
                  <button
                    key={owner.agent}
                    type="button"
                    onClick={() => setAgentFilter(owner.agent)}
                    className="block w-full rounded-md border border-white/10 bg-white/[0.025] p-3 text-left transition hover:border-[#3A8967]/40 hover:bg-white/[0.05]"
                  >
                    <span className="flex items-center justify-between gap-3">
                      <span className="flex items-center gap-2 text-sm font-semibold text-slate-950">
                        <span className="flex h-7 w-7 items-center justify-center rounded-full border border-white/10 bg-white/[0.06] text-[10px]">
                          {agentInitials(owner.agent)}
                        </span>
                        {owner.agent}
                      </span>
                      <span className="text-xs font-semibold text-slate-500">{owner.count} open</span>
                    </span>
                    <span className="mt-3 block h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
                      <span className="block h-full rounded-full bg-[#3A8967]" style={{ width: `${width}%` }} />
                    </span>
                  </button>
                );
              })}
              <div className="pt-1">
                {statusSummary.map((column) => (
                  <div key={column.value} className="mt-2 flex items-center justify-between text-xs">
                    <span className="text-slate-500">{column.label}</span>
                    <span className="font-semibold text-slate-950">{column.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </Panel>
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
                <h2 className="mt-2 text-2xl font-semibold tracking-tight text-slate-950">Issues</h2>
                <p className="mt-1 text-sm leading-6 text-slate-600">
                  Jira-style execution board for weekly priorities, ownership and decisions.
                </p>
              </div>
              <button type="button" className="nl-button-primary" onClick={() => openNewIssue("todo")}>
                New issue
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
                  <div className="sticky top-[64px] z-10 border-b border-white/10 bg-[#070807]/95 p-3 backdrop-blur">
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
                      <button type="button" className="nl-icon-button h-8 w-8" onClick={() => openNewIssue(column.value)} aria-label={`Create issue in ${column.label}`}>
                        +
                      </button>
                    </div>
                  </div>

                  <div className="flex flex-1 flex-col gap-3 p-3">
                    {columnItems.length === 0 ? (
                      draggedId ? (
                        <EmptyState title="Drop here" description="Move the selected issue into this lane." />
                      ) : (
                        <button
                          type="button"
                          onClick={() => openNewIssue(column.value)}
                          className="rounded-md border border-dashed border-white/15 bg-white/[0.025] px-3 py-8 text-center text-sm font-semibold text-white/55 transition hover:border-[#3A8967]/40 hover:bg-[#2F7D5C]/10 hover:text-white"
                        >
                          Add issue
                        </button>
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
                  </div>
                </div>
              );
            })}
          </div>
          </div>
        </section>
      </div>

      {issueModalOpen ? (
        <div className="fixed inset-0 z-[8000] flex items-start justify-center overflow-auto bg-black/75 px-4 py-10 backdrop-blur-md" role="dialog" aria-modal="true">
          <div className="w-full max-w-2xl overflow-hidden rounded-lg border border-white/10 bg-[#080B0A] shadow-[0_42px_120px_rgba(0,0,0,0.62)]">
            <div className="flex items-start justify-between gap-4 border-b border-white/10 p-5">
              <div>
                <p className="nl-kicker">{editingId ? issueKey({ id: editingId }) : "New issue"}</p>
                <h2 className="mt-2 text-2xl font-semibold tracking-tight text-slate-950">
                  {editingId ? "Edit priority issue" : "Create priority issue"}
                </h2>
              </div>
              <button type="button" className="nl-icon-button" onClick={closeIssueModal} aria-label="Close issue form">
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
                {saving ? "Saving..." : editingId ? "Update issue" : "Create issue"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}
