import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { deleteJson, fetchJson, patchJson, postJson } from "@/lib/api";

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
    title: "HQ",
    href: "/",
    metric: "Agency command",
    desc: "Coordinate priorities, owners and decisions across the HD Sports team.",
  },
  {
    title: "HD PLAYERS",
    href: "/hd-players",
    metric: "Player portfolio",
    desc: "Centralize player rooms, documents, season data and market strategy.",
  },
  {
    title: "MERCATO 2026",
    href: "/mercato-2026",
    metric: "Market execution",
    desc: "Manage club needs, matching shortlists and agent follow-up for 2026.",
  },
  {
    title: "NETWORK",
    href: "/crm",
    metric: "Football relationships",
    desc: "Manage clubs, players, free-role contacts and prospecting workflows.",
  },
  {
    title: "SCOUTING",
    href: "/scouting-lab",
    metric: "Scouting intelligence",
    desc: "Turn ranking, reports, projections and visuals into clear recommendations.",
  },
];

const emptyForm = {
  title: "",
  description: "",
  agent_name: "Yannis",
  priority: "medium",
  status: "todo",
  start_date: new Date().toISOString().slice(0, 10),
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

export default function Home() {
  const [items, setItems] = useState([]);
  const [form, setForm] = useState(emptyForm);
  const [editingId, setEditingId] = useState(null);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [agentFilter, setAgentFilter] = useState("all");
  const [draggedId, setDraggedId] = useState(null);
  const [dragOverStatus, setDragOverStatus] = useState("");
  const [playerCount, setPlayerCount] = useState(0);

  const loadItems = async () => {
    try {
      const [priorityData, playerData] = await Promise.all([
        fetchJson("/hq/priorities"),
        fetchJson("/hd-players"),
      ]);
      setItems(priorityData.items || []);
      setPlayerCount((playerData.items || []).length);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  };

  useEffect(() => {
    loadItems();
  }, []);

  const visibleItems = useMemo(() => {
    if (agentFilter === "all") return items;
    return items.filter((item) => (item.agent_name || "Yannis") === agentFilter);
  }, [agentFilter, items]);

  const byStatus = useMemo(() => {
    const groups = Object.fromEntries(KANBAN_COLUMNS.map((column) => [column.value, []]));
    visibleItems.forEach((item) => {
      groups[normalizeStatus(item.status)].push(item);
    });
    return groups;
  }, [visibleItems]);

  const activeCount = useMemo(
    () => items.filter((item) => !["done", "completed"].includes(normalizeStatus(item.status))).length,
    [items]
  );

  const urgentCount = useMemo(() => items.filter((item) => item.priority === "urgent").length, [items]);

  const startEdit = (item) => {
    setEditingId(item.id);
    setForm({
      title: item.title || "",
      description: item.description || "",
      agent_name: item.agent_name || "Yannis",
      priority: item.priority || "medium",
      status: normalizeStatus(item.status),
      start_date: item.start_date || new Date().toISOString().slice(0, 10),
      end_date: item.end_date || "",
      related_page: item.related_page || "/",
    });
  };

  const resetForm = () => {
    setEditingId(null);
    setForm({ ...emptyForm, start_date: new Date().toISOString().slice(0, 10) });
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
      resetForm();
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
    <main className="nl-page px-4 py-8 md:py-10">
      <div className="mx-auto max-w-[1500px] space-y-6">
        <section className="surface-panel rounded-lg p-6 md:p-8">
          <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_420px] lg:items-end">
            <div>
              <p className="nl-kicker">HQ</p>
              <h1 className="mt-3 max-w-4xl text-4xl font-extrabold leading-tight text-slate-950 md:text-6xl">
                HD Sports command room.
              </h1>
              <p className="mt-4 max-w-2xl text-base leading-7 text-slate-600">
                Next Legend brings the agency portfolio, mercato priorities and scouting intelligence into one premium operating system.
              </p>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg border border-slate-200 bg-white p-4">
                <p className="text-xs font-extrabold uppercase tracking-[0.14em] text-slate-500">Active tasks</p>
                <p className="mt-2 text-3xl font-extrabold text-slate-950">{activeCount}</p>
              </div>
              <div className="rounded-lg border border-rose-200 bg-rose-50 p-4">
                <p className="text-xs font-extrabold uppercase tracking-[0.14em] text-rose-700">Urgent</p>
                <p className="mt-2 text-3xl font-extrabold text-rose-900">{urgentCount}</p>
              </div>
              <div className="rounded-lg border border-teal-200 bg-teal-50 p-4">
                <p className="text-xs font-extrabold uppercase tracking-[0.14em] text-teal-700">Agents</p>
                <p className="mt-2 text-3xl font-extrabold text-teal-950">{AGENTS.length}</p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                <p className="text-xs font-extrabold uppercase tracking-[0.14em] text-slate-500">Players</p>
                <p className="mt-2 text-3xl font-extrabold text-slate-950">{playerCount}</p>
              </div>
            </div>
          </div>
        </section>

        <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {MODULES.map((module) => (
            <Link key={module.title} href={module.href} className="surface-panel group rounded-lg p-5 transition hover:-translate-y-0.5 hover:border-teal-500">
              <span className="text-xs font-extrabold uppercase tracking-[0.16em] text-teal-700">{module.metric}</span>
              <h2 className="mt-3 text-2xl font-extrabold text-slate-950">{module.title}</h2>
              <p className="mt-2 text-sm leading-6 text-slate-600">{module.desc}</p>
              <span className="mt-5 inline-flex text-sm font-extrabold text-slate-950">Enter workspace</span>
            </Link>
          ))}
        </section>

        <section className="surface-panel rounded-lg p-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="nl-kicker">Agent priorities</p>
              <h2 className="mt-2 text-2xl font-extrabold text-slate-950">Execution board</h2>
              <p className="mt-1 text-sm leading-6 text-slate-600">
                Create, assign, filter and move the actions that drive HD Sports decisions each week.
              </p>
            </div>
            <div className="flex flex-col gap-2 sm:items-end">
              <div className="flex flex-wrap gap-2">
                {["all", ...AGENTS].map((agent) => {
                  const active = agentFilter === agent;
                  return (
                    <button
                      key={agent}
                      type="button"
                      className={`rounded-full border px-3 py-1 text-xs font-extrabold transition ${
                        active
                          ? "border-teal-600 bg-teal-600 text-white shadow-sm"
                          : "border-slate-200 bg-white text-slate-700 hover:border-teal-400"
                      }`}
                      onClick={() => setAgentFilter(agent)}
                    >
                      {agent === "all" ? "All agents" : agent}
                    </button>
                  );
                })}
              </div>
              <p className="text-xs font-bold text-slate-500">
                Showing {visibleItems.length} of {items.length} tasks
              </p>
            </div>
          </div>

          <div className="mt-5 rounded-lg border border-slate-200 bg-slate-50 p-4">
            <div className="grid gap-3 lg:grid-cols-[minmax(220px,1.4fr)_150px_150px_150px_150px]">
              <input className="nl-field" name="task_title" aria-label="Task title" value={form.title} onChange={(e) => setForm((p) => ({ ...p, title: e.target.value }))} placeholder="Task title" />
              <select className="nl-field" name="task_agent" aria-label="Agent" value={form.agent_name} onChange={(e) => setForm((p) => ({ ...p, agent_name: e.target.value }))}>
                {AGENTS.map((agent) => <option key={agent}>{agent}</option>)}
              </select>
              <select className="nl-field" name="task_priority" aria-label="Priority" value={form.priority} onChange={(e) => setForm((p) => ({ ...p, priority: e.target.value }))}>
                {PRIORITIES.map((priority) => <option key={priority.value} value={priority.value}>{priority.label}</option>)}
              </select>
              <select className="nl-field" name="task_status" aria-label="Status" value={form.status} onChange={(e) => setForm((p) => ({ ...p, status: e.target.value }))}>
                {KANBAN_COLUMNS.map((column) => <option key={column.value} value={column.value}>{column.label}</option>)}
              </select>
              <input className="nl-field" name="task_due_date" aria-label="Due date" type="date" value={form.end_date} onChange={(e) => setForm((p) => ({ ...p, end_date: e.target.value }))} />
            </div>
            <div className="mt-3 grid gap-3 lg:grid-cols-[170px_minmax(180px,260px)_minmax(260px,1fr)_auto] lg:items-start">
              <input className="nl-field" name="task_start_date" aria-label="Start date" type="date" value={form.start_date} onChange={(e) => setForm((p) => ({ ...p, start_date: e.target.value }))} />
              <input className="nl-field" name="task_related_page" aria-label="Linked page" value={form.related_page} onChange={(e) => setForm((p) => ({ ...p, related_page: e.target.value }))} placeholder="/mercato-2026" />
              <textarea className="nl-field min-h-[46px]" name="task_description" aria-label="Task notes" value={form.description} onChange={(e) => setForm((p) => ({ ...p, description: e.target.value }))} placeholder="Task notes, club context or next action" />
              <div className="flex gap-2">
                <button type="button" className="nl-button-primary whitespace-nowrap" onClick={saveItem} disabled={saving}>
                  {saving ? "Saving..." : editingId ? "Update task" : "Create task"}
                </button>
                {editingId ? <button type="button" className="nl-button-secondary" onClick={resetForm}>Cancel</button> : null}
              </div>
            </div>
            {error ? <p className="mt-3 text-sm font-semibold text-rose-700">{error}</p> : null}
          </div>

          <div className="mt-5 grid gap-4 xl:grid-cols-5">
            {KANBAN_COLUMNS.map((column) => {
              const columnItems = byStatus[column.value] || [];
              return (
                <div
                  key={column.value}
                  className={`min-h-[360px] rounded-lg border p-3 transition ${
                    dragOverStatus === column.value
                      ? "border-teal-500 bg-teal-50 shadow-[inset_0_0_0_2px_rgba(20,184,166,0.22)]"
                      : "border-slate-200 bg-slate-50/80"
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
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <h3 className="text-sm font-extrabold uppercase tracking-[0.14em] text-slate-950">{column.label}</h3>
                      <p className="mt-1 text-xs leading-5 text-slate-500">{column.helper}</p>
                    </div>
                    <span className="rounded-full border border-slate-200 bg-white px-2 py-1 text-xs font-extrabold text-slate-700">{columnItems.length}</span>
                  </div>
                  <div className="mt-3 space-y-3">
                    {columnItems.length === 0 ? (
                      <p className="rounded-md border border-dashed border-slate-300 bg-white p-3 text-sm text-slate-500">
                        {draggedId ? "Drop the priority here." : "No priority in this lane."}
                      </p>
                    ) : null}
                    {columnItems.map((item) => (
                      <article
                        key={item.id}
                        className={`cursor-grab rounded-lg border border-slate-200 bg-white p-3 shadow-sm transition active:cursor-grabbing ${
                          String(draggedId) === String(item.id) ? "scale-[0.99] opacity-60 ring-2 ring-teal-300" : "hover:-translate-y-0.5 hover:border-teal-300"
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
                        <div className="flex items-start justify-between gap-2">
                          <h4 className="text-sm font-extrabold leading-5 text-slate-950">{item.title}</h4>
                          <span className={`rounded-full border px-2 py-0.5 text-[11px] font-extrabold uppercase ${priorityClass(item.priority)}`}>
                            {item.priority || "medium"}
                          </span>
                        </div>
                        <div className="mt-3 flex flex-wrap gap-2">
                          <span className="rounded-full border border-slate-200 bg-slate-50 px-2 py-1 text-xs font-bold text-slate-700">
                            {item.agent_name || "Yannis"}
                          </span>
                          {item.end_date ? (
                            <span className="rounded-full border border-slate-200 bg-slate-50 px-2 py-1 text-xs font-bold text-slate-700">
                              Due {item.end_date}
                            </span>
                          ) : null}
                        </div>
                        {item.description ? <p className="mt-3 text-sm leading-5 text-slate-600">{item.description}</p> : null}
                        <select
                          className="nl-field mt-3 h-9 text-xs font-bold"
                          name={`task_status_${item.id}`}
                          aria-label={`Move ${item.title}`}
                          value={normalizeStatus(item.status)}
                          onChange={(event) => moveItem(item, event.target.value)}
                        >
                          {KANBAN_COLUMNS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                        </select>
                        <div className="mt-3 flex flex-wrap gap-2">
                          <button type="button" className="rounded-md border border-slate-200 bg-white px-2 py-1 text-xs font-extrabold text-slate-700 hover:border-teal-400" onClick={() => startEdit(item)}>Edit</button>
                          <button type="button" className="rounded-md border border-slate-200 bg-white px-2 py-1 text-xs font-extrabold text-slate-700 hover:border-rose-300" onClick={() => removeItem(item.id)}>Delete</button>
                          {item.related_page ? <Link href={item.related_page} className="rounded-md border border-slate-200 bg-white px-2 py-1 text-xs font-extrabold text-slate-700 hover:border-teal-400">Open</Link> : null}
                        </div>
                      </article>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      </div>
    </main>
  );
}
