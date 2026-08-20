import { useEffect, useState } from "react";
import Link from "next/link";
import ClubLogo from "@/components/ClubLogo";
import { fetchJson, postJson } from "@/lib/api";

const AGENTS = ["Steven", "Don", "Yannis", "Lidahi"];
const PRIORITIES = ["A", "B", "C", "D"];

const emptyPlayer = {
  display_name: "",
  current_club: "",
  position: "",
  plan: "",
  priority: "B",
  demanded_transfer_fee: "",
  next_step: "",
  assigned_agent: "Yannis",
  photo_url: "",
};

const money = (value) => {
  if (value === null || value === undefined || value === "") return "-";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";
  if (Math.abs(numeric) >= 1000000) return `${Math.round((numeric / 1000000) * 10) / 10}M`;
  if (Math.abs(numeric) >= 1000) return `${Math.round(numeric / 1000)}K`;
  return `${Math.round(numeric)}`;
};

const priorityClass = (priority) => {
  if (priority === "A") return "border-rose-200 bg-rose-50 text-rose-800";
  if (priority === "B") return "border-amber-200 bg-amber-50 text-amber-800";
  if (priority === "C") return "border-teal-200 bg-teal-50 text-teal-800";
  return "border-slate-200 bg-slate-50 text-slate-700";
};

const initials = (name) =>
  String(name || "")
    .split(" ")
    .map((part) => part[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

export default function HdPlayersPage() {
  const [items, setItems] = useState([]);
  const [query, setQuery] = useState("");
  const [agent, setAgent] = useState("");
  const [form, setForm] = useState(emptyPlayer);
  const [showCreate, setShowCreate] = useState(false);
  const [message, setMessage] = useState("");
  const [saving, setSaving] = useState(false);

  const load = async () => {
    const data = await fetchJson("/hd-players", { q: query, agent });
    setItems(data.items || []);
  };

  useEffect(() => {
    load().catch((err) => setMessage(err.message));
  }, [query, agent]);

  const createPlayer = async () => {
    if (!form.display_name.trim()) {
      setMessage("Player name is required.");
      return;
    }
    setSaving(true);
    try {
      const created = await postJson("/hd-players", {
        ...form,
        demanded_transfer_fee: form.demanded_transfer_fee ? Number(form.demanded_transfer_fee) : null,
      });
      window.location.href = `/hd-players/${created.id}`;
    } catch (err) {
      setMessage(err.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <main className="nl-page px-4 py-8">
      <div className="mx-auto max-w-[1500px] space-y-6">
        <header className="surface-panel rounded-lg p-6 md:p-8">
          <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
            <div>
              <p className="nl-kicker">HD Sports portfolio</p>
              <h1 className="mt-2 text-4xl font-extrabold text-slate-950 md:text-5xl">
                Player rooms built for representation.
              </h1>
              <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">
                Manage every represented player with market strategy, documents, scouting evidence and next actions in one place.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <input className="nl-field w-full sm:w-64" name="hd_search" aria-label="Search player or club" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search player or club" />
              <select className="nl-field w-full sm:w-44" name="hd_agent_filter" aria-label="Agent filter" value={agent} onChange={(e) => setAgent(e.target.value)}>
                <option value="">All agents</option>
                {AGENTS.map((name) => <option key={name}>{name}</option>)}
              </select>
              <button type="button" className="nl-button-primary" onClick={() => setShowCreate((value) => !value)}>
                New player
              </button>
            </div>
          </div>
        </header>

        {showCreate ? (
          <section className="surface-panel rounded-lg p-5">
            <p className="nl-kicker">New player room</p>
            <div className="mt-4 grid gap-3 md:grid-cols-4">
              <input className="nl-field md:col-span-2" name="hd_player_name" aria-label="Player name" value={form.display_name} onChange={(e) => setForm((p) => ({ ...p, display_name: e.target.value }))} placeholder="Player name" />
              <input className="nl-field" name="hd_current_club" aria-label="Current club" value={form.current_club} onChange={(e) => setForm((p) => ({ ...p, current_club: e.target.value }))} placeholder="Current club" />
              <input className="nl-field" name="hd_position" aria-label="Position" value={form.position} onChange={(e) => setForm((p) => ({ ...p, position: e.target.value }))} placeholder="Position" />
              <input className="nl-field md:col-span-2" name="hd_plan" aria-label="Plan" value={form.plan} onChange={(e) => setForm((p) => ({ ...p, plan: e.target.value }))} placeholder="Market plan" />
              <select className="nl-field" name="hd_priority" aria-label="Priority" value={form.priority} onChange={(e) => setForm((p) => ({ ...p, priority: e.target.value }))}>
                {PRIORITIES.map((priority) => <option key={priority}>{priority}</option>)}
              </select>
              <select className="nl-field" name="hd_assigned_agent" aria-label="Assigned agent" value={form.assigned_agent} onChange={(e) => setForm((p) => ({ ...p, assigned_agent: e.target.value }))}>
                {AGENTS.map((name) => <option key={name}>{name}</option>)}
              </select>
              <input className="nl-field" name="hd_transfer_fee" aria-label="Demanded transfer fee" value={form.demanded_transfer_fee} onChange={(e) => setForm((p) => ({ ...p, demanded_transfer_fee: e.target.value }))} placeholder="Demanded transfer fee" />
              <input className="nl-field md:col-span-2" name="hd_next_step" aria-label="Next step" value={form.next_step} onChange={(e) => setForm((p) => ({ ...p, next_step: e.target.value }))} placeholder="Next step" />
              <input className="nl-field md:col-span-2" name="hd_photo_url" aria-label="Photo URL" value={form.photo_url} onChange={(e) => setForm((p) => ({ ...p, photo_url: e.target.value }))} placeholder="Photo URL" />
            </div>
            <button type="button" className="nl-button-primary mt-4" onClick={createPlayer} disabled={saving}>
              {saving ? "Creating..." : "Create player room"}
            </button>
          </section>
        ) : null}

        <section className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
          {items.map((item) => (
            <Link
              key={item.id}
              href={`/hd-players/${item.id}`}
              className="surface-panel group overflow-hidden rounded-lg transition hover:-translate-y-0.5 hover:border-teal-500"
            >
              <div className="relative h-64 bg-[radial-gradient(circle_at_top_left,rgba(20,184,166,0.18),transparent_34%),linear-gradient(135deg,#f8fafc,#e2e8f0)]">
                {item.photo_url ? (
                  <img src={item.photo_url} alt="" className="h-full w-full object-cover transition duration-300 group-hover:scale-[1.03]" />
                ) : (
                  <div className="flex h-full items-center justify-center text-6xl font-black text-slate-300">
                    {initials(item.display_name)}
                  </div>
                )}
                <div className="absolute left-4 top-4 flex gap-2">
                  <span className={`rounded-full border px-3 py-1 text-xs font-black ${priorityClass(item.priority)}`}>
                    Priority {item.priority || "B"}
                  </span>
                  <span className="rounded-full border border-slate-200 bg-white/90 px-3 py-1 text-xs font-black text-slate-700">
                    {item.assigned_agent || "Unassigned"}
                  </span>
                </div>
              </div>

              <div className="space-y-4 p-5">
                <div>
                  <h2 className="text-2xl font-extrabold text-slate-950">{item.display_name}</h2>
                  <div className="mt-2 flex items-center gap-2 text-sm font-semibold text-slate-500">
                    <ClubLogo name={item.current_club} className="h-7 w-7" />
                    <span className="min-w-0 truncate">
                      {item.current_club || "Club -"} {item.position ? `• ${item.position}` : ""}
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-2">
                  <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
                    <p className="text-[11px] font-black uppercase text-slate-500">Fee</p>
                    <p className="mt-1 font-extrabold text-slate-950">{money(item.demanded_transfer_fee)}</p>
                  </div>
                  <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
                    <p className="text-[11px] font-black uppercase text-slate-500">Docs</p>
                    <p className="mt-1 font-extrabold text-slate-950">{(item.documents || []).length}</p>
                  </div>
                  <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
                    <p className="text-[11px] font-black uppercase text-slate-500">Data link</p>
                    <p className="mt-1 truncate font-extrabold text-slate-950">{item.linked_player_name ? "Linked" : "To link"}</p>
                  </div>
                </div>

                <div className="rounded-md border border-slate-200 bg-white p-3">
                  <p className="text-[11px] font-black uppercase tracking-[0.14em] text-slate-500">Plan</p>
                  <p className="mt-1 text-sm font-semibold text-slate-800">{item.plan || "-"}</p>
                  <p className="mt-3 text-[11px] font-black uppercase tracking-[0.14em] text-slate-500">Next step</p>
                  <p className="mt-1 text-sm text-slate-600">{item.next_step || "-"}</p>
                </div>
              </div>
            </Link>
          ))}
        </section>

        {message ? <p className="text-sm font-semibold text-rose-700">{message}</p> : null}
      </div>
    </main>
  );
}
