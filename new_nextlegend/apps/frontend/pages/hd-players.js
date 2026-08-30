import { useEffect, useState } from "react";
import Link from "next/link";
import ClubLogo from "@/components/ClubLogo";
import { fetchJson, postJson } from "@/lib/api";

const AGENTS = ["Steven", "Don", "Yannis", "Lidahi"];

const emptyPlayer = {
  display_name: "",
  current_club: "",
  position: "",
  plan: "",
  priority: "B",
  demanded_transfer_fee: "",
  next_step: "",
  assigned_agent: "Yannis",
  birth_date: "",
};

const money = (value) => {
  if (value === null || value === undefined || value === "") return "-";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";
  if (Math.abs(numeric) >= 1000000) return `${Math.round((numeric / 1000000) * 10) / 10}M`;
  if (Math.abs(numeric) >= 1000) return `${Math.round(numeric / 1000)}K`;
  return `${Math.round(numeric)}`;
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
        <header className="surface-panel relative overflow-hidden rounded-lg p-6 md:p-8">
          <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-amber-200/50 to-transparent" />
          <div className="flex flex-col gap-5">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
              <div>
              <p className="nl-kicker">HD Sports portfolio</p>
              <h1 className="mt-2 text-3xl font-semibold text-slate-950 md:text-5xl">
                Player rooms built for representation.
              </h1>
              <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">
                Manage every represented player with market strategy, documents, scouting evidence and next actions in one place.
              </p>
              </div>
              <button type="button" className="nl-button-primary w-full shrink-0 lg:w-auto" onClick={() => setShowCreate((value) => !value)}>
                New Player
              </button>
            </div>
            <div className="flex flex-wrap gap-2">
              <input className="nl-field w-full sm:w-64" name="hd_search" aria-label="Search player or club" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search player or club" />
              <select className="nl-field w-full sm:w-44" name="hd_agent_filter" aria-label="Agent filter" value={agent} onChange={(e) => setAgent(e.target.value)}>
                <option value="">All agents</option>
                {AGENTS.map((name) => <option key={name}>{name}</option>)}
              </select>
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
              <select className="nl-field" name="hd_assigned_agent" aria-label="Assigned agent" value={form.assigned_agent} onChange={(e) => setForm((p) => ({ ...p, assigned_agent: e.target.value }))}>
                {AGENTS.map((name) => <option key={name}>{name}</option>)}
              </select>
              <input className="nl-field" name="hd_transfer_fee" aria-label="Demanded transfer fee" value={form.demanded_transfer_fee} onChange={(e) => setForm((p) => ({ ...p, demanded_transfer_fee: e.target.value }))} placeholder="Demanded transfer fee" />
              <input className="nl-field" name="hd_birth_date" aria-label="Birth date" type="date" value={form.birth_date} onChange={(e) => setForm((p) => ({ ...p, birth_date: e.target.value }))} />
              <input className="nl-field md:col-span-2" name="hd_next_step" aria-label="Next step" value={form.next_step} onChange={(e) => setForm((p) => ({ ...p, next_step: e.target.value }))} placeholder="Next step" />
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
              className="surface-panel group overflow-hidden rounded-lg transition hover:-translate-y-0.5 hover:border-amber-200/35"
            >
              <div className="relative h-64 bg-[linear-gradient(135deg,#0b0f18,#151923)]">
                {item.photo_url ? (
                  <img src={storageHref(item.photo_url)} alt="" className="h-full w-full object-cover transition duration-300 group-hover:scale-[1.03]" />
                ) : (
                  <div className="flex h-full items-center justify-center text-6xl font-black text-white/[0.22]">
                    {initials(item.display_name)}
                  </div>
                )}
                <div className="absolute left-4 top-4 flex gap-2">
                  <span className="rounded-full border border-white/10 bg-black/50 px-3 py-1 text-xs font-black text-white/80 backdrop-blur">
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
