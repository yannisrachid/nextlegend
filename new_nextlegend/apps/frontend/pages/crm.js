import { useEffect, useMemo, useRef, useState } from "react";
import { apiUrl, deleteJson, fetchJson, patchJson, postJson } from "@/lib/api";
import { loadClubLogoData, resolveClubLogoUrl } from "@/lib/clubLogos";

const TABS = [
  { id: "clubs", label: "Clubs" },
  { id: "players", label: "Players" },
  { id: "contacts", label: "Contacts" },
  { id: "prospection", label: "Prospection" },
  { id: "map", label: "Map" },
];

const STAGES = [
  { id: "prequalification", label: "Prequalification" },
  { id: "relance1", label: "Follow-up 1" },
  { id: "relance2", label: "Follow-up 2" },
  { id: "relance3", label: "Follow-up 3" },
];

const emptyClub = { name: "", city: "", country: "", logo: "", email: "", phone: "", website: "" };
const emptyPlayer = { first_name: "", last_name: "", age: 0, position: "", nationality: "", club_id: "", photo: "", email: "", phone: "" };
const emptyContact = { first_name: "", last_name: "", role: "", type: "CLUB", club_id: "", player_id: "", email: "", phone: "", notes: "" };
const emptyProspect = { contact_id: "", stage: "prequalification", notes: "" };

const plural = { club: "clubs", player: "players", contact: "contacts" };
const text = (value, fallback = "Unlinked") => value || fallback;
const normalizeSearchText = (value) =>
  String(value || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();
const fullName = (item, prefix = "") =>
  [item?.[`${prefix}first_name`] || item?.first_name, item?.[`${prefix}last_name`] || item?.last_name]
    .filter(Boolean)
    .join(" ");
const ageLabel = (age) => {
  const value = Number(age || 0);
  return value > 0 ? `${value} yrs` : "Unknown age";
};
const initials = (name) =>
  String(name || "")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("") || "FC";

const mapInitials = (name) => {
  const rawName = String(name || "");
  const normalizedName = rawName.toLowerCase();
  if (normalizedName.includes("olympique") && normalizedName.includes("marseille")) return "OM";

  const weakWords = new Set(["a", "ac", "afc", "as", "athletic", "bk", "ca", "calcio", "cd", "cf", "club", "de", "del", "des", "di", "fc", "if", "la", "le", "los", "of", "olympique", "real", "sc", "sk", "ssc", "the", "ud", "us"]);
  const parts = rawName
    .replace(/[^\p{L}\p{N}\s.-]/gu, " ")
    .split(/[\s.-]+/)
    .filter(Boolean);
  const meaningful = parts.filter((part) => !weakWords.has(part.toLowerCase()));
  const source = meaningful.length ? meaningful : parts;
  if (!source.length) return "FC";
  if (source.length === 1) return source[0].slice(0, 2).toUpperCase();
  return source.slice(0, 2).map((part) => part[0]?.toUpperCase()).join("") || "FC";
};

const directLogoUrl = (src) => {
  const raw = String(src || "").trim();
  if (!raw) return "";
  const wikiFile = raw.match(/^https?:\/\/[^/]*wikipedia\.org\/wiki\/File:(.+)$/i);
  if (wikiFile?.[1]) return `https://commons.wikimedia.org/wiki/Special:Redirect/file/${wikiFile[1]}`;
  return raw;
};

const escapeHtml = (value) =>
  String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

function Field({ label, children }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[11px] font-black uppercase tracking-[0.16em] text-slate-500">{label}</span>
      {children}
    </label>
  );
}

function Pagination({ page, totalPages, onPage }) {
  return (
    <div className="mt-5 flex flex-wrap items-center justify-between gap-3 text-sm">
      <span className="font-bold text-slate-500">Page {page} of {Math.max(totalPages || 1, 1)}</span>
      <div className="flex gap-2">
        <button type="button" className="nl-button-secondary px-3 py-1" disabled={page <= 1} onClick={() => onPage(page - 1)}>Previous</button>
        <button type="button" className="nl-button-secondary px-3 py-1" disabled={page >= totalPages} onClick={() => onPage(page + 1)}>Next</button>
      </div>
    </div>
  );
}

function Logo({ src, name, onClick, size = "h-12 w-12" }) {
  const [fallbackUrl, setFallbackUrl] = useState("");
  const [fallbackLoaded, setFallbackLoaded] = useState(false);
  const [directFailed, setDirectFailed] = useState(false);
  const directUrl = directLogoUrl(src);
  const logoUrl = fallbackLoaded ? (fallbackUrl || (directUrl && !directFailed ? directUrl : "")) : "";

  useEffect(() => {
    let active = true;
    setDirectFailed(false);
    setFallbackLoaded(false);
    loadClubLogoData().then((data) => {
      if (!active) return;
      setFallbackUrl(resolveClubLogoUrl(name, data));
      setFallbackLoaded(true);
    });
    return () => {
      active = false;
    };
  }, [name, src]);

  const content = logoUrl ? (
    <img
      src={logoUrl}
      alt={name ? `${name} logo` : "Club logo"}
      className="h-full w-full object-contain p-1.5"
      loading="lazy"
      onError={() => {
        if (directUrl && !directFailed) setDirectFailed(true);
      }}
    />
  ) : (
    <span className="flex h-full w-full items-center justify-center bg-teal-50 text-sm font-black text-teal-800">
      {initials(name)}
    </span>
  );
  const className = `${size} shrink-0 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm transition hover:-translate-y-0.5 hover:border-teal-400 hover:shadow-md`;

  if (!onClick) {
    return <span className={className} title={name || "Club logo"}>{content}</span>;
  }

  return (
    <button
      type="button"
      onClick={onClick}
      className={className}
      title={name || "Open"}
    >
      {content}
    </button>
  );
}

function StatCard({ label, value, sub }) {
  return (
    <div className="surface-subtle rounded-lg p-4">
      <p className="text-[11px] font-black uppercase tracking-[0.16em] text-slate-500">{label}</p>
      <p className="mt-2 text-3xl font-black tracking-tight text-slate-950">{value ?? 0}</p>
      {sub ? <p className="mt-1 text-xs font-bold text-slate-500">{sub}</p> : null}
    </div>
  );
}

function EntityToolbar({ kind, search, setSearch, runSearch, exportPath, onCreate }) {
  return (
    <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
      <div>
        <p className="nl-kicker">CRM {kind}</p>
        <h2 className="mt-1 text-2xl font-black capitalize tracking-tight text-slate-950">{kind}</h2>
      </div>
      <div className="flex min-w-0 flex-col gap-2 sm:flex-row">
        <input
          className="nl-field sm:w-80"
          placeholder={`Search ${kind}`}
          value={search[kind] || ""}
          onChange={(e) => setSearch((current) => ({ ...current, [kind]: e.target.value }))}
          onKeyDown={(e) => e.key === "Enter" && runSearch(kind)}
        />
        {search[kind] ? (
          <button
            type="button"
            className="nl-button-secondary"
            onClick={() => setSearch((current) => ({ ...current, [kind]: "" }))}
          >
            Clear
          </button>
        ) : null}
        <a className="nl-button-secondary" href={apiUrl(exportPath)}>Export</a>
        <button type="button" className="nl-button-primary" onClick={onCreate}>Create {kind.slice(0, -1)}</button>
      </div>
    </div>
  );
}

function Tag({ children, tone = "slate" }) {
  const tones = {
    slate: "border-slate-200 bg-slate-50 text-slate-700",
    teal: "border-teal-200 bg-teal-50 text-teal-800",
    amber: "border-amber-200 bg-amber-50 text-amber-800",
    rose: "border-rose-200 bg-rose-50 text-rose-800",
  };
  return <span className={`rounded-full border px-2.5 py-1 text-[11px] font-black uppercase tracking-[0.08em] ${tones[tone]}`}>{children}</span>;
}

function EmptyState({ label }) {
  return <div className="rounded-lg border border-dashed border-white/15 bg-white/[0.035] p-8 text-center text-sm font-bold text-slate-500">{label}</div>;
}

export default function CRM() {
  const [tab, setTab] = useState("clubs");
  const [summary, setSummary] = useState({});
  const [options, setOptions] = useState({ clubs: [], players: [], contacts: [] });
  const [lists, setLists] = useState({ clubs: {}, players: {}, contacts: {}, prospects: { data: [] }, cities: { data: [] }, map: { data: [] } });
  const [search, setSearch] = useState({ clubs: "", players: "", contacts: "" });
  const [page, setPage] = useState({ clubs: 1, players: 1, contacts: 1 });
  const [clubForm, setClubForm] = useState(emptyClub);
  const [playerForm, setPlayerForm] = useState(emptyPlayer);
  const [contactForm, setContactForm] = useState(emptyContact);
  const [prospectForm, setProspectForm] = useState(emptyProspect);
  const [prospectContactQuery, setProspectContactQuery] = useState("");
  const [editing, setEditing] = useState({ type: "", id: "" });
  const [formModal, setFormModal] = useState({ type: "" });
  const [detail, setDetail] = useState({ type: "", id: "", data: null, loading: false });
  const [mapSelection, setMapSelection] = useState(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const loadSummary = async () => setSummary(await fetchJson("/crm/summary"));
  const loadOptions = async () => setOptions(await fetchJson("/crm/options"));

  const loadList = async (kind, overrides = {}) => {
    const nextPage = overrides.page || page[kind] || 1;
    const nextSearch = Object.prototype.hasOwnProperty.call(overrides, "search") ? overrides.search : search[kind];
    const data = await fetchJson(`/crm/${kind}`, { page: nextPage, pageSize: 25, search: nextSearch });
    setLists((current) => ({ ...current, [kind]: data }));
  };

  const loadProspects = async () => {
    const data = await fetchJson("/crm/prospects");
    setLists((current) => ({ ...current, prospects: data }));
  };

  const loadCities = async () => {
    const data = await fetchJson("/crm/cities");
    setLists((current) => ({ ...current, cities: data }));
  };

  const loadMap = async () => {
    const data = await fetchJson("/crm/map-clusters");
    setLists((current) => ({ ...current, map: data }));
    setMapSelection((current) => current || (data.data || [])[0] || null);
  };

  const refreshAll = async () => {
    try {
      await Promise.all([loadSummary(), loadOptions(), loadList("clubs"), loadList("players"), loadList("contacts"), loadProspects(), loadCities(), loadMap()]);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  };

  useEffect(() => {
    refreshAll();
  }, []);

  useEffect(() => {
    if (["clubs", "players", "contacts"].includes(tab)) loadList(tab).catch((err) => setError(err.message));
    if (tab === "prospection") loadProspects().catch((err) => setError(err.message));
    if (tab === "map") loadMap().catch((err) => setError(err.message));
  }, [tab]);

  useEffect(() => {
    if (!["clubs", "players", "contacts"].includes(tab)) return undefined;
    const query = search[tab] || "";
    const timeout = window.setTimeout(() => {
      setPage((current) => ({ ...current, [tab]: 1 }));
      loadList(tab, { page: 1, search: query }).catch((err) => setError(err.message));
    }, 250);
    return () => window.clearTimeout(timeout);
  }, [search.clubs, search.players, search.contacts, tab]);

  const clubs = lists.clubs?.data || [];
  const players = lists.players?.data || [];
  const contacts = lists.contacts?.data || [];
  const prospects = lists.prospects?.data || [];
  const mapClusters = lists.map?.data || [];

  const contactsWithoutProspect = useMemo(() => {
    const prospectContactIds = new Set(prospects.map((item) => item.contact_id));
    return (options.contacts || []).filter((item) => !prospectContactIds.has(item.id));
  }, [prospects, options.contacts]);

  const byStage = useMemo(() => {
    const groups = Object.fromEntries(STAGES.map((stage) => [stage.id, []]));
    prospects.forEach((item) => groups[item.stage || "prequalification"]?.push(item));
    return groups;
  }, [prospects]);

  const openDetail = async (type, id) => {
    if (!id) return;
    setDetail({ type, id, data: null, loading: true });
    try {
      const data = await fetchJson(`/crm/${plural[type]}/${id}`);
      setDetail({ type, id, data, loading: false });
      setError("");
    } catch (err) {
      setDetail({ type: "", id: "", data: null, loading: false });
      setError(err.message);
    }
  };

  const saveClub = async () => {
    const payload = { ...clubForm };
    if (editing.type === "club") await patchJson(`/crm/clubs/${editing.id}`, payload);
    else await postJson("/crm/clubs", payload);
    setClubForm(emptyClub);
    setEditing({ type: "", id: "" });
    setFormModal({ type: "" });
    setMessage("Club saved.");
    await refreshAll();
  };

  const savePlayer = async () => {
    const payload = { ...playerForm, age: Number(playerForm.age || 0) };
    if (editing.type === "player") await patchJson(`/crm/players/${editing.id}`, payload);
    else await postJson("/crm/players", payload);
    setPlayerForm(emptyPlayer);
    setEditing({ type: "", id: "" });
    setFormModal({ type: "" });
    setMessage("Player saved.");
    await refreshAll();
  };

  const saveContact = async () => {
    const payload = { ...contactForm, club_id: contactForm.club_id || null, player_id: contactForm.player_id || null };
    if (editing.type === "contact") await patchJson(`/crm/contacts/${editing.id}`, payload);
    else await postJson("/crm/contacts", payload);
    setContactForm(emptyContact);
    setEditing({ type: "", id: "" });
    setFormModal({ type: "" });
    setMessage("Contact saved.");
    await refreshAll();
  };

  const saveProspect = async () => {
    await postJson("/crm/prospects", prospectForm);
    setProspectForm(emptyProspect);
    setProspectContactQuery("");
    setMessage("Prospect added.");
    await refreshAll();
  };

  const moveProspect = async (item, stage) => {
    await patchJson(`/crm/prospects/${item.id}`, { contact_id: item.contact_id, stage, notes: item.notes || "" });
    await loadProspects();
  };

  const removeEntity = async (kind, item) => {
    const labels = { clubs: "club", players: "player", contacts: "contact" };
    const label = labels[kind] || "item";
    const name = kind === "clubs" ? item.name : fullName(item);
    if (!window.confirm(`Delete ${label}${name ? ` "${name}"` : ""}?`)) return;
    await deleteJson(`/crm/${kind}/${item.id}`);
    setMessage(`${label[0].toUpperCase()}${label.slice(1)} deleted.`);
    if (editing.id === item.id) setEditing({ type: "", id: "" });
    if (detail.id === item.id) setDetail({ type: "", id: "", data: null, loading: false });
    await refreshAll();
  };

  const editClub = (club) => {
    setTab("clubs");
    setEditing({ type: "club", id: club.id });
    setClubForm({ ...emptyClub, ...club });
    setFormModal({ type: "club" });
  };

  const editPlayer = (player) => {
    setTab("players");
    setEditing({ type: "player", id: player.id });
    setPlayerForm({ ...emptyPlayer, ...player });
    setFormModal({ type: "player" });
  };

  const editContact = (contact) => {
    setTab("contacts");
    setEditing({ type: "contact", id: contact.id });
    setContactForm({ ...emptyContact, ...contact, club_id: contact.club_id || "", player_id: contact.player_id || "" });
    setFormModal({ type: "contact" });
  };

  const openCreate = (type) => {
    setEditing({ type: "", id: "" });
    if (type === "club") {
      setTab("clubs");
      setClubForm(emptyClub);
    }
    if (type === "player") {
      setTab("players");
      setPlayerForm(emptyPlayer);
    }
    if (type === "contact") {
      setTab("contacts");
      setContactForm(emptyContact);
    }
    setFormModal({ type });
  };

  const runSearch = async (kind) => {
    setPage((current) => ({ ...current, [kind]: 1 }));
    await loadList(kind, { page: 1 });
  };

  const setEntityPage = async (kind, nextPage) => {
    setPage((current) => ({ ...current, [kind]: nextPage }));
    await loadList(kind, { page: nextPage });
  };

  return (
    <main className="nl-page px-4 py-8 md:py-10">
      <div className="mx-auto max-w-[1560px] space-y-6">
        <section className="surface-panel relative overflow-hidden rounded-lg p-6 md:p-8">
          <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-amber-200/50 to-transparent" />
          <div className="relative grid gap-8 xl:grid-cols-[minmax(0,1fr)_620px] xl:items-end">
            <div>
              <p className="nl-kicker">Network CRM</p>
              <h1 className="mt-3 max-w-4xl text-3xl font-semibold tracking-tight text-slate-950 md:text-5xl">
                Football relationship OS.
              </h1>
              <p className="mt-4 max-w-2xl text-sm font-semibold leading-6 text-slate-600 md:text-base">
                Clubs, players, free-role contacts and prospecting in one operational workspace. Click any club, logo or player to inspect the relationship graph.
              </p>
            </div>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <StatCard label="Clubs" value={summary.clubs} sub="CRM entities" />
              <StatCard label="Players" value={summary.players} sub="Club linked" />
              <StatCard label="Contacts" value={summary.contacts} sub={`${summary.contacts_with_email || 0} emails`} />
              <StatCard label="Unlinked" value={summary.unlinked_contacts} sub="Accepted" />
            </div>
          </div>
        </section>

        <section className="sticky top-[76px] z-20 rounded-lg border border-white/10 bg-black/75 p-1.5 shadow-[0_16px_44px_rgba(0,0,0,0.30)] backdrop-blur-xl">
          <nav className="flex gap-1 overflow-x-auto">
            {TABS.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setTab(item.id)}
                className={`whitespace-nowrap rounded-md px-4 py-2.5 text-sm font-semibold transition ${
                  tab === item.id ? "bg-white text-black shadow-sm" : "text-white/60 hover:bg-white/[0.07] hover:text-white"
                }`}
              >
                {item.label}
              </button>
            ))}
          </nav>
        </section>

        {message ? <p className="rounded-2xl border border-teal-200 bg-teal-50 px-4 py-3 text-sm font-bold text-teal-900">{message}</p> : null}
        {error ? <p className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm font-bold text-rose-900">{error}</p> : null}

        {tab === "clubs" ? (
          <ClubsView
            clubs={clubs}
            page={page.clubs}
            totalPages={lists.clubs?.totalPages || 0}
            search={search}
            setSearch={setSearch}
            runSearch={runSearch}
            setEntityPage={setEntityPage}
            openCreate={() => openCreate("club")}
            editClub={editClub}
            openDetail={openDetail}
            removeEntity={removeEntity}
          />
        ) : null}

        {tab === "players" ? (
          <PlayersView
            players={players}
            page={page.players}
            totalPages={lists.players?.totalPages || 0}
            search={search}
            setSearch={setSearch}
            runSearch={runSearch}
            setEntityPage={setEntityPage}
            openCreate={() => openCreate("player")}
            editPlayer={editPlayer}
            openDetail={openDetail}
            removeEntity={removeEntity}
          />
        ) : null}

        {tab === "contacts" ? (
          <ContactsView
            contacts={contacts}
            page={page.contacts}
            totalPages={lists.contacts?.totalPages || 0}
            search={search}
            setSearch={setSearch}
            runSearch={runSearch}
            setEntityPage={setEntityPage}
            openCreate={() => openCreate("contact")}
            editContact={editContact}
            openDetail={openDetail}
            removeEntity={removeEntity}
          />
        ) : null}

        {tab === "prospection" ? (
          <ProspectionView
            contactsWithoutProspect={contactsWithoutProspect}
            prospectForm={prospectForm}
            setProspectForm={setProspectForm}
            prospectContactQuery={prospectContactQuery}
            setProspectContactQuery={setProspectContactQuery}
            byStage={byStage}
            saveProspect={saveProspect}
            moveProspect={moveProspect}
            loadProspects={loadProspects}
            openDetail={openDetail}
          />
        ) : null}

        {tab === "map" ? (
          <MapView
            map={lists.map || { data: [] }}
            mapClusters={mapClusters}
            mapSelection={mapSelection}
            setMapSelection={setMapSelection}
            loadMap={loadMap}
            openDetail={openDetail}
          />
        ) : null}

        <DetailModal
          detail={detail}
          close={() => setDetail({ type: "", id: "", data: null, loading: false })}
          openDetail={openDetail}
          editClub={editClub}
          editPlayer={editPlayer}
          editContact={editContact}
        />

        <EntityFormModal
          formModal={formModal}
          editing={editing}
          close={() => {
            setFormModal({ type: "" });
            setEditing({ type: "", id: "" });
          }}
          clubForm={clubForm}
          setClubForm={setClubForm}
          saveClub={saveClub}
          playerForm={playerForm}
          setPlayerForm={setPlayerForm}
          savePlayer={savePlayer}
          contactForm={contactForm}
          setContactForm={setContactForm}
          saveContact={saveContact}
          clubs={options.clubs}
          players={options.players}
        />
      </div>
    </main>
  );
}

function ClubsView(props) {
  const { clubs, page, totalPages, search, setSearch, runSearch, setEntityPage, openCreate, editClub, openDetail, removeEntity } = props;
  return (
    <section>
      <ListPanel>
        <EntityToolbar kind="clubs" search={search} setSearch={setSearch} runSearch={runSearch} exportPath="/crm/clubs/export.xlsx" onCreate={openCreate} />
        <div className="mt-5 grid gap-3 xl:grid-cols-2">
          {clubs.length ? clubs.map((club) => (
            <article
              key={club.id}
              role="button"
              tabIndex={0}
              onClick={() => openDetail("club", club.id)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") openDetail("club", club.id);
              }}
              className="group relative overflow-hidden rounded-2xl border border-slate-200 bg-white p-4 shadow-sm transition duration-200 hover:-translate-y-0.5 hover:border-teal-300 hover:shadow-[0_18px_45px_rgba(15,23,42,0.10)]"
            >
              <div className="relative z-10 flex gap-4">
                <Logo src={club.logo} name={club.name} size="h-12 w-12" />
                <div className="min-w-0 flex-1">
                  <div className="flex min-w-0 items-start justify-between gap-3">
                    <div className="min-w-0">
                      <h3 className="truncate text-lg font-black tracking-tight text-slate-950 transition group-hover:text-teal-800">{club.name}</h3>
                      <p className="mt-0.5 truncate text-sm font-bold text-slate-500">{club.city || "No city"} - {club.country || "No source"}</p>
                    </div>
                    <div className="flex shrink-0 gap-1.5">
                      <button type="button" className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-black uppercase tracking-[0.1em] text-slate-600 hover:border-teal-300 hover:bg-teal-50 hover:text-teal-800" onClick={(event) => { event.stopPropagation(); editClub(club); }}>Edit</button>
                      <button type="button" className="rounded-lg border border-rose-200 bg-white px-2.5 py-1 text-[11px] font-black uppercase tracking-[0.1em] text-rose-700 hover:bg-rose-50" onClick={(event) => { event.stopPropagation(); removeEntity("clubs", club); }}>Delete</button>
                    </div>
                  </div>
                  <div className="mt-3 flex flex-wrap items-center gap-2">
                    <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-black text-slate-700">{club.player_count || 0} players</span>
                    <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-black text-slate-700">{club.contact_count || 0} contacts</span>
                    {club.website ? (
                      <a className="rounded-full border border-teal-200 bg-teal-50 px-2.5 py-1 text-xs font-black text-teal-800 hover:bg-teal-100" href={club.website} target="_blank" rel="noreferrer" onClick={(event) => event.stopPropagation()}>
                        Website
                      </a>
                    ) : null}
                  </div>
                </div>
              </div>
            </article>
          )) : <EmptyState label="No clubs found." />}
        </div>
        <Pagination page={page} totalPages={totalPages} onPage={(next) => setEntityPage("clubs", next)} />
      </ListPanel>
    </section>
  );
}

function PlayersView(props) {
  const { players, page, totalPages, search, setSearch, runSearch, setEntityPage, openCreate, editPlayer, openDetail, removeEntity } = props;
  return (
    <section>
      <ListPanel>
        <EntityToolbar kind="players" search={search} setSearch={setSearch} runSearch={runSearch} exportPath="/crm/players/export.xlsx" onCreate={openCreate} />
        <div className="mt-5 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {players.length ? players.map((player) => (
            <article
              key={player.id}
              role="button"
              tabIndex={0}
              onClick={() => openDetail("player", player.id)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") openDetail("player", player.id);
              }}
              className="group cursor-pointer overflow-hidden rounded-[1.75rem] border border-slate-200 bg-white shadow-sm outline-none transition duration-300 hover:-translate-y-1 hover:border-teal-300 hover:shadow-[0_24px_70px_rgba(15,23,42,0.14)] focus:border-teal-400 focus:ring-4 focus:ring-teal-100"
            >
              <div className="relative border-b border-slate-100 bg-[radial-gradient(circle_at_80%_0%,rgba(15,118,110,0.14),transparent_34%),linear-gradient(135deg,#ffffff,#f8fafc)] p-5">
                <div className="absolute right-4 top-4 flex gap-2">
                  <button type="button" className="nl-button-secondary px-3 py-1" onClick={(event) => { event.stopPropagation(); editPlayer(player); }}>Edit</button>
                  <button type="button" className="rounded-md border border-rose-200 bg-white/80 px-3 py-1 text-xs font-black uppercase tracking-[0.12em] text-rose-700 hover:bg-rose-50" onClick={(event) => { event.stopPropagation(); removeEntity("players", player); }}>Delete</button>
                </div>
                <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-teal-100 bg-teal-50 text-xl font-black text-teal-800 shadow-sm">
                  {initials(fullName(player))}
                </div>
                <div className="mt-5 min-w-0 pr-28">
                  <h3 className="truncate text-2xl font-black tracking-tight text-slate-950 transition group-hover:text-teal-800">{fullName(player) || player.first_name}</h3>
                  <p className="mt-1 truncate text-sm font-bold text-slate-500">{player.position || "No position"} - {player.nationality || "Unknown"} - {ageLabel(player.age)}</p>
                </div>
              </div>
              <div className="p-5">
                <button type="button" className="flex w-full items-center gap-3 rounded-2xl border border-slate-200 bg-slate-50 p-3 text-left transition hover:border-teal-300 hover:bg-teal-50" onClick={(event) => { event.stopPropagation(); openDetail("club", player.club_id); }}>
                  <Logo src={null} name={player.club_name} size="h-10 w-10" />
                  <span className="min-w-0">
                    <span className="block truncate text-sm font-black text-slate-950">{player.club_name}</span>
                    <span className="block truncate text-xs font-bold text-slate-500">{player.club_city || "No city"} - {player.club_country || "No source"}</span>
                  </span>
                </button>
                <div className="mt-4 flex flex-wrap gap-2">
                  <Tag tone="teal">Player</Tag>
                  <Tag>{player.position || "No position"}</Tag>
                </div>
              </div>
            </article>
          )) : <EmptyState label="No players found." />}
        </div>
        <Pagination page={page} totalPages={totalPages} onPage={(next) => setEntityPage("players", next)} />
      </ListPanel>
    </section>
  );
}

function ContactsView(props) {
  const { contacts, page, totalPages, search, setSearch, runSearch, setEntityPage, openCreate, editContact, openDetail, removeEntity } = props;
  return (
    <section>
      <ListPanel>
        <EntityToolbar kind="contacts" search={search} setSearch={setSearch} runSearch={runSearch} exportPath="/crm/contacts/export.xlsx" onCreate={openCreate} />
        <div className="mt-5 grid gap-3">
          {contacts.length ? contacts.map((contact) => (
            <article
              key={contact.id}
              role="button"
              tabIndex={0}
              onClick={() => openDetail("contact", contact.id)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") openDetail("contact", contact.id);
              }}
              className="grid cursor-pointer gap-4 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm outline-none transition hover:-translate-y-0.5 hover:border-teal-300 hover:shadow-lg focus:border-teal-400 focus:ring-4 focus:ring-teal-100 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)_auto] lg:items-center"
            >
              <div className="min-w-0">
                <h3 className="truncate text-left text-lg font-black text-slate-950">{fullName(contact)}</h3>
                <div className="mt-2 flex flex-wrap gap-2">
                  <Tag tone={contact.type === "PLAYER" ? "amber" : "teal"}>{contact.type}</Tag>
                  <Tag>{contact.role}</Tag>
                </div>
              </div>
              <div className="min-w-0 text-sm font-semibold text-slate-600">
                <p className="truncate">{contact.email || "No email"}</p>
                <p className="truncate">{contact.phone || "No phone"}</p>
                {contact.club_id ? <button type="button" className="mt-1 truncate font-black text-teal-700 hover:text-teal-900" onClick={(event) => { event.stopPropagation(); openDetail("club", contact.club_id); }}>{contact.club_name}</button> : null}
                {!contact.club_id && contact.player_id ? <button type="button" className="mt-1 truncate font-black text-teal-700 hover:text-teal-900" onClick={(event) => { event.stopPropagation(); openDetail("player", contact.player_id); }}>{fullName(contact, "player_")}</button> : null}
                {!contact.club_id && !contact.player_id ? <p className="mt-1 font-black text-amber-700">Unlinked</p> : null}
              </div>
              <div className="flex flex-wrap justify-end gap-2">
                <button type="button" className="nl-button-secondary px-3 py-1" onClick={(event) => { event.stopPropagation(); editContact(contact); }}>Edit</button>
                <button type="button" className="rounded-md border border-rose-200 px-3 py-1 text-xs font-black uppercase tracking-[0.12em] text-rose-700 hover:bg-rose-50" onClick={(event) => { event.stopPropagation(); removeEntity("contacts", contact); }}>Delete</button>
              </div>
            </article>
          )) : <EmptyState label="No contacts found." />}
        </div>
        <Pagination page={page} totalPages={totalPages} onPage={(next) => setEntityPage("contacts", next)} />
      </ListPanel>
    </section>
  );
}

const contactSearchLabel = (contact) =>
  [fullName(contact), contact.role, contact.club_name || fullName(contact, "player_") || "Unlinked"]
    .filter(Boolean)
    .join(" - ");

function ContactSearchSelect({ contacts, selectedId, query, setQuery, onSelect, onClear }) {
  const [open, setOpen] = useState(false);
  const normalizedQuery = String(query || "").trim().toLowerCase();
  const selected = contacts.find((contact) => contact.id === selectedId);
  const matches = useMemo(() => {
    if (!normalizedQuery) return contacts.slice(0, 12);
    return contacts
      .filter((contact) => {
        const haystack = [
          fullName(contact),
          contact.role,
          contact.type,
          contact.email,
          contact.phone,
          contact.club_name,
          fullName(contact, "player_"),
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();
        return haystack.includes(normalizedQuery);
      })
      .slice(0, 12);
  }, [contacts, normalizedQuery]);

  return (
    <div className="relative">
      <Field label="contact">
        <div className="flex gap-2">
          <input
            className="nl-field"
            value={query}
            placeholder="Type a name, role, club, email..."
            onFocus={() => setOpen(true)}
            onBlur={() => window.setTimeout(() => setOpen(false), 120)}
            onChange={(event) => {
              setQuery(event.target.value);
              setOpen(true);
              if (selectedId) onClear(event.target.value);
            }}
          />
          {query || selectedId ? (
            <button type="button" className="nl-button-secondary shrink-0 px-3" onClick={() => { onClear(); setOpen(false); }}>
              Clear
            </button>
          ) : null}
        </div>
      </Field>
      {selected ? (
        <p className="mt-1 text-xs font-black uppercase tracking-[0.12em] text-teal-700">Selected: {contactSearchLabel(selected)}</p>
      ) : (
        <p className="mt-1 text-xs font-bold text-slate-500">Select a matching contact before adding it to the pipeline.</p>
      )}
      {open ? (
        <div className="absolute left-0 right-0 top-[78px] z-30 max-h-80 overflow-auto rounded-2xl border border-slate-200 bg-white p-2 shadow-[0_24px_70px_rgba(15,23,42,0.18)]">
          {matches.length ? matches.map((contact) => (
            <button
              key={contact.id}
              type="button"
              className="block w-full rounded-xl px-3 py-2 text-left transition hover:bg-teal-50"
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => {
                onSelect(contact);
                setOpen(false);
              }}
            >
              <span className="block truncate text-sm font-black text-slate-950">{fullName(contact)}</span>
              <span className="mt-0.5 block truncate text-xs font-bold text-slate-500">
                {contact.role || "No role"} - {contact.club_name || fullName(contact, "player_") || "Unlinked"}{contact.email ? ` - ${contact.email}` : ""}
              </span>
            </button>
          )) : (
            <div className="rounded-xl border border-dashed border-slate-300 bg-slate-50 px-3 py-4 text-sm font-bold text-slate-500">
              No contact matches this search.
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}

function ProspectionView({
  contactsWithoutProspect,
  prospectForm,
  setProspectForm,
  prospectContactQuery,
  setProspectContactQuery,
  byStage,
  saveProspect,
  moveProspect,
  loadProspects,
  openDetail,
}) {
  return (
    <section className="space-y-5">
      <ListPanel>
        <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_180px_minmax(0,1fr)_140px]">
          <ContactSearchSelect
            contacts={contactsWithoutProspect}
            selectedId={prospectForm.contact_id}
            query={prospectContactQuery}
            setQuery={setProspectContactQuery}
            onSelect={(contact) => {
              setProspectForm({ ...prospectForm, contact_id: contact.id });
              setProspectContactQuery(contactSearchLabel(contact));
            }}
            onClear={(nextQuery = "") => {
              setProspectForm({ ...prospectForm, contact_id: "" });
              setProspectContactQuery(nextQuery);
            }}
          />
          <Field label="stage">
            <select className="nl-field" value={prospectForm.stage} onChange={(e) => setProspectForm({ ...prospectForm, stage: e.target.value })}>
              {STAGES.map((stage) => <option key={stage.id} value={stage.id}>{stage.label}</option>)}
            </select>
          </Field>
          <Field label="notes">
            <input className="nl-field" value={prospectForm.notes || ""} onChange={(e) => setProspectForm({ ...prospectForm, notes: e.target.value })} />
          </Field>
          <button type="button" className="nl-button-primary self-end disabled:cursor-not-allowed disabled:opacity-50" disabled={!prospectForm.contact_id} onClick={saveProspect}>Add</button>
        </div>
      </ListPanel>
      <div className="grid gap-4 xl:grid-cols-4">
        {STAGES.map((stage) => (
          <div key={stage.id} className="rounded-3xl border border-slate-200 bg-slate-50/90 p-4 shadow-sm">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-black text-slate-950">{stage.label}</h2>
              <Tag>{byStage[stage.id]?.length || 0}</Tag>
            </div>
            <div className="space-y-3">
              {(byStage[stage.id] || []).map((item) => (
                <article key={item.id} className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                  <button type="button" className="text-left font-black text-slate-950 hover:text-teal-700" onClick={() => openDetail("contact", item.contact_id)}>{fullName(item, "contact_")}</button>
                  <p className="text-sm font-bold text-teal-700">{item.contact_role}</p>
                  <p className="mt-2 text-sm text-slate-600">{item.contact_email || item.contact_phone || "No direct contact data"}</p>
                  <p className="mt-1 text-sm font-semibold text-slate-500">{text(item.club_name || fullName(item, "player_"))}</p>
                  {item.notes ? <p className="mt-3 rounded-xl bg-slate-50 p-3 text-sm text-slate-600">{item.notes}</p> : null}
                  <select className="nl-field mt-3" value={item.stage} onChange={(e) => moveProspect(item, e.target.value)}>
                    {STAGES.map((next) => <option key={next.id} value={next.id}>{next.label}</option>)}
                  </select>
                  <button type="button" className="mt-3 text-xs font-black uppercase tracking-[0.14em] text-rose-700" onClick={async () => { await deleteJson(`/crm/prospects/${item.id}`); await loadProspects(); }}>Remove</button>
                </article>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function MapView({ map, mapClusters, mapSelection, setMapSelection, loadMap, openDetail }) {
  const [mapSearch, setMapSearch] = useState("");
  const normalizedMapSearch = normalizeSearchText(mapSearch.trim());
  const mapMatches = useMemo(() => {
    if (normalizedMapSearch.length < 2) return [];
    return mapClusters
      .map((cluster) => {
        const clubs = cluster.clubs || [];
        const haystack = normalizeSearchText([
          cluster.city,
          cluster.country,
          ...clubs.map((club) => club.name),
        ].join(" "));
        return haystack.includes(normalizedMapSearch) ? cluster : null;
      })
      .filter(Boolean)
      .slice(0, 8);
  }, [mapClusters, normalizedMapSearch]);

  useEffect(() => {
    if (normalizedMapSearch.length < 2 || !mapMatches.length) return;
    setMapSelection(mapMatches[0]);
  }, [mapMatches, normalizedMapSearch.length, setMapSelection]);

  const selected = mapSelection || mapClusters[0];
  return (
    <section className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
      <div className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-[0_24px_70px_rgba(15,23,42,0.10)]">
        <div className="flex flex-wrap items-end justify-between gap-3 border-b border-slate-200 p-5">
          <div>
            <p className="nl-kicker">Club map</p>
            <h2 className="mt-1 text-2xl font-black text-slate-950">Mapped football locations</h2>
            <p className="mt-1 text-sm font-semibold text-slate-500">
              Zoom, pan and click logo markers. {map.mapped_clubs || 0} clubs mapped. {map.unmapped_clubs || 0} clubs unmapped because source location data is dirty.
            </p>
          </div>
          <div className="relative w-full max-w-md">
            <input
              id="crm-map-search"
              name="crm-map-search"
              className="nl-field pr-24"
              placeholder="Search a club or city, e.g. Marseille"
              value={mapSearch}
              onChange={(event) => setMapSearch(event.target.value)}
            />
            <button type="button" className="absolute right-2 top-1/2 -translate-y-1/2 text-xs font-black uppercase tracking-[0.12em] text-teal-700" onClick={loadMap}>Refresh</button>
            {normalizedMapSearch.length >= 2 ? (
              <div className="absolute left-0 right-0 top-[calc(100%+8px)] z-[1200] overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-[0_24px_70px_rgba(15,23,42,0.18)]">
                {mapMatches.length ? mapMatches.map((cluster) => (
                  <button
                    key={`${cluster.city}|${cluster.country}`}
                    type="button"
                    className="block w-full px-4 py-3 text-left hover:bg-teal-50"
                    onClick={() => {
                      setMapSelection(cluster);
                      setMapSearch(`${cluster.city}, ${cluster.country}`);
                    }}
                  >
                    <span className="block font-black text-slate-950">{cluster.city}, {cluster.country}</span>
                    <span className="block truncate text-xs font-bold uppercase tracking-[0.12em] text-slate-500">
                      {cluster.club_count} club(s) - {(cluster.clubs || []).slice(0, 3).map((club) => club.name).join(", ")}
                    </span>
                  </button>
                )) : <div className="px-4 py-3 text-sm font-bold text-slate-500">No mapped match yet.</div>}
              </div>
            ) : null}
          </div>
        </div>
        <ClubLeafletMap
          clusters={mapClusters}
          selected={mapSelection}
          setMapSelection={setMapSelection}
          openDetail={openDetail}
        />
      </div>
      <aside className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
        <p className="nl-kicker">Selected city</p>
        {selected ? (
          <div className="mt-3">
            <h3 className="text-3xl font-black tracking-tight text-slate-950">{selected.city}</h3>
            <p className="font-bold text-slate-500">{selected.country} - {selected.club_count} club(s)</p>
            <div className="mt-5 space-y-2">
              {(selected.clubs || []).map((club) => (
                <button key={club.id} type="button" className="flex w-full items-center gap-3 rounded-2xl border border-slate-200 bg-slate-50 p-3 text-left transition hover:border-teal-300 hover:bg-teal-50" onClick={() => openDetail("club", club.id)}>
                  <Logo src={club.logo} name={club.name} size="h-10 w-10" />
                  <span className="min-w-0 truncate font-black text-slate-950">{club.name}</span>
                </button>
              ))}
            </div>
          </div>
        ) : <EmptyState label="No mapped locations yet." />}
      </aside>
    </section>
  );
}

function ClubLeafletMap({ clusters, selected, setMapSelection, openDetail }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const layerRef = useRef(null);
  const leafletRef = useRef(null);
  const fitDoneRef = useRef(false);
  const [logoData, setLogoData] = useState(null);
  const selectedKey = selected ? `${selected.city}|${selected.country}` : "";

  useEffect(() => {
    let active = true;
    loadClubLogoData().then((data) => {
      if (active) setLogoData(data || {});
    });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    import("leaflet").then((module) => {
      if (!active || !containerRef.current || mapRef.current) return;
      const L = module.default || module;
      leafletRef.current = L;
      const map = L.map(containerRef.current, {
        center: [46.6, 4.5],
        zoom: 4,
        minZoom: 2,
        maxZoom: 18,
        scrollWheelZoom: true,
        zoomControl: false,
      });
      L.control.zoom({ position: "topright" }).addTo(map);
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      }).addTo(map);
      layerRef.current = L.layerGroup().addTo(map);
      mapRef.current = map;
      setTimeout(() => map.invalidateSize(), 0);
    });
    return () => {
      active = false;
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
        layerRef.current = null;
        leafletRef.current = null;
        fitDoneRef.current = false;
      }
    };
  }, []);

  useEffect(() => {
    const L = leafletRef.current;
    const map = mapRef.current;
    const layer = layerRef.current;
    if (!L || !map || !layer || !logoData) return;

    layer.clearLayers();
    const bounds = [];
    clusters.forEach((cluster) => {
      const lat = Number(cluster.lat);
      const lon = Number(cluster.lon);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
      const key = `${cluster.city}|${cluster.country}`;
      const marker = L.marker([lat, lon], {
        icon: L.divIcon({
          className: "",
          html: clusterMarkerHtml(cluster, logoData, key === selectedKey),
          iconSize: [52, 52],
          iconAnchor: [26, 26],
          popupAnchor: [0, -24],
        }),
        title: `${cluster.city}, ${cluster.country}`,
      });
      marker.bindPopup(clusterPopupHtml(cluster, logoData), { maxWidth: 340, className: "crm-map-popup" });
      marker.on("click", () => setMapSelection(cluster));
      marker.on("popupopen", (event) => {
        const element = event.popup.getElement();
        element?.querySelectorAll("[data-crm-club-id]").forEach((button) => {
          button.addEventListener("click", () => openDetail("club", button.getAttribute("data-crm-club-id")));
        });
      });
      marker.addTo(layer);
      bounds.push([lat, lon]);
    });

    if (bounds.length && !fitDoneRef.current) {
      map.fitBounds(bounds, { padding: [34, 34], maxZoom: 6 });
      fitDoneRef.current = true;
    }
  }, [clusters, logoData, openDetail, selectedKey, setMapSelection]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !selected) return;
    const lat = Number(selected.lat);
    const lon = Number(selected.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
    map.flyTo([lat, lon], Math.max(map.getZoom(), 10), { duration: 0.8 });
  }, [selectedKey, selected]);

  return (
    <div className="relative h-[680px] w-full bg-slate-100">
      <div ref={containerRef} className="h-full w-full" />
      <div className="pointer-events-none absolute bottom-4 left-4 rounded-2xl border border-slate-200 bg-white/90 px-4 py-3 text-xs font-black uppercase tracking-[0.12em] text-slate-600 shadow-lg backdrop-blur">
        Scroll to zoom - drag to pan - click logos
      </div>
    </div>
  );
}

function clubLogoMarkup(club, logoData, className = "crm-map-logo") {
  const logoUrl = resolveClubLogoUrl(club.name, logoData) || directLogoUrl(club.logo);
  const fallback = escapeHtml(mapInitials(club.name));
  if (logoUrl) {
    return `
      <span class="${className} crm-map-logo-has-image">
        <img class="crm-map-logo-img" src="${escapeHtml(logoUrl)}" alt="${escapeHtml(club.name)} logo" onerror="this.parentElement.classList.add('crm-map-logo-broken')" />
        <span class="crm-map-logo-fallback-text">${fallback}</span>
      </span>
    `;
  }
  return `<span class="${className} crm-map-logo-fallback"><span class="crm-map-logo-fallback-text">${fallback}</span></span>`;
}

function clusterMarkerHtml(cluster, logoData, selected) {
  const clubs = (cluster.clubs || []).slice(0, 3);
  const count = Number(cluster.club_count || cluster.clubs?.length || 0);
  return `
    <div class="crm-map-marker ${selected ? "crm-map-marker-selected" : ""}">
      <div class="crm-map-logo-stack">
        ${clubs.map((club) => clubLogoMarkup(club, logoData)).join("")}
      </div>
      ${count > 1 ? `<span class="crm-map-count">${count}</span>` : ""}
    </div>
  `;
}

function clusterPopupHtml(cluster, logoData) {
  const clubs = cluster.clubs || [];
  return `
    <div class="crm-map-popup-content">
      <div class="crm-map-popup-title">${escapeHtml(cluster.city)}, ${escapeHtml(cluster.country)}</div>
      <div class="crm-map-popup-subtitle">${escapeHtml(cluster.club_count || clubs.length)} club(s)</div>
      <div class="crm-map-popup-clubs">
        ${clubs.slice(0, 12).map((club) => `
          <button type="button" class="crm-map-popup-club" data-crm-club-id="${escapeHtml(club.id)}">
            ${clubLogoMarkup(club, logoData, "crm-map-popup-logo")}
            <span>${escapeHtml(club.name)}</span>
          </button>
        `).join("")}
      </div>
    </div>
  `;
}

function FormPanel({ title, subtitle, children }) {
  return (
    <aside className="surface-panel rounded-lg p-5">
      <p className="nl-kicker">Workspace</p>
      <h2 className="mt-1 text-2xl font-black text-slate-950">{title}</h2>
      <p className="mt-1 text-sm font-semibold text-slate-500">{subtitle}</p>
      <div className="mt-5 grid gap-3">{children}</div>
    </aside>
  );
}

function ListPanel({ children }) {
  return <section className="surface-panel rounded-lg p-5">{children}</section>;
}

function ModalShell({ title, eyebrow = "Network CRM", children, close, size = "max-w-4xl" }) {
  return (
    <div className="fixed inset-0 z-[5000] flex items-center justify-center bg-black/70 p-3 backdrop-blur-md md:p-6" role="dialog" aria-modal="true">
      <div className={`max-h-[92vh] w-full ${size} overflow-hidden rounded-lg border border-white/10 bg-[#080b12] shadow-[0_40px_120px_rgba(0,0,0,0.55)]`}>
        <div className="flex items-start justify-between gap-4 border-b border-white/10 bg-white/[0.035] p-5 md:p-6">
          <div>
            <p className="nl-kicker">{eyebrow}</p>
            <h2 className="mt-1 text-3xl font-black tracking-tight text-slate-950">{title}</h2>
          </div>
          <button type="button" className="nl-button-secondary px-3 py-1" onClick={close}>Close</button>
        </div>
        <div className="max-h-[calc(92vh-112px)] overflow-auto p-5 md:p-6">
          {children}
        </div>
      </div>
    </div>
  );
}

function EntityFormModal(props) {
  const {
    formModal,
    editing,
    close,
    clubForm,
    setClubForm,
    saveClub,
    playerForm,
    setPlayerForm,
    savePlayer,
    contactForm,
    setContactForm,
    saveContact,
    clubs,
    players,
  } = props;
  if (!formModal.type) return null;
  const isEdit = Boolean(editing.id);
  const title = `${isEdit ? "Edit" : "Create"} ${formModal.type}`;
  return (
    <ModalShell title={title} eyebrow="Workspace" close={close} size="max-w-3xl">
      {formModal.type === "club" ? (
        <div className="grid gap-4 md:grid-cols-2">
          {Object.keys(emptyClub).map((key) => (
            <Field key={key} label={key}>
              <input className="nl-field" value={clubForm[key] || ""} onChange={(e) => setClubForm({ ...clubForm, [key]: e.target.value })} />
            </Field>
          ))}
          <div className="flex gap-2 md:col-span-2">
            <button type="button" className="nl-button-primary flex-1" onClick={saveClub}>{isEdit ? "Save changes" : "Create club"}</button>
            <button type="button" className="nl-button-secondary" onClick={close}>Cancel</button>
          </div>
        </div>
      ) : null}

      {formModal.type === "player" ? (
        <div className="grid gap-4 md:grid-cols-2">
          {["first_name", "last_name", "age", "position", "nationality", "photo", "email", "phone"].map((key) => (
            <Field key={key} label={key.replace("_", " ")}>
              <input className="nl-field" type={key === "age" ? "number" : "text"} value={playerForm[key] || ""} onChange={(e) => setPlayerForm({ ...playerForm, [key]: e.target.value })} />
            </Field>
          ))}
          <Field label="club">
            <select className="nl-field" value={playerForm.club_id || ""} onChange={(e) => setPlayerForm({ ...playerForm, club_id: e.target.value })}>
              <option value="">Select club</option>
              {clubs.map((club) => <option key={club.id} value={club.id}>{club.name} - {club.city}</option>)}
            </select>
          </Field>
          <div className="flex gap-2 md:col-span-2">
            <button type="button" className="nl-button-primary flex-1" onClick={savePlayer}>{isEdit ? "Save changes" : "Create player"}</button>
            <button type="button" className="nl-button-secondary" onClick={close}>Cancel</button>
          </div>
        </div>
      ) : null}

      {formModal.type === "contact" ? (
        <div className="grid gap-4 md:grid-cols-2">
          {["first_name", "last_name", "role", "email", "phone"].map((key) => (
            <Field key={key} label={key.replace("_", " ")}>
              <input className="nl-field" value={contactForm[key] || ""} onChange={(e) => setContactForm({ ...contactForm, [key]: e.target.value })} />
            </Field>
          ))}
          <Field label="type">
            <select className="nl-field" value={contactForm.type} onChange={(e) => setContactForm({ ...contactForm, type: e.target.value })}>
              <option value="CLUB">CLUB</option>
              <option value="PLAYER">PLAYER</option>
            </select>
          </Field>
          <Field label="linked club">
            <select className="nl-field" value={contactForm.club_id || ""} onChange={(e) => setContactForm({ ...contactForm, club_id: e.target.value })}>
              <option value="">No club relation</option>
              {clubs.map((club) => <option key={club.id} value={club.id}>{club.name} - {club.city}</option>)}
            </select>
          </Field>
          <Field label="linked player">
            <select className="nl-field" value={contactForm.player_id || ""} onChange={(e) => setContactForm({ ...contactForm, player_id: e.target.value })}>
              <option value="">No player relation</option>
              {players.map((player) => <option key={player.id} value={player.id}>{fullName(player)} - {player.club_name}</option>)}
            </select>
          </Field>
          <Field label="notes">
            <textarea className="nl-field min-h-[130px]" value={contactForm.notes || ""} onChange={(e) => setContactForm({ ...contactForm, notes: e.target.value })} />
          </Field>
          <div className="flex gap-2 md:col-span-2">
            <button type="button" className="nl-button-primary flex-1" onClick={saveContact}>{isEdit ? "Save changes" : "Create contact"}</button>
            <button type="button" className="nl-button-secondary" onClick={close}>Cancel</button>
          </div>
        </div>
      ) : null}
    </ModalShell>
  );
}

function DetailModal({ detail, close, openDetail, editClub, editPlayer, editContact }) {
  if (!detail.type && !detail.loading) return null;
  const item = detail.data;
  const title = item
    ? detail.type === "club"
      ? item.name
      : fullName(item) || detail.type
    : detail.type || "Loading";
  return (
    <ModalShell title={title} eyebrow="Entity detail" close={close}>
      {detail.loading ? <p className="mt-5 text-sm font-bold text-slate-500">Loading...</p> : null}
      {!detail.loading && item && detail.type === "club" ? <ClubDetail item={item} openDetail={openDetail} editClub={editClub} /> : null}
      {!detail.loading && item && detail.type === "player" ? <PlayerDetail item={item} openDetail={openDetail} editPlayer={editPlayer} /> : null}
      {!detail.loading && item && detail.type === "contact" ? <ContactDetail item={item} openDetail={openDetail} editContact={editContact} /> : null}
    </ModalShell>
  );
}

function ClubDetail({ item, openDetail, editClub }) {
  return (
    <div className="mt-5 space-y-5">
      <div className="flex items-center gap-4">
        <Logo src={item.logo} name={item.name} onClick={() => {}} size="h-16 w-16" />
        <div className="min-w-0">
          <h3 className="truncate text-2xl font-black text-slate-950">{item.name}</h3>
          <p className="font-bold text-slate-500">{item.city} - {item.country}</p>
        </div>
      </div>
      <button type="button" className="nl-button-primary w-full" onClick={() => editClub(item)}>Edit club</button>
      <InfoGrid items={[['Players', item.player_count || 0], ['Contacts', item.contact_count || 0], ['Email', item.email || '-'], ['Phone', item.phone || '-']]} />
      {item.website ? <a className="nl-button-secondary w-full" href={item.website} target="_blank" rel="noreferrer">Open website</a> : null}
      <RelationList title="Players" items={item.players || []} label={(row) => `${fullName(row)} - ${row.position || '-'}`} onClick={(row) => openDetail("player", row.id)} />
      <RelationList title="Contacts" items={item.contacts || []} label={(row) => `${fullName(row)} - ${row.role || '-'}`} onClick={(row) => openDetail("contact", row.id)} />
    </div>
  );
}

function PlayerDetail({ item, openDetail, editPlayer }) {
  return (
    <div className="mt-5 space-y-5">
      <div>
        <h3 className="text-2xl font-black text-slate-950">{fullName(item)}</h3>
        <p className="font-bold text-slate-500">{item.position} - {item.nationality} - {ageLabel(item.age)}</p>
      </div>
      <button type="button" className="nl-button-primary w-full" onClick={() => editPlayer(item)}>Edit player</button>
      <button type="button" className="flex w-full items-center gap-3 rounded-2xl border border-slate-200 bg-slate-50 p-3 text-left hover:border-teal-300 hover:bg-teal-50" onClick={() => openDetail("club", item.club_id)}>
        <Logo src={null} name={item.club_name} size="h-10 w-10" />
        <span><span className="block font-black text-slate-950">{item.club_name}</span><span className="text-xs font-bold text-slate-500">Open club</span></span>
      </button>
      <InfoGrid items={[['Email', item.email || '-'], ['Phone', item.phone || '-'], ['City', item.club_city || '-'], ['Source', item.club_country || '-']]} />
      <RelationList title="Contacts" items={item.contacts || []} label={(row) => `${fullName(row)} - ${row.role || '-'}`} onClick={(row) => openDetail("contact", row.id)} />
    </div>
  );
}

function ContactDetail({ item, openDetail, editContact }) {
  const playerName = fullName(item, "player_");
  return (
    <div className="mt-5 space-y-5">
      <div>
        <h3 className="text-2xl font-black text-slate-950">{fullName(item)}</h3>
        <div className="mt-2 flex flex-wrap gap-2"><Tag tone={item.type === "PLAYER" ? "amber" : "teal"}>{item.type}</Tag><Tag>{item.role}</Tag></div>
      </div>
      <button type="button" className="nl-button-primary w-full" onClick={() => editContact(item)}>Edit contact</button>
      <InfoGrid items={[['Email', item.email || '-'], ['Phone', item.phone || '-']]} />
      {item.club_id ? <button type="button" className="nl-button-secondary w-full" onClick={() => openDetail("club", item.club_id)}>Open {item.club_name}</button> : null}
      {item.player_id ? <button type="button" className="nl-button-secondary w-full" onClick={() => openDetail("player", item.player_id)}>Open {playerName}</button> : null}
      {!item.club_id && !item.player_id ? <div className="rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm font-bold text-amber-800">This contact is intentionally unlinked.</div> : null}
      {item.notes ? <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 text-sm leading-6 text-slate-700">{item.notes}</div> : null}
      {item.prospect ? <Tag tone="teal">Prospect: {item.prospect.stage}</Tag> : null}
    </div>
  );
}

function InfoGrid({ items }) {
  return (
    <div className="grid grid-cols-2 gap-2">
      {items.map(([label, value]) => (
        <div key={label} className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
          <p className="text-[11px] font-black uppercase tracking-[0.12em] text-slate-500">{label}</p>
          <p className="mt-1 break-words text-sm font-black text-slate-950">{value}</p>
        </div>
      ))}
    </div>
  );
}

function RelationList({ title, items, label, onClick }) {
  return (
    <div>
      <div className="mb-2 flex items-center justify-between"><h4 className="font-black text-slate-950">{title}</h4><Tag>{items.length}</Tag></div>
      <div className="max-h-64 space-y-2 overflow-auto pr-1">
        {items.length ? items.map((item) => (
          <button key={item.id} type="button" className="block w-full truncate rounded-xl border border-slate-200 bg-white px-3 py-2 text-left text-sm font-bold text-slate-700 hover:border-teal-300 hover:bg-teal-50" onClick={() => onClick(item)}>{label(item)}</button>
        )) : <p className="rounded-xl border border-dashed border-slate-300 bg-slate-50 px-3 py-2 text-sm font-bold text-slate-500">No linked records.</p>}
      </div>
    </div>
  );
}
