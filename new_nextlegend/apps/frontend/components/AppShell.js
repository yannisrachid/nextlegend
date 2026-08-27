import Link from "next/link";
import { useRouter } from "next/router";
import { useEffect, useMemo, useState } from "react";

const PRIMARY_NAV = [
  { href: "/", label: "HQ", icon: "grid" },
  { href: "/hd-players", label: "HD Players", icon: "users" },
  { href: "/crm", label: "Network", icon: "network" },
  { href: "/scouting-lab", label: "Scouting", icon: "radar" },
];

const SCOUTING_NAV = [
  { href: "/ranking", label: "Ranking", icon: "list" },
  { href: "/report", label: "Reports", icon: "file" },
  { href: "/comparison", label: "Compare", icon: "split" },
  { href: "/projection", label: "Projection", icon: "trend" },
  { href: "/stats-research", label: "Research", icon: "search" },
  { href: "/vizualisation", label: "Visuals", icon: "chart" },
  { href: "/prospect", label: "Prospect", icon: "target" },
  { href: "/ai", label: "AI Assistant", icon: "spark" },
  { href: "/admin", label: "Admin", icon: "shield", adminOnly: true },
];

const ROUTE_META = [
  { match: /^\/$/, title: "HQ", eyebrow: "Workspace", desc: "Priorities, portfolio and team execution." },
  { match: /^\/hd-players/, title: "HD Players", eyebrow: "Portfolio", desc: "Player rooms, documents and representation strategy." },
  { match: /^\/crm/, title: "Network", eyebrow: "CRM", desc: "Clubs, players, contacts and relationship graph." },
  { match: /^\/scouting-lab$/, title: "Scouting Lab", eyebrow: "Intelligence", desc: "Research modules and decision support." },
  { match: /^\/ranking/, title: "Ranking", eyebrow: "Scouting", desc: "Targeted cohorts and market-ranked profiles." },
  { match: /^\/report/, title: "Reports", eyebrow: "Scouting", desc: "Director-ready player dossiers." },
  { match: /^\/comparison/, title: "Compare", eyebrow: "Scouting", desc: "Side-by-side recruitment boards." },
  { match: /^\/projection/, title: "Projection", eyebrow: "Scouting", desc: "League fit and translation analysis." },
  { match: /^\/stats-research/, title: "Research", eyebrow: "Scouting", desc: "Metric discovery and statistical patterns." },
  { match: /^\/vizualisation/, title: "Visuals", eyebrow: "Scouting", desc: "Export-ready data visualizations." },
  { match: /^\/prospect/, title: "Prospect", eyebrow: "Scouting", desc: "Watchlists and scouting pipelines." },
  { match: /^\/ai/, title: "AI Assistant", eyebrow: "Scouting", desc: "Structured scouting briefs." },
  { match: /^\/admin/, title: "Access Control", eyebrow: "Admin", desc: "Users and permissions." },
];

const iconPaths = {
  grid: ["M4 4h7v7H4z", "M13 4h7v7h-7z", "M4 13h7v7H4z", "M13 13h7v7h-7z"],
  users: ["M16 21v-2a4 4 0 0 0-4-4H7a4 4 0 0 0-4 4v2", "M9.5 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8", "M22 21v-2a4 4 0 0 0-3-3.87", "M16 3.13a4 4 0 0 1 0 7.75"],
  network: ["M12 5a3 3 0 1 0 0.01 0", "M5 19a3 3 0 1 0 0.01 0", "M19 19a3 3 0 1 0 0.01 0", "M10.2 7.4 6.6 9.2", "M13.8 7.4 7.2 9.2", "M8 19h8"],
  radar: ["M12 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16", "M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8", "M12 12h8"],
  list: ["M8 6h13", "M8 12h13", "M8 18h13", "M3 6h.01", "M3 12h.01", "M3 18h.01"],
  file: ["M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z", "M14 2v6h6", "M8 13h8", "M8 17h5"],
  split: ["M16 3h5v5", "M4 20 21 3", "M21 16v5h-5", "M15 15l6 6", "M4 4l5 5"],
  trend: ["M3 17 9 11l4 4 8-8", "M14 7h7v7"],
  search: ["M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16", "m21 21-4.35-4.35"],
  chart: ["M4 19V5", "M4 19h16", "M8 16v-5", "M12 16V8", "M16 16v-9"],
  target: ["M12 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16", "M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8", "M12 12h.01"],
  spark: ["M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z", "M19 3v4", "M21 5h-4", "M5 17v4", "M7 19H3"],
  shield: ["M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10", "M9 12l2 2 4-4"],
  menu: ["M4 7h16", "M4 12h16", "M4 17h16"],
  close: ["m6 6 12 12", "m18 6-12 12"],
  command: ["M18 9a3 3 0 1 0 0-6 3 3 0 0 0 0 6", "M6 9a3 3 0 1 1 0-6 3 3 0 0 1 0 6", "M18 21a3 3 0 1 1 0-6 3 3 0 0 1 0 6", "M6 21a3 3 0 1 0 0-6 3 3 0 0 0 0 6", "M9 6h6", "M9 18h6", "M6 9v6", "M18 9v6"],
};

const Icon = ({ name, className = "h-4 w-4" }) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden="true">
    {(iconPaths[name] || iconPaths.grid).map((path) => (
      <path key={path} d={path} />
    ))}
  </svg>
);

function useRouteMeta(pathname) {
  return ROUTE_META.find((item) => item.match.test(pathname)) || ROUTE_META[0];
}

function NavLink({ item, active, onClick }) {
  return (
    <Link
      href={item.href}
      onClick={onClick}
      className={`group flex items-center gap-3 rounded-md border px-3 py-2.5 text-sm font-medium transition ${
        active
          ? "border-[#3A8967]/40 bg-[#2F7D5C]/20 text-[#DDF3E8]"
          : "border-transparent text-[#A0A8A3] hover:border-white/10 hover:bg-white/[0.045] hover:text-[#F3F5F4]"
      }`}
    >
      <span className={`flex h-7 w-7 items-center justify-center rounded-md border transition ${
        active ? "border-[#3A8967]/40 bg-[#2F7D5C]/20 text-[#7BC39B]" : "border-white/10 bg-white/[0.03] text-[#6F7772] group-hover:text-[#A0A8A3]"
      }`}>
        <Icon name={item.icon} className="h-3.5 w-3.5" />
      </span>
      <span className="min-w-0 truncate">{item.label}</span>
    </Link>
  );
}

function CommandPalette({ open, onClose, items }) {
  const router = useRouter();
  const [query, setQuery] = useState("");

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (event) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, open]);

  useEffect(() => {
    if (open) setQuery("");
  }, [open]);

  const results = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return items;
    return items.filter((item) => `${item.label} ${item.section || ""}`.toLowerCase().includes(normalized));
  }, [items, query]);

  if (!open) return null;

  const go = (href) => {
    onClose();
    router.push(href);
  };

  return (
    <div className="fixed inset-0 z-[7000] flex items-start justify-center bg-black/70 px-4 pt-[12vh] backdrop-blur-md" role="dialog" aria-modal="true">
      <div className="w-full max-w-2xl overflow-hidden rounded-lg border border-white/10 bg-[#080B0A] shadow-[0_40px_120px_rgba(0,0,0,0.62)]">
        <div className="flex items-center gap-3 border-b border-white/10 px-4 py-3">
          <Icon name="command" className="h-4 w-4 text-[#559A78]" />
          <input
            autoFocus
            className="h-10 min-w-0 flex-1 bg-transparent text-sm text-[#F3F5F4] outline-none placeholder:text-[#6F7772]"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search pages and modules"
          />
          <button type="button" onClick={onClose} className="nl-icon-button" aria-label="Close command palette">
            <Icon name="close" className="h-4 w-4" />
          </button>
        </div>
        <div className="max-h-[420px] overflow-auto p-2">
          {results.length === 0 ? (
            <div className="rounded-md border border-dashed border-white/10 bg-white/[0.03] px-4 py-8 text-center">
              <p className="text-sm font-medium text-[#F3F5F4]">No result</p>
              <p className="mt-1 text-xs text-[#6F7772]">Try another page or module name.</p>
            </div>
          ) : (
            results.map((item) => (
              <button
                key={item.href}
                type="button"
                onClick={() => go(item.href)}
                className="flex w-full items-center gap-3 rounded-md px-3 py-2.5 text-left text-sm text-[#A0A8A3] transition hover:bg-white/[0.055] hover:text-[#F3F5F4]"
              >
                <span className="flex h-8 w-8 items-center justify-center rounded-md border border-white/10 bg-white/[0.035] text-[#559A78]">
                  <Icon name={item.icon} className="h-4 w-4" />
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate font-medium">{item.label}</span>
                  <span className="block truncate text-xs text-[#6F7772]">{item.section}</span>
                </span>
                <span className="text-xs text-[#6F7772]">Open</span>
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

function RightNavigation({ items, scoutingItems, me, onLogout, activeFor, onNavigate }) {
  return (
    <aside className="flex h-full flex-col gap-5 overflow-auto border-l border-white/10 bg-[#060807]/95 px-4 py-4 text-[#F3F5F4] shadow-[inset_1px_0_0_rgba(255,255,255,0.035)]">
      <Link href="/" onClick={onNavigate} className="flex items-center gap-3 rounded-md border border-white/10 bg-white/[0.035] p-3 transition hover:bg-white/[0.055]">
        <span className="flex h-10 w-10 items-center justify-center rounded-md bg-white">
          <img src="/logo_nl.png" alt="Next Legend" className="h-7 w-7 object-contain" />
        </span>
        <span>
          <span className="block text-sm font-semibold text-[#F3F5F4]">Next Legend</span>
          <span className="block text-[11px] font-semibold uppercase tracking-[0.14em] text-[#6F7772]">HD Sports</span>
        </span>
      </Link>

      <div className="space-y-2">
        <p className="px-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-[#6F7772]">Main</p>
        <nav className="space-y-1">
          {items.map((item) => (
            <NavLink key={item.href} item={item} active={activeFor(item)} onClick={onNavigate} />
          ))}
        </nav>
      </div>

      <div className="space-y-2">
        <p className="px-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-[#6F7772]">Scouting modules</p>
        <nav className="space-y-1">
          {scoutingItems.map((item) => (
            <NavLink key={item.href} item={item} active={activeFor(item)} onClick={onNavigate} />
          ))}
        </nav>
      </div>

      <div className="mt-auto space-y-3 border-t border-white/10 pt-4">
        <div className="rounded-md border border-white/10 bg-white/[0.03] p-3">
          <p className="text-sm font-medium text-[#F3F5F4]">{me?.display_name || me?.username || "HD Sports"}</p>
          <p className="mt-1 truncate text-xs text-[#6F7772]">{me?.email || me?.role || "Workspace user"}</p>
        </div>
        <button type="button" onClick={onLogout} className="nl-button-secondary w-full justify-center">
          Logout
        </button>
      </div>
    </aside>
  );
}

export default function AppShell({ children, onLogout, shouldUseScoutingLab, me }) {
  const router = useRouter();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [commandOpen, setCommandOpen] = useState(false);
  const meta = useRouteMeta(router.pathname);

  const scoutingItems = useMemo(
    () => SCOUTING_NAV.filter((item) => !item.adminOnly || me?.role === "admin"),
    [me?.role]
  );

  const commandItems = useMemo(
    () => [
      ...PRIMARY_NAV.map((item) => ({ ...item, section: "Workspace" })),
      ...scoutingItems.map((item) => ({ ...item, section: "Scouting module" })),
    ],
    [scoutingItems]
  );

  useEffect(() => {
    const onKey = (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setCommandOpen(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const isActiveItem = (item) => {
    if (item.href === "/scouting-lab") return shouldUseScoutingLab && router.pathname === "/scouting-lab";
    return router.pathname === item.href || (item.href !== "/" && router.pathname.startsWith(item.href));
  };

  return (
    <div className="min-h-screen bg-[#050706]">
      <div className="lg:pr-[292px]">
        <header className="sticky top-0 z-50 border-b border-white/10 bg-[#050706]/90 backdrop-blur-xl">
          <div className="flex min-h-[64px] items-center justify-between gap-3 px-4 md:px-6">
            <div className="min-w-0">
              <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[#6F7772]">{meta.eyebrow}</p>
              <div className="flex min-w-0 items-baseline gap-3">
                <h1 className="truncate text-base font-semibold text-[#F3F5F4] md:text-lg">{meta.title}</h1>
                <p className="hidden max-w-[540px] truncate text-xs text-[#A0A8A3] xl:block">{meta.desc}</p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button type="button" onClick={() => setCommandOpen(true)} className="hidden h-9 items-center gap-2 rounded-md border border-white/10 bg-white/[0.035] px-3 text-sm text-[#A0A8A3] transition hover:bg-white/[0.06] hover:text-[#F3F5F4] md:inline-flex">
                <Icon name="command" className="h-3.5 w-3.5" />
                Search
                <span className="ml-2 rounded border border-white/10 px-1.5 py-0.5 text-[10px] text-[#6F7772]">Cmd K</span>
              </button>
              <button type="button" className="nl-icon-button lg:hidden" onClick={() => setMobileMenuOpen(true)} aria-label="Open navigation">
                <Icon name="menu" className="h-4 w-4" />
              </button>
            </div>
          </div>
        </header>

        <div className="min-h-[calc(100vh-64px)]">{children}</div>
      </div>

      <div className="fixed bottom-0 right-0 top-0 z-[60] hidden w-[292px] lg:block">
        <RightNavigation
          items={PRIMARY_NAV}
          scoutingItems={scoutingItems}
          me={me}
          onLogout={onLogout}
          activeFor={isActiveItem}
          onNavigate={() => setMobileMenuOpen(false)}
        />
      </div>

      {mobileMenuOpen ? (
        <div className="fixed inset-0 z-[6500] bg-black/70 backdrop-blur-md lg:hidden">
          <div className="absolute inset-y-0 right-0 w-[min(86vw,330px)]">
            <div className="absolute left-3 top-3 z-10">
              <button type="button" className="nl-icon-button" onClick={() => setMobileMenuOpen(false)} aria-label="Close navigation">
                <Icon name="close" className="h-4 w-4" />
              </button>
            </div>
            <RightNavigation
              items={PRIMARY_NAV}
              scoutingItems={scoutingItems}
              me={me}
              onLogout={onLogout}
              activeFor={isActiveItem}
              onNavigate={() => setMobileMenuOpen(false)}
            />
          </div>
        </div>
      ) : null}

      <CommandPalette open={commandOpen} onClose={() => setCommandOpen(false)} items={commandItems} />
    </div>
  );
}
