import Link from "next/link";
import { useRouter } from "next/router";
import { useState } from "react";

const NAV_ITEMS = [
  { href: "/", label: "HQ" },
  { href: "/hd-players", label: "HD Players" },
  { href: "/crm", label: "Network" },
  { href: "/scouting-lab", label: "Scouting" },
];

const MenuIcon = ({ className = "" }) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={className} aria-hidden="true">
    <path d="M4 7h16" />
    <path d="M4 12h16" />
    <path d="M4 17h16" />
  </svg>
);

const CloseIcon = ({ className = "" }) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={className} aria-hidden="true">
    <path d="m6 6 12 12" />
    <path d="m18 6-12 12" />
  </svg>
);

export default function AppShell({ children, onLogout, shouldUseScoutingLab }) {
  const router = useRouter();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const isActiveItem = (item) => {
    if (item.href === "/scouting-lab") return shouldUseScoutingLab;
    return router.pathname === item.href || (item.href !== "/" && router.pathname.startsWith(item.href));
  };

  const nav = (
    <nav className="flex min-w-0 items-center gap-1 overflow-x-auto rounded-md border border-white/10 bg-white/[0.04] p-1">
      {NAV_ITEMS.map((item) => {
        const active = isActiveItem(item);
        return (
          <Link
            key={item.href}
            href={item.href}
            onClick={() => setMobileMenuOpen(false)}
            className={`whitespace-nowrap rounded px-3 py-2 text-sm font-semibold transition ${
              active
                ? "bg-white text-black shadow-[0_12px_30px_rgba(255,255,255,0.12)]"
                : "text-white/60 hover:bg-white/[0.07] hover:text-white"
            }`}
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );

  return (
    <>
      <header className="sticky top-0 z-50 border-b border-white/10 bg-black/[0.85] text-white backdrop-blur-xl">
        <div className="mx-auto flex max-w-[1500px] items-center justify-between gap-4 px-4 py-3">
          <Link href="/" className="group flex min-w-fit items-center gap-3" onClick={() => setMobileMenuOpen(false)}>
            <span className="flex h-10 w-10 items-center justify-center rounded-md border border-white/10 bg-white shadow-[0_12px_28px_rgba(245,158,11,0.12)]">
              <img src="/logo_nl.png" alt="Next Legend" className="h-7 w-7 object-contain" />
            </span>
            <span className="leading-tight">
              <span className="block text-lg font-semibold tracking-normal text-white">Next Legend</span>
              <span className="block text-[11px] font-bold uppercase tracking-[0.16em] text-amber-100/50">
                HD Sports intelligence
              </span>
            </span>
          </Link>

          <div className="hidden flex-1 items-center justify-center lg:flex">{nav}</div>

          <div className="hidden items-center gap-3 lg:flex">
            <button type="button" onClick={onLogout} className="nl-button-ghost px-3">
              Logout
            </button>
          </div>

          <button
            type="button"
            className="inline-flex h-10 w-10 items-center justify-center rounded-md border border-white/10 bg-white/[0.04] text-white transition hover:bg-white/[0.08] lg:hidden"
            onClick={() => setMobileMenuOpen((open) => !open)}
            aria-label="Toggle navigation"
            aria-expanded={mobileMenuOpen}
          >
            {mobileMenuOpen ? <CloseIcon className="h-5 w-5" /> : <MenuIcon className="h-5 w-5" />}
          </button>
        </div>

        {mobileMenuOpen ? (
          <div className="border-t border-white/10 bg-black/95 px-4 py-4 shadow-[0_24px_60px_rgba(0,0,0,0.44)] lg:hidden">
            <div className="mx-auto flex max-w-[1500px] flex-col gap-3">
              {nav}
              <button type="button" onClick={onLogout} className="nl-button-ghost w-full">
                Logout
              </button>
            </div>
          </div>
        ) : null}
      </header>
      {children}
    </>
  );
}
