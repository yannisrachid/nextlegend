import Link from "next/link";
import { useRouter } from "next/router";

const LAB_ITEMS = [
  { href: "/scouting-lab", label: "Lab HQ", desc: "Scouting overview" },
  { href: "/ranking", label: "Ranking", desc: "Player cohorts" },
  { href: "/report", label: "Reports", desc: "Director dossiers" },
  { href: "/comparison", label: "Compare", desc: "Profile matchups" },
  { href: "/projection", label: "Projection", desc: "League fit" },
  { href: "/stats-research", label: "Research", desc: "Metric discovery" },
  { href: "/vizualisation", label: "Visuals", desc: "Export assets" },
  { href: "/prospect", label: "Prospect", desc: "Watchlists" },
  { href: "/ai", label: "AI", desc: "Scouting briefs" },
];

export default function ScoutingLabShell({ children, me }) {
  const router = useRouter();
  const isLabHome = router.pathname === "/scouting-lab";
  const items = LAB_ITEMS;

  return (
    <div className="scouting-lab-shell">
      <aside className="scouting-lab-sidebar">
        {isLabHome ? (
          <div className="min-w-0 shrink-0 xl:w-[180px]">
            <p className="nl-kicker">Scouting Lab</p>
            <h2 className="mt-1 text-lg font-semibold text-foreground">Research desk</h2>
          </div>
        ) : null}
        <nav aria-label="Scouting modules" className="flex w-full min-w-0 flex-1 gap-1 overflow-x-auto xl:w-auto">
          {items.map((item) => {
            const isActive =
              router.pathname === item.href ||
              (item.href !== "/scouting-lab" && router.pathname.startsWith(item.href));
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={isActive ? "page" : undefined}
                title={item.desc}
                className={`shrink-0 rounded-md border px-3 py-2 transition ${
                  isActive
                    ? "border-primary/25 bg-accent text-primary"
                    : "border-transparent bg-transparent text-muted-foreground hover:border-primary/25 hover:bg-secondary hover:text-foreground"
                }`}
              >
                <span className="block text-sm font-semibold">{item.label}</span>
              </Link>
            );
          })}
        </nav>
      </aside>
      <section className="scouting-lab-content">{children}</section>
    </div>
  );
}
