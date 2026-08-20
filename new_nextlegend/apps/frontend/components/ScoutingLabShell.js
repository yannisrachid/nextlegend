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
  { href: "/admin", label: "Admin", desc: "Access control", adminOnly: true },
];

export default function ScoutingLabShell({ children, me }) {
  const router = useRouter();
  const items = LAB_ITEMS.filter((item) => !item.adminOnly || me?.role === "admin");

  return (
    <div className="scouting-lab-shell">
      <aside className="scouting-lab-sidebar">
        <div className="mb-5">
          <p className="nl-kicker">Scouting Lab</p>
          <h2 className="mt-2 text-xl font-extrabold text-slate-950">Research desk</h2>
        </div>
        <nav className="flex gap-2 overflow-x-auto pb-2 lg:flex-col lg:overflow-visible lg:pb-0">
          {items.map((item) => {
            const isActive =
              router.pathname === item.href ||
              (item.href !== "/scouting-lab" && router.pathname.startsWith(item.href));
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`min-w-[150px] rounded-md border px-3 py-3 transition lg:min-w-0 ${
                  isActive
                    ? "border-teal-600 bg-teal-50 text-teal-900"
                    : "border-slate-200 bg-white text-slate-700 hover:border-teal-500 hover:text-slate-950"
                }`}
              >
                <span className="block text-sm font-extrabold">{item.label}</span>
                <span className="mt-1 block text-xs font-semibold text-slate-500">{item.desc}</span>
              </Link>
            );
          })}
        </nav>
      </aside>
      <section className="scouting-lab-content">{children}</section>
    </div>
  );
}
