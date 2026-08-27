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
        <div className="min-w-[180px]">
          <p className="nl-kicker">Scouting Lab</p>
          <h2 className="mt-1 text-lg font-semibold text-white">Research desk</h2>
        </div>
        <nav className="flex min-w-0 flex-1 gap-1 overflow-x-auto">
          {items.map((item) => {
            const isActive =
              router.pathname === item.href ||
              (item.href !== "/scouting-lab" && router.pathname.startsWith(item.href));
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`min-w-[138px] rounded-md border px-3 py-2 transition ${
                  isActive
                    ? "border-[#3A8967]/40 bg-[#2F7D5C]/20 text-[#DDF3E8]"
                    : "border-white/10 bg-white/[0.035] text-white/70 hover:border-[#3A8967]/40 hover:bg-white/[0.06] hover:text-white"
                }`}
              >
                <span className="block text-sm font-semibold">{item.label}</span>
                <span className={`mt-1 block text-xs font-medium ${isActive ? "text-[#8CC7A7]" : "text-white/40"}`}>{item.desc}</span>
              </Link>
            );
          })}
        </nav>
      </aside>
      <section className="scouting-lab-content">{children}</section>
    </div>
  );
}
