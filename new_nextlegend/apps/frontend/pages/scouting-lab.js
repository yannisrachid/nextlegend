import Link from "next/link";

const MODULES = [
  { href: "/ranking", title: "Ranking", desc: "Build targeted player cohorts by league, role and performance signals." },
  { href: "/report", title: "Reports", desc: "Open director-ready player dossiers with radars, percentiles and market context." },
  { href: "/comparison", title: "Compare", desc: "Evaluate profiles side by side with the metrics that matter for recruitment." },
  { href: "/projection", title: "Projection", desc: "Estimate how a player profile translates into a new competitive environment." },
  { href: "/stats-research", title: "Stats Research", desc: "Explore metric relationships and identify undervalued player patterns." },
  { href: "/vizualisation", title: "Visuals", desc: "Create clear, export-ready visuals for reports, decks and club conversations." },
  { href: "/prospect", title: "Prospect", desc: "Track watchlists, club needs and priority scouting pipelines." },
  { href: "/ai", title: "AI Assistant", desc: "Ask structured scouting questions and turn data into actionable briefs." },
];

export default function ScoutingLabPage() {
  return (
    <main className="nl-page px-4 py-8">
      <div className="mx-auto max-w-[1280px] space-y-6">
        <section className="surface-panel rounded-lg p-6 md:p-8">
          <p className="nl-kicker">Scouting Lab</p>
          <h1 className="mt-3 text-4xl font-extrabold text-slate-950 md:text-5xl">
            The intelligence layer behind every HD Sports decision.
          </h1>
          <p className="mt-4 max-w-3xl text-sm leading-6 text-slate-600">
            Research players, compare profiles, prepare reports and turn raw performance data into market-ready recommendations.
          </p>
        </section>

        <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {MODULES.map((module) => (
            <Link key={module.href} href={module.href} className="surface-panel rounded-lg p-5 transition hover:-translate-y-0.5 hover:border-teal-500">
              <h2 className="text-xl font-extrabold text-slate-950">{module.title}</h2>
              <p className="mt-2 text-sm leading-6 text-slate-600">{module.desc}</p>
              <span className="mt-5 inline-flex text-sm font-extrabold text-teal-700">Launch module</span>
            </Link>
          ))}
        </section>
      </div>
    </main>
  );
}
