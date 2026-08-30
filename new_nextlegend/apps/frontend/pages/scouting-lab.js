import Link from "next/link";
import { PageHeader, Panel } from "@/components/ui/product";

const MODULES = [
  { href: "/ranking", title: "Ranking", metric: "Cohorts", desc: "Build targeted player pools by league, position and performance signals.", primary: true },
  { href: "/report", title: "Reports", metric: "Dossiers", desc: "Open director-ready player dossiers with radars, percentiles and market context.", primary: true },
  { href: "/comparison", title: "Compare", metric: "Boards", desc: "Evaluate profiles side by side with the metrics that matter for recruitment.", primary: true },
  { href: "/projection", title: "Projection", metric: "Fit", desc: "Estimate how a player profile translates into a new competitive environment." },
  { href: "/stats-research", title: "Stats Research", metric: "Signals", desc: "Explore metric relationships and identify undervalued player patterns." },
  { href: "/vizualisation", title: "Visuals", metric: "Exports", desc: "Create clear, export-ready visuals for reports, decks and club conversations." },
  { href: "/prospect", title: "Prospect", metric: "Pipeline", desc: "Track watchlists, club needs and priority scouting pipelines." },
  { href: "/ai", title: "AI Assistant", metric: "Briefs", desc: "Ask structured scouting questions and turn data into actionable briefs." },
];

export default function ScoutingLabPage() {
  return (
    <main className="nl-page px-4 py-8">
      <div className="mx-auto max-w-[1280px] space-y-6">
        <PageHeader
          eyebrow="Scouting Lab"
          title="The intelligence layer behind every HD Sports decision."
          description="Research players, compare profiles, prepare reports and turn raw performance data into market-ready recommendations."
        />

        <section className="grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
          <Panel className="p-5">
            <div className="grid gap-3 md:grid-cols-3">
              {MODULES.filter((module) => module.primary).map((module) => (
                <Link key={module.href} href={module.href} className="rounded-lg border border-[#3A8967]/30 bg-[#2F7D5C]/15 p-4 transition hover:-translate-y-0.5 hover:border-[#3A8967]/50 hover:bg-[#2F7D5C]/20">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[#8CC7A7]">{module.metric}</p>
                  <h2 className="mt-2 text-xl font-semibold text-slate-950">{module.title}</h2>
                  <p className="mt-2 text-sm leading-6 text-slate-600">{module.desc}</p>
                </Link>
              ))}
            </div>
          </Panel>

          <Panel className="p-5">
            <p className="nl-kicker">Workflow</p>
            <div className="mt-4 space-y-3">
              {["Build a cohort", "Open the dossier", "Compare targets", "Export the evidence"].map((label, index) => (
                <div key={label} className="flex items-center gap-3 rounded-md border border-white/10 bg-white/[0.03] px-3 py-2">
                  <span className="flex h-7 w-7 items-center justify-center rounded-md bg-[#2F7D5C]/20 text-xs font-semibold text-[#8CC7A7]">
                    {index + 1}
                  </span>
                  <span className="text-sm font-medium text-slate-700">{label}</span>
                </div>
              ))}
            </div>
          </Panel>
        </section>

        <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
          {MODULES.filter((module) => !module.primary).map((module) => (
            <Link key={module.href} href={module.href} className="surface-panel rounded-lg p-4 transition hover:-translate-y-0.5 hover:border-[#3A8967]/40 hover:bg-white/[0.06]">
              <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[#8CC7A7]">{module.metric}</p>
              <h2 className="mt-2 text-lg font-semibold text-slate-950">{module.title}</h2>
              <p className="mt-2 text-sm leading-6 text-slate-600">{module.desc}</p>
              <span className="mt-4 inline-flex text-sm font-semibold text-[#8CC7A7]">Launch</span>
            </Link>
          ))}
        </section>
      </div>
    </main>
  );
}
