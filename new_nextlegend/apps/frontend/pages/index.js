import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen bg-hero-pattern text-slate-100 py-16 px-4">
      <div className="max-w-5xl mx-auto space-y-8">
        <header className="space-y-3">
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
            NextLegend v2
          </p>
          <h1 className="text-4xl md:text-5xl font-bold text-white tracking-tight">
            Player intelligence, rebuilt for scale.
          </h1>
          <p className="text-slate-300 max-w-2xl">
            Explore ranking pages powered by the new pipeline and API. Compare
            roles, filter by league, and share shortlists with confidence.
          </p>
        </header>

        <div className="glass-panel rounded-2xl p-6 border border-white/5 space-y-4">
          <h2 className="text-2xl font-semibold text-white">
            Jump into Ranking
          </h2>
          <p className="text-slate-300">
            The v2 ranking view includes pagination, league filters, and instant
            role context from the serving DB.
          </p>
          <Link
            href="/ranking"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-semibold"
          >
            Open Ranking
          </Link>
        </div>
      </div>
    </main>
  );
}
