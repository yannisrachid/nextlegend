import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { fetchJson } from "@/lib/api";

export default function Home() {
  const [opsMetrics, setOpsMetrics] = useState(null);
  const [seasonCount, setSeasonCount] = useState(0);
  const [competitionCount, setCompetitionCount] = useState(0);
  const [metricCount, setMetricCount] = useState(0);
  const [metricsError, setMetricsError] = useState("");

  useEffect(() => {
    const load = async () => {
      try {
        const [ops, seasons, competitions, metrics] = await Promise.all([
          fetchJson("/ops/metrics"),
          fetchJson("/meta/seasons"),
          fetchJson("/meta/competitions"),
          fetchJson("/meta/stats-research/metrics"),
        ]);
        setOpsMetrics(ops);
        setSeasonCount((seasons || []).length);
        setCompetitionCount((competitions || []).length);
        setMetricCount((metrics?.metrics || []).length);
      } catch (err) {
        console.error(err);
        setMetricsError("Unable to load ops metrics.");
      }
    };
    load();
  }, []);

  const formatNumber = (value) => {
    if (value === null || value === undefined) return "--";
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return String(value);
    return numeric.toLocaleString();
  };

  const stats = useMemo(() => {
    const counts = opsMetrics?.counts || {};
    return [
      { label: "Players in database", value: formatNumber(counts.players) },
      { label: "Competitions covered", value: formatNumber(competitionCount) },
      { label: "Seasons tracked", value: formatNumber(seasonCount) },
      { label: "Metrics tracked", value: formatNumber(metricCount) },
    ];
  }, [opsMetrics, competitionCount, seasonCount, metricCount]);

  const lastRun = opsMetrics?.last_pipeline_run;
  const lastRunDate = lastRun?.started_at
    ? new Date(lastRun.started_at).toLocaleDateString()
    : "--";
  const lastRunRows = lastRun?.rows_processed
    ? formatNumber(lastRun.rows_processed)
    : "--";

  const tools = [
    {
      title: "Scouting Report",
      description:
        "Generate player dossiers with radar, role fit, and Transfermarkt context.",
      href: "/report",
    },
    {
      title: "Ranking",
      description:
        "Filter cohorts by league and role, then shortlist with confidence.",
      href: "/ranking",
    },
    {
      title: "Comparison",
      description:
        "Side-by-side metrics, summary scores, and radar context shifts.",
      href: "/comparison",
    },
    {
      title: "Projection",
      description:
        "Translate performance across leagues and simulate target environments.",
      href: "/projection",
    },
    {
      title: "Vizualisation",
      description:
        "Create pizza charts and export visuals for reports and decks.",
      href: "/vizualisation",
    },
    {
      title: "Stats Research",
      description:
        "Explore bivariate relationships and discover hidden profiles.",
      href: "/stats-research",
    },
    {
      title: "Prospect Hub",
      description:
        "Track prospects, club needs, and priority pipelines.",
      href: "/prospect",
    },
    {
      title: "AI Assistant",
      description:
        "Chat with agentic scouting workflows using your database.",
      href: "/ai",
    },
  ];

  return (
    <main className="min-h-screen bg-hero-pattern text-slate-100 py-16 px-4">
      <div className="max-w-6xl mx-auto space-y-10">
        <header className="text-center space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-white/10 bg-slate-900/50 text-xs uppercase tracking-[0.3em] text-slate-300">
            NextLegend by Your Legend
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-white tracking-tight">
            Scout with Intelligence
          </h1>
          <p className="text-slate-300 max-w-2xl mx-auto">
            One workspace for reports, rankings, comparisons, projections, and
            AI scouting. Built for live pipelines and weekly refreshes.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-3">
            <Link
              href="/report"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-semibold"
            >
              Open Report
            </Link>
            <Link
              href="/ranking"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-md border border-slate-600 text-slate-100 hover:border-slate-400"
            >
              Open Ranking
            </Link>
          </div>
        </header>

        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {stats.map((item) => (
            <div
              key={item.label}
              className="glass-panel rounded-2xl p-5 border border-white/5 text-center"
            >
              <div className="text-3xl font-semibold text-primary">
                {item.value}
              </div>
              <div className="mt-2 text-xs uppercase tracking-[0.2em] text-slate-400">
                {item.label}
              </div>
            </div>
          ))}
        </section>

        <section className="glass-panel rounded-2xl p-6 border border-white/5">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                Data operations
              </p>
              <h2 className="text-2xl font-semibold text-white mt-2">
                Pipeline status
              </h2>
              <p className="text-slate-300 mt-1">
                Last refresh: {lastRunDate} - {lastRunRows} rows
              </p>
              {metricsError ? (
                <p className="text-xs text-amber-300 mt-2">{metricsError}</p>
              ) : null}
            </div>
            <div className="flex items-center gap-3">
              <Link
                href="/health"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md border border-slate-600 text-slate-100 hover:border-slate-400"
              >
                Health checks
              </Link>
              <Link
                href="/prospect"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-slate-100 text-slate-900 font-semibold hover:bg-white"
              >
                Prospect Hub
              </Link>
            </div>
          </div>
        </section>

        <section className="space-y-4">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <h2 className="text-2xl font-semibold text-white">
              Tools for Advanced Analysis
            </h2>
            <span className="text-xs uppercase tracking-[0.2em] text-slate-400">
              Explore modules
            </span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {tools.map((tool) => (
              <Link
                key={tool.title}
                href={tool.href}
                className="glass-panel rounded-2xl p-5 border border-white/5 hover:border-primary/60 transition"
              >
                <h3 className="text-lg font-semibold text-white">
                  {tool.title}
                </h3>
                <p className="text-slate-300 mt-2 text-sm">
                  {tool.description}
                </p>
                <span className="text-xs uppercase tracking-[0.2em] text-primary mt-4 inline-flex">
                  Open module
                </span>
              </Link>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
