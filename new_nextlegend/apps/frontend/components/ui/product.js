export function Panel({ children, className = "", as: Tag = "section" }) {
  return <Tag className={`surface-panel rounded-lg ${className}`}>{children}</Tag>;
}

export function PageHeader({ eyebrow, title, description, actions, className = "" }) {
  return (
    <Panel className={`nl-page-header ${className}`}>
      <div className="relative flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div className="min-w-0">
          {eyebrow ? <p className="nl-kicker">{eyebrow}</p> : null}
          <h1 className="mt-2 max-w-4xl text-3xl font-semibold leading-tight text-slate-950 md:text-4xl">
            {title}
          </h1>
          {description ? <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">{description}</p> : null}
        </div>
        {actions ? <div className="flex shrink-0 flex-wrap gap-2">{actions}</div> : null}
      </div>
    </Panel>
  );
}

export function MetricCard({ label, value, sub, tone = "default" }) {
  const toneClass =
    tone === "success"
      ? "border-[#3A8967]/40 bg-[#2F7D5C]/15"
      : tone === "warning"
        ? "border-amber-300/25 bg-amber-400/10"
        : tone === "danger"
          ? "border-rose-400/25 bg-rose-400/10"
          : "";

  return (
    <div className={`nl-metric-card ${toneClass}`}>
      <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">{label}</p>
      <p className="mt-2 text-3xl font-semibold tracking-tight text-slate-950">{value ?? 0}</p>
      {sub ? <p className="mt-1 text-xs font-medium text-slate-500">{sub}</p> : null}
    </div>
  );
}

export function EmptyState({ title, description, action }) {
  return (
    <div className="nl-empty-state">
      <p className="text-sm font-semibold text-slate-950">{title}</p>
      {description ? <p className="mt-1 text-xs leading-5 text-slate-500">{description}</p> : null}
      {action ? <div className="mt-4">{action}</div> : null}
    </div>
  );
}

export function SkeletonRows({ rows = 4 }) {
  return (
    <div className="space-y-3">
      {Array.from({ length: rows }).map((_, index) => (
        <div key={index} className="grid grid-cols-[44px_minmax(0,1fr)_120px] gap-3 rounded-lg border border-white/10 bg-white/[0.025] p-3">
          <div className="nl-skeleton h-10 w-10 rounded-full" />
          <div className="space-y-2">
            <div className="nl-skeleton h-3 w-2/5" />
            <div className="nl-skeleton h-3 w-3/5" />
          </div>
          <div className="nl-skeleton h-8 w-full" />
        </div>
      ))}
    </div>
  );
}
