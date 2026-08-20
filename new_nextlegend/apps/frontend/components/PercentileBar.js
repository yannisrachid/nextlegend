const clampPercentile = (value, max) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return null;
  return Math.max(0, Math.min(max, numeric));
};

const percentileTone = (percent) => {
  if (percent === null) {
    return {
      label: "N/A",
      fill: "bg-slate-600",
      text: "text-slate-400",
      track: "bg-slate-800/80",
      glow: "",
    };
  }
  if (percent <= 30) {
    return {
      label: "Low",
      fill: "bg-red-500",
      text: "text-red-300",
      track: "bg-red-950/40",
      glow: "shadow-[0_0_16px_rgba(239,68,68,0.22)]",
    };
  }
  if (percent <= 60) {
    return {
      label: "Medium",
      fill: "bg-amber-400",
      text: "text-amber-200",
      track: "bg-amber-950/35",
      glow: "shadow-[0_0_16px_rgba(251,191,36,0.20)]",
    };
  }
  if (percent <= 80) {
    return {
      label: "Good",
      fill: "bg-lime-400",
      text: "text-lime-200",
      track: "bg-lime-950/35",
      glow: "shadow-[0_0_16px_rgba(163,230,53,0.20)]",
    };
  }
  return {
    label: "Excellent",
    fill: "bg-teal-400",
    text: "text-teal-200",
    track: "bg-teal-950/40",
    glow: "shadow-[0_0_18px_rgba(45,212,191,0.26)]",
  };
};

export default function PercentileBar({
  label,
  value,
  max = 100,
  compact = false,
  showLabel = true,
  className = "",
}) {
  const safeMax = Number.isFinite(Number(max)) && Number(max) > 0 ? Number(max) : 100;
  const clamped = clampPercentile(value, safeMax);
  const percent = clamped === null ? null : Math.round((clamped / safeMax) * 100);
  const displayValue = percent === null ? "N/A" : String(Math.round(clamped));
  const tone = percentileTone(percent);
  const barWidth = percent === null ? 0 : percent;
  const segments = [10, 20, 30, 40, 50, 60, 70, 80, 90];

  return (
    <div className={`w-full min-w-[150px] ${className}`}>
      <div className={`mb-1 flex items-center justify-between gap-3 ${compact ? "text-[11px]" : "text-xs"}`}>
        {showLabel ? (
          <span className="min-w-0 truncate font-semibold text-slate-200">{label}</span>
        ) : (
          <span className="sr-only">{label}</span>
        )}
        <span className={`shrink-0 font-black tabular-nums ${tone.text}`}>{displayValue}</span>
      </div>
      <div
        className={`relative overflow-hidden rounded-full border border-white/10 ${tone.track} ${
          compact ? "h-2.5" : "h-3.5"
        }`}
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={safeMax}
        aria-valuenow={clamped === null ? undefined : clamped}
        aria-valuetext={clamped === null ? "N/A" : `${Math.round(clamped)} percentile`}
      >
        <div
          className={`h-full rounded-full ${tone.fill} ${tone.glow} transition-[width] duration-500 ease-out`}
          style={{ width: `${barWidth}%` }}
        />
        {segments.map((position) => (
          <span
            key={position}
            className="pointer-events-none absolute top-0 h-full w-px bg-slate-950/35"
            style={{ left: `${position}%` }}
            aria-hidden="true"
          />
        ))}
      </div>
      {!compact ? (
        <div className="mt-1 flex items-center justify-between text-[10px] uppercase tracking-[0.14em] text-slate-500">
          <span>{tone.label}</span>
          <span>100</span>
        </div>
      ) : null}
    </div>
  );
}
