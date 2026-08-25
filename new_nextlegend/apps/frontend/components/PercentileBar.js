import { percentileColor } from "@/lib/percentileColors";

const clampValue = (value, max) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return null;
  return Math.max(0, Math.min(max, numeric));
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
  const clamped = clampValue(value, safeMax);
  const percent = clamped === null ? null : Math.round((clamped / safeMax) * 100);
  const displayValue = percent === null ? "N/A" : String(Math.round(clamped));
  const tone = percentileColor(percent);
  const barWidth = percent === null ? 0 : percent;
  const segments = [10, 20, 30, 40, 50, 60, 70, 80, 90];

  return (
    <div className={`w-full min-w-[150px] ${className}`}>
      <div className={`mb-1 flex items-center justify-between gap-3 ${compact ? "text-[11px]" : "text-xs"}`}>
        {showLabel ? (
          <span className="min-w-0 truncate font-semibold text-slate-700">{label}</span>
        ) : (
          <span className="sr-only">{label}</span>
        )}
        <span className={`shrink-0 font-black tabular-nums ${tone.text}`}>{displayValue}</span>
      </div>
      <div
        className={`relative overflow-hidden rounded-full border border-slate-200 ${tone.track} ${
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
            className="pointer-events-none absolute top-0 h-full w-px bg-white/80"
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
