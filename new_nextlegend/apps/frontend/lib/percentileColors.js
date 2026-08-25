export const clampPercentile = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return null;
  return Math.max(0, Math.min(100, numeric));
};

export const percentileColor = (value) => {
  const pct = clampPercentile(value);
  if (pct === null) {
    return { label: "N/A", text: "text-slate-400", bg: "bg-slate-100", border: "border-slate-200", fill: "bg-slate-400", track: "bg-slate-100", hex: "#94a3b8", glow: "" };
  }
  if (pct >= 95) {
    return { label: "Elite", text: "text-cyan-700", bg: "bg-cyan-50", border: "border-cyan-200", fill: "bg-cyan-500", track: "bg-cyan-100", hex: "#0891b2", glow: "" };
  }
  if (pct >= 80) {
    return { label: "Very strong", text: "text-teal-700", bg: "bg-teal-50", border: "border-teal-200", fill: "bg-teal-600", track: "bg-teal-100", hex: "#0f766e", glow: "" };
  }
  if (pct >= 60) {
    return { label: "Strong", text: "text-emerald-700", bg: "bg-emerald-50", border: "border-emerald-200", fill: "bg-emerald-500", track: "bg-emerald-100", hex: "#059669", glow: "" };
  }
  if (pct >= 40) {
    return { label: "Average", text: "text-amber-700", bg: "bg-amber-50", border: "border-amber-200", fill: "bg-amber-400", track: "bg-amber-100", hex: "#b45309", glow: "" };
  }
  if (pct >= 20) {
    return { label: "Low", text: "text-orange-700", bg: "bg-orange-50", border: "border-orange-200", fill: "bg-orange-500", track: "bg-orange-100", hex: "#c2410c", glow: "" };
  }
  return { label: "Critical", text: "text-red-700", bg: "bg-red-50", border: "border-red-200", fill: "bg-red-500", track: "bg-red-100", hex: "#b91c1c", glow: "" };
};

export const strengthLevel = (value) => {
  const pct = clampPercentile(value);
  if (pct === null || pct < 80) return null;
  if (pct >= 95) return "ELITE";
  if (pct >= 90) return "VERY STRONG";
  return "STRONG";
};

export const weaknessLevel = (value) => {
  const pct = clampPercentile(value);
  if (pct === null || pct > 20) return null;
  if (pct <= 5) return "CRITICAL";
  if (pct <= 10) return "VERY LOW";
  return "LOW";
};
