import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/router";
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
} from "recharts";
import { apiUrl, fetchJson, fetchJsonCached, postJson, deleteJson } from "@/lib/api";
import ClubLogo from "@/components/ClubLogo";
import PercentileBar from "@/components/PercentileBar";
import { loadClubLogoData, resolveClubLogoUrl } from "@/lib/clubLogos";
import { METRIC_LABELS } from "@/lib/metricLabels";
import { englishRole, normalizeRoleForUse } from "@/lib/roles";

const DEFAULT_RADAR_METRICS = [
  "goals_per_90",
  "xa_per_90",
  "accurate_passes_percent",
  "passes_to_penalty_area_per_90",
  "progressive_passes_per_90",
  "progressive_runs_per_90",
  "successful_dribbles_percent",
  "def_duels_won_percent",
  "interceptions_padj",
  "aerial_duels_won_percent",
];

const TM_BASE_URL = "https://www.transfermarkt.com";
const EXCLUDED_METRIC_PREFIXES = [
  "calendar",
  "page_number",
  "row_number",
  "player",
  "player_id",
  "birth_year",
  "age",
  "matches_played",
  "minutes_played",
  "team",
  "competition",
  "assigned_role",
];

const Card = ({ children, className = "" }) => (
  <div className={`glass-panel rounded-xl p-4 border border-white/5 ${className}`}>
    {children}
  </div>
);

const Label = ({ children, htmlFor }) => (
  <label htmlFor={htmlFor} className="text-xs uppercase tracking-[0.2em] text-slate-400">
    {children}
  </label>
);

const Select = ({ value, onChange, children, id, name, ariaLabel }) => (
  <select
    id={id}
    name={name || id}
    aria-label={ariaLabel}
    className="w-full min-w-0 max-w-full bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 pr-8 text-slate-100 overflow-hidden text-ellipsis whitespace-nowrap"
    value={value}
    onChange={onChange}
  >
    {children}
  </select>
);

const Badge = ({ children }) => (
  <span className="px-2 py-1 rounded-full bg-slate-800 text-xs text-slate-200 border border-white/5">
    {children}
  </span>
);

const RadarTooltip = ({ active, payload }) => {
  if (!active || !payload || payload.length === 0) return null;
  const point = payload[0]?.payload;
  if (!point) return null;
  return (
    <div className="rounded-md border border-slate-700 bg-slate-900/95 px-3 py-2 text-xs text-slate-100 shadow-xl">
      <div className="font-semibold text-white">{point.metric}</div>
      <div className="text-slate-300">
        {point.contextLabel}: {Number(payload[0]?.value ?? point.value).toFixed(0)}
      </div>
      <div className="text-slate-400">
        Value: {point.raw != null ? Number(point.raw).toFixed(2) : "—"}
      </div>
    </div>
  );
};

const toAbsoluteUrl = (value) => {
  if (!value) return "";
  const url = String(value).trim();
  if (!url) return "";
  if (url.startsWith("http://") || url.startsWith("https://")) {
    return url;
  }
  if (url.startsWith("/")) {
    return `${TM_BASE_URL}${url}`;
  }
  return url;
};

const toCanvasSafeImageUrl = (value) => {
  const url = toAbsoluteUrl(value);
  if (!url || url.startsWith("data:")) return url;
  try {
    const parsed = new URL(url, window.location.href);
    if (parsed.origin === window.location.origin) {
      return url;
    }
    return apiUrl("/image-proxy", { url });
  } catch {
    return "";
  }
};

const extractUrls = (value) => {
  if (!value) return [];
  if (Array.isArray(value)) {
    return value
      .map((item) => (typeof item === "string" ? item : item?.url))
      .filter(Boolean);
  }
  if (typeof value === "object") {
    return Object.values(value)
      .map((item) => (typeof item === "string" ? item : item?.url))
      .filter(Boolean);
  }
  const raw = String(value).trim();
  if (!raw) return [];
  const urls = raw.match(/https?:\/\/[^\s,;]+/gi) || [];
  if (urls.length > 0) return urls;
  return raw
    .split(/[;,]/g)
    .map((item) => item.trim())
    .filter((item) => item.startsWith("http"));
};

const resolveSocialType = (url) => {
  const lower = String(url || "").toLowerCase();
  if (lower.includes("instagram.com")) return "instagram";
  if (lower.includes("twitter.com") || lower.includes("x.com")) return "x";
  if (lower.includes("facebook.com")) return "facebook";
  if (lower.includes("tiktok.com")) return "tiktok";
  if (lower.includes("youtube.com") || lower.includes("youtu.be")) return "youtube";
  if (lower.includes("linkedin.com")) return "linkedin";
  if (lower.includes("twitch.tv")) return "twitch";
  return "link";
};

const SocialIcon = ({ type }) => {
  switch (type) {
    case "instagram":
      return (
        <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true">
          <path
            fill="currentColor"
            d="M7 3h10a4 4 0 014 4v10a4 4 0 01-4 4H7a4 4 0 01-4-4V7a4 4 0 014-4zm10 2H7a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2zm-5 3.2a3.8 3.8 0 110 7.6 3.8 3.8 0 010-7.6zm0 1.8a2 2 0 100 4 2 2 0 000-4zm4.4-2a1 1 0 110 2 1 1 0 010-2z"
          />
        </svg>
      );
    case "x":
      return (
        <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true">
          <path
            fill="currentColor"
            d="M18.9 3h2.7l-6 6.9L22 21h-5.3l-4.2-6.3L6.8 21H4.1l6.5-7.5L2 3h5.4l3.8 5.7L18.9 3z"
          />
        </svg>
      );
    case "facebook":
      return (
        <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true">
          <path
            fill="currentColor"
            d="M13.5 8.5V7.1c0-.7.5-1.1 1.2-1.1h1.8V3.1h-2.4c-2.4 0-3.6 1.4-3.6 3.6v1.8H8v2.9h2.5V21h3V11.4h2.7l.4-2.9h-3.1z"
          />
        </svg>
      );
    case "tiktok":
      return (
        <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true">
          <path
            fill="currentColor"
            d="M16.5 3c.4 1.7 1.7 3 3.5 3.4v2.8c-1.3.1-2.7-.3-3.5-.9v6.5a5.4 5.4 0 11-5.4-5.4c.3 0 .7 0 1 .1v2.9a2.5 2.5 0 10 2.3 2.5V3h2.1z"
          />
        </svg>
      );
    case "youtube":
      return (
        <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true">
          <path
            fill="currentColor"
            d="M21.6 7.7a2.8 2.8 0 00-2-2c-1.7-.4-8.6-.4-8.6-.4s-6.9 0-8.6.4a2.8 2.8 0 00-2 2A29.5 29.5 0 000 12a29.5 29.5 0 00.4 4.3 2.8 2.8 0 002 2c1.7.4 8.6.4 8.6.4s6.9 0 8.6-.4a2.8 2.8 0 002-2A29.5 29.5 0 0022 12a29.5 29.5 0 00-.4-4.3zM9.8 15.5V8.5l6 3.5-6 3.5z"
          />
        </svg>
      );
    case "linkedin":
      return (
        <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true">
          <path
            fill="currentColor"
            d="M4.9 3.5a2.2 2.2 0 11-.1 4.4 2.2 2.2 0 010-4.4zM3 9h3.7v12H3V9zm7 0h3.5v1.7h.1c.5-.9 1.8-1.9 3.6-1.9 3.9 0 4.6 2.4 4.6 5.5V21h-3.7v-5.2c0-1.2 0-2.8-1.7-2.8-1.7 0-2 1.3-2 2.7V21H10V9z"
          />
        </svg>
      );
    case "twitch":
      return (
        <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true">
          <path
            fill="currentColor"
            d="M4 3h16v9l-4 4h-4l-2 2H7v-2H4V3zm3 2v9h4v2l2-2h4l2-2V5H7zm4 3h2v4h-2V8zm4 0h2v4h-2V8z"
          />
        </svg>
      );
    default:
      return (
        <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true">
          <path
            fill="currentColor"
            d="M10.6 13.4a4 4 0 010-5.7l2.1-2.1a4 4 0 115.7 5.7l-1.1 1.1-1.4-1.4 1.1-1.1a2 2 0 10-2.8-2.8l-2.1 2.1a2 2 0 102.8 2.8l.7-.7 1.4 1.4-.7.7a4 4 0 01-5.7 0zm-3 3l-2.1 2.1a4 4 0 11-5.7-5.7l1.1-1.1 1.4 1.4-1.1 1.1a2 2 0 102.8 2.8l2.1-2.1a2 2 0 10-2.8-2.8l-.7.7-1.4-1.4.7-.7a4 4 0 015.7 5.7z"
          />
        </svg>
      );
  }
};

const formatMetricLabel = (key) => {
  if (METRIC_LABELS[key]) return METRIC_LABELS[key];
  if (key.startsWith("summary_")) {
    const label = key
      .slice("summary_".length)
      .split("_")
      .map((chunk) => chunk.charAt(0).toUpperCase() + chunk.slice(1))
      .join(" ");
    return `Summary ${label}`;
  }
  return key.replace(/_/g, " ");
};

const normalizeMetricKey = (value) => {
  if (!value) return "";
  return String(value)
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]/gi, "")
    .toLowerCase();
};

const formatCompactNumber = (value) => {
  if (value === null || value === undefined || value === "") return "";
  let numeric = value;
  if (typeof numeric === "string") {
    const raw = numeric.trim();
    if (!raw) return "";
    const lower = raw.toLowerCase();
    const match = lower.match(/([0-9]+(?:\\.[0-9]+)?)/);
    if (!match) return raw;
    numeric = Number(match[1]);
    if (!Number.isFinite(numeric)) return raw;
    if (lower.includes("bn") || lower.includes("b")) {
      numeric *= 1e9;
    } else if (lower.includes("m")) {
      numeric *= 1e6;
    } else if (lower.includes("k")) {
      numeric *= 1e3;
    }
  }
  if (typeof numeric !== "number" || !Number.isFinite(numeric)) {
    return String(value);
  }
  const abs = Math.abs(numeric);
  const format = (num, suffix) => {
    const rounded = num >= 10 ? Math.round(num) : Math.round(num * 10) / 10;
    const label = rounded % 1 === 0 ? rounded.toFixed(0) : rounded.toFixed(1);
    return `${label} ${suffix}`;
  };
  if (abs >= 1e9) return format(abs / 1e9, "B");
  if (abs >= 1e6) return format(abs / 1e6, "M");
  if (abs >= 1e3) return format(abs / 1e3, "K");
  return `${Math.round(abs)}`;
};

const formatTransferDate = (input) => {
  if (!input) return "Date to confirm";
  const parsed = new Date(input);
  if (Number.isNaN(parsed.getTime())) return String(input);
  return new Intl.DateTimeFormat("en", { day: "2-digit", month: "short", year: "numeric" }).format(parsed);
};

const transferFeeLabel = (input) => {
  const clean = String(input || "").trim();
  if (!clean) return "Undisclosed";
  const numeric = Number(clean.replace(/[^\d.-]/g, ""));
  if (Number.isFinite(numeric) && numeric > 0 && /^[\d\s.,€£$-]+$/.test(clean)) {
    return formatCompactNumber(numeric);
  }
  return clean;
};

const getInitials = (value) => {
  if (!value) return "—";
  const parts = String(value).trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "—";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase();
};

const clampPercentile = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return null;
  return Math.max(0, Math.min(100, numeric));
};

const topPercentLabel = (percentile) => {
  const value = clampPercentile(percentile);
  if (value === null) return "N/A";
  const top = Math.max(1, 100 - Math.round(value));
  return `Top ${top}%`;
};

const formatEvidenceValue = (value) => {
  if (value === null || value === undefined || value === "") return "N/A";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return String(value);
  if (Math.abs(numeric) >= 1000) return formatCompactNumber(numeric);
  return numeric % 1 === 0 ? numeric.toFixed(0) : numeric.toFixed(1);
};

const simplifyMetricLabel = (label) =>
  String(label || "")
    .replace(/\s+per\s+90/gi, " / 90")
    .replace(/\s+percent$/i, " %")
    .replace(/^Summary\s+/i, "")
    .trim();

const ALWAYS_EXCLUDED_EVIDENCE = [
  "height",
  "height_cm",
  "weight",
  "birth",
  "market_value",
  "contract",
  "yellow_card",
  "red_card",
  "foul",
  "goals_conceded",
  "xg_against",
];
const WIDE_EVIDENCE = ["cross", "deep_cross", "goal_area_cross", "passes_to_penalty_area", "xa"];
const BALL_CARRY_EVIDENCE = ["dribble", "progressive_run", "acceleration"];

const isRelevantEvidenceMetric = (key, player = {}) => {
  const normalizedKey = String(key || "").toLowerCase();
  if (ALWAYS_EXCLUDED_EVIDENCE.some((item) => normalizedKey.includes(item))) return false;
  const position = String(player.position || "").toUpperCase();
  const role = normalizeRoleForUse(player.assigned_role || "");
  const isWide =
    ["LB", "RB", "LWB", "RWB", "LW", "RW", "LM", "RM"].some((item) => position.includes(item)) ||
    role.includes("wing") ||
    role.includes("wide") ||
    role.includes("left back") ||
    role.includes("right back");
  const isCentreBack =
    ["CB", "LCB", "RCB"].some((item) => position === item || position.includes(item)) ||
    role.includes("centre back") ||
    role.includes("center back");
  if (isCentreBack && WIDE_EVIDENCE.some((item) => normalizedKey.includes(item))) return false;
  if (isCentreBack && !role.includes("wide") && BALL_CARRY_EVIDENCE.some((item) => normalizedKey.includes(item))) return false;
  if (!isWide && normalizedKey.includes("accurate_crosses_percent")) return false;
  return true;
};

const getProductionStat = (metrics = {}) => {
  const choices = [
    { key: "goals", label: "Goals" },
    { key: "assists", label: "Assists" },
  ];
  const found = choices.find((item) => {
    const value = metrics[item.key];
    const numeric = Number(value);
    return value !== null && value !== undefined && value !== "" && Number.isFinite(numeric) && numeric > 0;
  });
  if (!found) return null;
  return { label: found.label, value: Number(metrics[found.key]).toFixed(0) };
};

const buildDirectorInsights = ({ report, allMetricsData, topRoles }) => {
  if (!report) {
    return { headline: "", roleBadge: "Profile", proofPoints: [], reasons: [] };
  }
  const primaryRole = topRoles?.[0] || {};
  const rolePct = clampPercentile(primaryRole.pct_global_adjusted ?? primaryRole.pct_global ?? primaryRole.pct_league);
  const roleName = englishRole(primaryRole.profile || report.player?.assigned_role) || report.player?.position || "Football profile";
  const roleBadge = rolePct === null ? roleName : `${topPercentLabel(rolePct)} ${roleName}`;

  const rows = (allMetricsData || [])
    .map((row) => {
      const percentile = clampPercentile(row.global ?? row.league);
      return percentile === null
        ? null
        : {
            metricKey: row.metricKey,
            label: simplifyMetricLabel(row.metric),
            raw: row.raw,
            percentile,
            context: row.global != null ? "global database" : "league",
          };
    })
    .filter(Boolean)
    .filter((row) => isRelevantEvidenceMetric(row.metricKey || row.label, report.player))
    .filter((row) => row.percentile >= 70)
    .sort((a, b) => b.percentile - a.percentile)
    .slice(0, 5);

  const fallbackRows = (allMetricsData || [])
    .map((row) => {
      const percentile = clampPercentile(row.global ?? row.league);
      return percentile === null
        ? null
        : {
            metricKey: row.metricKey,
            label: simplifyMetricLabel(row.metric),
            raw: row.raw,
            percentile,
            context: row.global != null ? "global database" : "league",
          };
    })
    .filter(Boolean)
    .filter((row) => isRelevantEvidenceMetric(row.metricKey || row.label, report.player))
    .sort((a, b) => b.percentile - a.percentile)
    .slice(0, 5);

  const proofPoints = rows.length ? rows : fallbackRows;
  const topLabel = proofPoints[0]?.label || roleName;
  const headline = `${report.player?.name || "Player"} profiles as ${roleBadge}, led by ${topLabel}.`;
  const reasons = proofPoints.slice(0, 3).map((point) => ({
    title: `${topPercentLabel(point.percentile)} for ${point.label}`,
    body: `He ranks ahead of ${Math.round(point.percentile)}% of comparable players in the ${point.context}.`,
  }));

  return { headline, roleBadge, proofPoints, reasons };
};

const wrapCanvasText = (ctx, text, x, y, maxWidth, lineHeight, maxLines = 4) => {
  const words = String(text || "").split(/\s+/).filter(Boolean);
  const lines = [];
  let line = "";
  words.forEach((word) => {
    const testLine = line ? `${line} ${word}` : word;
    if (ctx.measureText(testLine).width > maxWidth && line) {
      lines.push(line);
      line = word;
    } else {
      line = testLine;
    }
  });
  if (line) lines.push(line);
  lines.slice(0, maxLines).forEach((item, index) => {
    const output = index === maxLines - 1 && lines.length > maxLines ? `${item.replace(/\s+\S+$/, "")}...` : item;
    ctx.fillText(output, x, y + index * lineHeight);
  });
  return y + Math.min(lines.length, maxLines) * lineHeight;
};

const drawRoundedRect = (ctx, x, y, width, height, radius) => {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
};

const drawCoverImage = (ctx, image, x, y, width, height) => {
  const imageRatio = image.width / image.height;
  const targetRatio = width / height;
  let sourceWidth = image.width;
  let sourceHeight = image.height;
  let sourceX = 0;
  let sourceY = 0;
  if (imageRatio > targetRatio) {
    sourceWidth = image.height * targetRatio;
    sourceX = (image.width - sourceWidth) / 2;
  } else {
    sourceHeight = image.width / targetRatio;
    sourceY = (image.height - sourceHeight) / 2;
  }
  ctx.drawImage(image, sourceX, sourceY, sourceWidth, sourceHeight, x, y, width, height);
};

const drawContainImage = (ctx, image, x, y, width, height) => {
  const scale = Math.min(width / image.width, height / image.height);
  const drawWidth = image.width * scale;
  const drawHeight = image.height * scale;
  ctx.drawImage(image, x + (width - drawWidth) / 2, y + (height - drawHeight) / 2, drawWidth, drawHeight);
};

const percentileCanvasTone = (percentile) => {
  const value = clampPercentile(percentile);
  if (value === null) {
    return { fill: "#475569", track: "#1e293b", text: "#94a3b8", label: "N/A" };
  }
  if (value <= 30) return { fill: "#ef4444", track: "#fee2e2", text: "#b91c1c", label: "Low" };
  if (value <= 60) return { fill: "#fbbf24", track: "#fef3c7", text: "#b45309", label: "Medium" };
  if (value <= 80) return { fill: "#a3e635", track: "#ecfccb", text: "#4d7c0f", label: "Good" };
  return { fill: "#2dd4bf", track: "#ccfbf1", text: "#0f766e", label: "Excellent" };
};

const drawPercentileCanvasBar = (ctx, x, y, width, height, percentile) => {
  const value = clampPercentile(percentile) ?? 0;
  const tone = percentileCanvasTone(value);
  ctx.fillStyle = tone.track;
  drawRoundedRect(ctx, x, y, width, height, height / 2);
  ctx.fill();
  ctx.fillStyle = tone.fill;
  drawRoundedRect(ctx, x, y, (value / 100) * width, height, height / 2);
  ctx.fill();
  ctx.fillStyle = "rgba(15, 23, 42, 0.35)";
  for (let marker = 10; marker < 100; marker += 10) {
    const markerX = x + (marker / 100) * width;
    ctx.fillRect(markerX, y, 2, height);
  }
};

const loadCanvasImage = (url) =>
  new Promise((resolve) => {
    if (!url) {
      resolve(null);
      return;
    }
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => resolve(image);
    image.onerror = () => resolve(null);
    image.src = url;
  });

const buildDirectorReportCanvas = async ({ report, tmPhotoUrl, clubLogoUrl, directorInsights }) => {
  const canvas = document.createElement("canvas");
  canvas.width = 1400;
  canvas.height = 1900;
  const ctx = canvas.getContext("2d");
  const player = report.player || {};
  const proofPoints = directorInsights.proofPoints || [];
  const photo = await loadCanvasImage(tmPhotoUrl);
  const clubLogo = await loadCanvasImage(clubLogoUrl);

  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, canvas.width, 420);

  ctx.fillStyle = "#14b8a6";
  ctx.font = "700 30px Arial";
  ctx.fillText("NEXTLEGEND SCOUTING REPORT", 80, 92);
  ctx.fillStyle = "#ffffff";
  ctx.font = "900 76px Arial";
  wrapCanvasText(ctx, player.name || "Player", 80, 185, 820, 82, 2);

  ctx.fillStyle = "#cbd5e1";
  ctx.font = "500 30px Arial";
  const subtitle = [player.team, player.competition_name, player.calendar].filter(Boolean).join(" - ");
  if (clubLogo) {
    ctx.save();
    drawRoundedRect(ctx, 80, 282, 56, 56, 12);
    ctx.clip();
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(80, 282, 56, 56);
    drawContainImage(ctx, clubLogo, 88, 290, 40, 40);
    ctx.restore();
    ctx.fillText(subtitle || "Season context", 154, 320);
  } else {
    ctx.fillText(subtitle || "Season context", 80, 320);
  }
  ctx.fillStyle = "#99f6e4";
  ctx.font = "900 34px Arial";
  ctx.fillText(directorInsights.roleBadge || player.assigned_role || "Profile", 80, 372);

  if (photo) {
    ctx.save();
    drawRoundedRect(ctx, 1035, 55, 260, 285, 24);
    ctx.clip();
    drawCoverImage(ctx, photo, 1035, 55, 260, 285);
    ctx.restore();
  } else {
    ctx.fillStyle = "#1e293b";
    drawRoundedRect(ctx, 1035, 55, 260, 285, 24);
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.font = "900 72px Arial";
    ctx.textAlign = "center";
    ctx.fillText(getInitials(player.name), 1165, 215);
    ctx.textAlign = "left";
  }

  const productionStat = getProductionStat(report.raw_metrics || report.metrics || {});
  const kpis = [
    ["League score", player.assigned_role_pct_league == null ? "N/A" : Number(player.assigned_role_pct_league).toFixed(0)],
    ["Adjusted score", player.global_score_adjusted == null ? "N/A" : Number(player.global_score_adjusted).toFixed(1)],
    ["Minutes", player.minutes_played == null ? "N/A" : Number(player.minutes_played).toFixed(0)],
    ...(productionStat ? [[productionStat.label, productionStat.value]] : []),
  ];
  const kpiGap = 24;
  const kpiWidth = (1240 - kpiGap * (kpis.length - 1)) / kpis.length;
  kpis.forEach((item, index) => {
    const x = 80 + index * (kpiWidth + kpiGap);
    ctx.fillStyle = "#ffffff";
    drawRoundedRect(ctx, x, 480, kpiWidth, 140, 18);
    ctx.fill();
    ctx.fillStyle = "#64748b";
    ctx.font = "800 22px Arial";
    ctx.fillText(item[0].toUpperCase(), x + 24, 528);
    ctx.fillStyle = "#0f172a";
    ctx.font = "900 48px Arial";
    ctx.fillText(item[1], x + 24, 590);
  });

  ctx.fillStyle = "#0f172a";
  ctx.font = "900 44px Arial";
  ctx.fillText("Executive read", 80, 720);
  ctx.fillStyle = "#334155";
  ctx.font = "500 30px Arial";
  wrapCanvasText(ctx, directorInsights.headline, 80, 775, 1220, 42, 3);

  ctx.fillStyle = "#0f172a";
  ctx.font = "900 40px Arial";
  ctx.fillText("Why he stands out", 80, 955);

  proofPoints.slice(0, 5).forEach((point, index) => {
    const y = 1010 + index * 135;
    const tone = percentileCanvasTone(point.percentile);
    ctx.fillStyle = "#ffffff";
    drawRoundedRect(ctx, 80, y, 1240, 104, 18);
    ctx.fill();
    ctx.fillStyle = tone.text;
    ctx.font = "900 28px Arial";
    ctx.fillText(`${topPercentLabel(point.percentile)} ${point.label}`, 110, y + 40);
    ctx.fillStyle = "#64748b";
    ctx.font = "600 22px Arial";
    ctx.fillText(`Value: ${formatEvidenceValue(point.raw)} - better than ${Math.round(point.percentile)}% (${point.context})`, 110, y + 75);
    ctx.fillStyle = tone.text;
    ctx.font = "900 26px Arial";
    ctx.textAlign = "right";
    ctx.fillText(String(Math.round(clampPercentile(point.percentile) ?? 0)), 1290, y + 28);
    ctx.textAlign = "left";
    drawPercentileCanvasBar(ctx, 590, y + 42, 700, 18, point.percentile);
    ctx.fillStyle = "#64748b";
    ctx.font = "800 16px Arial";
    ctx.fillText(tone.label.toUpperCase(), 590, y + 82);
  });

  ctx.fillStyle = "#0f172a";
  ctx.font = "900 40px Arial";
  ctx.fillText("Simple translation", 80, 1720);
  ctx.fillStyle = "#334155";
  ctx.font = "500 26px Arial";
  wrapCanvasText(
    ctx,
    "Percentiles compare the player with similar professional players. Top 5% means only 5 players out of 100 perform better on that indicator.",
    80,
    1770,
    1220,
    36,
    3
  );

  ctx.fillStyle = "#64748b";
  ctx.font = "700 22px Arial";
  ctx.fillText("Generated by Next Legend - HD Sports", 80, 1855);
  return canvas;
};

const downloadCanvas = (canvas, filename) => {
  const link = document.createElement("a");
  link.href = canvas.toDataURL("image/png");
  link.download = filename;
  link.click();
};

const buildDirectorReportHtml = ({ report, tmPhotoUrl, clubLogoUrl, directorInsights }) => {
  const player = report.player || {};
  const proof = (directorInsights.proofPoints || []).slice(0, 5);
  const productionStat = getProductionStat(report.raw_metrics || report.metrics || {});
  const image = tmPhotoUrl
    ? `<img src="${tmPhotoUrl}" alt="" class="photo" />`
    : `<div class="photo initials">${getInitials(player.name)}</div>`;
  const clubLogo = clubLogoUrl ? `<img src="${clubLogoUrl}" alt="" class="club-logo" />` : "";
  return `<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>${player.name || "Player"} report</title>
<style>
  @page { size: A4 portrait; margin: 10mm; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #f8fafc; color: #0f172a; font-family: Arial, sans-serif; }
  .sheet { width: 190mm; min-height: 277mm; margin: 0 auto; background: #f8fafc; }
  .hero { background: #0f172a; color: white; border-radius: 10px; padding: 28px; display: grid; grid-template-columns: 1fr 120px; gap: 20px; }
  .kicker { color: #5eead4; font-size: 11px; font-weight: 900; letter-spacing: .18em; text-transform: uppercase; }
  h1 { margin: 18px 0 8px; font-size: 42px; line-height: 1; }
  .sub { color: #cbd5e1; font-size: 15px; margin: 0 0 16px; }
  .badge { display: inline-block; background: #134e4a; color: #ccfbf1; border-radius: 999px; padding: 8px 12px; font-weight: 900; }
  .photo { width: 120px; height: 120px; border-radius: 10px; object-fit: cover; background: #1e293b; display: flex; align-items: center; justify-content: center; color: white; font-size: 30px; font-weight: 900; }
  .club-line { display: flex; align-items: center; gap: 10px; margin: 8px 0 12px; }
  .club-logo { width: 34px; height: 34px; object-fit: contain; border-radius: 8px; background: white; border: 1px solid #e2e8f0; padding: 4px; }
  .kpis { display: grid; grid-template-columns: repeat(${productionStat ? 4 : 3}, 1fr); gap: 10px; margin: 14px 0; }
  .kpi, .card { background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; }
  .kpi span { display: block; color: #64748b; font-size: 10px; font-weight: 900; letter-spacing: .12em; text-transform: uppercase; }
  .kpi strong { display: block; margin-top: 8px; font-size: 26px; }
  h2 { font-size: 24px; margin: 22px 0 10px; }
  .lead { font-size: 18px; line-height: 1.45; color: #334155; }
  .proof { display: grid; gap: 9px; }
  .proof-title { display: flex; justify-content: space-between; gap: 14px; font-weight: 900; color: #0f766e; }
  .proof p { margin: 6px 0 0; color: #475569; font-size: 13px; }
  .bar { height: 8px; background: #e2e8f0; border-radius: 99px; overflow: hidden; margin-top: 10px; }
  .fill { height: 100%; background: #0f766e; border-radius: 99px; }
  .note { color: #64748b; font-size: 12px; line-height: 1.45; }
</style>
</head>
<body>
  <main class="sheet">
    <section class="hero">
      <div>
        <div class="kicker">Next Legend scouting report</div>
        <h1>${player.name || "Player"}</h1>
        <div class="club-line">${clubLogo}<p class="sub">${[player.team, player.competition_name, player.calendar].filter(Boolean).join(" - ")}</p></div>
        <span class="badge">${directorInsights.roleBadge}</span>
      </div>
      ${image}
    </section>
    <section class="kpis">
      <div class="kpi"><span>League score</span><strong>${player.assigned_role_pct_league == null ? "N/A" : Number(player.assigned_role_pct_league).toFixed(0)}</strong></div>
      <div class="kpi"><span>Adjusted score</span><strong>${player.global_score_adjusted == null ? "N/A" : Number(player.global_score_adjusted).toFixed(1)}</strong></div>
      <div class="kpi"><span>Minutes</span><strong>${player.minutes_played == null ? "N/A" : Number(player.minutes_played).toFixed(0)}</strong></div>
      ${productionStat ? `<div class="kpi"><span>${productionStat.label}</span><strong>${productionStat.value}</strong></div>` : ""}
    </section>
    <h2>Executive read</h2>
    <p class="lead">${directorInsights.headline}</p>
    <h2>Why he stands out</h2>
    <section class="proof">
      ${proof
        .map(
          (point) => `<article class="card">
            <div class="proof-title"><span>${topPercentLabel(point.percentile)} ${point.label}</span><span>${Math.round(point.percentile)}%</span></div>
            <p>Value: ${formatEvidenceValue(point.raw)} - ahead of ${Math.round(point.percentile)}% of comparable players in the ${point.context}.</p>
            <div class="bar"><div class="fill" style="width:${Math.round(point.percentile)}%"></div></div>
          </article>`
        )
        .join("")}
    </section>
    <h2>Simple translation</h2>
    <p class="note">Percentiles compare the player with similar professional players. Top 5% means only 5 players out of 100 perform better on that indicator.</p>
  </main>
  <script>window.onload = () => setTimeout(() => window.print(), 250);</script>
</body>
</html>`;
};

export default function ReportPage() {
  const router = useRouter();
  const hydratedQuery = useRef(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [playerQuery, setPlayerQuery] = useState("");
  const [playerResults, setPlayerResults] = useState([]);
  const [selectedPlayerId, setSelectedPlayerId] = useState("");
  const [selectedPlayerSeasonId, setSelectedPlayerSeasonId] = useState("");
  const [showResults, setShowResults] = useState(false);
  const [report, setReport] = useState(null);
  const [similarities, setSimilarities] = useState([]);
  const [similarLoading, setSimilarLoading] = useState(false);
  const [sortConfig, setSortConfig] = useState({ key: "league", dir: "desc" });
  const [showAllMetrics, setShowAllMetrics] = useState(false);
  const [similarSort, setSimilarSort] = useState({ key: "similarity", dir: "desc" });
  const [similarPage, setSimilarPage] = useState(0);
  const [similarHasNext, setSimilarHasNext] = useState(false);
  const [similarFilters, setSimilarFilters] = useState({
    ageMin: "",
    ageMax: "",
    big5Only: false,
    currentSeasonOnly: false,
  });
  const [radarContext, setRadarContext] = useState("global");
  const [isProspect, setIsProspect] = useState(false);
  const [prospectBusy, setProspectBusy] = useState(false);
  const [clubNeeds, setClubNeeds] = useState([]);
  const [clubNeedsLoading, setClubNeedsLoading] = useState(false);
  const [assignNeedId, setAssignNeedId] = useState("");
  const [assignMessage, setAssignMessage] = useState("");
  const [assignBusy, setAssignBusy] = useState(false);
  const [mercatoRequests, setMercatoRequests] = useState([]);
  const [mercatoNeedId, setMercatoNeedId] = useState("");
  const [mercatoNote, setMercatoNote] = useState("");
  const [mercatoMessage, setMercatoMessage] = useState("");
  const [mercatoLoading, setMercatoLoading] = useState(false);
  const [mercatoBusy, setMercatoBusy] = useState(false);
  const [exportBusy, setExportBusy] = useState(false);
  const [clubLogoUrl, setClubLogoUrl] = useState("");

  const similarLimit = 10;

  useEffect(() => {
    let active = true;
    const team = report?.player?.team;
    if (!team) {
      setClubLogoUrl("");
      return () => {
        active = false;
      };
    }
    loadClubLogoData().then((data) => {
      if (active) setClubLogoUrl(resolveClubLogoUrl(team, data));
    });
    return () => {
      active = false;
    };
  }, [report?.player?.team]);

  useEffect(() => {
    if (!playerQuery || playerQuery.trim().length < 2) {
      setPlayerResults([]);
      return;
    }
    const handle = setTimeout(async () => {
      try {
        const res = await fetchJson("/players", { q: playerQuery.trim() });
        const unique = new Map();
        (res || []).forEach((item) => {
          const normalize = (value) =>
            String(value || "")
              .trim()
              .toLowerCase();
          const key = [
            normalize(item.name),
            normalize(item.team),
            normalize(item.competition_name),
            normalize(item.calendar),
          ].join("|");
          if (!unique.has(key)) {
            unique.set(key, item);
          }
        });
        setPlayerResults(Array.from(unique.values()));
      } catch (err) {
        console.error(err);
      }
    }, 200);
    return () => clearTimeout(handle);
  }, [playerQuery]);

  useEffect(() => {
    if (!selectedPlayerId) {
      setReport(null);
      setSimilarities([]);
      setSimilarPage(0);
      setSimilarHasNext(false);
      setSimilarLoading(false);
      return;
    }
    const loadReport = async () => {
      setLoading(true);
      setError("");
      try {
        const data = await fetchJson(`/players/${selectedPlayerId}/report`, {
          player_season_id: selectedPlayerSeasonId || undefined,
        });
        setReport(data);
        if (data?.player) {
          const label = `${data.player.name} - ${data.player.team || "—"} - ${data.player.competition_name || "—"} - ${data.player.calendar || "—"}`;
          setPlayerQuery(label);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    loadReport();
  }, [selectedPlayerId, selectedPlayerSeasonId]);

  useEffect(() => {
    if (!selectedPlayerId) {
      setIsProspect(false);
      return;
    }
    fetchJson(`/prospects/${selectedPlayerId}`)
      .then((res) => setIsProspect(Boolean(res?.is_prospect)))
      .catch(() => setIsProspect(false));
  }, [selectedPlayerId]);

  useEffect(() => {
    if (!isProspect) {
      setClubNeeds([]);
      setAssignNeedId("");
      return;
    }
    const loadNeeds = async () => {
      setClubNeedsLoading(true);
      try {
        const res = await fetchJson("/prospect/club-needs");
        setClubNeeds(res?.needs || []);
      } catch (err) {
        console.error(err);
      } finally {
        setClubNeedsLoading(false);
      }
    };
    loadNeeds();
  }, [isProspect]);

  useEffect(() => {
    if (!selectedPlayerId) {
      setMercatoRequests([]);
      setMercatoNeedId("");
      return;
    }
    const loadMercato = async () => {
      setMercatoLoading(true);
      try {
        const res = await fetchJson("/mercato/requests");
        setMercatoRequests(res?.items || []);
      } catch (err) {
        console.error(err);
      } finally {
        setMercatoLoading(false);
      }
    };
    loadMercato();
  }, [selectedPlayerId]);

  useEffect(() => {
    if (!selectedPlayerId) {
      return;
    }
    if (!report?.similarities_enabled) {
      setSimilarities([]);
      setSimilarHasNext(false);
      setSimilarLoading(false);
      return;
    }
    const loadSimilarities = async () => {
      setSimilarLoading(true);
      setError("");
      try {
        const sims = await fetchJson(
          `/players/${selectedPlayerId}/similarities`,
          {
            player_season_id: selectedPlayerSeasonId || undefined,
            limit: similarLimit,
            offset: similarPage * similarLimit,
            age_min: similarFilters.ageMin || undefined,
            age_max: similarFilters.ageMax || undefined,
            big5_only: similarFilters.big5Only ? "true" : undefined,
            current_season_only: similarFilters.currentSeasonOnly ? "true" : undefined,
          }
        );
        setSimilarities(sims);
        setSimilarHasNext(sims.length === similarLimit);
      } catch (err) {
        setError(err.message);
      } finally {
        setSimilarLoading(false);
      }
    };
    loadSimilarities();
  }, [
    selectedPlayerId,
    selectedPlayerSeasonId,
    similarPage,
    similarFilters,
    report?.similarities_enabled,
  ]);

  const playerOptions = useMemo(() => {
    return playerResults.map((p) => ({
      id: String(p.id),
      seasonId: p.player_season_id ? String(p.player_season_id) : "",
      label: `${p.name} - ${p.team || "—"} - ${p.competition_name || "—"} - ${p.calendar || "—"}`,
    }));
  }, [playerResults]);

  const handlePlayerSelect = (player) => {
    setSelectedPlayerId(player.id);
    setSelectedPlayerSeasonId(player.seasonId || "");
    setPlayerQuery(player.label);
    setShowResults(false);
    setSimilarPage(0);
  };

  const handleProspectToggle = async () => {
    if (!selectedPlayerId || prospectBusy) return;
    setProspectBusy(true);
    try {
      if (isProspect) {
        await deleteJson(`/prospects/${selectedPlayerId}`);
        setIsProspect(false);
      } else {
        await postJson("/prospects", { player_id: Number(selectedPlayerId) });
        setIsProspect(true);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setProspectBusy(false);
    }
  };

  const handleAssignNeed = async () => {
    if (!assignNeedId || !selectedPlayerId || assignBusy) return;
    setAssignBusy(true);
    setAssignMessage("");
    try {
      const res = await postJson(`/prospect/club-needs/${assignNeedId}/players`, {
        player_id: Number(selectedPlayerId),
      });
      setAssignMessage(res?.added ? "Player assigned to need." : "Player already in that need.");
    } catch (err) {
      console.error(err);
      setAssignMessage("Unable to assign player.");
    } finally {
      setAssignBusy(false);
    }
  };

  const handleAssignMercato = async () => {
    if (!mercatoNeedId || !selectedPlayerId || mercatoBusy) return;
    setMercatoBusy(true);
    setMercatoMessage("");
    try {
      const res = await postJson(`/mercato/needs/${mercatoNeedId}/candidates`, {
        player_id: Number(selectedPlayerId),
        source: "report",
        status: "suggested",
        agent_note: mercatoNote || null,
      });
      setMercatoMessage(res?.added ? "Player assigned to Mercato need." : "Player already assigned to that Mercato need.");
      setMercatoNote("");
    } catch (err) {
      console.error(err);
      setMercatoMessage("Unable to assign player to Mercato.");
    } finally {
      setMercatoBusy(false);
    }
  };

  useEffect(() => {
    if (!router.isReady || hydratedQuery.current) return;
    const queryId = router.query.player_id || router.query.playerId;
    const querySeasonId = router.query.player_season_id || router.query.playerSeasonId;
    if (!queryId) return;
    const playerId = String(queryId);
    const seasonId = querySeasonId ? String(querySeasonId) : "";
    hydratedQuery.current = true;
    setSelectedPlayerId(playerId);
    setSelectedPlayerSeasonId(seasonId);
    fetchJson(`/players/${playerId}`)
      .then((data) => {
        if (!data) return;
        const label = `${data.name} - ${data.team || "—"} - ${data.competition_name || "—"} - ${data.calendar || "—"}`;
        setPlayerQuery(label);
      })
      .catch(() => {});
  }, [router.isReady, router.query, setSelectedPlayerId]);

  const metrics = report?.metrics || {};
  const rawMetrics = report?.raw_metrics || metrics;
  const tmFields = report?.tm_fields || {};
  const tmProfileUrl = toAbsoluteUrl(tmFields.tm_profile_url);
  const tmAgentUrl = toAbsoluteUrl(tmFields.tm_agent_url);
  const tmPhotoUrl = toAbsoluteUrl(tmFields.app_photo_url || tmFields.tm_profile_image_url || tmFields.profile_image_url);
  const socialRaw =
    tmFields.tm_social_media ||
    tmFields.tm_socials ||
    tmFields.tm_social_links ||
    tmFields.tm_social;
  const socialLinks = useMemo(() => {
    const urls = extractUrls(socialRaw);
    const unique = [];
    const seen = new Set();
    urls.forEach((url) => {
      const clean = toAbsoluteUrl(url);
      if (!clean) return;
      if (seen.has(clean)) return;
      seen.add(clean);
      const type = resolveSocialType(clean);
      unique.push({
        url: clean,
        type,
        label: type === "x" ? "X" : type.charAt(0).toUpperCase() + type.slice(1),
      });
    });
    return unique;
  }, [socialRaw]);
  const tmDetails = [
    { label: "Market value", value: formatCompactNumber(tmFields.tm_market_value) },
    { label: "Contract expires", value: tmFields.tm_club_contract_expires },
    { label: "Birth date", value: tmFields.tm_birth_date },
    { label: "Birth city", value: tmFields.tm_birth_city },
    { label: "Birth country", value: tmFields.tm_birth_country },
    { label: "Citizenship", value: tmFields.tm_citizenship },
    { label: "Foot", value: tmFields.tm_foot },
    { label: "Outfitter", value: tmFields.tm_outfitter },
  ];
  const hasTmData = Object.values(tmFields).some(
    (value) => value !== null && value !== undefined && String(value).trim() !== ""
  );
  const radarMetricKeys = useMemo(() => {
    if (Array.isArray(report?.radar_metrics) && report.radar_metrics.length > 0) {
      return report.radar_metrics;
    }
    return DEFAULT_RADAR_METRICS;
  }, [report]);
  const radarStats = useMemo(
    () =>
      radarMetricKeys.map((key) => ({
        key,
        label: formatMetricLabel(key),
      })),
    [radarMetricKeys]
  );
  const radarData = useMemo(() => {
    return radarStats.map((stat) => {
      const leagueKey = `${stat.key}_pct_league`;
      const globalKey = `${stat.key}_pct_global`;
      const leagueValue = metrics[leagueKey];
      const globalValue = metrics[globalKey];
      const leagueDisplay = leagueValue ?? globalValue ?? 0;
      const globalDisplay = globalValue ?? leagueValue ?? 0;
      return {
        metric: stat.label,
        value:
          radarContext === "league"
            ? Number(leagueDisplay) || 0
            : Number(globalDisplay) || 0,
        league: metrics[leagueKey],
        global: metrics[globalKey],
        leagueDisplay: Number(leagueDisplay) || 0,
        globalDisplay: Number(globalDisplay) || 0,
        raw: rawMetrics[stat.key],
        contextLabel: radarContext === "league" ? "League percentile" : "Global percentile",
      };
    });
  }, [metrics, rawMetrics, radarContext, radarStats]);

  const sortRows = (rows) => {
    const sorted = [...rows];
    const key = sortConfig.key;
    const dir = sortConfig.dir;
    const valueFor = (row) => {
      if (key === "metric") return row.metric;
      if (key === "raw") return row.raw;
      if (key === "league") return row.league;
      if (key === "global") return row.global;
      return row.metric;
    };
    sorted.sort((a, b) => {
      const left = valueFor(a);
      const right = valueFor(b);
      if (left == null && right == null) return 0;
      if (left == null) return 1;
      if (right == null) return -1;
      if (key === "metric") {
        const order = String(left).localeCompare(String(right), undefined, { sensitivity: "base" });
        return dir === "desc" ? -order : order;
      }
      const leftNum = Number(left);
      const rightNum = Number(right);
      if (Number.isFinite(leftNum) && Number.isFinite(rightNum)) {
        const order = leftNum - rightNum;
        return dir === "desc" ? -order : order;
      }
      const order = String(left).localeCompare(String(right), undefined, { sensitivity: "base" });
      return dir === "desc" ? -order : order;
    });
    return sorted;
  };

  const tableData = useMemo(() => {
    return sortRows(radarData);
  }, [radarData, sortConfig]);

  const roleProfileKeys = useMemo(() => {
    const profiles = report?.role_scores || [];
    const keys = new Set();
    profiles.forEach((item) => {
      const normalized = normalizeMetricKey(item.profile);
      if (normalized) {
        keys.add(normalized);
      }
    });
    return keys;
  }, [report]);

  const allMetricsData = useMemo(() => {
    const keys = Object.keys(metrics || {}).filter(
      (key) =>
        !key.endsWith("_pct_league") &&
        !key.endsWith("_pct_global") &&
        !EXCLUDED_METRIC_PREFIXES.some((prefix) => key.startsWith(prefix)) &&
        !key.startsWith("tm_") &&
        !key.toLowerCase().includes("profile") &&
        !key.includes(" - ") &&
        !roleProfileKeys.has(normalizeMetricKey(key))
    );
    const rows = keys
      .map((key) => {
        const raw = metrics[key];
        const league = metrics[`${key}_pct_league`];
        const global = metrics[`${key}_pct_global`];
        const hasValue =
          raw !== null && raw !== undefined
          || league !== null && league !== undefined
          || global !== null && global !== undefined;
        if (!hasValue) return null;
        return {
          metric: formatMetricLabel(key),
          metricKey: key,
          raw,
          league,
          global,
        };
      })
      .filter(Boolean);
    return sortRows(rows);
  }, [metrics, sortConfig, roleProfileKeys]);

  const handleSort = (key) => {
    setSortConfig((prev) => {
      if (prev.key === key) {
        return { key, dir: prev.dir === "asc" ? "desc" : "asc" };
      }
      return { key, dir: "asc" };
    });
  };

  const sortIndicator = (key) => {
    if (sortConfig.key !== key) return "";
    return sortConfig.dir === "asc" ? "▲" : "▼";
  };

  const handleSimilarSort = (key) => {
    setSimilarSort((prev) => {
      if (prev.key === key) {
        return { key, dir: prev.dir === "asc" ? "desc" : "asc" };
      }
      return { key, dir: "desc" };
    });
  };

  const similarSortIndicator = (key) => {
    if (similarSort.key !== key) return "";
    return similarSort.dir === "asc" ? "▲" : "▼";
  };

  const sortedSimilarities = useMemo(() => {
    const rows = [...similarities];
    const key = similarSort.key;
    const dir = similarSort.dir;
    const valueFor = (row) => {
      if (key === "player") return row.player_b_name || "";
      if (key === "similarity") return row.similarity;
      if (key === "adjusted") return row.global_score_adjusted;
      if (key === "league") return row.assigned_role_pct_league;
      if (key === "global") return row.assigned_role_pct_global;
      if (key === "age") return row.age;
      return row.similarity;
    };
    rows.sort((a, b) => {
      const left = valueFor(a);
      const right = valueFor(b);
      if (left == null && right == null) return 0;
      if (left == null) return 1;
      if (right == null) return -1;
      if (key === "player") {
        const order = String(left).localeCompare(String(right), undefined, { sensitivity: "base" });
        return dir === "desc" ? -order : order;
      }
      const leftNum = Number(left);
      const rightNum = Number(right);
      if (Number.isFinite(leftNum) && Number.isFinite(rightNum)) {
        const order = leftNum - rightNum;
        return dir === "desc" ? -order : order;
      }
      const order = String(left).localeCompare(String(right), undefined, { sensitivity: "base" });
      return dir === "desc" ? -order : order;
    });
    return rows;
  }, [similarities, similarSort]);

  const topRoles = useMemo(() => {
    const roles = Array.isArray(report?.role_scores) ? report.role_scores : [];
    const withPercentiles = roles.filter(
      (role) => role?.pct_league != null || role?.pct_global != null
    );
    if (withPercentiles.length > 0) {
      const withoutPercentiles = roles.filter(
        (role) => role?.pct_league == null && role?.pct_global == null
      );
      return [...withPercentiles, ...withoutPercentiles].slice(0, 3);
    }

    const fallbackProfile = report?.player?.assigned_role;
    const fallbackLeague = report?.player?.assigned_role_pct_league;
    const fallbackGlobal = report?.player?.assigned_role_pct_global;
    if (
      fallbackProfile &&
      (fallbackLeague != null || fallbackGlobal != null)
    ) {
      return [
        {
          profile: fallbackProfile,
          pct_league: fallbackLeague,
          pct_global: fallbackGlobal,
        },
        ...roles,
      ].slice(0, 3);
    }
    return roles.slice(0, 3);
  }, [report]);

  const availableSeasons = useMemo(() => {
    return Array.isArray(report?.available_seasons) ? report.available_seasons : [];
  }, [report]);

  const seasonSelectValue =
    selectedPlayerSeasonId || (report?.player?.player_season_id ? String(report.player.player_season_id) : "");

  const scoreHistoryData = useMemo(() => {
    const rows = Array.isArray(report?.score_history) ? report.score_history : [];
    return rows.map((row) => ({
      ...row,
      global_score_adjusted:
        row?.global_score_adjusted == null ? null : Number(row.global_score_adjusted),
    }));
  }, [report]);

  const directorInsights = useMemo(
    () => buildDirectorInsights({ report, allMetricsData, topRoles }),
    [allMetricsData, report, topRoles]
  );

  const handleDownloadPng = async () => {
    if (!report || exportBusy) return;
    setExportBusy(true);
    try {
      const canvas = await buildDirectorReportCanvas({
        report,
        tmPhotoUrl: toCanvasSafeImageUrl(tmPhotoUrl),
        clubLogoUrl: toCanvasSafeImageUrl(clubLogoUrl),
        directorInsights,
      });
      const filename = `${report.player?.name || "player"}-director-report.png`
        .replace(/[^\w.-]+/g, "_")
        .toLowerCase();
      downloadCanvas(canvas, filename);
    } finally {
      setExportBusy(false);
    }
  };

  const handleExportPdf = () => {
    if (!report || typeof window === "undefined") return;
    const popup = window.open("", "_blank", "width=980,height=1200");
    if (!popup) return;
    popup.document.open();
    popup.document.write(buildDirectorReportHtml({ report, tmPhotoUrl, clubLogoUrl, directorInsights }));
    popup.document.close();
  };

  return (
    <main className="nl-page py-10 px-4">
      <div className="max-w-6xl mx-auto space-y-6">
        <header className="flex flex-col gap-2">
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
            Report
          </p>
          <h1 className="text-4xl font-bold text-white tracking-tight">
            Director-ready player report
          </h1>
          <p className="text-slate-300 max-w-3xl">
            Search a player, review the role evidence and export a clear scouting summary for club decision-makers.
          </p>
        </header>

        <Card className="relative z-30">
          <div className="relative">
            <div className="flex flex-col gap-2">
              <Label htmlFor="report-player-search">Player</Label>
              <input
                id="report-player-search"
                name="player_search"
                aria-label="Player search"
                className="bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100"
                placeholder="Start typing a player name..."
                value={playerQuery}
                onChange={(e) => {
                  setPlayerQuery(e.target.value);
                  setSelectedPlayerId("");
                  setSelectedPlayerSeasonId("");
                  setShowResults(true);
                }}
                onFocus={() => {
                  if (selectedPlayerId) {
                    setPlayerQuery("");
                    setSelectedPlayerId("");
                    setSelectedPlayerSeasonId("");
                    setReport(null);
                    setSimilarities([]);
                  }
                  setShowResults(true);
                }}
                onClick={() => {
                  if (selectedPlayerId) {
                    setPlayerQuery("");
                    setSelectedPlayerId("");
                    setSelectedPlayerSeasonId("");
                    setReport(null);
                    setSimilarities([]);
                    setShowResults(true);
                  }
                }}
                onBlur={() => setTimeout(() => setShowResults(false), 150)}
              />
            </div>
            {showResults && playerQuery.trim().length >= 2 ? (
              <div className="absolute z-50 mt-2 w-full max-h-72 overflow-auto rounded-lg border border-slate-700 bg-slate-900/95 shadow-xl">
                {playerOptions.length === 0 ? (
                  <div className="px-3 py-2 text-sm text-slate-400">
                    No matches found.
                  </div>
                ) : (
                  playerOptions.map((player) => (
                    <button
                      key={`${player.id}-${player.seasonId || "latest"}`}
                      type="button"
                      className="w-full text-left px-3 py-2 text-sm text-slate-200 hover:bg-slate-800/80"
                      onMouseDown={(e) => e.preventDefault()}
                      onClick={() => handlePlayerSelect(player)}
                    >
                      {player.label}
                    </button>
                  ))
                )}
              </div>
            ) : null}
          </div>
        </Card>

        {error && !report && (
          <Card>
            <p className="text-danger">Error: {error}</p>
          </Card>
        )}

        {loading ? (
          <Card>
            <p className="text-slate-400">Loading report…</p>
          </Card>
        ) : report ? (
          <>
            <Card className="overflow-hidden border-emerald-300/20 bg-slate-950/80">
              <div className="grid gap-5 lg:grid-cols-[1.15fr_0.85fr]">
                <div>
                  <p className="text-xs font-black uppercase tracking-[0.24em] text-emerald-300">
                    Director-ready export
                  </p>
                  <h2 className="mt-3 text-2xl font-black text-white">
                    {directorInsights.roleBadge}
                  </h2>
                  <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-300">
                    {directorInsights.headline}
                  </p>
                  <div className="mt-4 grid gap-2 md:grid-cols-3">
                    {directorInsights.reasons.map((reason) => (
                      <div key={reason.title} className="rounded-lg border border-white/10 bg-white/[0.04] p-3">
                        <p className="text-sm font-black text-emerald-200">{reason.title}</p>
                        <p className="mt-2 text-xs leading-5 text-slate-400">{reason.body}</p>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="rounded-lg border border-white/10 bg-white p-4 text-slate-950">
                  <div className="flex items-start gap-3">
                    {tmPhotoUrl ? (
                      <img src={tmPhotoUrl} alt="" className="h-16 w-16 rounded-lg object-cover" />
                    ) : (
                      <div className="flex h-16 w-16 items-center justify-center rounded-lg bg-slate-900 text-lg font-black text-white">
                        {getInitials(report.player.name)}
                      </div>
                    )}
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-lg font-black">{report.player.name}</p>
                      <div className="mt-2 flex items-center gap-2 text-xs font-bold text-slate-500">
                        <ClubLogo name={report.player.team} className="h-6 w-6 rounded" />
                        <span className="min-w-0 truncate">
                          {[report.player.team, report.player.competition_name, report.player.calendar].filter(Boolean).join(" - ")}
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="mt-4 space-y-2">
                    {directorInsights.proofPoints.slice(0, 3).map((point) => (
                      <div key={point.label}>
                        <div className="flex items-center justify-between gap-3 text-xs font-black">
                          <span className="truncate">{topPercentLabel(point.percentile)} {point.label}</span>
                          <span>{Math.round(point.percentile)}%</span>
                        </div>
                        <div className="mt-1 h-2 overflow-hidden rounded-full bg-slate-200">
                          <div
                            className="h-full rounded-full bg-teal-700"
                            style={{ width: `${Math.round(point.percentile)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="mt-5 grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      className="rounded-md bg-slate-950 px-3 py-2 text-sm font-black text-white transition hover:bg-slate-800 disabled:opacity-60"
                      onClick={handleDownloadPng}
                      disabled={exportBusy}
                    >
                      {exportBusy ? "Preparing..." : "Download PNG"}
                    </button>
                    <button
                      type="button"
                      className="rounded-md border border-slate-300 px-3 py-2 text-sm font-black text-slate-950 transition hover:bg-slate-50"
                      onClick={handleExportPdf}
                    >
                      Export PDF
                    </button>
                  </div>
                </div>
              </div>
            </Card>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <Card className="lg:col-span-1 space-y-4">
                <div className="flex items-start gap-3">
                  {tmPhotoUrl ? (
                    <img
                      src={tmPhotoUrl}
                      alt={report.player.name}
                      className="h-24 w-24 rounded-full object-cover border border-white/10"
                    />
                  ) : (
                    <div className="h-24 w-24 rounded-full bg-slate-800 border border-white/10 flex items-center justify-center text-slate-200 font-semibold">
                      {getInitials(report.player.name)}
                    </div>
                  )}
                  <div>
                    <p className="text-sm text-slate-400">Player</p>
                    <h2 className="text-2xl font-semibold text-white flex items-center gap-2">
                      {report.player.name}
                      {isProspect ? (
                        <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-bold text-amber-800" aria-label="Prospect">
                          Prospect
                        </span>
                      ) : null}
                    </h2>
                    <div className="mt-2 flex items-center gap-2 text-slate-400">
                      <ClubLogo name={report.player.team} className="h-8 w-8 rounded" />
                      <span className="min-w-0 truncate">{report.player.team} • {report.player.competition_name}</span>
                    </div>
                    {tmProfileUrl ? (
                      <a
                        href={tmProfileUrl}
                        target="_blank"
                        rel="noreferrer"
                        className="text-xs text-primary hover:text-primary/80"
                      >
                        Transfermarkt profile
                      </a>
                    ) : null}
                    <button
                      type="button"
                      className={`mt-3 inline-flex items-center rounded-full border px-4 py-1 text-xs uppercase tracking-[0.2em] ${
                        isProspect
                          ? "border-yellow-400/70 text-yellow-300"
                          : "border-slate-700 text-slate-200"
                      }`}
                      onClick={handleProspectToggle}
                      disabled={prospectBusy}
                    >
                      {isProspect ? "Remove to Prospect" : "Add to Prospect"}
                    </button>
                    {isProspect ? (
                      <div className="mt-4 space-y-2">
                        <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                          Assign to club need
                        </p>
                        <div className="flex flex-col gap-2">
                          <select
                            id="report-club-need"
                            name="club_need_id"
                            aria-label="Assign to club need"
                            className="w-full bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100 text-sm"
                            value={assignNeedId}
                            onChange={(e) => setAssignNeedId(e.target.value)}
                            disabled={clubNeedsLoading}
                          >
                            <option value="">
                              {clubNeedsLoading ? "Loading needs..." : "Select a need"}
                            </option>
                            {clubNeeds.map((need) => (
                              <option key={need.id} value={need.id}>
                                {need.need_label} • {need.club_name || "Club"} • {need.priority_stage}
                              </option>
                            ))}
                          </select>
                          <div className="flex items-center gap-2">
                            <button
                              type="button"
                              className="text-xs uppercase tracking-[0.2em] px-3 py-2 border border-primary text-primary rounded-full disabled:opacity-60"
                              onClick={handleAssignNeed}
                              disabled={!assignNeedId || assignBusy}
                            >
                              Assign
                            </button>
                          </div>
                        </div>
                        {assignMessage ? (
                          <p className="text-xs text-slate-300">{assignMessage}</p>
                        ) : null}
                      </div>
                    ) : null}
                    <div className="mt-4 space-y-2">
                      <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                        Assign to a Mercato need
                      </p>
                      <div className="flex flex-col gap-2">
                        <select
                          id="report-mercato-need"
                          name="mercato_need_id"
                          aria-label="Assign to a Mercato need"
                          className="w-full bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100 text-sm"
                          value={mercatoNeedId}
                          onChange={(e) => setMercatoNeedId(e.target.value)}
                          disabled={mercatoLoading}
                        >
                          <option value="">
                            {mercatoLoading ? "Loading Mercato needs..." : "Select a Mercato need"}
                          </option>
                          {mercatoRequests.flatMap((requestItem) =>
                            (requestItem.needs || []).map((need) => (
                              <option key={need.id} value={need.id}>
                                {requestItem.club_name || "Club"} • {need.position || "Position"} • {requestItem.priority}
                              </option>
                            ))
                          )}
                        </select>
                        <textarea
                          id="report-mercato-note"
                          name="mercato_note"
                          aria-label="Optional agent note"
                          className="w-full bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100 text-sm"
                          rows={2}
                          value={mercatoNote}
                          onChange={(e) => setMercatoNote(e.target.value)}
                          placeholder="Optional agent note"
                        />
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            className="text-xs uppercase tracking-[0.2em] px-3 py-2 border border-primary text-primary rounded-full disabled:opacity-60"
                            onClick={handleAssignMercato}
                            disabled={!mercatoNeedId || mercatoBusy}
                          >
                            Assign Mercato
                          </button>
                        </div>
                      </div>
                      {mercatoMessage ? (
                        <p className="text-xs text-slate-300">{mercatoMessage}</p>
                      ) : null}
                    </div>
                  </div>
                </div>
                {availableSeasons.length > 0 ? (
                  <div className="space-y-2 min-w-0">
                    <Label htmlFor="report-season">Season</Label>
                    <Select
                      id="report-season"
                      ariaLabel="Report season"
                      value={seasonSelectValue}
                      onChange={(e) => {
                        setSelectedPlayerSeasonId(e.target.value);
                        setSimilarPage(0);
                      }}
                    >
                      {availableSeasons.map((season) => (
                        <option key={season.player_season_id} value={season.player_season_id}>
                          {season.calendar || "—"} • {season.competition_name || "—"} • {season.team || "—"}
                        </option>
                      ))}
                    </Select>
                  </div>
                ) : null}
                <div className="flex flex-wrap gap-2">
                  {report.player.assigned_role && (
                    <Badge>{englishRole(report.player.assigned_role)}</Badge>
                  )}
                  {report.player.position && (
                    <Badge>{report.player.position}</Badge>
                  )}
                  {report.player.age && <Badge>{report.player.age} yrs</Badge>}
                  <Badge>{report.player.minutes_played} mins</Badge>
                </div>
                {hasTmData ? (
                  <div className="space-y-2 border-t border-white/5 pt-3">
                    <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                      Transfermarkt
                    </p>
                    {(tmFields.tm_agent_name || tmAgentUrl) && (
                      <p className="text-sm text-slate-300">
                        Agent:{" "}
                        {tmAgentUrl ? (
                          <a
                            href={tmAgentUrl}
                            target="_blank"
                            rel="noreferrer"
                            className="text-primary hover:text-primary/80"
                          >
                            {tmFields.tm_agent_name || "Profile"}
                          </a>
                        ) : (
                          tmFields.tm_agent_name
                        )}
                      </p>
                    )}
                    {socialLinks.length > 0 ? (
                      <div className="flex flex-wrap gap-2 pt-1">
                        {socialLinks.map((link) => (
                          <a
                            key={link.url}
                            href={link.url}
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-2 rounded-full border border-slate-700/80 bg-slate-900/70 px-3 py-1 text-xs text-slate-200 hover:border-emerald-300/70 hover:text-emerald-200"
                          >
                            <SocialIcon type={link.type} />
                            <span>{link.label}</span>
                          </a>
                        ))}
                      </div>
                    ) : null}
                    <div className="grid grid-cols-1 gap-1 text-sm text-slate-300">
                      {tmDetails
                        .filter(
                          (item) =>
                            item.value !== null &&
                            item.value !== undefined &&
                            String(item.value).trim() !== ""
                        )
                        .map((item) => (
                          <div key={item.label} className="flex items-center justify-between gap-2">
                            <span className="text-slate-400">{item.label}</span>
                            <span className="text-slate-100">{item.value}</span>
                          </div>
                        ))}
                    </div>
                  </div>
                ) : (
                  <p className="text-xs text-slate-500">
                    Transfermarkt data not available yet.
                  </p>
                )}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <p className="text-xs uppercase text-slate-400">
                      Global score adjusted
                    </p>
                    <p className="text-2xl font-bold text-primary">
                      {report.player.global_score_adjusted?.toFixed(1) ?? "—"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs uppercase text-slate-400">
                      Role pct (league / adjusted)
                    </p>
                    <div className="mt-2 space-y-2">
                      <PercentileBar
                        label="League"
                        value={report.player.assigned_role_pct_league}
                        compact
                      />
                      <PercentileBar
                        label="Adjusted"
                        value={report.player.assigned_role_pct_global}
                        compact
                      />
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="lg:col-span-2">
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <h3 className="text-lg font-semibold text-white">
                    Percentile Radar
                  </h3>
                  <div className="flex items-center gap-2">
                    {[
                      { key: "global", label: "Global" },
                      { key: "league", label: "League" },
                    ].map((option) => (
                      <button
                        key={option.key}
                        type="button"
                        onClick={() => setRadarContext(option.key)}
                        className={`px-3 py-1 rounded-md text-xs uppercase tracking-[0.2em] border ${
                          radarContext === option.key
                            ? "border-emerald-400/70 bg-emerald-400/20 text-emerald-200"
                            : "border-slate-700 bg-slate-900/60 text-slate-300"
                        }`}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="h-96">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart
                      key={`radar-${radarContext}`}
                      data={radarData}
                      outerRadius="92%"
                    >
                      <PolarGrid stroke="#334155" strokeDasharray="3 3" />
                      <PolarAngleAxis dataKey="metric" tick={{ fill: "#94a3b8", fontSize: 10 }} />
                      <PolarRadiusAxis
                        angle={90}
                        domain={[0, 100]}
                        ticks={[0, 25, 50, 75, 100]}
                        tick={{ fill: "#94a3b8", fontSize: 8 }}
                        tickLine={{ stroke: "#475569", strokeOpacity: 0.6 }}
                        axisLine={{ stroke: "#475569", strokeOpacity: 0.6 }}
                      />
                      <Tooltip content={<RadarTooltip />} cursor={false} />
                      <Radar
                        name="Percentile"
                        dataKey="value"
                        stroke="#7bd389"
                        fill="rgba(123, 211, 137, 0.25)"
                        fillOpacity={0.35}
                        dot={{ r: 4, stroke: "#7bd389", strokeWidth: 1 }}
                        activeDot={{ r: 6 }}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </div>

            <Card>
              <h3 className="text-lg font-semibold text-white mb-3">
                Role Fit (Top 3)
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {topRoles.map((role, index) => (
                  <div
                    key={role.profile}
                    className={`border rounded-lg p-3 bg-slate-900/50 ${
                      index === 0 ? "border-emerald-400/80" : "border-white/5"
                    }`}
                  >
                    <p className="text-sm text-slate-400">Profile</p>
                    <p className="text-base font-semibold text-white">
                      {englishRole(role.profile)}
                    </p>
                    <div className="mt-3 space-y-2">
                      <PercentileBar label="League" value={role.pct_league} compact />
                      <PercentileBar label="Global" value={role.pct_global} compact />
                      <PercentileBar label="Adjusted" value={role.pct_global_adjusted} compact />
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            <Card>
              <h3 className="text-lg font-semibold text-white mb-3">
                Percentile Overview
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-xs uppercase text-slate-400 border-b border-white/5">
                      <th
                        className="text-left py-2 cursor-pointer select-none"
                        onClick={() => handleSort("metric")}
                      >
                        Metric {sortIndicator("metric")}
                      </th>
                      <th
                        className="text-right py-2 cursor-pointer select-none"
                        onClick={() => handleSort("raw")}
                      >
                        Value {sortIndicator("raw")}
                      </th>
                      <th
                        className="text-left py-2 pl-4 cursor-pointer select-none"
                        onClick={() => handleSort("league")}
                      >
                        League percentile {sortIndicator("league")}
                      </th>
                      <th
                        className="text-left py-2 pl-4 cursor-pointer select-none"
                        onClick={() => handleSort("global")}
                      >
                        Global percentile {sortIndicator("global")}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {tableData.map((row) => (
                      <tr key={row.metric} className="border-b border-white/5">
                        <td className="py-2 text-slate-200">{row.metric}</td>
                        <td className="py-2 text-right text-slate-100">
                          {row.raw != null ? Number(row.raw).toFixed(2) : "—"}
                        </td>
                        <td className="py-2 pl-4">
                          <PercentileBar label={`${row.metric} league percentile`} value={row.league} compact showLabel={false} />
                        </td>
                        <td className="py-2 pl-4">
                          <PercentileBar label={`${row.metric} global percentile`} value={row.global} compact showLabel={false} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>

            <Card>
              <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
                <h3 className="text-lg font-semibold text-white">
                  All Metrics
                </h3>
                <button
                  type="button"
                  className="text-xs uppercase tracking-[0.2em] text-slate-300 hover:text-white"
                  onClick={() => setShowAllMetrics((prev) => !prev)}
                >
                  {showAllMetrics ? "Hide full table" : "Show full table"}
                </button>
              </div>
              {showAllMetrics ? (
                <div className="overflow-x-auto">
                  <div className="max-h-96 overflow-y-auto pr-3">
                    <table className="w-full text-sm">
                      <thead className="sticky top-0 bg-slate-950/95">
                        <tr className="text-xs uppercase text-slate-400 border-b border-white/5">
                          <th
                            className="text-left py-2 cursor-pointer select-none"
                            onClick={() => handleSort("metric")}
                          >
                            Metric {sortIndicator("metric")}
                          </th>
                          <th
                            className="text-right py-2 cursor-pointer select-none"
                            onClick={() => handleSort("raw")}
                          >
                            Value {sortIndicator("raw")}
                          </th>
                          <th
                            className="text-left py-2 pl-4 cursor-pointer select-none"
                            onClick={() => handleSort("league")}
                          >
                            League percentile {sortIndicator("league")}
                          </th>
                          <th
                            className="text-left py-2 pl-4 cursor-pointer select-none"
                            onClick={() => handleSort("global")}
                          >
                            Global percentile {sortIndicator("global")}
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {allMetricsData.length === 0 ? (
                          <tr>
                            <td colSpan={4} className="py-4 text-center text-slate-400">
                              No metrics available.
                            </td>
                          </tr>
                        ) : (
                          allMetricsData.map((row) => (
                            <tr key={row.metricKey} className="border-b border-white/5">
                              <td className="py-2 text-slate-200">{row.metric}</td>
                              <td className="py-2 text-right text-slate-100">
                                {row.raw != null ? Number(row.raw).toFixed(2) : "—"}
                              </td>
                              <td className="py-2 pl-4">
                                <PercentileBar label={`${row.metric} league percentile`} value={row.league} compact showLabel={false} />
                              </td>
                              <td className="py-2 pl-4">
                                <PercentileBar label={`${row.metric} global percentile`} value={row.global} compact showLabel={false} />
                              </td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : (
                <p className="text-slate-400 text-sm">
                  Expand to browse all metrics with percentiles.
                </p>
              )}
            </Card>

            <Card>
              <h3 className="text-lg font-semibold text-white mb-3">
                Similar Players
              </h3>
              {!report?.similarities_enabled ? (
                <p className="text-slate-400">
                  Similarities are available only for the current season ({report?.current_season_label || "2025/2026"}).
                </p>
              ) : (
                <>
                  <div className="flex flex-wrap items-end gap-4 mb-4">
                    <div className="flex flex-col gap-2">
                      <Label htmlFor="similar-age-min">Age min</Label>
                      <input
                        id="similar-age-min"
                        name="similar_age_min"
                        aria-label="Similar players minimum age"
                        type="number"
                        min={0}
                        className="bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100 w-28"
                        value={similarFilters.ageMin}
                        onChange={(e) => {
                          setSimilarFilters((prev) => ({ ...prev, ageMin: e.target.value }));
                          setSimilarPage(0);
                        }}
                      />
                    </div>
                    <div className="flex flex-col gap-2">
                      <Label htmlFor="similar-age-max">Age max</Label>
                      <input
                        id="similar-age-max"
                        name="similar_age_max"
                        aria-label="Similar players maximum age"
                        type="number"
                        min={0}
                        className="bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100 w-28"
                        value={similarFilters.ageMax}
                        onChange={(e) => {
                          setSimilarFilters((prev) => ({ ...prev, ageMax: e.target.value }));
                          setSimilarPage(0);
                        }}
                      />
                    </div>
                    <label className="flex items-center gap-2 text-sm text-slate-300">
                      <input
                        id="similar-big5-only"
                        name="similar_big5_only"
                        aria-label="5 Big Leagues Only"
                        type="checkbox"
                        className="accent-emerald-400"
                        checked={similarFilters.big5Only}
                        onChange={(e) => {
                          setSimilarFilters((prev) => ({ ...prev, big5Only: e.target.checked }));
                          setSimilarPage(0);
                        }}
                      />
                      5 Big Leagues Only
                    </label>
                    <label className="flex items-center gap-2 text-sm text-slate-300">
                      <input
                        id="similar-current-season-only"
                        name="similar_current_season_only"
                        aria-label={`Current season only ${report?.current_season_label || "2025/2026"}`}
                        type="checkbox"
                        className="accent-emerald-400"
                        checked={similarFilters.currentSeasonOnly}
                        onChange={(e) => {
                          setSimilarFilters((prev) => ({ ...prev, currentSeasonOnly: e.target.checked }));
                          setSimilarPage(0);
                        }}
                      />
                      Current season only ({report?.current_season_label || "2025/2026"})
                    </label>
                  </div>
                  {similarLoading ? (
                    <p className="text-slate-400 text-sm mb-2">Loading similar players…</p>
                  ) : null}
                  {similarities.length === 0 ? (
                    <p className="text-slate-400">No similar players found.</p>
                  ) : (
                    <div className="space-y-3">
                      <div className="hidden md:grid grid-cols-[minmax(0,2.3fr)_repeat(5,minmax(0,1fr))] text-xs uppercase text-slate-400 border-b border-white/5 pb-2">
                        <button
                          type="button"
                          className="text-left cursor-pointer select-none"
                          onClick={() => handleSimilarSort("player")}
                        >
                          Player {similarSortIndicator("player")}
                        </button>
                        <button
                          type="button"
                          className="text-right cursor-pointer select-none"
                          onClick={() => handleSimilarSort("age")}
                        >
                          Age {similarSortIndicator("age")}
                        </button>
                        <button
                          type="button"
                          className="text-right cursor-pointer select-none"
                          onClick={() => handleSimilarSort("similarity")}
                        >
                          Similarity {similarSortIndicator("similarity")}
                        </button>
                        <button
                          type="button"
                          className="text-right cursor-pointer select-none"
                          onClick={() => handleSimilarSort("adjusted")}
                        >
                          Adjusted score {similarSortIndicator("adjusted")}
                        </button>
                        <button
                          type="button"
                          className="text-right cursor-pointer select-none"
                          onClick={() => handleSimilarSort("league")}
                        >
                          League pct {similarSortIndicator("league")}
                        </button>
                        <button
                          type="button"
                          className="text-right cursor-pointer select-none"
                          onClick={() => handleSimilarSort("global")}
                        >
                          Global pct {similarSortIndicator("global")}
                        </button>
                      </div>
                      <div className="space-y-3">
                        {sortedSimilarities.map((sim) => {
                          const tmFields = sim.tm_fields || {};
                          const tmPhotoUrl = toAbsoluteUrl(
                            tmFields.tm_profile_image_url || tmFields.profile_image_url
                          );
                          const tmProfileUrl = toAbsoluteUrl(
                            tmFields.tm_profile_url || sim.tm_profile_url
                          );
                          const reportUrl = `/report?player_id=${sim.player_b_id}`;
                          return (
                            <div
                              key={`${sim.player_b_id}-${sim.profile}`}
                              role="button"
                              tabIndex={0}
                              onClick={() => window.open(reportUrl, "_blank", "noopener,noreferrer")}
                              onKeyDown={(event) => {
                                if (event.key === "Enter" || event.key === " ") {
                                  event.preventDefault();
                                  window.open(reportUrl, "_blank", "noopener,noreferrer");
                                }
                              }}
                              className="grid grid-cols-1 md:grid-cols-[minmax(0,2.3fr)_repeat(5,minmax(0,1fr))] gap-3 items-center border border-white/5 rounded-lg p-3 bg-slate-900/50 cursor-pointer focus:outline-none focus:ring-2 focus:ring-emerald-400/60"
                            >
                              <div className="flex items-center gap-3">
                                {tmPhotoUrl ? (
                                  <img
                                    src={tmPhotoUrl}
                                    alt={sim.player_b_name}
                                    className="h-12 w-12 rounded-full object-cover border border-white/10"
                                  />
                                ) : (
                                  <div className="h-12 w-12 rounded-full bg-slate-800 border border-white/10 flex items-center justify-center text-slate-200 font-semibold">
                                    {getInitials(sim.player_b_name)}
                                  </div>
                                )}
                                <div>
                                  <p className="text-base font-semibold text-white">
                                    {sim.player_b_name}
                                    <span className="text-sm font-normal text-slate-300">
                                      {" "}
                                      • {sim.calendar || "—"}
                                    </span>
                                  </p>
                                  <p className="text-sm text-slate-400">
                                    {sim.team || "—"} • {sim.competition_name || "—"}
                                  </p>
                                  {tmProfileUrl ? (
                                    <a
                                      href={tmProfileUrl}
                                      target="_blank"
                                      rel="noreferrer"
                                      className="text-xs text-primary hover:text-primary/80"
                                      onClick={(event) => event.stopPropagation()}
                                    >
                                      Transfermarkt profile
                                    </a>
                                  ) : null}
                                </div>
                              </div>
                              <div className="text-right text-sm text-slate-100">
                                {sim.age != null ? Number(sim.age).toFixed(0) : "—"}
                              </div>
                              <div className="text-right text-sm text-slate-100">
                                {sim.similarity != null
                                  ? `${(Number(sim.similarity) * 100).toFixed(1)}%`
                                  : "—"}
                              </div>
                              <div className="text-right text-sm text-slate-100">
                                {sim.global_score_adjusted != null
                                  ? Number(sim.global_score_adjusted).toFixed(1)
                                  : "—"}
                              </div>
                              <div>
                                <PercentileBar
                                  label="League"
                                  value={sim.assigned_role_pct_league}
                                  compact
                                  showLabel={false}
                                />
                              </div>
                              <div>
                                <PercentileBar
                                  label="Global"
                                  value={sim.assigned_role_pct_global}
                                  compact
                                  showLabel={false}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                      <div className="flex items-center justify-between pt-2">
                        <button
                          type="button"
                          className="px-3 py-2 rounded-md border border-slate-700 bg-slate-900/60 disabled:opacity-50"
                          disabled={similarPage === 0}
                          onClick={() => setSimilarPage((prev) => Math.max(0, prev - 1))}
                        >
                          Prev
                        </button>
                        <span className="text-xs text-slate-400">
                          Page {similarPage + 1}
                        </span>
                        <button
                          type="button"
                          className="px-3 py-2 rounded-md border border-slate-700 bg-slate-900/60 disabled:opacity-50"
                          disabled={!similarHasNext}
                          onClick={() => setSimilarPage((prev) => prev + 1)}
                        >
                          Next
                        </button>
                      </div>
                    </div>
                  )}
                </>
              )}
            </Card>

            <Card>
              <h3 className="text-lg font-semibold text-white mb-3">
                Global Score Evolution
              </h3>
              {scoreHistoryData.length === 0 ? (
                <p className="text-slate-400">No historical score data available.</p>
              ) : (
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={scoreHistoryData} margin={{ top: 12, right: 18, left: 4, bottom: 8 }}>
                      <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
                      <XAxis dataKey="calendar" stroke="#94a3b8" tick={{ fill: "#94a3b8", fontSize: 12 }} />
                      <YAxis
                        stroke="#94a3b8"
                        tick={{ fill: "#94a3b8", fontSize: 12 }}
                        domain={[0, 100]}
                      />
                      <Tooltip
                        contentStyle={{ color: "#000000" }}
                        labelStyle={{ color: "#000000" }}
                        itemStyle={{ color: "#000000" }}
                        formatter={(value) => [
                          value == null ? "—" : Number(value).toFixed(1),
                          "Global score",
                        ]}
                      />
                      <Line
                        type="monotone"
                        dataKey="global_score_adjusted"
                        stroke="#7bd389"
                        strokeWidth={2}
                        dot={{ r: 4, stroke: "#7bd389", fill: "#0f172a", strokeWidth: 1.5 }}
                        activeDot={{ r: 6 }}
                        connectNulls
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </Card>

            <Card>
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Transfers</p>
                  <h3 className="mt-2 text-lg font-semibold text-white">
                    Career movement timeline
                  </h3>
                  <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-400">
                    Review verified club movements with source context before using this report in market conversations.
                  </p>
                </div>
                <span className="rounded-full border border-slate-700 bg-slate-900/70 px-3 py-1 text-xs font-black uppercase tracking-[0.12em] text-slate-300">
                  {(report.transfer_history || []).length} moves
                </span>
              </div>

              {(report.transfer_history || []).length ? (
                <div className="mt-5 space-y-3">
                  {(report.transfer_history || []).map((transfer) => (
                    <div
                      key={`${transfer.id}-${transfer.transfer_date || "date"}-${transfer.team_in_name || "in"}-${transfer.team_out_name || "out"}`}
                      className="rounded-xl border border-white/10 bg-slate-950/60 p-4"
                    >
                      <div className="flex flex-wrap items-start justify-between gap-4">
                        <div className="min-w-0 flex-1">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="rounded-full bg-white px-2.5 py-1 text-[11px] font-black uppercase tracking-[0.08em] text-slate-950">
                              {formatTransferDate(transfer.transfer_date)}
                            </span>
                            <span className="rounded-full border border-emerald-300/30 bg-emerald-400/10 px-2.5 py-1 text-[11px] font-black uppercase tracking-[0.08em] text-emerald-200">
                              {transfer.transfer_type || "Transfer"}
                            </span>
                            {transfer.match_type === "name_club" ? (
                              <span className="rounded-full border border-amber-300/30 bg-amber-400/10 px-2.5 py-1 text-[11px] font-black uppercase tracking-[0.08em] text-amber-200">
                                Name + club match
                              </span>
                            ) : null}
                          </div>
                          <div className="mt-4 grid gap-3 md:grid-cols-[minmax(0,1fr)_44px_minmax(0,1fr)] md:items-center">
                            <div className="flex min-w-0 items-center gap-3 rounded-lg border border-white/10 bg-slate-900/70 p-3">
                              <ClubLogo name={transfer.team_out_name} className="h-10 w-10 rounded" />
                              <div className="min-w-0">
                                <p className="text-[11px] font-black uppercase tracking-[0.12em] text-slate-500">From</p>
                                <p className="truncate text-sm font-extrabold text-white">{transfer.team_out_name || "Free agent"}</p>
                              </div>
                            </div>
                            <div className="hidden h-10 items-center justify-center rounded-full border border-white/10 bg-slate-900/70 text-lg font-black text-slate-400 md:flex">
                              →
                            </div>
                            <div className="flex min-w-0 items-center gap-3 rounded-lg border border-emerald-300/20 bg-emerald-400/10 p-3">
                              <ClubLogo name={transfer.team_in_name} className="h-10 w-10 rounded" />
                              <div className="min-w-0">
                                <p className="text-[11px] font-black uppercase tracking-[0.12em] text-emerald-200">To</p>
                                <p className="truncate text-sm font-extrabold text-white">{transfer.team_in_name || "Free agent"}</p>
                              </div>
                            </div>
                          </div>
                        </div>
                        <div className="min-w-[150px] rounded-lg border border-white/10 bg-slate-900/70 p-3 text-right">
                          <p className="text-[11px] font-black uppercase tracking-[0.12em] text-slate-500">Fee</p>
                          <p className="mt-1 text-lg font-extrabold text-white">{transferFeeLabel(transfer.transfer_fee)}</p>
                          <p className="mt-1 text-xs font-semibold text-slate-400">{transfer.league_name || transfer.team_name_context || "League to confirm"}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="mt-5 rounded-xl border border-dashed border-slate-700 bg-slate-900/50 p-5">
                  <p className="text-sm font-extrabold text-white">No verified transfer history yet.</p>
                  <p className="mt-1 text-sm font-semibold text-slate-400">
                    Import the Wyscout transfer file and confirm player-club mapping to enrich this report with career movements.
                  </p>
                </div>
              )}
            </Card>
          </>
        ) : (
          <Card>
            <p className="text-slate-400">
              Start typing a player name to see matching results.
            </p>
          </Card>
        )}
      </div>
    </main>
  );
}
