import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/router";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { apiUrl, fetchJson, fetchJsonCached, postJson } from "@/lib/api";
import ClubLogo from "@/components/ClubLogo";
import {
  getMetricConfig,
  getPositionMeta,
  getRadarMetricKeys,
  metricGroupOrderForPosition,
  metricGroups,
  normalizePositionGroup,
  POSITION_GROUPS,
} from "@/lib/reportMetrics";
import {
  DEFAULT_SCOUTING_SEASON,
  withDefaultSeason,
} from "@/lib/scoutingFilters";
import {
  AdvancedCharacteristics,
  availableMetricsSummary,
  buildCharacteristics,
  buildProfileCategories,
  CharacteristicsCard,
  PlayerProfileCard,
  PlayerRadarComparison,
  PlayerSearch,
  PlayerSeasonRadarComparison,
  PlayerStatsComparison,
  PositionCard,
  ReportCard,
  SeasonSelector,
  SeasonStatistics,
  SimilarPlayersCard,
  formatValue,
} from "@/components/report/PlayerReportComponents";

const selectedLabel = (item) =>
  [item?.name, item?.team, item?.competition_name, item?.calendar].filter(Boolean).join(" - ");

const TM_BASE_URL = "https://www.transfermarkt.com";

const toAbsoluteUrl = (value) => {
  if (!value) return "";
  const url = String(value).trim();
  if (!url) return "";
  if (url.startsWith("http://") || url.startsWith("https://")) return url;
  if (url.startsWith("/")) return `${TM_BASE_URL}${url}`;
  return url;
};

const getInitials = (value) => {
  const parts = String(value || "").trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "-";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase();
};

const hasValue = (value) => value !== null && value !== undefined && String(value).trim() !== "";

const formatCompactNumber = (value) => {
  if (!hasValue(value)) return "";
  const raw = String(value).trim();
  const numeric = Number(raw.replace(/[^\d.-]/g, ""));
  if (!Number.isFinite(numeric) || numeric <= 0 || !/^[\d\s.,€£$-]+$/.test(raw)) return raw;
  const abs = Math.abs(numeric);
  const fmt = (num, suffix) => `${num >= 10 ? Math.round(num) : Math.round(num * 10) / 10} ${suffix}`;
  if (abs >= 1e9) return fmt(abs / 1e9, "B");
  if (abs >= 1e6) return fmt(abs / 1e6, "M");
  if (abs >= 1e3) return fmt(abs / 1e3, "K");
  return String(Math.round(abs));
};

const extractUrls = (value) => {
  if (!value) return [];
  if (Array.isArray(value)) return value.map((item) => (typeof item === "string" ? item : item?.url)).filter(Boolean);
  if (typeof value === "object") return Object.values(value).map((item) => (typeof item === "string" ? item : item?.url)).filter(Boolean);
  const raw = String(value).trim();
  if (!raw) return [];
  const urls = raw.match(/https?:\/\/[^\s,;]+/gi) || [];
  if (urls.length) return urls;
  return raw.split(/[;,]/g).map((item) => item.trim()).filter((item) => item.startsWith("http"));
};

const resolveSocialType = (url) => {
  const lower = String(url || "").toLowerCase();
  if (lower.includes("instagram.com")) return "instagram";
  if (lower.includes("twitter.com") || lower.includes("x.com")) return "x";
  if (lower.includes("facebook.com")) return "facebook";
  if (lower.includes("tiktok.com")) return "tiktok";
  if (lower.includes("youtube.com") || lower.includes("youtu.be")) return "youtube";
  if (lower.includes("linkedin.com")) return "linkedin";
  return "link";
};

const SocialIcon = ({ type }) => {
  const common = "h-4 w-4";
  if (type === "instagram") {
    return <svg viewBox="0 0 24 24" className={common} aria-hidden="true"><path fill="currentColor" d="M7 2h10a5 5 0 0 1 5 5v10a5 5 0 0 1-5 5H7a5 5 0 0 1-5-5V7a5 5 0 0 1 5-5Zm0 2a3 3 0 0 0-3 3v10a3 3 0 0 0 3 3h10a3 3 0 0 0 3-3V7a3 3 0 0 0-3-3H7Zm5 3.8a4.2 4.2 0 1 1 0 8.4 4.2 4.2 0 0 1 0-8.4Zm0 2a2.2 2.2 0 1 0 0 4.4 2.2 2.2 0 0 0 0-4.4Zm4.7-2.4a1 1 0 1 1 0 2 1 1 0 0 1 0-2Z" /></svg>;
  }
  if (type === "x") {
    return <svg viewBox="0 0 24 24" className={common} aria-hidden="true"><path fill="currentColor" d="M18.2 2h3.1l-6.8 7.8L22.5 22h-6.2l-4.9-7.1L5.3 22H2.2l7.3-8.4L1.8 2h6.4l4.4 6.4L18.2 2Zm-1.1 17.9h1.7L7.2 4H5.4l11.7 15.9Z" /></svg>;
  }
  if (type === "tiktok") {
    return <svg viewBox="0 0 24 24" className={common} aria-hidden="true"><path fill="currentColor" d="M16.6 2c.5 2.3 1.9 3.8 4.3 4.1v3.1a7.4 7.4 0 0 1-4.2-1.3v6.4a6.2 6.2 0 1 1-6.2-6.2c.4 0 .8 0 1.2.1v3.3a2.9 2.9 0 1 0 2.1 2.8V2h2.8Z" /></svg>;
  }
  if (type === "youtube") {
    return <svg viewBox="0 0 24 24" className={common} aria-hidden="true"><path fill="currentColor" d="M23 7.2a3 3 0 0 0-2.1-2.1C19 4.6 12 4.6 12 4.6s-7 0-8.9.5A3 3 0 0 0 1 7.2 31 31 0 0 0 .5 12 31 31 0 0 0 1 16.8a3 3 0 0 0 2.1 2.1c1.9.5 8.9.5 8.9.5s7 0 8.9-.5a3 3 0 0 0 2.1-2.1 31 31 0 0 0 .5-4.8 31 31 0 0 0-.5-4.8ZM9.8 15.5v-7l6 3.5-6 3.5Z" /></svg>;
  }
  if (type === "facebook") {
    return <svg viewBox="0 0 24 24" className={common} aria-hidden="true"><path fill="currentColor" d="M14 8h3V4h-3c-3.3 0-5 2-5 5v2H6v4h3v7h4v-7h3.2l.8-4h-4V9c0-.7.3-1 1-1Z" /></svg>;
  }
  if (type === "linkedin") {
    return <svg viewBox="0 0 24 24" className={common} aria-hidden="true"><path fill="currentColor" d="M4.9 3.4a2.2 2.2 0 1 1 0 4.4 2.2 2.2 0 0 1 0-4.4ZM3 9h3.8v12H3V9Zm7 0h3.6v1.6h.1c.5-.9 1.8-1.9 3.7-1.9 4 0 4.7 2.6 4.7 6V21h-3.8v-5.6c0-1.3 0-3-1.9-3s-2.2 1.4-2.2 2.9V21H10V9Z" /></svg>;
  }
  return <svg viewBox="0 0 24 24" className={common} aria-hidden="true"><path fill="currentColor" d="M10.6 13.4a4 4 0 0 1 0-5.7l2.1-2.1a4 4 0 1 1 5.7 5.7l-1.1 1.1-1.4-1.4 1.1-1.1a2 2 0 1 0-2.8-2.8l-2.1 2.1a2 2 0 1 0 2.8 2.8l.7-.7 1.4 1.4-.7.7a4 4 0 0 1-5.7 0Zm-3.2 3.2a4 4 0 0 1 0-5.7l.7-.7 1.4 1.4-.7.7A2 2 0 1 0 11.6 15l2.1-2.1a2 2 0 0 0-2.8-2.8l-.7.7-1.4-1.4.7-.7a4 4 0 0 1 5.7 5.7l-2.1 2.1a4 4 0 0 1-5.7 0Z" /></svg>;
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
  if (Number.isFinite(numeric) && numeric > 0 && /^[\d\s.,€£$-]+$/.test(clean)) return formatCompactNumber(numeric);
  return clean;
};

const toCanvasSafeImageUrl = (value) => {
  const url = toAbsoluteUrl(value);
  if (!url || url.startsWith("data:") || typeof window === "undefined") return url;
  try {
    const parsed = new URL(url, window.location.href);
    if (parsed.origin === window.location.origin) return url;
    return apiUrl("/image-proxy", { url });
  } catch {
    return "";
  }
};

const loadCanvasImage = (url) =>
  new Promise((resolve) => {
    if (!url) return resolve(null);
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => resolve(image);
    image.onerror = () => resolve(null);
    image.src = url;
  });

const wrapCanvasText = (ctx, text, x, y, maxWidth, lineHeight, maxLines = 4) => {
  const words = String(text || "").split(/\s+/).filter(Boolean);
  const lines = [];
  let line = "";
  words.forEach((word) => {
    const test = line ? `${line} ${word}` : word;
    if (ctx.measureText(test).width > maxWidth && line) {
      lines.push(line);
      line = word;
    } else {
      line = test;
    }
  });
  if (line) lines.push(line);
  lines.slice(0, maxLines).forEach((item, index) => {
    ctx.fillText(index === maxLines - 1 && lines.length > maxLines ? `${item}...` : item, x, y + index * lineHeight);
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

const canvasClamp = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return null;
  return Math.max(0, Math.min(100, numeric));
};

const canvasMetricPct = (metrics, key, context = "global") =>
  metrics?.[`${key}_pct_${context}`] ?? metrics?.[`${key}_pct_global`] ?? metrics?.[`${key}_pct_league`];

const canvasTone = (value) => {
  const score = canvasClamp(value);
  if (score === null) return { color: "#94a3b8", bg: "#f1f5f9", text: "#64748b" };
  if (score >= 95) return { color: "#0891b2", bg: "#cffafe", text: "#155e75" };
  if (score >= 80) return { color: "#0f766e", bg: "#ccfbf1", text: "#115e59" };
  if (score >= 60) return { color: "#22c55e", bg: "#dcfce7", text: "#166534" };
  if (score >= 40) return { color: "#d97706", bg: "#fef3c7", text: "#92400e" };
  if (score >= 20) return { color: "#f97316", bg: "#ffedd5", text: "#9a3412" };
  return { color: "#dc2626", bg: "#fee2e2", text: "#991b1b" };
};

const drawCard = (ctx, x, y, width, height, options = {}) => {
  ctx.fillStyle = options.fill || "#ffffff";
  drawRoundedRect(ctx, x, y, width, height, options.radius || 24);
  ctx.fill();
  ctx.strokeStyle = options.stroke || "#e2e8f0";
  ctx.lineWidth = options.lineWidth || 2;
  ctx.stroke();
};

const drawKicker = (ctx, text, x, y, color = "#0f766e") => {
  ctx.fillStyle = color;
  ctx.font = "900 22px Arial";
  ctx.fillText(String(text || "").toUpperCase(), x, y);
};

const drawPill = (ctx, text, x, y, color = "#0f766e", bg = "#ccfbf1") => {
  ctx.font = "900 18px Arial";
  const width = Math.min(360, Math.max(94, ctx.measureText(String(text || "")).width + 32));
  ctx.fillStyle = bg;
  drawRoundedRect(ctx, x, y, width, 38, 19);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.fillText(String(text || "").toUpperCase(), x + 16, y + 25);
  return width;
};

const drawBar = (ctx, label, value, raw, x, y, width) => {
  const score = canvasClamp(value);
  const tone = canvasTone(score);
  ctx.fillStyle = "#0f172a";
  ctx.font = "900 18px Arial";
  ctx.fillText(label, x, y);
  ctx.fillStyle = "#64748b";
  ctx.font = "700 15px Arial";
  ctx.textAlign = "right";
  ctx.fillText(raw ? `${raw} raw` : score === null ? "-" : `${Math.round(score)} pct`, x + width, y);
  ctx.textAlign = "left";
  ctx.fillStyle = tone.bg;
  drawRoundedRect(ctx, x, y + 16, width, 10, 5);
  ctx.fill();
  if (score !== null) {
    ctx.fillStyle = tone.color;
    drawRoundedRect(ctx, x, y + 16, Math.max(10, (width * score) / 100), 10, 5);
    ctx.fill();
  }
  return y + 46;
};

const drawKeyValue = (ctx, label, value, x, y, width) => {
  ctx.fillStyle = "#64748b";
  ctx.font = "900 17px Arial";
  ctx.fillText(String(label || "").toUpperCase(), x, y);
  ctx.fillStyle = "#0f172a";
  ctx.font = "900 25px Arial";
  wrapCanvasText(ctx, value || "-", x, y + 34, width, 28, 2);
};

const drawCoverImage = (ctx, image, x, y, width, height, radius, fallbackText) => {
  ctx.save();
  drawRoundedRect(ctx, x, y, width, height, radius);
  ctx.clip();
  if (image) {
    const scale = Math.max(width / image.width, height / image.height);
    const sw = width / scale;
    const sh = height / scale;
    const sx = (image.width - sw) / 2;
    const sy = (image.height - sh) / 2;
    ctx.drawImage(image, sx, sy, sw, sh, x, y, width, height);
  } else {
    const gradient = ctx.createLinearGradient(x, y, x + width, y + height);
    gradient.addColorStop(0, "#ccfbf1");
    gradient.addColorStop(1, "#f8fafc");
    ctx.fillStyle = gradient;
    ctx.fillRect(x, y, width, height);
    ctx.fillStyle = "#0f766e";
    ctx.font = "900 74px Arial";
    ctx.textAlign = "center";
    ctx.fillText(getInitials(fallbackText), x + width / 2, y + height / 2 + 24);
    ctx.textAlign = "left";
  }
  ctx.restore();
  ctx.strokeStyle = "#99f6e4";
  ctx.lineWidth = 4;
  drawRoundedRect(ctx, x, y, width, height, radius);
  ctx.stroke();
};

const drawMiniPitch = (ctx, x, y, width, height, meta) => {
  drawCard(ctx, x, y, width, height, { fill: "#f0fdfa", stroke: "#99f6e4", radius: 24 });
  ctx.strokeStyle = "#14b8a6";
  ctx.lineWidth = 2;
  ctx.strokeRect(x + 24, y + 24, width - 48, height - 48);
  ctx.beginPath();
  ctx.arc(x + width / 2, y + height / 2, 52, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x + 24, y + height / 2);
  ctx.lineTo(x + width - 24, y + height / 2);
  ctx.stroke();
  ctx.strokeRect(x + width / 2 - 62, y + 24, 124, 54);
  ctx.strokeRect(x + width / 2 - 62, y + height - 78, 124, 54);
  const px = x + (width * (meta?.pitch?.x ?? 50)) / 100;
  const py = y + (height * (meta?.pitch?.y ?? 50)) / 100;
  ctx.fillStyle = "#0f766e";
  ctx.beginPath();
  ctx.arc(px, py, 14, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 5;
  ctx.stroke();
  ctx.fillStyle = "#0f172a";
  ctx.font = "900 42px Arial";
  ctx.textAlign = "center";
  ctx.fillText(meta?.short || "POS", x + width / 2, y + height - 22);
  ctx.textAlign = "left";
};

const drawRadar = (ctx, rows, x, y, size, title, subtitle) => {
  drawCard(ctx, x, y, size, size + 92, { fill: "#ffffff", stroke: "#e2e8f0", radius: 24 });
  drawKicker(ctx, title, x + 28, y + 38);
  ctx.fillStyle = "#64748b";
  ctx.font = "700 18px Arial";
  ctx.fillText(subtitle, x + 28, y + 68);
  const centerX = x + size / 2;
  const centerY = y + 92 + size / 2;
  const radius = size * 0.34;
  const filtered = rows.slice(0, 6);
  if (filtered.length < 3) {
    ctx.fillStyle = "#64748b";
    ctx.font = "700 22px Arial";
    ctx.fillText("Not enough percentile data for radar.", x + 30, centerY);
    return;
  }
  [0.25, 0.5, 0.75, 1].forEach((ring) => {
    ctx.beginPath();
    filtered.forEach((_, index) => {
      const angle = -Math.PI / 2 + (index * Math.PI * 2) / filtered.length;
      const px = centerX + Math.cos(angle) * radius * ring;
      const py = centerY + Math.sin(angle) * radius * ring;
      if (index === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.closePath();
    ctx.strokeStyle = "#dbeafe";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
  filtered.forEach((row, index) => {
    const angle = -Math.PI / 2 + (index * Math.PI * 2) / filtered.length;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(centerX + Math.cos(angle) * radius, centerY + Math.sin(angle) * radius);
    ctx.strokeStyle = "#e2e8f0";
    ctx.stroke();
    ctx.fillStyle = "#334155";
    ctx.font = "800 15px Arial";
    ctx.textAlign = Math.cos(angle) > 0.2 ? "left" : Math.cos(angle) < -0.2 ? "right" : "center";
    ctx.fillText(row.shortLabel || row.label, centerX + Math.cos(angle) * (radius + 24), centerY + Math.sin(angle) * (radius + 22));
  });
  ctx.beginPath();
  filtered.forEach((_, index) => {
    const angle = -Math.PI / 2 + (index * Math.PI * 2) / filtered.length;
    const px = centerX + Math.cos(angle) * radius * 0.5;
    const py = centerY + Math.sin(angle) * radius * 0.5;
    if (index === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.closePath();
  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.beginPath();
  filtered.forEach((row, index) => {
    const angle = -Math.PI / 2 + (index * Math.PI * 2) / filtered.length;
    const percentile = canvasClamp(row.percentile) ?? 0;
    const px = centerX + Math.cos(angle) * radius * (percentile / 100);
    const py = centerY + Math.sin(angle) * radius * (percentile / 100);
    if (index === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.closePath();
  ctx.fillStyle = "rgba(15,118,110,0.18)";
  ctx.fill();
  ctx.strokeStyle = "#0f766e";
  ctx.lineWidth = 4;
  ctx.stroke();
  ctx.textAlign = "left";
};

const buildPngVerdict = (player, generatedAt) => {
  const minutes = Number(player?.minutes_played);
  const sample = Number.isFinite(minutes) && minutes > 0
    ? `${formatValue(minutes, "integer")} minutes observed`
    : "Minutes sample unavailable";
  return `Observed statistical profile generated on ${generatedAt} from Wyscout season data. ${sample}. Position-based percentiles should be confirmed through video scouting before any recruitment decision.`;
};

const drawSeasonEvolution = (ctx, rows, x, y, width, height) => {
  drawCard(ctx, x, y, width, height, { fill: "#ffffff", stroke: "#e2e8f0", radius: 24 });
  drawKicker(ctx, "Season evolution", x + 32, y + 42);

  const seasons = [...(rows || [])]
    .filter((row) => hasValue(row?.calendar) || hasValue(row?.global_score_adjusted) || hasValue(row?.minutes_played))
    .sort((a, b) => seasonSortValue(a.calendar) - seasonSortValue(b.calendar))
    .slice(-5);

  ctx.fillStyle = "#64748b";
  ctx.font = "700 18px Arial";
  ctx.fillText("Latest available seasons with minutes and model rating trend.", x + 32, y + 74);

  if (!seasons.length) {
    ctx.fillStyle = "#f8fafc";
    drawRoundedRect(ctx, x + 32, y + 108, width - 64, 72, 18);
    ctx.fill();
    ctx.fillStyle = "#475569";
    ctx.font = "800 20px Arial";
    ctx.fillText("No season evolution available for this player.", x + 54, y + 153);
    return;
  }

  const chartX = x + 42;
  const chartY = y + 106;
  const chartW = width - 84;
  const chartH = 128;
  const scores = seasons.map((row) => Number(row.global_score_adjusted)).filter(Number.isFinite);
  const minScore = Math.max(0, Math.floor((Math.min(...scores, 50) - 5) / 5) * 5);
  const maxScore = Math.min(100, Math.ceil((Math.max(...scores, 100) + 5) / 5) * 5);
  const span = Math.max(10, maxScore - minScore);

  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1.5;
  [0, 0.5, 1].forEach((ratio) => {
    const gy = chartY + chartH * ratio;
    ctx.beginPath();
    ctx.moveTo(chartX, gy);
    ctx.lineTo(chartX + chartW, gy);
    ctx.stroke();
  });

  const points = seasons.map((row, index) => {
    const score = Number(row.global_score_adjusted);
    const px = chartX + (seasons.length === 1 ? chartW / 2 : (chartW * index) / (seasons.length - 1));
    const py = Number.isFinite(score) ? chartY + chartH - ((score - minScore) / span) * chartH : null;
    return { row, score, x: px, y: py };
  });

  let lineStarted = false;
  ctx.beginPath();
  points.forEach((point) => {
    if (point.y === null) return;
    if (!lineStarted) {
      ctx.moveTo(point.x, point.y);
      lineStarted = true;
    } else {
      ctx.lineTo(point.x, point.y);
    }
  });
  ctx.strokeStyle = "#0f766e";
  ctx.lineWidth = 5;
  ctx.stroke();

  points.forEach((point) => {
    if (point.y !== null) {
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(point.x, point.y, 10, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#0f766e";
      ctx.lineWidth = 5;
      ctx.stroke();
    }

    ctx.textAlign = "center";
    ctx.fillStyle = "#64748b";
    ctx.font = "800 14px Arial";
    ctx.fillText(point.row.calendar || "-", point.x, chartY + chartH + 28);
    ctx.fillStyle = "#0f172a";
    ctx.font = "900 18px Arial";
    ctx.fillText(Number.isFinite(point.score) ? formatValue(point.score, "score") : "-", point.x, chartY + chartH + 54);
  });
  ctx.textAlign = "left";
};

const drawReportPng = async ({
  report,
  photoUrl,
  tmDetails,
  characteristics,
  profileCategoriesData,
  similarities,
  scoreHistory,
  context,
  referenceGroup,
  full,
}) => {
  const canvas = document.createElement("canvas");
  canvas.width = 1600;
  canvas.height = full ? 2480 : 2440;
  const ctx = canvas.getContext("2d");
  const player = report.player || {};
  const metrics = report.metrics || {};
  const positionMeta = getPositionMeta(player.assigned_role, player.position);
  const positionGroup = referenceGroup || normalizePositionGroup(player.assigned_role, player.position);
  const sampleSize = report?.average_contexts?.[context]?.[positionGroup]?.sample_size;
  const photo = await loadCanvasImage(toCanvasSafeImageUrl(photoUrl));
  const generatedAt = new Intl.DateTimeFormat("en", { day: "2-digit", month: "short", year: "numeric" }).format(new Date());
  const radarKeys = getRadarMetricKeys(positionGroup, report?.radar_metrics || [])
    .map((key) => {
      const cfg = getMetricConfig(key);
      return {
        key,
        label: cfg.label,
        shortLabel: cfg.label.replace("Progressive", "Prog.").replace("Successful", "Succ.").replace("Accuracy", "Acc."),
        raw: metrics[key],
        percentile: canvasMetricPct(metrics, key, context),
        format: cfg.format,
      };
    })
    .filter((row) => canvasClamp(row.percentile) !== null);
  const keyRows = radarKeys.slice(0, full ? 8 : 6);
  const topStrengths = (characteristics.strengths || []).slice(0, full ? 8 : 5);
  const topWeaknesses = (characteristics.weaknesses || []).slice(0, full ? 5 : 3);
  const sortedSimilarities = [...(similarities || [])].sort((a, b) => Number(b.global_score_adjusted || -1) - Number(a.global_score_adjusted || -1)).slice(0, full ? 5 : 3);
  const transfers = [...(report.transfer_history || [])]
    .sort((a, b) => new Date(b.transfer_date || 0).getTime() - new Date(a.transfer_date || 0).getTime())
    .slice(0, full ? 6 : 3);

  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const hero = ctx.createLinearGradient(0, 0, canvas.width, 440);
  hero.addColorStop(0, "#0f172a");
  hero.addColorStop(0.55, "#134e4a");
  hero.addColorStop(1, "#ecfeff");
  ctx.fillStyle = hero;
  ctx.fillRect(0, 0, canvas.width, 440);
  ctx.fillStyle = "rgba(255,255,255,0.08)";
  ctx.beginPath();
  ctx.arc(1310, 35, 360, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#99f6e4";
  ctx.font = "900 25px Arial";
  ctx.fillText(full ? "COMPLETE PLAYER REPORT" : "SCOUT DECISION SNAPSHOT", 70, 72);
  ctx.fillStyle = "#ffffff";
  ctx.font = "900 76px Arial";
  wrapCanvasText(ctx, player.name || "Player", 70, 165, 880, 80, 2);
  ctx.fillStyle = "#d1fae5";
  ctx.font = "800 27px Arial";
  ctx.fillText([positionMeta.short, player.team, player.competition_name, player.calendar].filter(Boolean).join("  |  "), 72, 282);
  ctx.fillStyle = "#e2e8f0";
  ctx.font = "700 22px Arial";
  wrapCanvasText(ctx, buildPngVerdict(player, generatedAt), 72, 330, 950, 31, 3);

  drawCoverImage(ctx, photo, 1160, 54, 300, 340, 32, player.name);

  let y = 492;
  const kpis = [
    ["Season", player.calendar || "-"],
    ["Club", player.team || "-"],
    ["Competition", player.competition_name || "-"],
    ["Minutes", formatValue(player.minutes_played, "integer")],
    ["Matches", formatValue(player.matches_played, "integer")],
    ["Goals", formatValue(metrics.goals, "integer")],
    ["Assists", formatValue(metrics.assists, "integer")],
    ["xG", formatValue(metrics.xg, "number")],
    ["xA", formatValue(metrics.xa, "number")],
    ["Prog. passes /90", formatValue(metrics.progressive_passes_per_90, "number")],
    ["Prog. runs /90", formatValue(metrics.progressive_runs_per_90, "number")],
    ["Def. actions /90", formatValue(metrics.successful_def_actions_per_90, "number")],
  ];
  kpis.forEach(([label, value], index) => {
    const col = index % 4;
    const row = Math.floor(index / 4);
    const x = 70 + col * 365;
    const rowY = y + row * 112;
    drawCard(ctx, x, rowY, 330, 86, { fill: "#ffffff", stroke: "#e2e8f0", radius: 20 });
    ctx.fillStyle = "#64748b";
    ctx.font = "900 17px Arial";
    ctx.fillText(label.toUpperCase(), x + 22, rowY + 30);
    ctx.fillStyle = "#0f172a";
    ctx.font = "900 28px Arial";
    wrapCanvasText(ctx, String(value || "-"), x + 22, rowY + 66, 280, 28, 1);
  });

  y += 360;
  drawCard(ctx, 70, y, 460, 360, { fill: "#ffffff", stroke: "#e2e8f0", radius: 24 });
  drawKicker(ctx, "Position", 98, y + 42);
  ctx.fillStyle = "#0f172a";
  ctx.font = "900 34px Arial";
  ctx.fillText(positionMeta.label, 98, y + 86);
  drawMiniPitch(ctx, 142, y + 112, 314, 210, positionMeta);

  drawCard(ctx, 570, y, 470, 360, { fill: "#ffffff", stroke: "#e2e8f0", radius: 24 });
  drawKicker(ctx, "Performance dimensions", 598, y + 42);
  const dimensionRows = profileCategoriesData.slice(0, 6);
  const dimensionStep = Math.min(46, Math.floor(276 / Math.max(dimensionRows.length, 1)));
  let barY = y + 92;
  dimensionRows.forEach((category) => {
    drawBar(ctx, category.label, category.score, null, 598, barY, 392);
    barY += dimensionStep;
  });

  drawCard(ctx, 1080, y, 450, 360, { fill: "#ffffff", stroke: "#e2e8f0", radius: 24 });
  drawKicker(ctx, "Scout proof", 1108, y + 42);
  ctx.fillStyle = "#0f172a";
  ctx.font = "900 28px Arial";
  ctx.fillText(`${topStrengths.length} key strengths`, 1108, y + 82);
  let proofY = y + 122;
  topStrengths.slice(0, 4).forEach((row) => {
    const tone = canvasTone(row.percentile);
    ctx.fillStyle = tone.bg;
    drawRoundedRect(ctx, 1108, proofY, 394, 48, 18);
    ctx.fill();
    ctx.fillStyle = tone.text;
    ctx.font = "900 18px Arial";
    ctx.fillText(`${Math.round(row.percentile)} pct`, 1126, proofY + 31);
    ctx.fillStyle = "#0f172a";
    ctx.font = "800 18px Arial";
    wrapCanvasText(ctx, row.label, 1210, proofY + 31, 278, 20, 1);
    proofY += 58;
  });

  y += 408;
  drawRadar(ctx, radarKeys, 70, y, 650, "Visual benchmark", `${player.name || "Player"} vs average ${POSITION_GROUPS?.[positionGroup]?.label || positionMeta.label} (${context}, n=${sampleSize || "-"})`);
  drawCard(ctx, 760, y, 770, 742, { fill: "#ffffff", stroke: "#e2e8f0", radius: 24 });
  drawKicker(ctx, "Key statistical evidence", 792, y + 42);
  ctx.fillStyle = "#64748b";
  ctx.font = "700 19px Arial";
  ctx.fillText("Percentiles are position-based. Raw values are shown when available.", 792, y + 76);
  ctx.font = "900 15px Arial";
  ctx.fillStyle = "#0f766e";
  ctx.fillText("METRIC", 814, y + 122);
  ctx.textAlign = "right";
  ctx.fillText("RAW VALUE", 1328, y + 122);
  ctx.fillText("PERCENTILE", 1470, y + 122);
  ctx.textAlign = "left";
  let tableY = y + 146;
  keyRows.forEach((row) => {
    const tone = canvasTone(row.percentile);
    ctx.fillStyle = "#f8fafc";
    drawRoundedRect(ctx, 792, tableY, 706, 54, 16);
    ctx.fill();
    ctx.fillStyle = "#0f172a";
    ctx.font = "900 20px Arial";
    ctx.fillText(row.label, 814, tableY + 34);
    ctx.fillStyle = "#64748b";
    ctx.font = "800 18px Arial";
    ctx.textAlign = "right";
    ctx.fillText(formatValue(row.raw, row.format), 1328, tableY + 34);
    ctx.fillStyle = tone.color;
    ctx.font = "900 22px Arial";
    ctx.fillText(`${Math.round(canvasClamp(row.percentile))}`, 1470, tableY + 34);
    ctx.textAlign = "left";
    tableY += 66;
  });
  if (!keyRows.length) {
    ctx.fillStyle = "#64748b";
    ctx.font = "700 22px Arial";
    ctx.fillText("No available radar metric percentiles for this player.", 792, tableY);
  }

  y += 790;
  drawSeasonEvolution(ctx, scoreHistory, 70, y, 710, 320);

  drawCard(ctx, 820, y, 710, 320, { fill: "#ffffff", stroke: "#e2e8f0", radius: 24 });
  drawKicker(ctx, "Market context", 852, y + 42);
  const contextItems = [
    ...tmDetails.slice(0, 5).map((item) => [item.label, item.value || "Profile"]),
    ["Generated", generatedAt],
  ].slice(0, 6);
  contextItems.forEach(([label, value], index) => {
    const x = 852 + (index % 2) * 330;
    const itemY = y + 88 + Math.floor(index / 2) * 72;
    drawKeyValue(ctx, label, value, x, itemY, 275);
  });

  if (full) {
    y += 368;
    drawCard(ctx, 70, y, 710, 355, { fill: "#ffffff", stroke: "#e2e8f0", radius: 24 });
    drawKicker(ctx, "Statistical neighbours", 102, y + 42);
    ctx.fillStyle = "#64748b";
    ctx.font = "700 18px Arial";
    ctx.fillText("Sorted by Next Legend rating among the 10 closest profiles.", 102, y + 74);
    let simY = y + 112;
    sortedSimilarities.forEach((sim, index) => {
      ctx.fillStyle = "#f8fafc";
      drawRoundedRect(ctx, 102, simY, 646, 42, 14);
      ctx.fill();
      ctx.fillStyle = "#0f766e";
      ctx.font = "900 19px Arial";
      ctx.fillText(`#${index + 1}`, 122, simY + 28);
      ctx.fillStyle = "#0f172a";
      ctx.font = "900 19px Arial";
      ctx.fillText(sim.player_b_name || "-", 178, simY + 28);
      ctx.fillStyle = "#64748b";
      ctx.font = "800 17px Arial";
      ctx.textAlign = "right";
      ctx.fillText(formatValue(sim.global_score_adjusted, "score"), 724, simY + 28);
      ctx.textAlign = "left";
      simY += 52;
    });
    if (!sortedSimilarities.length) {
      ctx.fillStyle = "#64748b";
      ctx.font = "700 20px Arial";
      ctx.fillText("No statistical neighbours available.", 102, simY);
    }

    drawCard(ctx, 820, y, 710, 355, { fill: "#ffffff", stroke: "#e2e8f0", radius: 24 });
    drawKicker(ctx, "Transfer history", 852, y + 42);
    let transferY = y + 88;
    (transfers.length ? transfers : [{ team_out_name: "No transfer history available", team_in_name: "", transfer_fee: "", transfer_date: "" }]).forEach((transfer) => {
      ctx.fillStyle = "#f8fafc";
      drawRoundedRect(ctx, 852, transferY, 646, 56, 16);
      ctx.fill();
      ctx.fillStyle = "#0f172a";
      ctx.font = "900 18px Arial";
      wrapCanvasText(ctx, `${transfer.team_out_name || "Free agent"} -> ${transfer.team_in_name || "Free agent"}`, 872, transferY + 25, 440, 20, 1);
      ctx.fillStyle = "#64748b";
      ctx.font = "800 16px Arial";
      ctx.fillText(formatTransferDate(transfer.transfer_date), 872, transferY + 48);
      ctx.textAlign = "right";
      ctx.fillStyle = "#0f766e";
      ctx.font = "900 18px Arial";
      ctx.fillText(transferFeeLabel(transfer.transfer_fee), 1474, transferY + 35);
      ctx.textAlign = "left";
      transferY += 68;
    });
  }

  ctx.fillStyle = "#64748b";
  ctx.font = "700 16px Arial";
  ctx.fillText("Source: player season report. Missing values are not converted to zero.", 70, canvas.height - 44);
  ctx.textAlign = "left";
  return canvas;
};

const downloadCanvas = (canvas, filename) => {
  const link = document.createElement("a");
  link.download = filename;
  link.href = canvas.toDataURL("image/png");
  link.click();
};

const escapeHtml = (value) =>
  String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");

const pdfTone = (value) => {
  const score = canvasClamp(value);
  if (score === null) return { color: "#94a3b8", bg: "rgba(148,163,184,0.12)" };
  if (score >= 80) return { color: "#8CC7A7", bg: "rgba(47,125,92,0.22)" };
  if (score >= 60) return { color: "#86efac", bg: "rgba(34,197,94,0.14)" };
  if (score >= 40) return { color: "#fbbf24", bg: "rgba(245,158,11,0.14)" };
  if (score >= 20) return { color: "#fb923c", bg: "rgba(249,115,22,0.14)" };
  return { color: "#fda4af", bg: "rgba(244,63,94,0.14)" };
};

const pdfBar = (label, value, raw = null) => {
  const score = canvasClamp(value);
  const tone = pdfTone(score);
  const width = score === null ? 0 : score;
  return `
    <div class="bar-row">
      <div class="bar-head">
        <span>${escapeHtml(label)}</span>
        <strong style="color:${tone.color}">${score === null ? "-" : Math.round(score)}</strong>
      </div>
      <div class="bar-track"><div class="bar-fill" style="width:${width}%;background:${tone.color}"></div></div>
      ${raw !== null && raw !== undefined && raw !== "" ? `<div class="raw">${escapeHtml(raw)}</div>` : ""}
    </div>
  `;
};

const buildPdfMetricGroups = (metrics, positionGroup, context) =>
  metricGroupOrderForPosition(positionGroup)
    .map((groupKey) => {
      const group = metricGroups[groupKey];
      const rows = (group?.metrics || [])
        .map((metric) => ({
          ...metric,
          raw: metrics?.[metric.key],
          percentile: canvasMetricPct(metrics, metric.key, context),
        }))
        .filter((metric) => hasValue(metric.raw) || canvasClamp(metric.percentile) !== null);
      return { key: groupKey, label: group?.label || groupKey, rows };
    })
    .filter((group) => group.rows.length);

const buildFullReportPdfHtml = ({
  report,
  photoUrl,
  tmDetails,
  socialLinks,
  characteristics,
  profileCategoriesData,
  similarities,
  comparisonReport,
  scoreHistory,
  scoreSnapshots,
  metricsSummary,
  context,
  referenceGroup,
}) => {
  const player = report?.player || {};
  const metrics = report?.metrics || {};
  const comparison = comparisonReport?.player;
  const comparisonMetrics = comparisonReport?.metrics || {};
  const comparisonLabel = comparison?.name || "cohort average";
  const positionMeta = getPositionMeta(player.assigned_role, player.position);
  const positionGroup = referenceGroup || normalizePositionGroup(player.assigned_role, player.position);
  const radarKeys = getRadarMetricKeys(positionGroup, report?.radar_metrics || [])
    .map((key) => {
      const cfg = getMetricConfig(key);
      const avg = report?.average_contexts?.[context]?.[positionGroup]?.metrics?.[key] || {};
      return {
        key,
        label: cfg.label,
        format: cfg.format,
        raw: metrics[key],
        percentile: canvasMetricPct(metrics, key, context),
        avgRaw: comparison ? comparisonMetrics[key] : avg.raw,
        avgPercentile: comparison ? canvasMetricPct(comparisonMetrics, key, context) : avg.percentile,
      };
    })
    .filter((row) => hasValue(row.raw) || canvasClamp(row.percentile) !== null);
  const metricGroupsForPdf = buildPdfMetricGroups(metrics, positionGroup, context);
  const recentSeasons = (report?.season_metric_history || []).slice(-4);
  const similarRows = [...(similarities || [])]
    .sort((a, b) => Number(b.global_score_adjusted || -1) - Number(a.global_score_adjusted || -1))
    .slice(0, 10);
  const generatedAt = new Intl.DateTimeFormat("en", { day: "2-digit", month: "short", year: "numeric" }).format(new Date());
  const stats = [
    ["Season", player.calendar],
    ["Club", player.team],
    ["Competition", player.competition_name],
    ["Position", `${positionMeta.short} - ${positionMeta.label}`],
    ["Minutes", formatValue(player.minutes_played, "integer")],
    ["Matches", formatValue(player.matches_played, "integer")],
    ["Goals", formatValue(metrics.goals, "integer")],
    ["Assists", formatValue(metrics.assists, "integer")],
    ["xG", formatValue(metrics.xg, "number")],
    ["xA", formatValue(metrics.xa, "number")],
    ["Prog. passes /90", formatValue(metrics.progressive_passes_per_90, "number")],
    ["Prog. runs /90", formatValue(metrics.progressive_runs_per_90, "number")],
  ];

  return `<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>${escapeHtml(player.name || "Player report")}</title>
  <style>
    @page { size: A4; margin: 12mm; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #050706; color: #F3F5F4; font-family: Inter, Arial, sans-serif; font-size: 12px; }
    .page { min-height: 100vh; background: radial-gradient(circle at 18% 0%, rgba(85,154,120,0.18), transparent 28%), #050706; }
    .section { margin-bottom: 14px; border: 1px solid rgba(255,255,255,0.10); border-radius: 10px; background: rgba(255,255,255,0.045); padding: 14px; }
    .page-break { break-before: page; page-break-before: always; }
    .hero { display: grid; grid-template-columns: 1fr 120px; gap: 16px; align-items: start; padding: 18px; border: 1px solid rgba(85,154,120,0.30); border-radius: 12px; background: linear-gradient(135deg, rgba(47,125,92,0.18), rgba(255,255,255,0.035)); }
    .kicker { color: #8CC7A7; font-size: 9px; font-weight: 900; text-transform: uppercase; letter-spacing: .16em; }
    h1 { margin: 8px 0 8px; font-size: 34px; line-height: 1; letter-spacing: -0.03em; }
    h2 { margin: 6px 0 10px; font-size: 18px; }
    h3 { margin: 0 0 8px; font-size: 14px; color: #F3F5F4; }
    p { margin: 0; color: #A0A8A3; line-height: 1.45; }
    .photo { width: 120px; height: 150px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.12); object-fit: cover; background: rgba(255,255,255,0.06); }
    .grid { display: grid; gap: 8px; }
    .grid-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .grid-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .grid-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .tile { min-width: 0; border: 1px solid rgba(255,255,255,0.10); border-radius: 8px; background: rgba(0,0,0,0.20); padding: 9px; }
    .tile small { display: block; margin-bottom: 4px; color: #6F7772; font-size: 8px; font-weight: 900; text-transform: uppercase; letter-spacing: .14em; }
    .tile strong { display: block; overflow-wrap: anywhere; color: #F3F5F4; font-size: 12px; }
    .chip { display: inline-block; margin: 0 4px 4px 0; border: 1px solid rgba(255,255,255,0.10); border-radius: 7px; background: rgba(255,255,255,0.045); padding: 5px 7px; color: #DDE3DF; font-size: 10px; font-weight: 800; }
    .chip-green { border-color: rgba(58,137,103,0.35); background: rgba(47,125,92,0.15); color: #8CC7A7; }
    .bar-row { margin-bottom: 9px; }
    .bar-head { display: flex; justify-content: space-between; gap: 10px; margin-bottom: 4px; color: #DDE3DF; font-size: 11px; font-weight: 800; }
    .bar-track { height: 7px; overflow: hidden; border-radius: 999px; background: rgba(255,255,255,0.08); }
    .bar-fill { height: 100%; border-radius: inherit; }
    .raw { margin-top: 3px; color: #6F7772; font-size: 9px; font-weight: 700; }
    table { width: 100%; border-collapse: collapse; break-inside: avoid; }
    th { color: #8CC7A7; font-size: 8px; text-transform: uppercase; letter-spacing: .12em; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.10); padding: 7px 6px; }
    td { border-bottom: 1px solid rgba(255,255,255,0.075); padding: 7px 6px; color: #DDE3DF; vertical-align: top; }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    .muted { color: #6F7772; }
    .small { font-size: 10px; }
    .strength { border-left: 3px solid #559A78; padding-left: 9px; margin-bottom: 8px; }
    .weakness { border-left: 3px solid #fb7185; padding-left: 9px; margin-bottom: 8px; }
    @media print { body { print-color-adjust: exact; -webkit-print-color-adjust: exact; } }
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div>
        <div class="kicker">Full player report</div>
        <h1>${escapeHtml(player.name || "Player")}</h1>
        <p>${escapeHtml([positionMeta.short, player.team, player.competition_name, player.calendar].filter(Boolean).join(" | "))}</p>
        <div style="margin-top:12px">
          <span class="chip chip-green">${escapeHtml(positionMeta.label)}</span>
          <span class="chip">${escapeHtml(context)} context</span>
          <span class="chip">Generated ${escapeHtml(generatedAt)}</span>
        </div>
      </div>
      ${photoUrl ? `<img class="photo" src="${escapeHtml(photoUrl)}" alt="${escapeHtml(player.name || "Player")}" />` : `<div class="photo" style="display:grid;place-items:center;font-size:28px;font-weight:900">${escapeHtml(getInitials(player.name))}</div>`}
    </section>

    <section class="section">
      <div class="kicker">Statistics</div>
      <h2>Season executive summary</h2>
      <div class="grid grid-4">
        ${stats.map(([label, value]) => `<div class="tile"><small>${escapeHtml(label)}</small><strong>${escapeHtml(value || "-")}</strong></div>`).join("")}
      </div>
    </section>

    <section class="section">
      <div class="kicker">Market and identity</div>
      <h2>Player context</h2>
      <div class="grid grid-3">
        ${(tmDetails.length ? tmDetails : [{ label: "Transfermarkt", value: "No external data available" }]).map((item) => `<div class="tile"><small>${escapeHtml(item.label)}</small><strong>${escapeHtml(item.value || item.url || "-")}</strong></div>`).join("")}
      </div>
      ${socialLinks?.length ? `<div style="margin-top:10px">${socialLinks.map((link) => `<span class="chip">${escapeHtml(link.label)}</span>`).join("")}</div>` : ""}
    </section>

    <section class="section">
      <div class="kicker">Performance dimensions</div>
      <h2>Profile overview</h2>
      <div class="grid grid-2">
        ${profileCategoriesData.map((category) => pdfBar(category.label, category.score)).join("")}
      </div>
    </section>

    <section class="section">
      <div class="kicker">Characteristics</div>
      <h2>Strengths and weaknesses</h2>
      <div class="grid grid-2">
        <div>
          <h3>Strengths</h3>
          ${(characteristics.strengths || []).slice(0, 10).map((row) => `<div class="strength"><strong>${escapeHtml(row.label)}</strong><p class="small">${escapeHtml(formatValue(row.raw, getMetricConfig(row.key).format))} raw | ${escapeHtml(Math.round(row.percentile))} percentile</p></div>`).join("") || `<p>No major strength flagged.</p>`}
        </div>
        <div>
          <h3>Weaknesses</h3>
          ${(characteristics.weaknesses || []).slice(0, 10).map((row) => `<div class="weakness"><strong>${escapeHtml(row.label)}</strong><p class="small">${escapeHtml(formatValue(row.raw, getMetricConfig(row.key).format))} raw | ${escapeHtml(Math.round(row.percentile))} percentile</p></div>`).join("") || `<p>No major weakness flagged.</p>`}
        </div>
      </div>
    </section>

    <section class="section page-break">
      <div class="kicker">Visual comparison</div>
      <h2>${escapeHtml(player.name || "Player")} vs ${escapeHtml(comparisonLabel)}</h2>
      <table>
        <thead><tr><th>Metric</th><th class="num">Player raw</th><th class="num">Player pct</th><th class="num">${escapeHtml(comparison ? "Comparison raw" : "Cohort raw")}</th><th class="num">${escapeHtml(comparison ? "Comparison pct" : "Cohort pct")}</th></tr></thead>
        <tbody>
          ${radarKeys.map((row) => {
            const playerTone = pdfTone(row.percentile);
            const avgTone = pdfTone(row.avgPercentile);
            return `<tr><td>${escapeHtml(row.label)}</td><td class="num">${escapeHtml(formatValue(row.raw, row.format))}</td><td class="num" style="color:${playerTone.color}">${escapeHtml(formatValue(row.percentile, "integer"))}</td><td class="num">${escapeHtml(formatValue(row.avgRaw, row.format))}</td><td class="num" style="color:${avgTone.color}">${escapeHtml(formatValue(row.avgPercentile, "integer"))}</td></tr>`;
          }).join("")}
        </tbody>
      </table>
    </section>

    <section class="section">
      <div class="kicker">Advanced characteristics</div>
      <h2>Detailed metric percentiles</h2>
      ${metricGroupsForPdf.map((group) => `
        <div style="margin-top:12px;break-inside:avoid">
          <h3>${escapeHtml(group.label)}</h3>
          <table>
            <thead><tr><th>Metric</th><th class="num">Raw value</th><th class="num">Percentile</th></tr></thead>
            <tbody>
              ${group.rows.map((row) => {
                const tone = pdfTone(row.percentile);
                return `<tr><td>${escapeHtml(row.label)}</td><td class="num">${escapeHtml(formatValue(row.raw, row.format))}</td><td class="num" style="color:${tone.color}">${escapeHtml(formatValue(row.percentile, "integer"))}</td></tr>`;
              }).join("")}
            </tbody>
          </table>
        </div>
      `).join("")}
    </section>

    <section class="section page-break">
      <div class="kicker">Season radar</div>
      <h2>Latest 4 seasons</h2>
      <table>
        <thead><tr><th>Metric</th>${recentSeasons.map((season) => `<th class="num">${escapeHtml(season.calendar || season.team || "Season")}</th>`).join("")}</tr></thead>
        <tbody>
          ${radarKeys.map((row) => `<tr><td>${escapeHtml(row.label)}</td>${recentSeasons.map((season) => {
            const pct = canvasMetricPct(season.metrics || {}, row.key, context);
            const tone = pdfTone(pct);
            return `<td class="num" style="color:${tone.color}">${escapeHtml(formatValue(pct, "integer"))}</td>`;
          }).join("")}</tr>`).join("")}
        </tbody>
      </table>
    </section>

    <section class="section">
      <div class="kicker">Similar players</div>
      <h2>Statistical neighbours</h2>
      <table>
        <thead><tr><th>#</th><th>Player</th><th>Club</th><th>Competition</th><th class="num">Age</th><th class="num">Score</th></tr></thead>
        <tbody>
          ${similarRows.map((sim, index) => `<tr><td>${index + 1}</td><td>${escapeHtml(sim.player_b_name || "-")}</td><td>${escapeHtml(sim.team || "-")}</td><td>${escapeHtml([sim.competition_name, sim.calendar].filter(Boolean).join(" | ") || "-")}</td><td class="num">${escapeHtml(formatValue(sim.age, "integer"))}</td><td class="num">${escapeHtml(formatValue(sim.global_score_adjusted, "score"))}</td></tr>`).join("") || `<tr><td colspan="6">No similar players found.</td></tr>`}
        </tbody>
      </table>
    </section>

    <section class="section">
      <div class="kicker">Score history</div>
      <h2>Rating evolution</h2>
      <table>
        <thead><tr><th>Season</th><th>Club</th><th>Competition</th><th class="num">Minutes</th><th class="num">Score</th></tr></thead>
        <tbody>
          ${(scoreHistory || []).map((row) => `<tr><td>${escapeHtml(row.calendar || "-")}</td><td>${escapeHtml(row.team || "-")}</td><td>${escapeHtml(row.competition_name || "-")}</td><td class="num">${escapeHtml(formatValue(row.minutes_played, "integer"))}</td><td class="num">${escapeHtml(formatValue(row.global_score_adjusted, "score"))}</td></tr>`).join("") || `<tr><td colspan="5">No score history available.</td></tr>`}
        </tbody>
      </table>
    </section>

    <section class="section">
      <div class="kicker">In-season snapshots</div>
      <h2>Current season score tracking</h2>
      <table>
        <thead><tr><th>Snapshot</th><th>Club</th><th>Competition</th><th class="num">Minutes</th><th class="num">Coverage</th><th class="num">Score</th></tr></thead>
        <tbody>
          ${(scoreSnapshots || []).map((row) => `<tr><td>${escapeHtml(row.snapshot_date || row.snapshot_key || "-")}</td><td>${escapeHtml(row.team || "-")}</td><td>${escapeHtml(row.competition_name || "-")}</td><td class="num">${escapeHtml(formatValue(row.minutes_played, "integer"))}</td><td class="num">${escapeHtml(row.minutes_ratio == null ? "-" : `${Math.round(Number(row.minutes_ratio) * 100)}%`)}</td><td class="num">${escapeHtml(formatValue(row.global_score_adjusted, "score"))}</td></tr>`).join("") || `<tr><td colspan="6">No in-season snapshot available yet.</td></tr>`}
        </tbody>
      </table>
    </section>

    <section class="section">
      <div class="kicker">Data coverage</div>
      <p>Configured report metrics available: ${escapeHtml(metricsSummary.available.length)}. Missing or unavailable for this player/provider: ${escapeHtml(metricsSummary.missing.length)}. Missing data is shown as "-" and is never converted to zero.</p>
    </section>
  </main>
</body>
</html>`;
};

const seasonSortValue = (label) => {
  const text = String(label || "");
  const years = text.match(/20\d{2}/g) || [];
  if (!years.length) return 0;
  return Math.max(...years.map(Number));
};

const sortSeasons = (items = []) =>
  [...items].sort((a, b) => {
    const diff = seasonSortValue(b.calendar) - seasonSortValue(a.calendar);
    if (diff !== 0) return diff;
    return Number(b.minutes_played || 0) - Number(a.minutes_played || 0);
  });

const formatSnapshotLabel = (value, fallback) => {
  if (!value) return fallback || "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return fallback || String(value);
  return new Intl.DateTimeFormat("en", { day: "2-digit", month: "short" }).format(date);
};

function ScoreHistory({ data }) {
  if (!data?.length) return null;
  return (
    <ReportCard>
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="nl-kicker">Score history</p>
          <h3 className="mt-2 text-xl font-black text-white">Rating evolution</h3>
        </div>
      </div>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 12, right: 18, left: 0, bottom: 8 }}>
            <CartesianGrid stroke="rgba(255,255,255,0.10)" strokeDasharray="3 3" />
            <XAxis dataKey="calendar" stroke="#6F7772" tick={{ fill: "#A0A8A3", fontSize: 12 }} />
            <YAxis domain={[50, 100]} stroke="#6F7772" tick={{ fill: "#A0A8A3", fontSize: 12 }} />
            <Tooltip
              contentStyle={{ background: "#080B0A", border: "1px solid rgba(255,255,255,0.10)", borderRadius: 8, color: "#F3F5F4", boxShadow: "0 20px 45px rgba(0,0,0,0.34)" }}
              labelStyle={{ color: "#8CC7A7" }}
              formatter={(value) => [value == null ? "-" : Number(value).toFixed(1), "Rating"]}
            />
            <Line type="monotone" dataKey="global_score_adjusted" stroke="#559A78" strokeWidth={2.5} dot={{ r: 4, fill: "#080B0A", stroke: "#8CC7A7", strokeWidth: 2 }} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </ReportCard>
  );
}

function InSeasonScoreSnapshots({ data }) {
  if (!data?.length) return null;
  const first = data.find((row) => row.global_score_adjusted != null);
  const latest = [...data].reverse().find((row) => row.global_score_adjusted != null);
  const delta = first && latest
    ? Number(latest.global_score_adjusted || 0) - Number(first.global_score_adjusted || 0)
    : null;
  return (
    <ReportCard>
      <div className="mb-4 flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="nl-kicker">In-season snapshots</p>
          <h3 className="mt-2 text-xl font-black text-white">Current season score tracking</h3>
          <p className="mt-1 max-w-xl text-sm text-[#A0A8A3]">
            Biweekly model snapshots with score, minutes coverage and stored scoring metrics.
          </p>
        </div>
        {delta != null ? (
          <span className={`rounded-md border px-3 py-1 text-xs font-black ${delta >= 0 ? "border-[#3A8967]/35 bg-[#2F7D5C]/15 text-[#8CC7A7]" : "border-red-400/25 bg-red-500/10 text-red-200"}`}>
            {delta >= 0 ? "+" : ""}{delta.toFixed(1)} since first snapshot
          </span>
        ) : null}
      </div>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 12, right: 18, left: 0, bottom: 8 }}>
            <CartesianGrid stroke="rgba(255,255,255,0.10)" strokeDasharray="3 3" />
            <XAxis dataKey="label" stroke="#6F7772" tick={{ fill: "#A0A8A3", fontSize: 12 }} />
            <YAxis domain={[50, 100]} stroke="#6F7772" tick={{ fill: "#A0A8A3", fontSize: 12 }} />
            <Tooltip
              contentStyle={{ background: "#080B0A", border: "1px solid rgba(255,255,255,0.10)", borderRadius: 8, color: "#F3F5F4", boxShadow: "0 20px 45px rgba(0,0,0,0.34)" }}
              labelStyle={{ color: "#8CC7A7" }}
              formatter={(value, name) => {
                if (name === "global_score_adjusted") return [value == null ? "-" : Number(value).toFixed(1), "Score"];
                return [value == null ? "-" : `${Math.round(Number(value) * 100)}%`, "Minutes coverage"];
              }}
            />
            <Line type="monotone" dataKey="global_score_adjusted" stroke="#559A78" strokeWidth={2.6} dot={{ r: 4, fill: "#080B0A", stroke: "#8CC7A7", strokeWidth: 2 }} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-4 grid gap-2 sm:grid-cols-3">
        {data.slice(-3).map((row) => (
          <div key={`${row.snapshot_key}-${row.snapshot_date}`} className="rounded-lg border border-white/10 bg-white/[0.025] px-3 py-2">
            <p className="text-[11px] font-black uppercase tracking-[0.12em] text-[#6F7772]">{row.label}</p>
            <p className="mt-1 text-sm font-black text-white">{formatValue(row.global_score_adjusted, "score")} score</p>
            <p className="text-xs text-[#A0A8A3]">{formatValue(row.minutes_played, "integer")} min - {row.minutes_ratio == null ? "-" : `${Math.round(Number(row.minutes_ratio) * 100)}%`} coverage</p>
          </div>
        ))}
      </div>
    </ReportCard>
  );
}

function TransferHistoryCard({ transfers = [] }) {
  const items = [...transfers].sort((a, b) => {
    const aTime = a.transfer_date ? new Date(a.transfer_date).getTime() : 0;
    const bTime = b.transfer_date ? new Date(b.transfer_date).getTime() : 0;
    return bTime - aTime;
  });
  return (
    <ReportCard>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="nl-kicker">Transfers</p>
          <h3 className="mt-2 text-xl font-black text-white">Career movement timeline</h3>
          <p className="mt-1 text-sm text-slate-600">Most recent moves first, with fee and source context when available.</p>
        </div>
        <span className="rounded-md border border-[#3A8967]/30 bg-[#2F7D5C]/15 px-3 py-1 text-xs font-black text-[#8CC7A7]">{items.length} moves</span>
      </div>
      {items.length ? (
        <div className="mt-5 space-y-3">
          {items.map((transfer) => (
            <div key={`${transfer.id}-${transfer.transfer_date || "date"}-${transfer.team_in_name || "in"}-${transfer.team_out_name || "out"}`} className="rounded-lg border border-white/10 bg-white/[0.025] p-4 shadow-sm">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="rounded-full bg-slate-950 px-2.5 py-1 text-[11px] font-black uppercase tracking-[0.08em] text-white">{formatTransferDate(transfer.transfer_date)}</span>
                    <span className="rounded-md border border-[#3A8967]/30 bg-[#2F7D5C]/15 px-2.5 py-1 text-[11px] font-black uppercase tracking-[0.08em] text-[#8CC7A7]">{transfer.transfer_type || "Transfer"}</span>
                  </div>
                  <div className="mt-4 grid gap-3 md:grid-cols-[minmax(0,1fr)_44px_minmax(0,1fr)] md:items-center">
                    <div className="flex min-w-0 items-center gap-3 rounded-lg border border-white/10 bg-white/[0.035] p-3">
                      <ClubLogo name={transfer.team_out_name} className="h-10 w-10 rounded" />
                      <div className="min-w-0">
                        <p className="text-[11px] font-black uppercase tracking-[0.12em] text-slate-400">From</p>
                        <p className="truncate text-sm font-extrabold text-slate-950">{transfer.team_out_name || "Free agent"}</p>
                      </div>
                    </div>
                    <div className="hidden h-10 items-center justify-center rounded-md border border-white/10 bg-white/[0.035] text-lg font-black text-slate-400 md:flex">-&gt;</div>
                    <div className="flex min-w-0 items-center gap-3 rounded-lg border border-[#3A8967]/30 bg-[#2F7D5C]/15 p-3">
                      <ClubLogo name={transfer.team_in_name} className="h-10 w-10 rounded" />
                      <div className="min-w-0">
                        <p className="text-[11px] font-black uppercase tracking-[0.12em] text-[#8CC7A7]">To</p>
                        <p className="truncate text-sm font-extrabold text-slate-950">{transfer.team_in_name || "Free agent"}</p>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="min-w-[170px] rounded-lg border border-white/10 bg-white/[0.035] p-3 text-left lg:text-right">
                  <p className="text-[11px] font-black uppercase tracking-[0.12em] text-slate-400">Fee</p>
                  <p className="mt-1 text-lg font-extrabold text-slate-950">{transferFeeLabel(transfer.transfer_fee)}</p>
                  <p className="mt-1 text-xs font-semibold text-slate-500">{transfer.league_name || transfer.team_name_context || "League to confirm"}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="mt-5 rounded-lg border border-dashed border-white/10 bg-white/[0.025] p-5">
          <p className="text-sm font-extrabold text-white">No verified transfer history yet.</p>
          <p className="mt-1 text-sm font-semibold text-slate-500">Import Transfermarkt movement data to enrich this report.</p>
        </div>
      )}
    </ReportCard>
  );
}

export default function ReportPage() {
  const router = useRouter();
  const hydratedQuery = useRef(false);
  const [playerQuery, setPlayerQuery] = useState("");
  const [playerResults, setPlayerResults] = useState([]);
  const [showPlayerResults, setShowPlayerResults] = useState(false);
  const [searchSeasons, setSearchSeasons] = useState([]);
  const [selectedSearchSeason, setSelectedSearchSeason] = useState(DEFAULT_SCOUTING_SEASON);
  const [selectedPlayerId, setSelectedPlayerId] = useState("");
  const [selectedPlayerSeasonId, setSelectedPlayerSeasonId] = useState("");
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [percentileContext, setPercentileContext] = useState("global");
  const [rawMode, setRawMode] = useState(false);
  const [similarities, setSimilarities] = useState([]);
  const [similarLoading, setSimilarLoading] = useState(false);
  const [compareQuery, setCompareQuery] = useState("");
  const [compareResults, setCompareResults] = useState([]);
  const [showCompareResults, setShowCompareResults] = useState(false);
  const [selectedComparisonLabel, setSelectedComparisonLabel] = useState("");
  const [comparisonReport, setComparisonReport] = useState(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [exportBusy, setExportBusy] = useState(false);
  const [prospectLoading, setProspectLoading] = useState(false);
  const [isProspect, setIsProspect] = useState(false);
  const [prospectMessage, setProspectMessage] = useState("");

  useEffect(() => {
    if (!router.isReady || hydratedQuery.current) return;
    const queryId = router.query.player_id || router.query.playerId;
    const querySeasonId = router.query.player_season_id || router.query.playerSeasonId;
    if (!queryId) return;
    hydratedQuery.current = true;
    setSelectedPlayerId(String(queryId));
    setSelectedPlayerSeasonId(querySeasonId ? String(querySeasonId) : "");
  }, [router.isReady, router.query]);

  useEffect(() => {
    fetchJsonCached("/meta/seasons")
      .then((data) => setSearchSeasons(withDefaultSeason(data || [])))
      .catch(() => setSearchSeasons([]));
  }, []);

  useEffect(() => {
    if (playerQuery.trim().length < 2 || selectedPlayerId) {
      setPlayerResults([]);
      return;
    }
    const handle = setTimeout(async () => {
      try {
        const rows = await fetchJson("/players", {
          q: playerQuery.trim(),
          limit: 12,
          season: selectedSearchSeason || undefined,
        });
        setPlayerResults(rows || []);
      } catch (err) {
        console.error(err);
      }
    }, 180);
    return () => clearTimeout(handle);
  }, [playerQuery, selectedPlayerId, selectedSearchSeason]);

  useEffect(() => {
    if (!selectedPlayerId) {
      setReport(null);
      setSimilarities([]);
      setComparisonReport(null);
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
        setPlayerQuery(selectedLabel(data.player));
        if (!selectedPlayerSeasonId && data?.player?.player_season_id) {
          setSelectedPlayerSeasonId(String(data.player.player_season_id));
        }
      } catch (err) {
        setError(err.message || "Unable to load report.");
      } finally {
        setLoading(false);
      }
    };
    loadReport();
  }, [selectedPlayerId, selectedPlayerSeasonId]);

  useEffect(() => {
    if (!selectedPlayerId || !selectedPlayerSeasonId) return;
    const loadSimilarities = async () => {
      setSimilarLoading(true);
      try {
        const rows = await fetchJson(`/players/${selectedPlayerId}/similarities`, {
          player_season_id: selectedPlayerSeasonId,
          limit: 10,
        });
        setSimilarities(rows || []);
      } catch (err) {
        console.error(err);
        setSimilarities([]);
      } finally {
        setSimilarLoading(false);
      }
    };
    loadSimilarities();
  }, [selectedPlayerId, selectedPlayerSeasonId]);

  useEffect(() => {
    if (!selectedPlayerId) {
      setIsProspect(false);
      setProspectMessage("");
      return;
    }
    let cancelled = false;
    setIsProspect(false);
    setProspectMessage("");
    fetchJson(`/prospects/${selectedPlayerId}`)
      .then((res) => {
        if (!cancelled) {
          setIsProspect(Boolean(res?.is_prospect));
          setProspectMessage("");
        }
      })
      .catch(() => {
        if (!cancelled) setIsProspect(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedPlayerId]);

  useEffect(() => {
    if (compareQuery.trim().length < 2) {
      setCompareResults([]);
      setShowCompareResults(false);
      return;
    }
    if (selectedComparisonLabel && compareQuery === selectedComparisonLabel) {
      setShowCompareResults(false);
      return;
    }
    const handle = setTimeout(async () => {
      try {
        const rows = await fetchJson("/players", { q: compareQuery.trim(), limit: 12, season: report?.player?.calendar || undefined });
        setCompareResults(rows || []);
        setShowCompareResults(true);
      } catch (err) {
        console.error(err);
      }
    }, 180);
    return () => clearTimeout(handle);
  }, [compareQuery, report?.player?.calendar, selectedComparisonLabel]);

  const selectPlayer = (item) => {
    setSelectedPlayerId(String(item.id));
    setSelectedPlayerSeasonId(item.player_season_id ? String(item.player_season_id) : "");
    setPlayerQuery(selectedLabel(item));
    setShowPlayerResults(false);
    setComparisonReport(null);
    setSelectedComparisonLabel("");
    router.replace({ pathname: "/report", query: { player_id: item.id, player_season_id: item.player_season_id } }, undefined, { shallow: true });
  };

  const selectComparisonPlayer = async (item) => {
    const label = selectedLabel(item);
    setSelectedComparisonLabel(label);
    setCompareQuery(label);
    setShowCompareResults(false);
    setComparisonLoading(true);
    try {
      let seasonId = item.player_season_id;
      try {
        const seasons = await fetchJson(`/players/${item.id}/seasons`);
        const current = report?.player || {};
        const sameCompetition = seasons?.find((season) => season.calendar === current.calendar && season.competition_name === current.competition_name);
        const sameSeason = seasons?.find((season) => season.calendar === current.calendar);
        seasonId = sameCompetition?.player_season_id || sameSeason?.player_season_id || seasonId;
      } catch (err) {
        console.error(err);
      }
      const data = await fetchJson(`/players/${item.id}/report`, { player_season_id: seasonId });
      setComparisonReport(data);
    } catch (err) {
      console.error(err);
      setComparisonReport(null);
    } finally {
      setComparisonLoading(false);
    }
  };

  const exportPng = async (full = false) => {
    if (!report || exportBusy) return;
    setExportBusy(true);
    try {
      const canvas = await drawReportPng({
        report,
        photoUrl: tmPhotoUrl,
        tmDetails,
        characteristics,
        profileCategoriesData,
        similarities,
        scoreHistory,
        context: percentileContext,
        referenceGroup: selectedReferenceGroup,
        full,
      });
      const slug = String(report.player?.name || "player").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
      downloadCanvas(canvas, `report-${slug || "player"}-${full ? "full" : "scout"}.png`);
    } finally {
      setExportBusy(false);
    }
  };

  const exportPdf = () => {
    if (!report || exportBusy || typeof window === "undefined") return;
    setExportBusy(true);
    try {
      const pdfWindow = window.open("", "_blank");
      if (!pdfWindow) {
        setError("Unable to open the PDF export window. Please allow pop-ups for this site.");
        return;
      }
      const html = buildFullReportPdfHtml({
        report,
        photoUrl: tmPhotoUrl,
        tmDetails,
        socialLinks,
        characteristics,
        profileCategoriesData,
        similarities,
        comparisonReport,
        scoreHistory,
        scoreSnapshots,
        metricsSummary,
        context: percentileContext,
        referenceGroup: selectedReferenceGroup,
      });
      pdfWindow.document.open();
      pdfWindow.document.write(html);
      pdfWindow.document.close();
      pdfWindow.focus();
      setTimeout(() => {
        pdfWindow.print();
      }, 450);
    } finally {
      setTimeout(() => setExportBusy(false), 700);
    }
  };

  const addToProspects = async () => {
    if (!selectedPlayerId || prospectLoading || isProspect) return;
    setProspectLoading(true);
    setProspectMessage("");
    try {
      const selectedSeason = selectedPlayerSeasonId || report?.player?.player_season_id;
      const res = await postJson("/prospects", {
        player_id: Number(selectedPlayerId),
        player_season_id: selectedSeason ? Number(selectedSeason) : undefined,
      });
      setIsProspect(true);
      setProspectMessage(res?.added ? "Player added to prospects." : "Player is already in prospects.");
    } catch (err) {
      setProspectMessage(err.message || "Unable to add player to prospects.");
    } finally {
      setProspectLoading(false);
    }
  };

  const selectedSeasonId = selectedPlayerSeasonId || report?.player?.player_season_id || "";
  const seasons = useMemo(() => sortSeasons(report?.available_seasons || []), [report?.available_seasons]);
  const metrics = report?.metrics || {};
  const positionGroup = normalizePositionGroup(report?.player?.assigned_role, report?.player?.position);
  const positionMeta = getPositionMeta(report?.player?.assigned_role, report?.player?.position);
  const selectedReferenceGroup = positionGroup;
  const profileCategoriesData = useMemo(() => buildProfileCategories(metrics, percentileContext), [metrics, percentileContext]);
  const characteristics = useMemo(() => buildCharacteristics(metrics, positionGroup, percentileContext), [metrics, positionGroup, percentileContext]);
  const scoreHistory = useMemo(() => (report?.score_history || []).map((row) => ({ ...row, global_score_adjusted: row.global_score_adjusted == null ? null : Number(row.global_score_adjusted) })), [report?.score_history]);
  const scoreSnapshots = useMemo(
    () =>
      (report?.score_snapshots || []).map((row) => ({
        ...row,
        label: formatSnapshotLabel(row.snapshot_date, row.snapshot_key),
        global_score_adjusted: row.global_score_adjusted == null ? null : Number(row.global_score_adjusted),
        minutes_played: row.minutes_played == null ? null : Number(row.minutes_played),
        minutes_ratio: row.minutes_ratio == null ? null : Number(row.minutes_ratio),
      })),
    [report?.score_snapshots]
  );
  const metricsSummary = useMemo(() => availableMetricsSummary(metrics), [metrics]);
  const tmFields = report?.tm_fields || {};
  const tmProfileUrl = toAbsoluteUrl(report?.player?.tm_profile_url || tmFields.tm_profile_url);
  const tmAgentUrl = toAbsoluteUrl(tmFields.tm_agent_url || tmFields.agent_url);
  const tmPhotoUrl = toAbsoluteUrl(
    tmFields.app_photo_url ||
      tmFields.tm_profile_image_url ||
      tmFields.profile_image_url ||
      tmFields.tm_photo_url
  );
  const socialLinks = useMemo(() => {
    const raw = tmFields.tm_social_media || tmFields.tm_socials || tmFields.tm_social_links || tmFields.tm_social;
    const seen = new Set();
    return extractUrls(raw)
      .map((url) => toAbsoluteUrl(url))
      .filter(Boolean)
      .filter((url) => {
        if (seen.has(url)) return false;
        seen.add(url);
        return true;
      })
      .map((url) => {
        const type = resolveSocialType(url);
        const label = type === "x" ? "X" : type.charAt(0).toUpperCase() + type.slice(1);
        return { url, type, label };
      });
  }, [tmFields]);
  const tmDetails = [
    { label: "Market value", value: formatCompactNumber(tmFields.tm_market_value || tmFields.market_value) },
    { label: "Agent", value: tmFields.tm_agent_name || tmFields.agent_name, url: tmAgentUrl },
    { label: "Height", value: tmFields.tm_height || tmFields.tm_height_cm || metrics.height_cm },
    { label: "Weight", value: tmFields.tm_weight || tmFields.tm_weight_kg || metrics.weight_kg },
    { label: "Date of birth", value: tmFields.tm_birth_date || tmFields.birth_date },
    { label: "Place of birth", value: [tmFields.tm_birth_city || tmFields.birth_city, tmFields.tm_birth_country || tmFields.birth_country].filter(Boolean).join(", ") },
    { label: "Contract", value: tmFields.tm_club_contract_expires || tmFields.tm_contract_expires || tmFields.tm_contract_until },
    { label: "Citizenship", value: tmFields.tm_citizenship || tmFields.citizenship },
    { label: "Foot", value: tmFields.tm_foot || tmFields.foot },
    { label: "Outfitter", value: tmFields.tm_outfitter || tmFields.outfitter },
  ].filter((item) => hasValue(item.value) || hasValue(item.url));
  const hasTmData = tmDetails.length > 0 || socialLinks.length > 0 || hasValue(tmProfileUrl) || hasValue(tmPhotoUrl);

  return (
    <main className="nl-page px-4 py-8 text-slate-900 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-[1540px] space-y-6">
        <header className="surface-panel relative z-50 overflow-visible rounded-lg p-5 md:p-7">
          <div className="grid gap-5 2xl:grid-cols-[minmax(0,1fr)_420px] 2xl:items-start">
            <div className={report ? "grid gap-5 lg:grid-cols-[170px_minmax(0,1fr)]" : ""}>
              {report ? (
                <div className="w-full max-w-[170px]">
                  <div className="h-[210px] overflow-hidden rounded-lg border border-[#3A8967]/25 bg-[#06100C] p-2 shadow-sm">
                    <div className="h-full overflow-hidden rounded-md border border-white/10 bg-white/[0.04]">
                      {tmPhotoUrl ? (
                        <img src={tmPhotoUrl} alt={report.player.name || "Player"} className="h-full w-full object-cover" loading="lazy" />
                      ) : (
                        <div className="flex h-full w-full items-center justify-center text-4xl font-black text-slate-500">
                          {getInitials(report.player.name)}
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="mt-3 flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.035] px-3 py-2 shadow-sm">
                    <ClubLogo name={report.player.team} className="h-8 w-8 rounded-md" />
                    <span className="min-w-0 truncate text-xs font-black text-slate-300">{report.player.team || "No club"}</span>
                  </div>
                </div>
              ) : null}

              <div>
                <p className="nl-kicker">Player report</p>
                <h1 className="mt-3 max-w-4xl text-4xl font-black tracking-[-0.04em] text-white md:text-6xl">
                  {report?.player?.name || "Professional scouting dossier"}
                </h1>
                <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600 md:text-base">
                  Position-based scouting report using the selected season, competition context, raw metrics, and positional percentiles.
                </p>
                {report ? (
                  <>
                    <div className="mt-5 flex flex-wrap items-center gap-2">
                      <span className="rounded-md border border-[#3A8967]/35 bg-[#2F7D5C]/15 px-3 py-1 text-xs font-black uppercase tracking-[0.14em] text-[#8CC7A7]">{positionMeta.short} - {positionMeta.label}</span>
                      <span className="rounded-md border border-white/10 bg-white/[0.035] px-3 py-1 text-xs font-bold text-slate-300">{report.player.team || "No club"}</span>
                      <span className="rounded-md border border-white/10 bg-white/[0.035] px-3 py-1 text-xs font-bold text-slate-300">{report.player.competition_name || "No competition"}</span>
                      {tmProfileUrl ? (
                        <a href={tmProfileUrl} target="_blank" rel="noreferrer" className="rounded-md border border-sky-400/25 bg-sky-500/10 px-3 py-1 text-xs font-black uppercase tracking-[0.14em] text-sky-200">
                          Transfermarkt
                        </a>
                      ) : null}
                    </div>
                    {socialLinks.length > 0 ? (
                      <div className="mt-3 flex flex-wrap gap-2">
                        {socialLinks.map((link) => (
                          <a key={link.url} href={link.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-2 rounded-md border border-white/10 bg-white/[0.035] px-3 py-1 text-xs font-black text-slate-300 transition hover:border-[#3A8967]/35 hover:text-[#8CC7A7]">
                            <SocialIcon type={link.type} />
                            <span>{link.label}</span>
                          </a>
                        ))}
                      </div>
                    ) : null}
                    <div className="mt-5 flex flex-wrap gap-2">
                      <button type="button" onClick={addToProspects} disabled={prospectLoading || isProspect} className={isProspect ? "nl-button-secondary px-4 py-2 text-xs uppercase tracking-[0.14em]" : "nl-button-primary px-4 py-2 text-xs uppercase tracking-[0.14em]"}>
                        {prospectLoading ? "Adding..." : isProspect ? "In prospects" : "Add prospect"}
                      </button>
                      <button type="button" onClick={() => exportPng(false)} disabled={exportBusy} className="nl-button-primary px-4 py-2 text-xs uppercase tracking-[0.14em]">
                        Scout PNG
                      </button>
                      <button type="button" onClick={exportPdf} disabled={exportBusy} className="nl-button-secondary px-4 py-2 text-xs uppercase tracking-[0.14em]">
                        Full PDF
                      </button>
                    </div>
                    {prospectMessage ? (
                      <p className={`mt-3 text-xs font-semibold ${isProspect ? "text-[#8CC7A7]" : "text-amber-200"}`}>
                        {prospectMessage}
                      </p>
                    ) : null}
                    {hasTmData ? (
                      <div className="mt-5 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
                        {tmDetails.map((item) => (
                          <div key={item.label} className="rounded-lg border border-white/10 bg-white/[0.035] px-3 py-2">
                            <p className="text-[10px] font-black uppercase tracking-[0.14em] text-slate-400">{item.label}</p>
                            {item.url ? (
                              <a href={item.url} target="_blank" rel="noreferrer" className="mt-1 block truncate text-sm font-black text-[#8CC7A7] hover:text-white">
                                {item.value || "Profile"}
                              </a>
                            ) : (
                              <p className="mt-1 truncate text-sm font-black text-white">{item.value}</p>
                            )}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="mt-5 text-xs font-semibold text-slate-500">Transfermarkt data not available for this player yet.</p>
                    )}
                  </>
                ) : null}
              </div>
            </div>

            <div className="relative z-[9999] space-y-3 2xl:pt-2">
              <div className="rounded-lg border border-white/10 bg-white/[0.035] p-3">
                <label htmlFor="report-search-season" className="text-[10px] font-black uppercase tracking-[0.16em] text-slate-500">
                  Search season
                </label>
                <select
                  id="report-search-season"
                  className="nl-field mt-2"
                  value={selectedSearchSeason}
                  onChange={(event) => {
                    setSelectedSearchSeason(event.target.value);
                    setPlayerResults([]);
                    if (!selectedPlayerId) {
                      setShowPlayerResults(playerQuery.trim().length >= 2);
                    }
                  }}
                >
                  <option value="">All seasons</option>
                  {searchSeasons.map((season) => (
                    <option key={season} value={season}>
                      {season}
                    </option>
                  ))}
                </select>
              </div>
              <PlayerSearch
                query={playerQuery}
                results={playerResults}
                visible={showPlayerResults}
                onFocus={() => {
                  if (selectedPlayerId) {
                    setPlayerQuery("");
                    setSelectedPlayerId("");
                    setSelectedPlayerSeasonId("");
                  }
                  setShowPlayerResults(true);
                }}
                onQueryChange={(value) => {
                  setPlayerQuery(value);
                  setSelectedPlayerId("");
                  setSelectedPlayerSeasonId("");
                  setReport(null);
                  setShowPlayerResults(true);
                }}
                onSelect={selectPlayer}
              />
            </div>
          </div>
        </header>

        {error ? <ReportCard><p className="text-sm font-semibold text-red-700">{error}</p></ReportCard> : null}
        {loading ? <ReportCard><p className="text-sm text-slate-600">Loading report...</p></ReportCard> : null}

        {report ? (
          <>
            <SeasonSelector
              seasons={seasons}
              selectedSeasonId={selectedSeasonId}
              onSelect={(seasonId) => {
                setSelectedPlayerSeasonId(seasonId);
                setComparisonReport(null);
                setSelectedComparisonLabel("");
                setCompareQuery("");
                if (selectedPlayerId) {
                  router.replace({ pathname: "/report", query: { player_id: selectedPlayerId, player_season_id: seasonId } }, undefined, { shallow: true });
                }
              }}
            />
            <SeasonStatistics report={report} metrics={metrics} />

            <div className="grid gap-4 xl:grid-cols-[0.82fr_1.18fr_1fr]">
              <PositionCard player={report.player} />
              <PlayerProfileCard categories={profileCategoriesData} />
              <CharacteristicsCard characteristics={characteristics} />
            </div>

            <AdvancedCharacteristics
              metrics={metrics}
              player={report.player}
              rawMode={rawMode}
              setRawMode={setRawMode}
              context={percentileContext}
              setContext={setPercentileContext}
            />

            <div className="grid gap-4 xl:grid-cols-2">
              <PlayerRadarComparison
                report={report}
                comparisonReport={comparisonReport}
                context={percentileContext}
              />
              <PlayerStatsComparison
                report={report}
                comparisonReport={comparisonReport}
                context={percentileContext}
                query={compareQuery}
                results={compareResults}
                showResults={showCompareResults}
                loading={comparisonLoading}
                onSearch={(value) => {
                  setCompareQuery(value);
                  setSelectedComparisonLabel("");
                  setShowCompareResults(value.trim().length >= 2);
                }}
                onSelect={selectComparisonPlayer}
              />
            </div>

            <SimilarPlayersCard
              similarities={similarities}
              loading={similarLoading}
              onOpen={(sim) => {
                if (sim?.player_b_id) {
                  window.open(`/report?player_id=${sim.player_b_id}`, "_blank", "noopener,noreferrer");
                }
              }}
            />

            <div className="grid gap-4 xl:grid-cols-3">
              <PlayerSeasonRadarComparison
                report={report}
                context={percentileContext}
              />
              <ScoreHistory data={scoreHistory} />
              <InSeasonScoreSnapshots data={scoreSnapshots} />
            </div>

            <ReportCard>
              <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                <div>
                  <p className="text-[11px] font-black uppercase tracking-[0.22em] text-teal-700">Data coverage</p>
                  <p className="mt-2 text-sm text-slate-600">Configured report metrics available: {metricsSummary.available.length}. Missing or unavailable for this player/provider: {metricsSummary.missing.length}.</p>
                </div>
                <p className="text-xs text-slate-500">Missing data is shown as "-" and is never coerced to zero.</p>
              </div>
            </ReportCard>

            <TransferHistoryCard transfers={report.transfer_history || []} />
          </>
        ) : !loading ? (
          <ReportCard className="py-14 text-center">
            <p className="text-lg font-black text-slate-950">Search a player to open a professional scouting report.</p>
            <p className="mt-2 text-sm text-slate-500">All statistics and percentiles are loaded from the Next Legend backend.</p>
          </ReportCard>
        ) : null}
      </div>
    </main>
  );
}
