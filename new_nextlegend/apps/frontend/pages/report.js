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
import { apiUrl, fetchJson } from "@/lib/api";
import ClubLogo from "@/components/ClubLogo";
import { getMetricConfig, getPositionMeta, getRadarMetricKeys, normalizePositionGroup, POSITION_GROUPS } from "@/lib/reportMetrics";
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
  ctx.font = "900 23px Arial";
  ctx.fillText(label, x, y);
  ctx.fillStyle = "#64748b";
  ctx.font = "700 18px Arial";
  ctx.textAlign = "right";
  ctx.fillText(raw ? `${raw} raw` : score === null ? "-" : `${Math.round(score)} pct`, x + width, y);
  ctx.textAlign = "left";
  ctx.fillStyle = tone.bg;
  drawRoundedRect(ctx, x, y + 16, width, 18, 9);
  ctx.fill();
  if (score !== null) {
    ctx.fillStyle = tone.color;
    drawRoundedRect(ctx, x, y + 16, Math.max(10, (width * score) / 100), 18, 9);
    ctx.fill();
  }
  return y + 54;
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
  const filtered = rows.slice(0, 8);
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

const buildPngVerdict = (player, positionMeta, characteristics) => {
  const score = Number(player?.global_score_adjusted);
  const minutes = Number(player?.minutes_played);
  const strengths = characteristics?.strengths || [];
  const top = strengths.slice(0, 2).map((row) => row.label.toLowerCase());
  const scoreText = Number.isFinite(score) && score >= 85
    ? "high-priority follow-up"
    : Number.isFinite(score) && score >= 75
      ? "worth a deeper scouting check"
      : "profile to monitor with context";
  const evidence = top.length ? `Core evidence: ${top.join(" and ")}.` : "Evidence is limited by available percentile coverage.";
  const volume = Number.isFinite(minutes) && minutes >= 900
    ? "The sample is meaningful enough to trust the signal."
    : Number.isFinite(minutes) && minutes > 0
      ? "The minutes sample is still developing, so video validation matters."
      : "Minutes context is unavailable.";
  return `${positionMeta.short} ${scoreText}. ${evidence} ${volume}`;
};

const drawReportPng = async ({
  report,
  photoUrl,
  tmDetails,
  characteristics,
  profileCategoriesData,
  similarities,
  context,
  referenceGroup,
  full,
}) => {
  const canvas = document.createElement("canvas");
  canvas.width = 1600;
  canvas.height = full ? 2480 : 1800;
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
  const keyRows = radarKeys.slice(0, full ? 10 : 7);
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
  ctx.fillText(full ? "NEXT LEGEND - COMPLETE PLAYER REPORT" : "NEXT LEGEND - SCOUT DECISION SNAPSHOT", 70, 72);
  ctx.fillStyle = "#ffffff";
  ctx.font = "900 76px Arial";
  wrapCanvasText(ctx, player.name || "Player", 70, 165, 880, 80, 2);
  ctx.fillStyle = "#d1fae5";
  ctx.font = "800 27px Arial";
  ctx.fillText([positionMeta.short, player.team, player.competition_name, player.calendar].filter(Boolean).join("  |  "), 72, 282);
  ctx.fillStyle = "#e2e8f0";
  ctx.font = "700 22px Arial";
  wrapCanvasText(ctx, buildPngVerdict(player, positionMeta, characteristics), 72, 330, 950, 31, 3);

  drawCoverImage(ctx, photo, 1160, 54, 300, 340, 32, player.name);
  drawCard(ctx, 1010, 278, 188, 128, { fill: "#ffffff", stroke: "#99f6e4", radius: 26 });
  ctx.fillStyle = "#0f766e";
  ctx.font = "900 20px Arial";
  ctx.fillText("NL RATING", 1032, 316);
  ctx.fillStyle = "#0f172a";
  ctx.font = "900 58px Arial";
  ctx.fillText(formatValue(player.global_score_adjusted, "score"), 1032, 376);

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
  let barY = y + 88;
  profileCategoriesData.forEach((category) => {
    barY = drawBar(ctx, category.label, category.score, null, 598, barY, 392);
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
  let tableY = y + 116;
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
  drawCard(ctx, 70, y, 710, 320, { fill: "#ffffff", stroke: "#e2e8f0", radius: 24 });
  drawKicker(ctx, "Risk check", 102, y + 42);
  ctx.fillStyle = "#0f172a";
  ctx.font = "900 30px Arial";
  ctx.fillText("Relevant weaknesses only", 102, y + 82);
  let riskY = y + 124;
  (topWeaknesses.length ? topWeaknesses : [{ label: "No major relevant weakness flagged by the current percentile model.", percentile: null }]).forEach((row) => {
    const tone = canvasTone(row.percentile);
    ctx.fillStyle = row.percentile == null ? "#f8fafc" : tone.bg;
    drawRoundedRect(ctx, 102, riskY, 646, 48, 18);
    ctx.fill();
    ctx.fillStyle = row.percentile == null ? "#475569" : tone.text;
    ctx.font = "800 19px Arial";
    wrapCanvasText(ctx, row.percentile == null ? row.label : `${row.label} - ${Math.round(row.percentile)} pct`, 122, riskY + 31, 600, 20, 1);
    riskY += 58;
  });

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
  ctx.fillText("Source: Next Legend player season report. Missing values are not converted to zero.", 70, canvas.height - 44);
  ctx.textAlign = "right";
  ctx.fillText("nextlegend.ai", canvas.width - 70, canvas.height - 44);
  ctx.textAlign = "left";
  return canvas;
};

const downloadCanvas = (canvas, filename) => {
  const link = document.createElement("a");
  link.download = filename;
  link.href = canvas.toDataURL("image/png");
  link.click();
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

function ScoreHistory({ data }) {
  if (!data?.length) return null;
  return (
    <ReportCard>
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="text-[11px] font-black uppercase tracking-[0.22em] text-teal-700">Score history</p>
          <h3 className="mt-2 text-xl font-black text-slate-950">Next Legend rating evolution</h3>
        </div>
      </div>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 12, right: 18, left: 0, bottom: 8 }}>
            <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
            <XAxis dataKey="calendar" stroke="#64748b" tick={{ fill: "#64748b", fontSize: 12 }} />
            <YAxis domain={[50, 100]} stroke="#64748b" tick={{ fill: "#64748b", fontSize: 12 }} />
            <Tooltip
              contentStyle={{ background: "#ffffff", border: "1px solid #e2e8f0", borderRadius: 14, color: "#0f172a", boxShadow: "0 20px 45px rgba(15,23,42,0.12)" }}
              labelStyle={{ color: "#0f766e" }}
              formatter={(value) => [value == null ? "-" : Number(value).toFixed(1), "Rating"]}
            />
            <Line type="monotone" dataKey="global_score_adjusted" stroke="#0f766e" strokeWidth={2.5} dot={{ r: 4, fill: "#ffffff", stroke: "#0f766e", strokeWidth: 2 }} connectNulls />
          </LineChart>
        </ResponsiveContainer>
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
          <p className="text-[11px] font-black uppercase tracking-[0.22em] text-teal-700">Transfers</p>
          <h3 className="mt-2 text-xl font-black text-slate-950">Career movement timeline</h3>
          <p className="mt-1 text-sm text-slate-600">Most recent moves first, with fee and source context when available.</p>
        </div>
        <span className="rounded-full border border-slate-200 px-3 py-1 text-xs font-black text-teal-700">{items.length} moves</span>
      </div>
      {items.length ? (
        <div className="mt-5 space-y-3">
          {items.map((transfer) => (
            <div key={`${transfer.id}-${transfer.transfer_date || "date"}-${transfer.team_in_name || "in"}-${transfer.team_out_name || "out"}`} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="rounded-full bg-slate-950 px-2.5 py-1 text-[11px] font-black uppercase tracking-[0.08em] text-white">{formatTransferDate(transfer.transfer_date)}</span>
                    <span className="rounded-full border border-teal-200 bg-teal-50 px-2.5 py-1 text-[11px] font-black uppercase tracking-[0.08em] text-teal-700">{transfer.transfer_type || "Transfer"}</span>
                  </div>
                  <div className="mt-4 grid gap-3 md:grid-cols-[minmax(0,1fr)_44px_minmax(0,1fr)] md:items-center">
                    <div className="flex min-w-0 items-center gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
                      <ClubLogo name={transfer.team_out_name} className="h-10 w-10 rounded" />
                      <div className="min-w-0">
                        <p className="text-[11px] font-black uppercase tracking-[0.12em] text-slate-400">From</p>
                        <p className="truncate text-sm font-extrabold text-slate-950">{transfer.team_out_name || "Free agent"}</p>
                      </div>
                    </div>
                    <div className="hidden h-10 items-center justify-center rounded-full border border-slate-200 bg-white text-lg font-black text-slate-400 md:flex">-&gt;</div>
                    <div className="flex min-w-0 items-center gap-3 rounded-lg border border-teal-200 bg-teal-50 p-3">
                      <ClubLogo name={transfer.team_in_name} className="h-10 w-10 rounded" />
                      <div className="min-w-0">
                        <p className="text-[11px] font-black uppercase tracking-[0.12em] text-teal-700">To</p>
                        <p className="truncate text-sm font-extrabold text-slate-950">{transfer.team_in_name || "Free agent"}</p>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="min-w-[170px] rounded-lg border border-slate-200 bg-slate-50 p-3 text-left lg:text-right">
                  <p className="text-[11px] font-black uppercase tracking-[0.12em] text-slate-400">Fee</p>
                  <p className="mt-1 text-lg font-extrabold text-slate-950">{transferFeeLabel(transfer.transfer_fee)}</p>
                  <p className="mt-1 text-xs font-semibold text-slate-500">{transfer.league_name || transfer.team_name_context || "League to confirm"}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="mt-5 rounded-xl border border-dashed border-slate-300 bg-slate-50 p-5">
          <p className="text-sm font-extrabold text-slate-950">No verified transfer history yet.</p>
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
  const [referenceGroup, setReferenceGroup] = useState("");
  const [exportBusy, setExportBusy] = useState(false);

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
    if (playerQuery.trim().length < 2 || selectedPlayerId) {
      setPlayerResults([]);
      return;
    }
    const handle = setTimeout(async () => {
      try {
        const rows = await fetchJson("/players", { q: playerQuery.trim(), limit: 12 });
        setPlayerResults(rows || []);
      } catch (err) {
        console.error(err);
      }
    }, 180);
    return () => clearTimeout(handle);
  }, [playerQuery, selectedPlayerId]);

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
        context: percentileContext,
        referenceGroup: selectedReferenceGroup,
        full,
      });
      const slug = String(report.player?.name || "player").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
      downloadCanvas(canvas, `nextlegend-${slug || "report"}-${full ? "full" : "scout"}.png`);
    } finally {
      setExportBusy(false);
    }
  };

  const selectedSeasonId = selectedPlayerSeasonId || report?.player?.player_season_id || "";
  const seasons = useMemo(() => sortSeasons(report?.available_seasons || []), [report?.available_seasons]);
  const metrics = report?.metrics || {};
  const positionGroup = normalizePositionGroup(report?.player?.assigned_role, report?.player?.position);
  const positionMeta = getPositionMeta(report?.player?.assigned_role, report?.player?.position);
  const selectedReferenceGroup = referenceGroup || positionGroup;
  const profileCategoriesData = useMemo(() => buildProfileCategories(metrics, percentileContext), [metrics, percentileContext]);
  const characteristics = useMemo(() => buildCharacteristics(metrics, positionGroup, percentileContext), [metrics, positionGroup, percentileContext]);
  const scoreHistory = useMemo(() => (report?.score_history || []).map((row) => ({ ...row, global_score_adjusted: row.global_score_adjusted == null ? null : Number(row.global_score_adjusted) })), [report?.score_history]);
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

  useEffect(() => {
    if (report?.player?.player_season_id) {
      setReferenceGroup(positionGroup);
    }
  }, [report?.player?.player_season_id, positionGroup]);

  return (
    <main className="nl-page px-4 py-8 text-slate-900 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-[1540px] space-y-6">
        <header className="surface-panel relative z-50 overflow-visible rounded-lg p-5 md:p-7">
          <div className="grid gap-5 2xl:grid-cols-[minmax(0,1fr)_420px] 2xl:items-start">
            <div className={report ? "grid gap-5 lg:grid-cols-[170px_minmax(0,1fr)]" : ""}>
              {report ? (
                <div className="w-full max-w-[170px]">
                  <div className="h-[210px] overflow-hidden rounded-2xl border border-teal-200 bg-gradient-to-br from-teal-50 to-white p-2 shadow-sm">
                    <div className="h-full overflow-hidden rounded-xl border border-white bg-slate-100">
                      {tmPhotoUrl ? (
                        <img src={tmPhotoUrl} alt={report.player.name || "Player"} className="h-full w-full object-cover" loading="lazy" />
                      ) : (
                        <div className="flex h-full w-full items-center justify-center text-4xl font-black text-slate-500">
                          {getInitials(report.player.name)}
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="mt-3 flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 shadow-sm">
                    <ClubLogo name={report.player.team} className="h-8 w-8 rounded-md" />
                    <span className="min-w-0 truncate text-xs font-black text-slate-700">{report.player.team || "No club"}</span>
                  </div>
                </div>
              ) : null}

              <div>
                <p className="text-[11px] font-black uppercase tracking-[0.32em] text-teal-700">Next Legend report</p>
                <h1 className="mt-3 max-w-4xl text-4xl font-black tracking-[-0.04em] text-slate-950 md:text-6xl">
                  {report?.player?.name || "Professional scouting dossier"}
                </h1>
                <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600 md:text-base">
                  Position-based scouting report using the selected season, competition context, raw metrics, and positional percentiles.
                </p>
                {report ? (
                  <>
                    <div className="mt-5 flex flex-wrap items-center gap-2">
                      <span className="rounded-full border border-teal-600/40 bg-teal-50 px-3 py-1 text-xs font-black uppercase tracking-[0.14em] text-teal-700">{positionMeta.short} - {positionMeta.label}</span>
                      <span className="rounded-full border border-slate-200 px-3 py-1 text-xs font-bold text-slate-600">{report.player.team || "No club"}</span>
                      <span className="rounded-full border border-slate-200 px-3 py-1 text-xs font-bold text-slate-600">{report.player.competition_name || "No competition"}</span>
                      {tmProfileUrl ? (
                        <a href={tmProfileUrl} target="_blank" rel="noreferrer" className="rounded-full border border-blue-200 bg-blue-50 px-3 py-1 text-xs font-black uppercase tracking-[0.14em] text-blue-700">
                          Transfermarkt
                        </a>
                      ) : null}
                    </div>
                    {socialLinks.length > 0 ? (
                      <div className="mt-3 flex flex-wrap gap-2">
                        {socialLinks.map((link) => (
                          <a key={link.url} href={link.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-black text-slate-700 transition hover:border-teal-500 hover:text-teal-700">
                            <SocialIcon type={link.type} />
                            <span>{link.label}</span>
                          </a>
                        ))}
                      </div>
                    ) : null}
                    <div className="mt-5 flex flex-wrap gap-2">
                      <button type="button" onClick={() => exportPng(false)} disabled={exportBusy} className="rounded-lg bg-teal-700 px-4 py-2 text-xs font-black uppercase tracking-[0.14em] text-white shadow-sm transition hover:bg-teal-800 disabled:opacity-60">
                        Scout PNG
                      </button>
                      <button type="button" onClick={() => exportPng(true)} disabled={exportBusy} className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-xs font-black uppercase tracking-[0.14em] text-slate-800 shadow-sm transition hover:border-teal-600 hover:text-teal-700 disabled:opacity-60">
                        Full PNG
                      </button>
                    </div>
                    {hasTmData ? (
                      <div className="mt-5 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
                        {tmDetails.map((item) => (
                          <div key={item.label} className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                            <p className="text-[10px] font-black uppercase tracking-[0.14em] text-slate-400">{item.label}</p>
                            {item.url ? (
                              <a href={item.url} target="_blank" rel="noreferrer" className="mt-1 block truncate text-sm font-black text-teal-700 hover:text-teal-900">
                                {item.value || "Profile"}
                              </a>
                            ) : (
                              <p className="mt-1 truncate text-sm font-black text-slate-900">{item.value}</p>
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

            <div className="relative z-[9999] 2xl:pt-2">
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
              referenceGroup={selectedReferenceGroup}
              setReferenceGroup={setReferenceGroup}
            />

            <div className="grid gap-4 xl:grid-cols-2">
              <PlayerRadarComparison
                report={report}
                comparisonReport={comparisonReport}
                context={percentileContext}
                referenceGroup={selectedReferenceGroup}
              />
              <PlayerStatsComparison
                report={report}
                comparisonReport={comparisonReport}
                context={percentileContext}
                referenceGroup={selectedReferenceGroup}
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
                setSelectedPlayerId(String(sim.player_b_id));
                setSelectedPlayerSeasonId("");
                setPlayerQuery(sim.player_b_name || "");
                setComparisonReport(null);
                setSelectedComparisonLabel("");
                router.replace({ pathname: "/report", query: { player_id: sim.player_b_id } }, undefined, { shallow: true });
              }}
            />

            <div className="grid gap-4 xl:grid-cols-2">
              <PlayerSeasonRadarComparison
                report={report}
                context={percentileContext}
                referenceGroup={selectedReferenceGroup}
              />
              <ScoreHistory data={scoreHistory} />
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
