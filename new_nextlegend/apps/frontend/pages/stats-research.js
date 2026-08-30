import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import dynamic from "next/dynamic";
import { fetchJson, fetchJsonCached } from "@/lib/api";
import { METRIC_LABELS } from "@/lib/metricLabels";
import { POSITIONS_GLOSSARY } from "@/lib/positionsGlossary";
import {
  DEFAULT_MIN_MINUTES,
  DEFAULT_SCOUTING_COMPETITION,
  DEFAULT_SCOUTING_SEASON,
  formatFilterValue,
  parseIntegerInput,
  sortCompetitionNames,
  sortPositions,
  withDefaultSeason,
} from "@/lib/scoutingFilters";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const Card = ({ children, className = "" }) => (
  <div className={`glass-panel rounded-xl p-4 border border-white/5 ${className}`}>
    {children}
  </div>
);

const Label = ({ children }) => (
  <label className="text-xs uppercase tracking-[0.2em] text-slate-400">
    {children}
  </label>
);

const Select = ({ value, onChange, children, id, name, ariaLabel }) => (
  <select
    id={id}
    name={name || id}
    aria-label={ariaLabel}
    className="nl-field"
    value={value}
    onChange={onChange}
  >
    {children}
  </select>
);

const formatMetricLabel = (key) => {
  if (METRIC_LABELS[key]) return METRIC_LABELS[key];
  return key.replace(/_/g, " ");
};

const formatValue = (value, digits = 2) => {
  if (value === null || value === undefined || value === "") return "--";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return String(value);
  const abs = Math.abs(numeric);
  if (abs >= 100) return numeric.toFixed(0);
  return numeric.toFixed(digits);
};

const sortIndicator = (sortConfig, key) => {
  if (sortConfig.key !== key) return "";
  return sortConfig.dir === "asc" ? "^" : "v";
};

const percentile = (values, p) => {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  const weight = index - lower;
  return sorted[lower] + (sorted[upper] - sorted[lower]) * weight;
};

const axisRange = (values) => {
  const numeric = values.filter(Number.isFinite);
  if (!numeric.length) return [0, 1];
  const min = Math.min(...numeric);
  const max = Math.max(...numeric);
  if (min === max) {
    const pad = Math.max(Math.abs(min) * 0.1, 1);
    return [min - pad, max + pad];
  }
  const pad = (max - min) * 0.08;
  return [min - pad, max + pad];
};

const isGoodValue = (value, threshold, lowerIsBetter) => {
  if (!Number.isFinite(value) || !Number.isFinite(threshold)) return false;
  return lowerIsBetter ? value <= threshold : value >= threshold;
};

const rowKey = (row) => `${row.name || ""}|${row.team || ""}|${row.competition_name || ""}`;

export default function StatsResearchPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [leagueOptions, setLeagueOptions] = useState([DEFAULT_SCOUTING_COMPETITION, "All leagues"]);
  const [selectedLeague, setSelectedLeague] = useState(DEFAULT_SCOUTING_COMPETITION);
  const [seasons, setSeasons] = useState([]);
  const [selectedSeason, setSelectedSeason] = useState(DEFAULT_SCOUTING_SEASON);
  const [positions, setPositions] = useState([]);
  const [positionsOpen, setPositionsOpen] = useState(false);
  const [positionsAll, setPositionsAll] = useState(true);
  const [selectedPositions, setSelectedPositions] = useState([]);
  const [minMinutes, setMinMinutes] = useState(DEFAULT_MIN_MINUTES);
  const [metrics, setMetrics] = useState([]);
  const [lowerIsBetter, setLowerIsBetter] = useState(new Set());
  const [metricX, setMetricX] = useState("");
  const [metricY, setMetricY] = useState("");
  const [rows, setRows] = useState([]);
  const [tableSort, setTableSort] = useState({ key: "metric_x", dir: "desc" });
  const positionsButtonRef = useRef(null);
  const positionsMenuRef = useRef(null);
  const [positionsMenuStyle, setPositionsMenuStyle] = useState({});
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    const loadMeta = async () => {
      try {
        const [competitions, seasonsData, positionsData, metricsData] = await Promise.all([
          fetchJsonCached("/meta/competitions"),
          fetchJsonCached("/meta/seasons"),
          fetchJsonCached("/meta/positions"),
          fetchJsonCached("/meta/stats-research/metrics"),
        ]);
        const compNames = (competitions || []).map((item) => item.name).filter(Boolean);
        setLeagueOptions([...sortCompetitionNames(compNames), "All leagues"]);
        setSeasons(withDefaultSeason(seasonsData || []));
        const positionList = sortPositions(positionsData || []).map((code) => ({
          code,
          label: POSITIONS_GLOSSARY[code] || code,
        }));
        setPositions(positionList);
        const metricList = Array.from(
          new Set((metricsData?.metrics || []).filter(Boolean))
        ).filter((metric) => METRIC_LABELS[metric]);
        metricList.sort((a, b) => formatMetricLabel(a).localeCompare(formatMetricLabel(b), undefined, { sensitivity: "base" }));
        setMetrics(metricList);
        setLowerIsBetter(new Set(metricsData?.lower_is_better || []));
        if (metricList.length > 0) {
          setMetricX((prev) => prev || metricList[0]);
        }
        if (metricList.length > 1) {
          setMetricY((prev) => prev || metricList[1]);
        }
      } catch (err) {
        console.error(err);
        setError(err.message || "Failed to load metadata");
      }
    };
    loadMeta();
  }, []);

  useEffect(() => {
    setIsClient(true);
  }, []);

  useEffect(() => {
    if (!metricX || !metricY) return;
    const handle = setTimeout(async () => {
      setLoading(true);
      setError("");
      try {
        const data = await fetchJson("/stats-research", {
          metric_x: metricX,
          metric_y: metricY,
          league: selectedLeague !== "All leagues" ? selectedLeague : undefined,
          season: selectedSeason || undefined,
          positions: positionsAll ? undefined : selectedPositions.join(","),
          min_minutes: minMinutes,
        });
        setRows(data || []);
      } catch (err) {
        console.error(err);
        setError(err.message || "Failed to load stats");
      } finally {
        setLoading(false);
      }
    }, 200);
    return () => clearTimeout(handle);
  }, [metricX, metricY, selectedLeague, selectedSeason, positionsAll, selectedPositions, minMinutes]);

  useEffect(() => {
    if (!metricX || !metricY || metricX !== metricY) return;
    const fallback = metrics.find((metric) => metric !== metricX);
    if (fallback) setMetricY(fallback);
  }, [metricX, metricY, metrics]);

  useEffect(() => {
    if (!positionsOpen) return;
    const updatePosition = () => {
      const rect = positionsButtonRef.current?.getBoundingClientRect();
      if (!rect) return;
      setPositionsMenuStyle({
        position: "fixed",
        top: rect.bottom + 8,
        left: rect.left,
        width: rect.width,
        zIndex: 9999,
      });
    };
    updatePosition();
    window.addEventListener("scroll", updatePosition, true);
    window.addEventListener("resize", updatePosition);
    return () => {
      window.removeEventListener("scroll", updatePosition, true);
      window.removeEventListener("resize", updatePosition);
    };
  }, [positionsOpen]);

  useEffect(() => {
    if (!positionsOpen) return;
    const handleClickOutside = (event) => {
      if (
        positionsMenuRef.current?.contains(event.target) ||
        positionsButtonRef.current?.contains(event.target)
      ) {
        return;
      }
      setPositionsOpen(false);
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [positionsOpen]);

  const metricXLabel = formatMetricLabel(metricX);
  const metricYLabel = formatMetricLabel(metricY);

  const normalizedRows = useMemo(() => {
    const unique = new Map();
    (rows || []).forEach((row) => {
      const metricXVal = Number(row.metric_x);
      const metricYVal = Number(row.metric_y);
      if (!Number.isFinite(metricXVal) || !Number.isFinite(metricYVal)) return;
      const key = rowKey(row);
      const existing = unique.get(key);
      if (!existing) {
        unique.set(key, {
          ...row,
          metric_x: metricXVal,
          metric_y: metricYVal,
        });
        return;
      }
      if (metricXVal > existing.metric_x || (metricXVal === existing.metric_x && metricYVal > existing.metric_y)) {
        unique.set(key, {
          ...row,
          metric_x: metricXVal,
          metric_y: metricYVal,
        });
      }
    });
    return Array.from(unique.values());
  }, [rows]);

  const highlightInfo = useMemo(() => {
    const valuesX = normalizedRows.map((row) => row.metric_x);
    const valuesY = normalizedRows.map((row) => row.metric_y);
    const medianX = percentile(valuesX, 50);
    const medianY = percentile(valuesY, 50);
    const xLower = lowerIsBetter.has(metricX);
    const yLower = lowerIsBetter.has(metricY);
    const thresholdX = percentile(valuesX, xLower ? 25 : 75);
    const thresholdY = percentile(valuesY, yLower ? 25 : 75);
    let highlights = normalizedRows.filter(
      (row) =>
        isGoodValue(row.metric_x, thresholdX, xLower) &&
        isGoodValue(row.metric_y, thresholdY, yLower)
    );
    if (normalizedRows.length > 0) {
      const bestRow = normalizedRows.reduce((acc, row) => {
        if (!acc) return row;
        if (xLower) {
          return row.metric_x < acc.metric_x ? row : acc;
        }
        return row.metric_x > acc.metric_x ? row : acc;
      }, null);
      if (bestRow && !highlights.includes(bestRow)) {
        highlights = [bestRow, ...highlights];
      }
    }
    const unique = new Map();
    highlights.forEach((row) => {
      const key = rowKey(row);
      if (!unique.has(key)) {
        unique.set(key, row);
      }
    });
    const highlightedRows = Array.from(unique.values()).sort((a, b) => {
      const xRank = xLower ? a.metric_x - b.metric_x : b.metric_x - a.metric_x;
      if (xRank !== 0) return xRank;
      return yLower ? a.metric_y - b.metric_y : b.metric_y - a.metric_y;
    });
    const keys = new Set(highlightedRows.map(rowKey));
    return {
      highlights: highlightedRows,
      labelKeys: new Set(highlightedRows.slice(0, 14).map(rowKey)),
      keys,
      medianX,
      medianY,
      thresholdX,
      thresholdY,
      xLower,
      yLower,
    };
  }, [normalizedRows, lowerIsBetter, metricX, metricY]);

  const chartTitle = useMemo(() => {
    let title = `${metricXLabel} vs ${metricYLabel} - ${selectedLeague}`;
    if (!positionsAll && selectedPositions.length > 0) {
      const labels = selectedPositions
        .map((code) => POSITIONS_GLOSSARY[code] || code)
        .join(", ");
      title += ` | Positions: ${labels}`;
    }
    if (selectedSeason) {
      title += ` | Season: ${selectedSeason}`;
    }
    title += ` | Min ${minMinutes} mins`;
    return title;
  }, [metricXLabel, metricYLabel, selectedLeague, selectedSeason, positionsAll, selectedPositions, minMinutes]);

  const scatterData = useMemo(() => {
    const baseRows = normalizedRows.filter((row) => !highlightInfo.keys.has(rowKey(row)));
    const highlightedRows = highlightInfo.highlights;
    const toCustom = (row) => [
      row.name || "Unknown",
      row.team || "Unknown",
      row.competition_name || "Unknown",
      row.calendar || selectedSeason || "Unknown season",
      row.player_id || "",
    ];
    const baseTrace = {
      x: baseRows.map((row) => row.metric_x),
      y: baseRows.map((row) => row.metric_y),
      type: "scatter",
      mode: "markers",
      name: "Cohort",
      marker: {
        color: "rgba(160,168,163,0.42)",
        size: 8,
        opacity: 0.72,
        line: { color: "rgba(255,255,255,0.10)", width: 0.8 },
      },
      customdata: baseRows.map(toCustom),
      hovertemplate:
        "<b>%{customdata[0]}</b><br>" +
        "%{customdata[1]} · %{customdata[2]}<br>" +
        "%{customdata[3]}<br><br>" +
        `<span style="color:#A0A8A3">${metricXLabel}</span>: %{x:.2f}<br>` +
        `<span style="color:#A0A8A3">${metricYLabel}</span>: %{y:.2f}` +
        "<extra></extra>",
    };
    const highlightTrace = {
      x: highlightedRows.map((row) => row.metric_x),
      y: highlightedRows.map((row) => row.metric_y),
      type: "scatter",
      mode: "markers+text",
      name: "Target quadrant",
      text: highlightedRows.map((row) => (highlightInfo.labelKeys.has(rowKey(row)) ? row.name || "" : "")),
      textposition: "top center",
      textfont: { color: "#DDF3E8", size: 10, family: "Inter, system-ui, sans-serif" },
      marker: {
        color: "#559A78",
        size: 11,
        opacity: 0.96,
        line: { color: "rgba(221,243,232,0.78)", width: 1.25 },
      },
      customdata: highlightedRows.map(toCustom),
      hovertemplate:
        "<b>%{customdata[0]}</b><br>" +
        "%{customdata[1]} · %{customdata[2]}<br>" +
        "%{customdata[3]}<br><br>" +
        `<span style="color:#8CC7A7">Target quadrant</span><br>` +
        `<span style="color:#A0A8A3">${metricXLabel}</span>: %{x:.2f}<br>` +
        `<span style="color:#A0A8A3">${metricYLabel}</span>: %{y:.2f}` +
        "<extra></extra>",
      showlegend: false,
    };
    return [baseTrace, highlightTrace];
  }, [normalizedRows, highlightInfo, metricXLabel, metricYLabel, selectedSeason]);

  const plotLayout = useMemo(() => {
    const xValues = normalizedRows.map((row) => row.metric_x).filter(Number.isFinite);
    const yValues = normalizedRows.map((row) => row.metric_y).filter(Number.isFinite);
    const [xMin, xMax] = axisRange(xValues);
    const [yMin, yMax] = axisRange(yValues);
    const medianX = highlightInfo.medianX;
    const medianY = highlightInfo.medianY;
    const thresholdX = highlightInfo.thresholdX;
    const thresholdY = highlightInfo.thresholdY;
    const shapes = [];
    const annotations = [];
    if (Number.isFinite(thresholdX) && Number.isFinite(thresholdY)) {
      shapes.push({
        type: "rect",
        xref: "x",
        yref: "y",
        x0: highlightInfo.xLower ? xMin : thresholdX,
        x1: highlightInfo.xLower ? thresholdX : xMax,
        y0: highlightInfo.yLower ? yMin : thresholdY,
        y1: highlightInfo.yLower ? thresholdY : yMax,
        fillcolor: "rgba(85,154,120,0.08)",
        line: { width: 0 },
        layer: "below",
      });
      annotations.push({
        text: "Target quadrant",
        x: highlightInfo.xLower ? xMin : xMax,
        y: highlightInfo.yLower ? yMin : yMax,
        xref: "x",
        yref: "y",
        xanchor: highlightInfo.xLower ? "left" : "right",
        yanchor: highlightInfo.yLower ? "bottom" : "top",
        showarrow: false,
        font: { size: 11, color: "#8CC7A7" },
        bgcolor: "rgba(47,125,92,0.14)",
        bordercolor: "rgba(85,154,120,0.22)",
        borderpad: 4,
      });
    }
    if (Number.isFinite(medianX)) {
      shapes.push({
        type: "line",
        x0: medianX,
        x1: medianX,
        y0: 0,
        y1: 1,
        xref: "x",
        yref: "paper",
        line: { color: "rgba(243,245,244,0.38)", width: 1.25, dash: "dot" },
      });
      annotations.push({
        text: `Median ${metricXLabel}`,
        x: medianX,
        y: 1.01,
        xref: "x",
        yref: "paper",
        showarrow: false,
        font: { size: 10, color: "rgba(243,245,244,0.58)" },
        yanchor: "bottom",
      });
    }
    if (Number.isFinite(medianY)) {
      shapes.push({
        type: "line",
        x0: 0,
        x1: 1,
        y0: medianY,
        y1: medianY,
        xref: "paper",
        yref: "y",
        line: { color: "rgba(243,245,244,0.38)", width: 1.25, dash: "dot" },
      });
      annotations.push({
        text: `Median ${metricYLabel}`,
        x: 1.01,
        y: medianY,
        xref: "paper",
        yref: "y",
        showarrow: false,
        font: { size: 10, color: "rgba(243,245,244,0.58)" },
        xanchor: "left",
      });
    }
    return {
      autosize: true,
      plot_bgcolor: "rgba(5,7,6,0)",
      paper_bgcolor: "rgba(5,7,6,0)",
      font: { color: "#A0A8A3", family: "Inter, system-ui, sans-serif" },
      margin: { l: 56, r: 48, t: 26, b: 54 },
      hovermode: "closest",
      dragmode: "zoom",
      clickmode: "event+select",
      hoverlabel: {
        bgcolor: "#0A0C0B",
        bordercolor: "#3A8967",
        font: { color: "#F3F5F4", size: 12 },
        align: "left",
      },
      xaxis: {
        title: { text: metricXLabel, font: { size: 12, color: "#DDE3DF" } },
        range: [xMin, xMax],
        gridcolor: "rgba(255,255,255,0.055)",
        zeroline: false,
        linecolor: "rgba(255,255,255,0.14)",
        tickcolor: "rgba(255,255,255,0.14)",
        tickfont: { size: 11, color: "#8D968F" },
        ticks: "outside",
      },
      yaxis: {
        title: { text: metricYLabel, font: { size: 12, color: "#DDE3DF" } },
        range: [yMin, yMax],
        gridcolor: "rgba(255,255,255,0.055)",
        zeroline: false,
        linecolor: "rgba(255,255,255,0.14)",
        tickcolor: "rgba(255,255,255,0.14)",
        tickfont: { size: 11, color: "#8D968F" },
        ticks: "outside",
      },
      showlegend: false,
      shapes,
      annotations,
    };
  }, [highlightInfo, metricXLabel, metricYLabel, normalizedRows]);

  const tableRows = normalizedRows
    .map((row) => ({
      key: `${row.name || ""}|${row.team || ""}|${row.competition_name || ""}`,
      label: `${row.name || "Unknown"} - ${row.team || "Unknown"} - ${row.competition_name || "Unknown"}`,
      player_id: row.player_id,
      metric_x: row.metric_x,
      metric_y: row.metric_y,
    }))
    .sort((a, b) => {
      const sortKey = tableSort.key;
      const direction = tableSort.dir === "asc" ? 1 : -1;
      if (sortKey === "player") {
        const order = a.label.localeCompare(b.label, undefined, { sensitivity: "base" });
        if (order !== 0) return order * direction;
        if (a.metric_x === b.metric_x) return (a.metric_y - b.metric_y) * direction;
        return (a.metric_x - b.metric_x) * direction;
      }
      const aVal = sortKey === "metric_x" ? a.metric_x : a.metric_y;
      const bVal = sortKey === "metric_x" ? b.metric_x : b.metric_y;
      if (aVal === bVal) {
        const fallbackKey = sortKey === "metric_x" ? "metric_y" : "metric_x";
        const fallbackA = fallbackKey === "metric_x" ? a.metric_x : a.metric_y;
        const fallbackB = fallbackKey === "metric_x" ? b.metric_x : b.metric_y;
        return (fallbackA - fallbackB) * direction;
      }
      return (aVal - bVal) * direction;
    });

  const positionsSummary = positionsAll
    ? "All"
    : selectedPositions.map((code) => POSITIONS_GLOSSARY[code] || code).join(", ");

  const activeFilterCount = [
    selectedLeague !== DEFAULT_SCOUTING_COMPETITION,
    selectedSeason !== DEFAULT_SCOUTING_SEASON,
    !positionsAll,
    minMinutes !== DEFAULT_MIN_MINUTES,
  ].filter(Boolean).length;

  const resetFilters = () => {
    setSelectedLeague(DEFAULT_SCOUTING_COMPETITION);
    setSelectedSeason(DEFAULT_SCOUTING_SEASON);
    setPositionsAll(true);
    setSelectedPositions([]);
    setMinMinutes(DEFAULT_MIN_MINUTES);
  };

  const updateMinMinutes = (value) => {
    setMinMinutes(parseIntegerInput(value, 0));
  };

  const handleTableSort = (key) => {
    setTableSort((prev) => {
      if (prev.key === key) {
        return { key, dir: prev.dir === "asc" ? "desc" : "asc" };
      }
      const defaultDir = key === "player" ? "asc" : "desc";
      return { key, dir: defaultDir };
    });
  };

  const openPointReport = (event) => {
    const playerId = event?.points?.[0]?.customdata?.[4];
    if (!playerId) return;
    window.open(`/report/${playerId}`, "_blank", "noopener,noreferrer");
  };

  const renderPositionsMenu = () => {
    if (!positionsOpen || !isClient) return null;
    return createPortal(
      <div
        ref={positionsMenuRef}
        className="max-h-64 overflow-auto rounded-lg border border-slate-700 bg-slate-900/95 shadow-2xl p-3"
        style={positionsMenuStyle}
      >
        <label className="flex items-center gap-2 text-sm text-slate-200 mb-2">
          <input
            type="checkbox"
            checked={positionsAll}
            onChange={() => {
              setPositionsAll(true);
              setSelectedPositions([]);
            }}
          />
          All
        </label>
        <div className="space-y-2">
          {positions.map((pos) => (
            <label key={pos.code} className="flex items-center gap-2 text-sm text-slate-200">
              <input
                type="checkbox"
                checked={!positionsAll && selectedPositions.includes(pos.code)}
                onChange={() => {
                  setPositionsAll(false);
                  setSelectedPositions((prev) => {
                    if (prev.includes(pos.code)) {
                      return prev.filter((item) => item !== pos.code);
                    }
                    return [...prev, pos.code];
                  });
                }}
              />
              {pos.label}
            </label>
          ))}
        </div>
      </div>,
      document.body
    );
  };

  return (
    <main className="nl-page py-10 px-4">
      <div className="max-w-6xl mx-auto space-y-6">
        <header className="flex flex-col gap-2">
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
            Stats Research
          </p>
          <h1 className="text-4xl font-bold text-white tracking-tight">
            Metric discovery lab
          </h1>
          <p className="text-slate-300 max-w-3xl">
            Explore performance relationships across leagues, positions and minutes thresholds to uncover stronger scouting angles.
          </p>
        </header>

        <Card className="nl-filter-bar p-0">
          <div className="flex flex-col gap-4 border-b border-white/10 px-4 py-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="nl-kicker">Research filters</p>
              <h2 className="mt-1 text-lg font-semibold text-white">Define the analytical cohort</h2>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-md border border-[#3A8967]/30 bg-[#2F7D5C]/15 px-3 py-2 text-xs font-semibold text-[#8CC7A7]">
                {normalizedRows.length} players
              </span>
              <span className="rounded-md border border-white/10 bg-white/[0.04] px-3 py-2 text-xs font-semibold text-slate-500">
                {activeFilterCount} custom filters
              </span>
              <button type="button" className="nl-button-secondary px-3 py-2 text-xs" onClick={resetFilters}>
                Reset
              </button>
            </div>
          </div>

          <div className="space-y-4 p-4">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
            <div className="flex flex-col gap-2">
              <Label>League</Label>
              <Select id="stats-research-league" value={selectedLeague} onChange={(e) => setSelectedLeague(e.target.value)}>
                {leagueOptions.map((league) => (
                  <option key={league} value={league}>
                    {league}
                  </option>
                ))}
              </Select>
            </div>
            <div className="flex flex-col gap-2">
              <Label>Season</Label>
              <Select id="stats-research-season" value={selectedSeason} onChange={(e) => setSelectedSeason(e.target.value)}>
                <option value="">All seasons</option>
                {seasons.map((season) => (
                  <option key={season} value={season}>
                    {season}
                  </option>
                ))}
              </Select>
            </div>
            <div className="flex flex-col gap-2 relative">
              <Label>Positions</Label>
              <button
                type="button"
                onClick={() => setPositionsOpen((prev) => !prev)}
                className="nl-field text-left"
                ref={positionsButtonRef}
              >
                {positionsSummary || "All"}
              </button>
              {renderPositionsMenu()}
            </div>
            <div className="flex flex-col gap-2">
              <Label>Minimum minutes</Label>
              <input
                type="number"
                inputMode="numeric"
                min="0"
                max="10000"
                step="1"
                value={minMinutes}
                onChange={(e) => updateMinMinutes(e.target.value)}
                className="nl-field tabular-nums"
              />
            </div>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
            <div className="flex flex-col gap-2">
              <Label>Axis X</Label>
              <Select value={metricX} onChange={(e) => setMetricX(e.target.value)}>
                {metrics.map((metric) => (
                  <option key={metric} value={metric}>
                    {formatMetricLabel(metric)}
                  </option>
                ))}
              </Select>
            </div>
            <div className="flex flex-col gap-2">
              <Label>Axis Y</Label>
              <Select value={metricY} onChange={(e) => setMetricY(e.target.value)}>
                {metrics
                  .filter((metric) => metric !== metricX)
                  .map((metric) => (
                    <option key={metric} value={metric}>
                      {formatMetricLabel(metric)}
                    </option>
                  ))}
              </Select>
            </div>
          </div>
          <div className="flex items-center justify-end mt-4">
            <button
              type="button"
              onClick={() => {
                if (!metricX || !metricY) return;
                setMetricX(metricY);
                setMetricY(metricX);
              }}
              className="nl-button-secondary px-3 py-2 text-xs"
            >
              Inverse Axis
            </button>
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-2">
            {[
              ["League", formatFilterValue(selectedLeague, "All leagues")],
              ["Season", formatFilterValue(selectedSeason, "All seasons")],
              ["Positions", positionsSummary || "All"],
              ["Minutes", `${minMinutes}+`],
            ].map(([label, value]) => (
              <span key={label} className="rounded-md border border-white/10 bg-white/[0.035] px-2.5 py-1.5 text-[11px] font-semibold text-slate-500">
                <span className="text-white/45">{label}</span> <span className="text-white/80">{value}</span>
              </span>
            ))}
          </div>
          </div>
        </Card>

        {error && (
          <Card>
            <p className="text-danger">Error: {error}</p>
          </Card>
        )}

        {loading ? (
          <Card>
            <p className="text-slate-400">Loading stats...</p>
          </Card>
        ) : normalizedRows.length === 0 ? (
          <Card>
            <p className="text-slate-400">No players found with current filters.</p>
          </Card>
        ) : (
          <>
            <Card className="overflow-hidden p-0">
              <div className="flex flex-col gap-3 border-b border-white/10 px-4 py-4 md:flex-row md:items-center md:justify-between">
                <div>
                  <p className="text-sm font-semibold text-white">{chartTitle}</p>
                  <p className="mt-1 text-xs font-semibold text-slate-500">
                    Median guides, target quadrant and Q3 highlights are computed from the current cohort.
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-md border border-white/10 bg-white/[0.035] px-3 py-2 text-xs font-semibold text-slate-500">
                    {normalizedRows.length} plotted
                  </span>
                  <span className="rounded-md border border-[#3A8967]/30 bg-[#2F7D5C]/15 px-3 py-2 text-xs font-semibold text-[#8CC7A7]">
                    {highlightInfo.highlights.length} target
                  </span>
                </div>
              </div>
              <div className="h-[590px] px-2 py-3">
                <Plot
                  data={scatterData}
                  layout={plotLayout}
                  style={{ width: "100%", height: "100%" }}
                  config={{
                    displayModeBar: "hover",
                    displaylogo: false,
                    responsive: true,
                    scrollZoom: true,
                    modeBarButtonsToRemove: ["lasso2d", "select2d"],
                  }}
                  onClick={openPointReport}
                />
              </div>
            </Card>

            <Card>
              <h3 className="text-lg font-semibold text-white mb-3">Results</h3>
              <div className="overflow-auto max-h-[520px] pr-2">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-slate-900/95">
                    <tr className="text-xs uppercase text-slate-400 border-b border-white/5">
                      <th className="text-left py-2">
                        <button
                          type="button"
                          className="cursor-pointer select-none hover:text-slate-200"
                          onClick={() => handleTableSort("player")}
                        >
                          Player - Club - League {sortIndicator(tableSort, "player")}
                        </button>
                      </th>
                      <th className="text-left py-2">
                        <button
                          type="button"
                          className="cursor-pointer select-none hover:text-slate-200"
                          onClick={() => handleTableSort("metric_x")}
                        >
                          {metricXLabel} {sortIndicator(tableSort, "metric_x")}
                        </button>
                      </th>
                      <th className="text-left py-2">
                        <button
                          type="button"
                          className="cursor-pointer select-none hover:text-slate-200"
                          onClick={() => handleTableSort("metric_y")}
                        >
                          {metricYLabel} {sortIndicator(tableSort, "metric_y")}
                        </button>
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {tableRows.map((row) => {
                      const isHighlight = highlightInfo.keys.has(row.key);
                      return (
                        <tr
                          key={row.key}
                          className={`border-b border-white/5 ${
                            isHighlight ? "bg-emerald-400/20 text-emerald-50" : ""
                          }`}
                        >
                          <td className="py-2 text-slate-200">
                            {row.player_id ? (
                              <button
                                type="button"
                                className="text-left hover:text-emerald-200"
                                onClick={() =>
                                  window.open(
                                    `/report/${row.player_id}`,
                                    "_blank",
                                    "noopener,noreferrer"
                                  )
                                }
                              >
                                {row.label}
                              </button>
                            ) : (
                              row.label
                            )}
                          </td>
                          <td className="py-2 text-slate-300">{formatValue(row.metric_x)}</td>
                          <td className="py-2 text-slate-300">{formatValue(row.metric_y)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Card>
          </>
        )}
      </div>
    </main>
  );
}
