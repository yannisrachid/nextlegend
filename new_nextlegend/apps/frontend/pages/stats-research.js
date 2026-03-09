import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import dynamic from "next/dynamic";
import { fetchJson, fetchJsonCached } from "@/lib/api";
import { METRIC_LABELS } from "@/lib/metricLabels";
import { POSITIONS_GLOSSARY } from "@/lib/positionsGlossary";

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

const Select = ({ value, onChange, children }) => (
  <select
    className="bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100"
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

const seasonSortKey = (value) => {
  if (!value) return -Infinity;
  const raw = String(value).trim();
  if (!raw) return -Infinity;
  const range = raw.match(/^(\d{4})\s*[\/-]\s*(\d{2,4})$/);
  if (range) {
    const start = Number(range[1]);
    const endRaw = range[2];
    let end = Number(endRaw.length === 2 ? `${range[1].slice(0, 2)}${endRaw}` : endRaw.slice(-4));
    if (end < start) end += 100;
    return start * 10000 + end;
  }
  const single = raw.match(/^(\d{4})$/);
  if (single) {
    const year = Number(single[1]);
    return year * 10000 + year;
  }
  const digits = raw.match(/(\d{4})/g);
  if (digits?.length) return Number(digits[0]) * 10000 + Number(digits[digits.length - 1]);
  return -Infinity;
};

const sortSeasonValues = (values) =>
  [...(values || [])].sort((a, b) => {
    const diff = seasonSortKey(b) - seasonSortKey(a);
    if (diff !== 0) return diff;
    return String(b || "").localeCompare(String(a || ""), undefined, { sensitivity: "base" });
  });

export default function StatsResearchPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [leagueOptions, setLeagueOptions] = useState(["All leagues"]);
  const [selectedLeague, setSelectedLeague] = useState("All leagues");
  const [seasons, setSeasons] = useState([]);
  const [selectedSeason, setSelectedSeason] = useState("");
  const [positions, setPositions] = useState([]);
  const [positionsOpen, setPositionsOpen] = useState(false);
  const [positionsAll, setPositionsAll] = useState(true);
  const [selectedPositions, setSelectedPositions] = useState([]);
  const [minMinutes, setMinMinutes] = useState(270);
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
        const unique = [];
        const seen = new Set();
        compNames.forEach((name) => {
          if (seen.has(name)) return;
          seen.add(name);
          if (name === "Big 5 Leagues" || name === "Big 10 Competitions" || name === "First Divisions Only" || name === "Second Divisions Only") {
            if ((competitions || []).find((item) => item.name === name)?.seasons?.length) {
              unique.push(name);
            }
            return;
          }
          unique.push(name);
        });
        setLeagueOptions(["All leagues", ...unique]);
        setSeasons(sortSeasonValues((seasonsData || []).filter(Boolean)));
        const positionList = (positionsData || []).map((code) => ({
          code,
          label: POSITIONS_GLOSSARY[code] || code,
        }));
        positionList.sort((a, b) => a.label.localeCompare(b.label, undefined, { sensitivity: "base" }));
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
      const key = `${row.name || ""}|${row.team || ""}|${row.competition_name || ""}`;
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
    const thresholdX = percentile(valuesX, 85);
    const thresholdY = percentile(valuesY, 50);
    const xLower = lowerIsBetter.has(metricX);
    const yLower = lowerIsBetter.has(metricY);
    let highlights = [];
    if (thresholdX !== null && thresholdY !== null && !xLower && !yLower) {
      highlights = normalizedRows.filter(
        (row) => row.metric_x >= thresholdX && row.metric_y >= thresholdY
      );
    }
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
      const key = `${row.name || ""}|${row.team || ""}|${row.competition_name || ""}`;
      if (!unique.has(key)) {
        unique.set(key, row);
      }
    });
    const limited = Array.from(unique.values()).slice(0, 20);
    const keys = new Set(unique.keys());
    return { highlights: limited, keys };
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
    const baseCustom = normalizedRows.map((row) => [
      row.name || "Unknown",
      row.team || "Unknown",
      row.competition_name || "Unknown",
    ]);
    const highlightCustom = highlightInfo.highlights.map((row) => [
      row.name || "Unknown",
      row.team || "Unknown",
      row.competition_name || "Unknown",
    ]);
    const baseTrace = {
      x: normalizedRows.map((row) => row.metric_x),
      y: normalizedRows.map((row) => row.metric_y),
      type: "scatter",
      mode: "markers",
      marker: { color: "#60A5FA", size: 9, opacity: 0.35 },
      customdata: baseCustom,
      hovertemplate:
        "<b>%{customdata[0]}</b><br>%{customdata[1]} - %{customdata[2]}" +
        `<br>${metricXLabel}: %{x:.2f}<br>${metricYLabel}: %{y:.2f}<extra></extra>`,
    };
    const highlightTrace = {
      x: highlightInfo.highlights.map((row) => row.metric_x),
      y: highlightInfo.highlights.map((row) => row.metric_y),
      type: "scatter",
      mode: "markers+text",
      text: highlightInfo.highlights.map((row) => row.name || ""),
      textposition: "top center",
      textfont: { color: "#F8FAFC", size: 11 },
      marker: { color: "#34D399", size: 12, opacity: 0.95, line: { color: "#064E3B", width: 1.5 } },
      customdata: highlightCustom,
      hovertemplate:
        "<b>%{customdata[0]}</b><br>%{customdata[1]} - %{customdata[2]}" +
        `<br>${metricXLabel}: %{x:.2f}<br>${metricYLabel}: %{y:.2f}<extra></extra>`,
      showlegend: false,
    };
    return [baseTrace, highlightTrace];
  }, [normalizedRows, highlightInfo, metricXLabel, metricYLabel]);

  const plotLayout = useMemo(() => {
    const xValues = normalizedRows.map((row) => row.metric_x).filter(Number.isFinite);
    const yValues = normalizedRows.map((row) => row.metric_y).filter(Number.isFinite);
    const medianX = percentile(xValues, 50);
    const medianY = percentile(yValues, 50);
    const shapes = [];
    if (Number.isFinite(medianX)) {
      shapes.push({
        type: "line",
        x0: medianX,
        x1: medianX,
        y0: 0,
        y1: 1,
        xref: "x",
        yref: "paper",
        line: { color: "rgba(148,163,184,0.45)", width: 1.5, dash: "dot" },
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
        line: { color: "rgba(148,163,184,0.45)", width: 1.5, dash: "dot" },
      });
    }
    return {
      title: { text: chartTitle, font: { size: 14, color: "#E2E8F0" } },
      plot_bgcolor: "#0F172A",
      paper_bgcolor: "#0F172A",
      font: { color: "#E2E8F0" },
      margin: { l: 40, r: 20, t: 60, b: 40 },
      xaxis: { title: metricXLabel, gridcolor: "rgba(255,255,255,0.08)" },
      yaxis: { title: metricYLabel, gridcolor: "rgba(255,255,255,0.08)" },
      showlegend: false,
      shapes,
    };
  }, [chartTitle, metricXLabel, metricYLabel, normalizedRows]);

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

  const handleTableSort = (key) => {
    setTableSort((prev) => {
      if (prev.key === key) {
        return { key, dir: prev.dir === "asc" ? "desc" : "asc" };
      }
      const defaultDir = key === "player" ? "asc" : "desc";
      return { key, dir: defaultDir };
    });
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
    <main className="min-h-screen bg-hero-pattern text-slate-100 py-10 px-4">
      <div className="max-w-6xl mx-auto space-y-6">
        <header className="flex flex-col gap-2">
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
            Stats Research
          </p>
          <h1 className="text-4xl font-bold text-white tracking-tight">
            Stats Research
          </h1>
          <p className="text-slate-300 max-w-3xl">
            Explore metric relationships with league and position filters.
          </p>
        </header>

        <Card>
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
            <div className="flex flex-col gap-2">
              <Label>League</Label>
              <Select value={selectedLeague} onChange={(e) => setSelectedLeague(e.target.value)}>
                {leagueOptions.map((league) => (
                  <option key={league} value={league}>
                    {league}
                  </option>
                ))}
              </Select>
            </div>
            <div className="flex flex-col gap-2">
              <Label>Season</Label>
              <Select value={selectedSeason} onChange={(e) => setSelectedSeason(e.target.value)}>
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
                className="bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-left text-slate-100"
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
                min="0"
                step="30"
                value={minMinutes}
                onChange={(e) => setMinMinutes(Number(e.target.value))}
                className="bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100"
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
              className="px-3 py-2 rounded-md text-xs uppercase tracking-[0.2em] border border-slate-700 text-slate-200 hover:border-slate-500"
            >
              Inverse Axis
            </button>
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
            <Card>
              <div className="h-[560px]">
                <Plot
                  data={scatterData}
                  layout={plotLayout}
                  style={{ width: "100%", height: "100%" }}
                  config={{ displayModeBar: false, responsive: true }}
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
