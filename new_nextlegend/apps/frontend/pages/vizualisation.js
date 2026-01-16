import { useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { createPortal } from "react-dom";
import { fetchJson, fetchJsonCached, postJson } from "@/lib/api";
import { METRIC_LABELS } from "@/lib/metricLabels";
import { POSITIONS_GLOSSARY } from "@/lib/positionsGlossary";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const GLOBAL_THRESHOLD = 75;

const SECTIONS = [
  {
    key: "goal_creation",
    title: "Goal Creation",
    metrics: [
      "xa_per_90",
      "key_passes_per_90",
      "smart_passes_per_90",
      "shot_assists_per_90",
      "passes_to_penalty_area_per_90",
      "deep_completions_per_90",
      "through_passes_per_90",
      "progressive_passes_per_90",
    ],
  },
  {
    key: "attacking_threat",
    title: "Attacking Threat",
    metrics: [
      "goals_per_90",
      "xg_per_90",
      "shots_per_90",
      "shots_on_target_percent",
      "touches_in_penalty_area_per_90",
      "progressive_runs_per_90",
      "accelerations_per_90",
      "non_penalty_goals_per_90",
    ],
  },
  {
    key: "crossing_delivery",
    title: "Crossing & Delivery",
    metrics: [
      "crosses_per_90",
      "accurate_crosses_percent",
      "deep_crosses_per_90",
      "crosses_to_goalkeeper_per_90",
      "crosses_to_box_per_90",
      "crosses_to_penalty_area_per_90",
    ],
  },
  {
    key: "defensive_contribution",
    title: "Defensive Contribution",
    metrics: [
      "successful_def_actions_per_90",
      "def_duels_per_90",
      "def_duels_won_percent",
      "interceptions_per_90",
      "sliding_tackles_per_90",
      "aerial_duels_won_percent",
      "blocked_shots_per_90",
      "recoveries_per_90",
    ],
  },
  {
    key: "pressing_activity",
    title: "Pressing & Work Rate",
    metrics: [
      "offensive_duels_per_90",
      "duels_per_90",
      "pressures_per_90",
      "fouls_per_90",
      "counterpressing_recoveries_per_90",
      "ball_recoveries_in_final_third_per_90",
    ],
  },
  {
    key: "build_up_play",
    title: "Build-Up & Progression",
    metrics: [
      "passes_per_90",
      "accurate_passes_percent",
      "progressive_passes_per_90",
      "passes_to_final_third_per_90",
      "passes_to_penalty_area_per_90",
      "progressive_carries_per_90",
      "long_passes_per_90",
      "accurate_long_passes_percent",
    ],
  },
  {
    key: "possession_retention",
    title: "Possession Retention",
    metrics: [
      "dribbles_per_90",
      "successful_dribbles_percent",
      "ball_losses_per_90",
      "dispossessed_per_90",
      "miscontrols_per_90",
      "passes_received_per_90",
    ],
  },
];

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

const buildDynamicSections = (percentiles, enforceTop) => {
  if (!enforceTop) return { sections: SECTIONS, reason: "Full analytical breakdown retained." };
  const sections = [];
  SECTIONS.forEach((section) => {
    const metrics = section.metrics.filter((metric) => (percentiles?.[metric] ?? 0) >= GLOBAL_THRESHOLD);
    if (metrics.length >= 2) {
      sections.push({ ...section, metrics });
    }
  });
  if (sections.length > 0) {
    return {
      sections,
      reason: "Focus on metrics above the 75th percentile.",
    };
  }
  const highImpact = Object.entries(percentiles || {})
    .filter(([, value]) => value >= GLOBAL_THRESHOLD)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([metric]) => metric);
  if (highImpact.length > 0) {
    sections.push({ key: "high_impact", title: "High Impact Metrics", metrics: highImpact });
  }
  return {
    sections,
    reason: "Focus on the highest impact metrics available.",
  };
};

const PizzaChart = ({ labels, values, playerLabel, subtitle }) => {
  const graphRef = useRef(null);
  const [exporting, setExporting] = useState(false);
  const [copyStatus, setCopyStatus] = useState("");
  const [showZoom, setShowZoom] = useState(false);
  const rounded = values.map((value) => Math.round(value));
  const colors = rounded.map((value) => (value >= GLOBAL_THRESHOLD ? "#01ca0a" : "#4ecc54"));
  const sliceCount = Math.max(labels.length, 1);
  const sliceWidth = 360 / sliceCount;
  const angles = labels.map((_, index) => index * sliceWidth + sliceWidth / 2);
  const textRadius = rounded.map((value) => {
      const clamped = Math.max(0, Math.min(100, value));
      if (clamped >= 98) return 96;
      if (clamped <= 8) return clamped + 2;
      return clamped;
    });
  const textSizes = rounded.map((value) => {
    if (value < 20) return 9;
    if (value < 40) return 11;
    return 12;
  });
  const badgeSizes = rounded.map((value) => {
    if (value < 20) return 18;
    if (value < 40) return 22;
    return 26;
  });
  const data = [
    {
      type: "barpolar",
      r: rounded,
      theta: angles,
      width: labels.map(() => sliceWidth),
        marker: { color: colors, line: { color: "#0B1120", width: 1.6 } },
        opacity: 0.95,
        hoverinfo: "skip",
      },
    {
      type: "scatterpolar",
      r: textRadius,
      theta: angles,
      mode: "markers+text",
      marker: {
        color: colors,
        size: badgeSizes,
        line: { color: "#0B1120", width: 1 },
      },
      text: rounded.map((value) => `${value}`),
      textfont: { color: "#FFFFFF", size: textSizes },
      textposition: "middle center",
      cliponaxis: false,
      hoverinfo: "skip",
    },
  ];
    const layout = {
      polar: {
        domain: { x: [0.08, 0.92], y: [0.2, 0.92] },
        radialaxis: {
          range: [0, 100],
          showticklabels: false,
        ticks: "",
        showline: false,
          gridcolor: "rgba(255,255,255,0.08)",
          showgrid: true,
        },
        angularaxis: {
          tickmode: "array",
          tickvals: angles,
          ticktext: labels,
          tickfont: { color: "#E2E8F0", size: 10 },
          rotation: 90,
          direction: "clockwise",
          showline: false,
          showgrid: false,
        },
        bgcolor: "#0F172A",
      },
      showlegend: false,
      margin: { l: 90, r: 90, t: 160, b: 170 },
      paper_bgcolor: "#0F172A",
      plot_bgcolor: "#0F172A",
      annotations: [
        {
          text: playerLabel,
          x: 0.5,
          y: 1.12,
          xref: "paper",
          yref: "paper",
          showarrow: false,
          font: { size: 16, color: "#F2F2F2" },
        },
        {
          text: subtitle,
          x: 0.5,
          y: 1.06,
          xref: "paper",
          yref: "paper",
          showarrow: false,
          font: { size: 12, color: "#F2F2F2" },
        },
        {
          text: "Data: StatsBomb / Your Legend",
          x: 0.98,
          y: -0.08,
          xref: "paper",
          yref: "paper",
          showarrow: false,
          font: { size: 9, color: "#F2F2F2" },
          align: "right",
        },
        {
          text: "Statistics scaled per 90 minutes",
          x: 0.02,
          y: -0.14,
          xref: "paper",
          yref: "paper",
          showarrow: false,
          font: { size: 9, color: "#F2F2F2" },
          align: "left",
        },
      ],
    };
  const getPlotly = async () => {
    const module = await import("plotly.js-dist-min");
    return module.default ?? module;
  };
  const exportImage = async () => {
    if (!graphRef.current) return null;
    const Plotly = await getPlotly();
    return Plotly.toImage(graphRef.current, {
      format: "png",
      scale: 2,
    });
  };
  const zoomLayout = {
    ...layout,
    margin: { l: 120, r: 120, t: 210, b: 180 },
    polar: {
      ...layout.polar,
      domain: { x: [0.08, 0.92], y: [0.2, 0.9] },
    },
  };
  const handleCopy = async (event) => {
    event.stopPropagation();
    if (exporting) return;
    setExporting(true);
    setCopyStatus("");
    try {
      const dataUrl = await exportImage();
      if (!dataUrl) return;
      let copied = false;
      if (navigator.clipboard?.write && window.ClipboardItem && window.isSecureContext) {
        try {
          const response = await fetch(dataUrl);
          const blob = await response.blob();
          await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
          copied = true;
        } catch (error) {
          copied = false;
        }
      }
      if (!copied && navigator.clipboard?.writeText) {
        try {
          await navigator.clipboard.writeText(dataUrl);
          copied = true;
        } catch (error) {
          copied = false;
        }
      }
      if (!copied) {
        setCopyStatus("Copy blocked by browser. Please use Download.");
        return;
      }
      setCopyStatus("Copied!");
      setTimeout(() => setCopyStatus(""), 1800);
    } finally {
      setExporting(false);
    }
  };
  const handleDownload = async (event) => {
    event.stopPropagation();
    if (exporting) return;
    setExporting(true);
    try {
      const dataUrl = await exportImage();
      if (!dataUrl) return;
      const link = document.createElement("a");
      link.href = dataUrl;
      link.download = `${playerLabel || "pizza"}-${subtitle || "chart"}.png`.replace(/\s+/g, "_");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } finally {
      setExporting(false);
    }
  };
    return (
    <div className="relative h-[780px] rounded-xl overflow-hidden bg-[#0F172A]">
      <div className="absolute right-3 top-3 z-10 flex items-center gap-2">
        <button
          type="button"
          className="text-xs uppercase tracking-[0.2em] text-slate-200 border border-white/10 px-3 py-1 rounded-full hover:border-emerald-400"
          onClick={handleCopy}
        >
          Copy
        </button>
        <button
          type="button"
          className="text-xs uppercase tracking-[0.2em] text-slate-200 border border-white/10 px-3 py-1 rounded-full hover:border-emerald-400"
          onClick={handleDownload}
        >
          Download
        </button>
      </div>
      {copyStatus ? (
        <div className="absolute right-3 top-12 z-10 text-xs text-slate-300">
          {copyStatus}
        </div>
      ) : null}
      <div className="h-full cursor-zoom-in" onClick={() => setShowZoom(true)}>
        <Plot
          data={data}
          layout={layout}
          style={{ width: "100%", height: "100%" }}
          config={{ displayModeBar: false, responsive: true }}
          onInitialized={(_, graphDiv) => {
            graphRef.current = graphDiv;
          }}
          onUpdate={(_, graphDiv) => {
            graphRef.current = graphDiv;
          }}
        />
      </div>
      {showZoom && typeof document !== "undefined"
        ? createPortal(
            <div
              className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-6"
              onClick={() => setShowZoom(false)}
            >
              <div
                className="relative w-full max-w-5xl h-[82vh] bg-slate-950 border border-white/10 rounded-2xl shadow-xl"
                onClick={(event) => event.stopPropagation()}
              >
                <div className="absolute right-4 top-4 flex items-center gap-2">
                  <button
                    type="button"
                    className="text-xs uppercase tracking-[0.2em] text-slate-200 border border-white/10 px-3 py-1 rounded-full hover:border-emerald-400"
                    onClick={handleCopy}
                  >
                    Copy
                  </button>
                  <button
                    type="button"
                    className="text-xs uppercase tracking-[0.2em] text-slate-200 border border-white/10 px-3 py-1 rounded-full hover:border-emerald-400"
                    onClick={handleDownload}
                  >
                    Download
                  </button>
                  <button
                    type="button"
                    className="text-xs uppercase tracking-[0.2em] text-slate-200 border border-white/10 px-3 py-1 rounded-full hover:border-emerald-400"
                    onClick={() => setShowZoom(false)}
                  >
                    Close
                  </button>
                </div>
                {copyStatus ? (
                  <div className="absolute right-4 top-14 text-xs text-slate-300">
                    {copyStatus}
                  </div>
                ) : null}
                <div className="h-full pt-12">
                  <Plot
                    data={data}
                    layout={zoomLayout}
                    style={{ width: "100%", height: "100%" }}
                    config={{ displayModeBar: false, responsive: true }}
                  />
                </div>
              </div>
            </div>,
            document.body
          )
        : null}
    </div>
  );
};

export default function VizualisationPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [playerQuery, setPlayerQuery] = useState("");
  const [playerResults, setPlayerResults] = useState([]);
  const [selectedPlayerId, setSelectedPlayerId] = useState("");
  const [showResults, setShowResults] = useState(false);
  const [vizData, setVizData] = useState(null);
  const [context, setContext] = useState("League");
  const [positions, setPositions] = useState([]);
  const [positionsOpen, setPositionsOpen] = useState(false);
  const [positionsAll, setPositionsAll] = useState(true);
  const [selectedPositions, setSelectedPositions] = useState([]);
  const [minMinutes, setMinMinutes] = useState(270);
  const [showOnlyTopGlobal, setShowOnlyTopGlobal] = useState(false);
  const [sectionSelections, setSectionSelections] = useState({});
  const [sectionShowTop, setSectionShowTop] = useState({});
  const positionsButtonRef = useRef(null);
  const positionsMenuRef = useRef(null);
  const [positionsMenuStyle, setPositionsMenuStyle] = useState({});
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  useEffect(() => {
    fetchJsonCached("/meta/positions")
      .then((data) => {
        const list = (data || []).map((code) => ({
          code,
          label: POSITIONS_GLOSSARY[code] || code,
        }));
        list.sort((a, b) => a.label.localeCompare(b.label, undefined, { sensitivity: "base" }));
        setPositions(list);
      })
      .catch((err) => console.error(err));
  }, []);

  useEffect(() => {
    if (!playerQuery || playerQuery.trim().length < 2) {
      setPlayerResults([]);
      setSelectedPlayerId("");
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
      setVizData(null);
      return;
    }
    const metrics = Array.from(new Set(SECTIONS.flatMap((section) => section.metrics)));
    const load = async () => {
      setLoading(true);
      setError("");
      try {
        const data = await postJson("/viz/percentiles", {
          player_id: Number(selectedPlayerId),
          metrics,
          context,
          positions: positionsAll ? [] : selectedPositions,
          min_minutes: minMinutes,
        });
        setVizData(data);
      } catch (err) {
        setError(err.message || "Failed to load percentiles");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [selectedPlayerId, context, positionsAll, selectedPositions, minMinutes]);

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

  useEffect(() => {
    setSectionSelections({});
    setSectionShowTop({});
  }, [selectedPlayerId, context, positionsAll, selectedPositions, minMinutes, showOnlyTopGlobal]);

  const playerOptions = useMemo(() => {
    return playerResults.map((player) => ({
      id: String(player.id),
      label: `${player.name} - ${player.team || "--"} - ${player.competition_name || "--"} - ${player.calendar || "--"}`,
    }));
  }, [playerResults]);

  const handlePlayerSelect = (player) => {
    setSelectedPlayerId(player.id);
    setPlayerQuery(player.label);
    setShowResults(false);
  };

  const percentiles = vizData?.percentiles || {};
  const values = vizData?.values || {};
  const playerInfo = vizData?.player || {};
  const playerLabel = playerInfo.name
    ? `${playerInfo.name} - ${playerInfo.team || "--"}`
    : "Player";
  const peerGroup = context === "League" ? `${playerInfo.competition_name || "League"} peer group` : "global peer group";
  const subtitleSuffix = `Percentile vs ${peerGroup} (min ${minMinutes} mins)`;

  const { sections: activeSections, reason } = useMemo(
    () => buildDynamicSections(percentiles, showOnlyTopGlobal),
    [percentiles, showOnlyTopGlobal]
  );

  const positionsSummary = positionsAll
    ? "All"
    : selectedPositions.map((code) => POSITIONS_GLOSSARY[code] || code).join(", ");

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
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Vizualisation</p>
          <h1 className="text-4xl font-bold text-white tracking-tight">Vizualisation</h1>
          <p className="text-slate-300 max-w-3xl">
            Recompute percentiles within a custom cohort and explore pizza charts by theme.
          </p>
        </header>

        <Card className="relative z-30">
          <div className="relative">
            <div className="flex flex-col gap-2">
              <Label>Player</Label>
              <input
                className="bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100"
                placeholder="Start typing a player name..."
                value={playerQuery}
                onChange={(e) => {
                  setPlayerQuery(e.target.value);
                  setSelectedPlayerId("");
                  setShowResults(true);
                }}
                onFocus={() => {
                  if (selectedPlayerId) {
                    setPlayerQuery("");
                    setSelectedPlayerId("");
                    setVizData(null);
                  }
                  setShowResults(true);
                }}
                onClick={() => {
                  if (selectedPlayerId) {
                    setPlayerQuery("");
                    setSelectedPlayerId("");
                    setVizData(null);
                    setShowResults(true);
                  }
                }}
                onBlur={() => setTimeout(() => setShowResults(false), 150)}
              />
            </div>
            {showResults && playerQuery.trim().length >= 2 ? (
              <div className="absolute z-50 mt-2 w-full max-h-72 overflow-auto rounded-lg border border-slate-700 bg-slate-900/95 shadow-xl">
                {playerOptions.length === 0 ? (
                  <div className="px-3 py-2 text-sm text-slate-400">No matches found.</div>
                ) : (
                  playerOptions.map((player) => (
                    <button
                      key={player.id}
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

        <Card>
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
            <div className="flex flex-col gap-2">
              <Label>Context</Label>
              <Select value={context} onChange={(e) => setContext(e.target.value)}>
                <option value="League">League</option>
                <option value="Global">Global</option>
              </Select>
            </div>
            <div className="flex flex-col gap-2 relative">
              <Label>Compare against positions</Label>
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
              <Label>Minimum minutes played</Label>
              <input
                type="number"
                min="0"
                step="30"
                value={minMinutes}
                onChange={(e) => setMinMinutes(Number(e.target.value))}
                className="bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100"
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label>Generate with only +75 percentile values</Label>
              <label className="flex items-center gap-2 text-sm text-slate-300">
                <input
                  type="checkbox"
                  className="accent-emerald-400"
                  checked={showOnlyTopGlobal}
                  onChange={(e) => setShowOnlyTopGlobal(e.target.checked)}
                />
                Enabled
              </label>
            </div>
          </div>
          {vizData ? (
          <p className="text-xs text-slate-400 mt-3">
            Scout Analyst Engine: {reason}
          </p>
          ) : null}
        </Card>

        {error && (
          <Card>
            <p className="text-danger">Error: {error}</p>
          </Card>
        )}

        {loading ? (
          <Card>
            <p className="text-slate-400">Loading visualisation...</p>
          </Card>
        ) : !selectedPlayerId ? (
          <Card>
            <p className="text-slate-400">Select a player to see visualisations.</p>
          </Card>
        ) : activeSections.length === 0 ? (
          <Card>
            <p className="text-slate-400">No relevant metrics found for the current filters.</p>
          </Card>
        ) : (
          activeSections.map((section) => {
            const sectionKey = section.key;
            const availableMetrics = section.metrics.filter((metric) => percentiles[metric] != null);
            if (availableMetrics.length === 0) {
              return (
                <Card key={sectionKey}>
                  <p className="text-slate-400">No metrics available for {section.title}.</p>
                </Card>
              );
            }
            const defaultSelection = availableMetrics.slice(0, Math.min(availableMetrics.length, 8));
            const currentSelection = sectionSelections[sectionKey] || defaultSelection;
            const showOnlyTop = sectionShowTop[sectionKey] ?? showOnlyTopGlobal;
            const metricsToPlot = currentSelection.filter((metric) => {
              const value = percentiles[metric];
              if (value == null) return false;
              if (showOnlyTop && value < GLOBAL_THRESHOLD) return false;
              return true;
            });
            const labels = metricsToPlot.map(formatMetricLabel);
            const valuesToPlot = metricsToPlot.map((metric) => percentiles[metric]);

            return (
              <div key={sectionKey}>
                <div className="border-t border-white/5 my-6" />
                <h3 className="text-xl font-semibold text-white mb-4">{section.title}</h3>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Card>
                    {metricsToPlot.length === 0 ? (
                      <p className="text-slate-400">No metrics to display with current filters.</p>
                    ) : (
                      <div className="w-full">
                        <PizzaChart
                          labels={labels}
                          values={valuesToPlot}
                          playerLabel={playerLabel}
                          subtitle={`${section.title} - ${subtitleSuffix}`}
                        />
                      </div>
                    )}
                  </Card>
                  <Card>
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-sm uppercase tracking-[0.2em] text-slate-400">Metrics</h4>
                      <label className="flex items-center gap-2 text-xs text-slate-300">
                        <input
                          type="checkbox"
                          className="accent-emerald-400"
                          checked={showOnlyTop}
                          onChange={(e) =>
                            setSectionShowTop((prev) => ({
                              ...prev,
                              [sectionKey]: e.target.checked,
                            }))
                          }
                        />
                        Show only +75 percentile values
                      </label>
                    </div>
                    <div className="max-h-[360px] overflow-auto space-y-2">
                      {availableMetrics.map((metric) => {
                        const isChecked = currentSelection.includes(metric);
                        const percentile = percentiles[metric];
                        const value = values[metric];
                        return (
                          <label key={metric} className="flex items-center justify-between gap-3 text-sm text-slate-200">
                            <span className="flex items-center gap-2">
                              <input
                                type="checkbox"
                                checked={isChecked}
                                onChange={() => {
                                  setSectionSelections((prev) => {
                                    const base = prev[sectionKey] || defaultSelection;
                                    if (base.includes(metric)) {
                                      return { ...prev, [sectionKey]: base.filter((item) => item !== metric) };
                                    }
                                    return { ...prev, [sectionKey]: [...base, metric] };
                                  });
                                }}
                              />
                              {formatMetricLabel(metric)}
                            </span>
                            <span className="text-xs text-slate-400">
                              {percentile != null ? Math.round(percentile) : "--"} / {value != null ? Number(value).toFixed(2) : "--"}
                            </span>
                          </label>
                        );
                      })}
                    </div>
                  </Card>
                </div>
              </div>
            );
          })
        )}
      </div>
    </main>
  );
}
