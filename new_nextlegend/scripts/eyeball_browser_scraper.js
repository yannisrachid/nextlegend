(() => {
  const DEFAULT_EXPECTED_TOTAL = 20495;
  const DEFAULT_DELAY_MS = 700;
  const DEFAULT_MAX_PAGES = 2500;

  const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));
  const clean = (value) => String(value || "").replace(/\s+/g, " ").trim();
  const visible = (element) => {
    if (!element) return false;
    const style = window.getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    return style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
  };

  const csvEscape = (value) => {
    const text = String(value ?? "");
    return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };

  const downloadCsv = (rows, headers, fileName) => {
    const csv = [
      headers.map(csvEscape).join(","),
      ...rows.map((row) => headers.map((header) => csvEscape(row[header])).join(",")),
    ].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

  const byColIndex = (a, b) => {
    const ai = Number(a.getAttribute("aria-colindex") || a.getAttribute("col-index") || 0);
    const bi = Number(b.getAttribute("aria-colindex") || b.getAttribute("col-index") || 0);
    if (ai || bi) return ai - bi;
    return a.getBoundingClientRect().left - b.getBoundingClientRect().left;
  };

  const extractors = [
    {
      name: "html-table",
      headers: () => [...document.querySelectorAll("table thead th")]
        .filter(visible)
        .map((node) => clean(node.innerText)),
      rows: () => [...document.querySelectorAll("table tbody tr")]
        .filter(visible)
        .map((row) => [...row.querySelectorAll("td, th")]
          .filter(visible)
          .map((cell) => clean(cell.innerText))),
    },
    {
      name: "ag-grid",
      headers: () => [...document.querySelectorAll(".ag-header-cell:not(.ag-header-cell-group)")]
        .filter(visible)
        .sort(byColIndex)
        .map((node) => clean(node.querySelector(".ag-header-cell-text")?.innerText || node.innerText)),
      rows: () => {
        const rowsByIndex = new Map();
        const containers = [
          ".ag-pinned-left-cols-container .ag-row",
          ".ag-center-cols-container .ag-row",
          ".ag-pinned-right-cols-container .ag-row",
        ];
        containers.forEach((selector) => {
          [...document.querySelectorAll(selector)].filter(visible).forEach((row) => {
            const index = row.getAttribute("row-index") || row.getAttribute("aria-rowindex") || clean(row.innerText);
            const cells = [...row.querySelectorAll(".ag-cell")].filter(visible).sort(byColIndex);
            const existing = rowsByIndex.get(index) || [];
            rowsByIndex.set(index, [...existing, ...cells.map((cell) => clean(cell.innerText))]);
          });
        });
        return [...rowsByIndex.values()].filter((row) => row.some(Boolean));
      },
    },
    {
      name: "mui-data-grid",
      headers: () => [...document.querySelectorAll(".MuiDataGrid-columnHeader")]
        .filter(visible)
        .sort(byColIndex)
        .map((node) => clean(node.querySelector(".MuiDataGrid-columnHeaderTitle")?.innerText || node.innerText)),
      rows: () => [...document.querySelectorAll(".MuiDataGrid-row")]
        .filter(visible)
        .map((row) => [...row.querySelectorAll(".MuiDataGrid-cell")]
          .filter(visible)
          .sort(byColIndex)
          .map((cell) => clean(cell.innerText))),
    },
    {
      name: "aria-grid",
      headers: () => [...document.querySelectorAll('[role="columnheader"]')]
        .filter(visible)
        .sort(byColIndex)
        .map((node) => clean(node.innerText)),
      rows: () => [...document.querySelectorAll('[role="row"]')]
        .filter(visible)
        .map((row) => [...row.querySelectorAll('[role="gridcell"], [role="cell"]')]
          .filter(visible)
          .sort(byColIndex)
          .map((cell) => clean(cell.innerText)))
        .filter((row) => row.length > 0),
    },
  ];

  const getExtractor = () => {
    const ranked = extractors
      .map((extractor) => ({
        extractor,
        headers: extractor.headers().filter(Boolean),
        rows: extractor.rows().filter((row) => row.some(Boolean)),
      }))
      .sort((a, b) => (b.headers.length + b.rows.length * 2) - (a.headers.length + a.rows.length * 2));
    return ranked[0];
  };

  const normalizeHeaders = (headers, width) => {
    const base = headers.length ? headers : Array.from({ length: width }, (_, index) => `column_${index + 1}`);
    return base.map((header, index) => clean(header) || `column_${index + 1}`);
  };

  const pageRowsToRecords = (rows, headers) => rows.map((values) => {
    const record = {};
    headers.forEach((header, index) => {
      record[header] = values[index] || "";
    });
    return record;
  });

  const findNextButton = () => {
    const controls = [...document.querySelectorAll("button, a, [role='button'], .pagination-next, [aria-label], [title]")]
      .filter(visible);
    const candidates = controls.filter((node) => {
      const label = clean([
        node.innerText,
        node.getAttribute("aria-label"),
        node.getAttribute("title"),
        node.getAttribute("data-testid"),
        node.className,
      ].join(" ")).toLowerCase();
      return (
        /\b(next|suivant|following|page suivante|go to next)\b/.test(label)
        || label === ">"
        || label === "\u203a"
        || label.includes("chevron-right")
        || label.includes("pagination-next")
      );
    });
    return candidates.find((node) => !isDisabled(node)) || candidates[0] || null;
  };

  const isDisabled = (node) => {
    if (!node) return true;
    return Boolean(
      node.disabled
      || node.getAttribute("aria-disabled") === "true"
      || node.className.toString().toLowerCase().includes("disabled")
      || node.closest("[disabled], [aria-disabled='true']"),
    );
  };

  const currentSignature = () => {
    const active = getExtractor();
    return active.rows.slice(0, 5).map((row) => row.join("|")).join("||");
  };

  const waitForPageChange = async (previousSignature, timeoutMs = 8000) => {
    const startedAt = Date.now();
    while (Date.now() - startedAt < timeoutMs) {
      await sleep(150);
      if (currentSignature() && currentSignature() !== previousSignature) return true;
    }
    return false;
  };

  const scrapePage = () => {
    const active = getExtractor();
    const maxWidth = Math.max(active.headers.length, ...active.rows.map((row) => row.length), 0);
    const headers = normalizeHeaders(active.headers, maxWidth);
    const rows = active.rows
      .filter((row) => row.some(Boolean))
      .map((row) => row.slice(0, headers.length));
    return {
      extractor: active.extractor.name,
      headers,
      rows: pageRowsToRecords(rows, headers),
      rawRows: rows,
      rowCount: rows.length,
    };
  };

  const inspect = () => {
    const result = scrapePage();
    const next = findNextButton();
    const scrollContainers = [...document.querySelectorAll("*")]
      .filter((node) => visible(node) && node.scrollWidth > node.clientWidth + 20)
      .slice(0, 8)
      .map((node) => ({
        tag: node.tagName.toLowerCase(),
        className: clean(node.className),
        scrollWidth: node.scrollWidth,
        clientWidth: node.clientWidth,
      }));
    console.log("Eyeball scraper inspection", {
      extractor: result.extractor,
      headers: result.headers,
      rowCount: result.rowCount,
      firstRow: result.rawRows[0],
      nextButton: next ? {
        tag: next.tagName.toLowerCase(),
        text: clean(next.innerText),
        ariaLabel: next.getAttribute("aria-label"),
        className: clean(next.className),
        disabled: isDisabled(next),
      } : null,
      horizontalScrollContainers: scrollContainers,
    });
    console.table(result.rows.slice(0, 5));
    return result;
  };

  const scrapeAll = async ({
    expectedTotal = DEFAULT_EXPECTED_TOTAL,
    delayMs = DEFAULT_DELAY_MS,
    maxPages = DEFAULT_MAX_PAGES,
    fileName = `eyeball_player_stats_${new Date().toISOString().slice(0, 10)}.csv`,
  } = {}) => {
    const allRows = [];
    let headers = [];

    for (let page = 1; page <= maxPages; page += 1) {
      const result = scrapePage();
      if (!result.rowCount) {
        throw new Error(`No rows detected on page ${page}. Run nextLegendEyeballScraper.inspect() first.`);
      }
      headers = headers.length >= result.headers.length ? headers : result.headers;
      result.rows.forEach((row) => allRows.push(row));

      console.log(`[Eyeball] page ${page}: +${result.rowCount} rows, total ${allRows.length}`);
      if (expectedTotal && allRows.length >= expectedTotal) break;

      const next = findNextButton();
      if (!next || isDisabled(next)) break;
      const previousSignature = currentSignature();
      next.click();
      await sleep(delayMs);
      await waitForPageChange(previousSignature);
    }

    downloadCsv(allRows, headers, fileName);
    console.log(`[Eyeball] done: ${allRows.length} rows exported to ${fileName}`);
    return { rows: allRows.length, headers, fileName };
  };

  const flatten = (value, prefix = "", out = {}) => {
    if (Array.isArray(value)) {
      out[prefix] = value.map((item) => (typeof item === "object" ? JSON.stringify(item) : String(item ?? ""))).join(" | ");
      return out;
    }
    if (value && typeof value === "object") {
      const entries = Object.entries(value);
      if (!entries.length && prefix) out[prefix] = "";
      entries.forEach(([key, child]) => flatten(child, prefix ? `${prefix}_${key}` : key, out));
      return out;
    }
    if (prefix) out[prefix] = value ?? "";
    return out;
  };

  const createPanel = () => {
    document.getElementById("nextlegend-eyeball-download-panel")?.remove();
    const panel = document.createElement("div");
    panel.id = "nextlegend-eyeball-download-panel";
    panel.style.cssText = [
      "position:fixed",
      "right:18px",
      "bottom:18px",
      "z-index:2147483647",
      "background:#050706",
      "color:#f3f5f4",
      "border:1px solid rgba(255,255,255,.16)",
      "border-radius:10px",
      "padding:14px 16px",
      "font:13px system-ui,sans-serif",
      "box-shadow:0 16px 60px rgba(0,0,0,.4)",
      "min-width:310px",
      "max-width:420px",
    ].join(";");
    panel.innerHTML = '<div style="font-weight:700;margin-bottom:6px">Next Legend Eyeball export</div><div data-status>Starting...</div>';
    document.body.appendChild(panel);
    return {
      panel,
      setStatus: (html) => {
        panel.querySelector("[data-status]").innerHTML = html;
      },
    };
  };

  const createDownloadLink = (panel, content, fileName, contentType, label) => {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = fileName;
    link.textContent = label;
    link.style.cssText = [
      "display:inline-flex",
      "margin-top:10px",
      "margin-right:8px",
      "background:#3A8967",
      "color:#fff",
      "text-decoration:none",
      "border-radius:7px",
      "padding:8px 10px",
      "font-weight:700",
    ].join(";");
    panel.appendChild(link);
    return { fileName, bytes: blob.size };
  };

  const readCapturedPlayerSearchPayload = (expectedTotal) => {
    const logs = [...(window.__nextLegendEyeballNetworkLogs || [])].reverse();
    const match = logs.find((log) => {
      try {
        const body = JSON.parse(log.requestBody || "{}");
        const response = JSON.parse(log.responseText || "{}");
        return log.status === 200 && (!expectedTotal || response.total === expectedTotal) && body.pagination;
      } catch (_) {
        return false;
      }
    });
    return match ? JSON.parse(match.requestBody) : null;
  };

  const scrapeApiToDownloadPanel = ({
    endpoint = "https://portal-api.eyeball.club/portal/player/search",
    expectedTotal = 20495,
    season = 2026,
    countryCodes = ["FR"],
    limit = 50,
    concurrency = 4,
    basePayload = null,
  } = {}) => {
    const runId = `eyeball_${season}_${countryCodes.join("-").toLowerCase()}_${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}`;
    const payloadTemplate = basePayload
      || readCapturedPlayerSearchPayload(expectedTotal)
      || { sort: [], season, player: { countryCodes }, stats: {}, pagination: { page: 0, limit: 18 } };
    const { panel, setStatus } = createPanel();
    const state = {
      runId,
      status: "running",
      startedAt: new Date().toISOString(),
      expectedTotal,
      limit,
      concurrency,
      totalPages: null,
      pagesFetched: 0,
      rowsFetched: 0,
      errors: [],
      csvBytes: 0,
      csvName: null,
    };
    window.__nextLegendEyeballMemoryExport = state;

    const fetchPage = async (page, attempt = 1) => {
      const payload = JSON.parse(JSON.stringify(payloadTemplate));
      payload.pagination = { ...(payload.pagination || {}), page, limit };
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "content-type": "application/json", accept: "application/json" },
        credentials: "include",
        body: JSON.stringify(payload),
      });
      const text = await response.text();
      if (!response.ok) {
        if (attempt < 5) {
          await sleep(800 * attempt);
          return fetchPage(page, attempt + 1);
        }
        throw new Error(`Page ${page} failed with ${response.status}: ${text.slice(0, 200)}`);
      }
      const json = JSON.parse(text);
      return { page, total: json.total, pages: json.pages, items: json._embedded?.items || [] };
    };

    (async () => {
      try {
        const first = await fetchPage(0);
        const lastPage = Number(first.pages || Math.ceil(first.total / limit) - 1);
        state.totalPages = lastPage + 1;
        setStatus(`Fetching 1/${state.totalPages} pages...`);
        const results = [first];
        state.pagesFetched = 1;
        state.rowsFetched = first.items.length;
        const pages = [];
        for (let page = 1; page <= lastPage; page += 1) pages.push(page);
        let cursor = 0;
        const workers = Array.from({ length: concurrency }, async () => {
          while (cursor < pages.length) {
            const page = pages[cursor++];
            const result = await fetchPage(page);
            results.push(result);
            state.pagesFetched += 1;
            state.rowsFetched += result.items.length;
            if (state.pagesFetched % 10 === 0 || state.pagesFetched === state.totalPages) {
              setStatus(`Fetching ${state.pagesFetched}/${state.totalPages} pages...<br>${state.rowsFetched} rows`);
            }
          }
        });
        await Promise.all(workers);
        results.sort((a, b) => a.page - b.page);
        const rawRows = results.flatMap((result) => result.items.map((item, index) => ({
          item,
          apiPage: result.page,
          apiRowIndex: index,
        })));
        const flatRows = rawRows.map(({ item, apiPage, apiRowIndex }, globalIndex) => {
          const flat = flatten(item);
          flat.api_page = apiPage;
          flat.api_row_index = apiRowIndex;
          flat.export_row_index = globalIndex;
          if (item.id) flat.player_url = `https://portal.eyeball.club/player/${item.id}`;
          return flat;
        });
        const priority = [
          "export_row_index",
          "api_page",
          "api_row_index",
          "id",
          "player_url",
          "firstName",
          "lastName",
          "birthday",
          "clubName",
          "teamName",
          "teamAgeGroup",
          "teamLevel",
          "countryCode",
          "nationality_countryCode",
          "nationality_countryLabel",
          "position",
          "positionCount",
          "strongFoot",
          "height",
          "weight",
          "gamesCount",
          "minutesPlayed",
          "rating_value",
        ];
        const headerSet = new Set(flatRows.flatMap((row) => Object.keys(row)));
        const headers = [
          ...priority.filter((key) => headerSet.has(key)),
          ...[...headerSet].filter((key) => !priority.includes(key)).sort(),
        ];
        const csv = [
          headers.map(csvEscape).join(","),
          ...flatRows.map((row) => headers.map((header) => csvEscape(row[header])).join(",")),
        ].join("\n");
        const metadata = {
          runId,
          endpoint,
          exportedAt: new Date().toISOString(),
          basePayload: payloadTemplate,
          expectedTotal,
          apiTotal: first.total,
          requestedLimit: limit,
          apiLastPage: lastPage,
          pagesFetched: results.length,
          exportedRows: flatRows.length,
          headers,
        };
        const today = new Date().toISOString().slice(0, 10);
        const csvName = `eyeball_player_stats_${season}_${countryCodes.join("-").toLowerCase()}_full_${flatRows.length}_${today}.csv`;
        const metaName = `eyeball_player_stats_${season}_${countryCodes.join("-").toLowerCase()}_full_${flatRows.length}_${today}.metadata.json`;
        const csvLink = createDownloadLink(panel, csv, csvName, "text/csv;charset=utf-8", "Download CSV");
        createDownloadLink(panel, JSON.stringify(metadata, null, 2), metaName, "application/json;charset=utf-8", "Download metadata");
        Object.assign(state, {
          status: "ready",
          finishedAt: new Date().toISOString(),
          exportedRows: flatRows.length,
          headersCount: headers.length,
          csvName,
          metaName,
          csvBytes: csvLink.bytes,
        });
        window.__nextLegendEyeballCsv = csv;
        window.__nextLegendEyeballMetadata = metadata;
        setStatus(`Ready: ${flatRows.length} rows, ${headers.length} columns.<br>Click Download CSV below.`);
      } catch (error) {
        state.status = "failed";
        state.finishedAt = new Date().toISOString();
        state.errors.push(String(error?.message || error));
        setStatus(`Failed:<br>${String(error?.message || error)}`);
      }
    })();

    return state;
  };

  window.nextLegendEyeballScraper = { inspect, scrapePage, scrapeAll, scrapeApiToDownloadPanel };
  console.log("Next Legend Eyeball scraper loaded. Run nextLegendEyeballScraper.inspect() or nextLegendEyeballScraper.scrapeApiToDownloadPanel().");
})();
