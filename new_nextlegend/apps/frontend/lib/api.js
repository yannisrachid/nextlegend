const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
const cacheStore = new Map();

export async function fetchJson(path, params = {}) {
  const url = new URL(path, API_BASE);
  Object.entries(params).forEach(([key, val]) => {
    if (val === undefined || val === null || val === "") return;
    url.searchParams.set(key, val);
  });
  const res = await fetch(url.toString());
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

export async function fetchJsonCached(
  path,
  params = {},
  { ttlMs = 5 * 60 * 1000 } = {}
) {
  const cacheKey = `${path}?${JSON.stringify(params)}`;
  const cached = cacheStore.get(cacheKey);
  if (cached && cached.expiresAt > Date.now()) {
    return cached.data;
  }
  const data = await fetchJson(path, params);
  cacheStore.set(cacheKey, { data, expiresAt: Date.now() + ttlMs });
  return data;
}
