const ENV_API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL;
const FALLBACK_API_BASE = "http://localhost:8000";
const INTERNAL_DOCKER_HOSTS = ["http://api:8000", "https://api:8000"];

const resolveApiBase = () => {
  if (typeof window !== "undefined") {
    if (!ENV_API_BASE || INTERNAL_DOCKER_HOSTS.includes(ENV_API_BASE)) {
      const hostname = window.location.hostname === "0.0.0.0"
        ? "localhost"
        : window.location.hostname;
      return `${window.location.protocol}//${hostname}:8000`;
    }
  }
  return ENV_API_BASE || FALLBACK_API_BASE;
};
const cacheStore = new Map();

export async function fetchJson(path, params = {}) {
  const url = new URL(path, resolveApiBase());
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
