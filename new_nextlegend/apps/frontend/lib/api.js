const ENV_API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL;
const FALLBACK_API_BASE = "http://localhost:8000";
const INTERNAL_DOCKER_HOSTS = ["http://api:8000", "https://api:8000"];

const resolveApiBase = () => {
  if (typeof window !== "undefined") {
    if (!ENV_API_BASE || INTERNAL_DOCKER_HOSTS.includes(ENV_API_BASE)) {
      const hostname = window.location.hostname;
      if (hostname === "app.nextlegend.fr") {
        return "https://api.nextlegend.fr";
      }
      const localHost = hostname === "0.0.0.0" ? "localhost" : hostname;
      return `${window.location.protocol}//${localHost}:8000`;
    }
  }
  return ENV_API_BASE || FALLBACK_API_BASE;
};
const cacheStore = new Map();

export function apiUrl(path, params = {}) {
  const url = new URL(path, resolveApiBase());
  Object.entries(params).forEach(([key, val]) => {
    if (val === undefined || val === null || val === "") return;
    url.searchParams.set(key, val);
  });
  return url.toString();
}

export async function fetchJson(path, params = {}) {
  const res = await fetch(apiUrl(path, params), { credentials: "include" });
  if (!res.ok) {
    const text = await res.text();
    if (
      res.status === 401 &&
      typeof window !== "undefined" &&
      !path.startsWith("/auth")
    ) {
      window.location.href = "/login";
    }
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

export async function postJson(path, body = {}) {
  const url = new URL(path, resolveApiBase());
  const res = await fetch(url.toString(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    credentials: "include",
  });
  if (!res.ok) {
    const text = await res.text();
    if (
      res.status === 401 &&
      typeof window !== "undefined" &&
      !path.startsWith("/auth")
    ) {
      window.location.href = "/login";
    }
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

export async function deleteJson(path) {
  const url = new URL(path, resolveApiBase());
  const res = await fetch(url.toString(), {
    method: "DELETE",
    credentials: "include",
  });
  if (!res.ok) {
    const text = await res.text();
    if (
      res.status === 401 &&
      typeof window !== "undefined" &&
      !path.startsWith("/auth")
    ) {
      window.location.href = "/login";
    }
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

export async function patchJson(path, body = {}) {
  const url = new URL(path, resolveApiBase());
  const res = await fetch(url.toString(), {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    credentials: "include",
  });
  if (!res.ok) {
    const text = await res.text();
    if (
      res.status === 401 &&
      typeof window !== "undefined" &&
      !path.startsWith("/auth")
    ) {
      window.location.href = "/login";
    }
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
