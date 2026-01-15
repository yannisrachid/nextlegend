import "@/styles/globals.css";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/router";
import { fetchJson, postJson } from "@/lib/api";
import { AuthContext } from "@/lib/auth";

const NAV_ITEMS = [
  { href: "/", label: "Home" },
  { href: "/ranking", label: "Ranking" },
  { href: "/report", label: "Report" },
  { href: "/comparison", label: "Comparison" },
  { href: "/projection", label: "Projection" },
  { href: "/stats-research", label: "Stats Research" },
  { href: "/vizualisation", label: "Vizualisation" },
  { href: "/prospect", label: "Prospect" },
  { href: "/ai", label: "AI" },
];

export default function App({ Component, pageProps }) {
  const router = useRouter();
  const isLogin = router.pathname === "/login";
  const [me, setMe] = useState(null);
  const [authStatus, setAuthStatus] = useState("loading");

  const handleLogout = async () => {
    try {
      await postJson("/auth/logout");
    } catch (err) {
      console.error(err);
    } finally {
      window.location.href = "/login";
    }
  };

  const refreshAuth = useCallback(async () => {
    let active = true;
    setAuthStatus("loading");
    try {
      const data = await fetchJson("/auth/me");
      if (!active) return null;
      setMe(data);
      setAuthStatus("authenticated");
      return data;
    } catch (err) {
      if (!active) return null;
      setMe(null);
      setAuthStatus("unauthenticated");
      return null;
    } finally {
      active = false;
    }
  }, []);

  useEffect(() => {
    refreshAuth();
  }, [refreshAuth]);

  useEffect(() => {
    if (authStatus === "loading") return;
    if (isLogin && authStatus === "authenticated") {
      router.replace("/");
      return;
    }
    if (!isLogin && authStatus === "unauthenticated") {
      router.replace("/login");
    }
  }, [authStatus, isLogin, router]);

  const navItems = me?.role === "admin"
    ? [...NAV_ITEMS, { href: "/admin", label: "Admin" }]
    : NAV_ITEMS;

  return (
    <AuthContext.Provider value={{ me, status: authStatus, refreshAuth }}>
      {!isLogin ? (
        <header className="sticky top-0 z-40 border-b border-white/5 bg-slate-900/80 backdrop-blur">
          <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-primary text-lg font-semibold text-glow">
                NextLegend
              </span>
              <span className="text-xs uppercase tracking-[0.3em] text-slate-500">
                v2
              </span>
            </div>
            <nav className="flex items-center gap-4">
              {navItems.map((item) => {
                const isActive = router.pathname === item.href;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`text-sm font-medium ${
                      isActive
                        ? "text-primary"
                        : "text-slate-300 hover:text-white"
                    }`}
                  >
                    {item.label}
                  </Link>
                );
              })}
              <button
                type="button"
                onClick={handleLogout}
                className="text-sm font-medium text-slate-300 hover:text-white"
              >
                Logout
              </button>
            </nav>
          </div>
        </header>
      ) : null}
      <Component {...pageProps} />
    </AuthContext.Provider>
  );
}
