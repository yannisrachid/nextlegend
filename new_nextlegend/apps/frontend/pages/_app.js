import "@/styles/globals.css";
import Link from "next/link";
import { useEffect, useState } from "react";
import { useRouter } from "next/router";
import { fetchJson, postJson } from "@/lib/api";

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

  const handleLogout = async () => {
    try {
      await postJson("/auth/logout");
    } catch (err) {
      console.error(err);
    } finally {
      window.location.href = "/login";
    }
  };

  useEffect(() => {
    if (isLogin) return;
    let active = true;
    fetchJson("/auth/me")
      .then((data) => {
        if (!active) return;
        setMe(data);
      })
      .catch(() => {
        if (!active) return;
        setMe(null);
      });
    return () => {
      active = false;
    };
  }, [isLogin]);

  const navItems = me?.role === "admin"
    ? [...NAV_ITEMS, { href: "/admin", label: "Admin" }]
    : NAV_ITEMS;

  return (
    <>
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
    </>
  );
}
