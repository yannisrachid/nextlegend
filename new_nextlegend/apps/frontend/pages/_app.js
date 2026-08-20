import "@/styles/globals.css";
import Link from "next/link";
import Head from "next/head";
import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/router";
import { fetchJson, postJson } from "@/lib/api";
import { AuthContext } from "@/lib/auth";
import ScoutingLabShell from "@/components/ScoutingLabShell";

const NAV_ITEMS = [
  { href: "/", label: "HQ" },
  { href: "/hd-players", label: "HD PLAYERS" },
  { href: "/mercato-2026", label: "MERCATO 2026" },
  { href: "/scouting-lab", label: "SCOUTING LAB" },
];

const SCOUTING_LAB_PATHS = [
  "/scouting-lab",
  "/ranking",
  "/report",
  "/comparison",
  "/projection",
  "/stats-research",
  "/vizualisation",
  "/prospect",
  "/ai",
  "/admin",
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

  const shouldUseScoutingLab =
    !isLogin && SCOUTING_LAB_PATHS.some((path) => router.pathname === path || router.pathname.startsWith(`${path}/`));

  return (
    <AuthContext.Provider value={{ me, status: authStatus, refreshAuth }}>
      <Head>
        <title>Next Legend | HD Sports</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" type="image/png" href="/logo_nl.png" />
        <link rel="shortcut icon" href="/logo_nl.png" />
      </Head>
      {!isLogin ? (
        <header className="sticky top-0 z-40 border-b border-slate-200/80 bg-white/80 backdrop-blur-xl">
          <div className="mx-auto flex max-w-[1500px] flex-col gap-3 px-4 py-3 lg:flex-row lg:items-center lg:justify-between">
            <Link href="/" className="group flex min-w-fit items-center gap-3">
              <span className="flex h-10 w-10 items-center justify-center rounded-md border border-teal-700/20 bg-white shadow-sm">
                <img src="/logo_nl.png" alt="Next Legend" className="h-7 w-7 object-contain" />
              </span>
              <span className="leading-tight">
                <span className="block text-lg font-extrabold text-slate-950 text-glow">
                  Next Legend
                </span>
                <span className="block text-[11px] font-bold uppercase tracking-[0.18em] text-teal-700">
                  HD Sports intelligence
                </span>
              </span>
            </Link>
            <div className="flex min-w-0 flex-col gap-2 sm:flex-row sm:items-center sm:gap-3">
              <nav className="flex w-full min-w-0 items-center gap-1 overflow-x-auto rounded-md border border-slate-200 bg-slate-50/80 p-1 sm:flex-1">
                {NAV_ITEMS.map((item) => {
                  const isActive =
                    router.pathname === item.href ||
                    (item.href === "/scouting-lab"
                      ? shouldUseScoutingLab
                      : item.href !== "/" && router.pathname.startsWith(item.href));
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      className={`whitespace-nowrap rounded px-3 py-2 text-sm font-bold transition duration-200 ${
                        isActive
                          ? "bg-white text-teal-800 shadow-sm"
                          : "text-slate-600 hover:bg-white/80 hover:text-slate-950"
                      }`}
                    >
                      {item.label}
                    </Link>
                  );
                })}
              </nav>
              <button
                type="button"
                onClick={handleLogout}
                className="nl-button-secondary self-start px-3 sm:self-auto"
              >
                Logout
              </button>
            </div>
          </div>
        </header>
      ) : null}
      {shouldUseScoutingLab ? (
        <ScoutingLabShell me={me}>
          <Component {...pageProps} />
        </ScoutingLabShell>
      ) : (
        <Component {...pageProps} />
      )}
    </AuthContext.Provider>
  );
}
