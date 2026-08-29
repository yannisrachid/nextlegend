import "@/styles/globals.css";
import "leaflet/dist/leaflet.css";
import Head from "next/head";
import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/router";
import { fetchJson, postJson } from "@/lib/api";
import { AuthContext } from "@/lib/auth";
import AppShell from "@/components/AppShell";
import ScoutingLabShell from "@/components/ScoutingLabShell";

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
    if (isLogin) {
      setMe(null);
      setAuthStatus("unauthenticated");
      return;
    }
    refreshAuth();
  }, [isLogin, refreshAuth]);

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
        <AppShell onLogout={handleLogout} shouldUseScoutingLab={shouldUseScoutingLab} me={me}>
          {shouldUseScoutingLab ? (
            <ScoutingLabShell me={me}>
              <Component {...pageProps} />
            </ScoutingLabShell>
          ) : (
            <Component {...pageProps} />
          )}
        </AppShell>
      ) : (
        <Component {...pageProps} />
      )}
    </AuthContext.Provider>
  );
}
