import { useState } from "react";
import { useRouter } from "next/router";
import { postJson } from "@/lib/api";
import { useAuth } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const { refreshAuth } = useAuth();

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!username.trim() || !password) return;
    setLoading(true);
    setError("");
    try {
      const legacyUserId =
        typeof window !== "undefined"
          ? window.localStorage.getItem("nl_ai_user_id")
          : null;
      await postJson("/auth/login", {
        username: username.trim(),
        password,
        legacy_user_id: legacyUserId || undefined,
      });
      const authed = await refreshAuth();
      if (authed) {
        router.replace("/");
      }
    } catch (err) {
      setError("Invalid credentials.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="nl-page flex min-h-screen items-center justify-center px-4 py-10">
      <div className="grid w-full max-w-5xl overflow-hidden rounded-lg border border-white/10 bg-white shadow-[0_28px_90px_rgba(0,0,0,0.34)] lg:grid-cols-[1.05fr_0.95fr]">
        <section className="hidden min-h-[560px] flex-col justify-between bg-black p-8 text-white lg:flex">
          <div>
            <img src="/logo_nl.png" alt="Next Legend" className="h-12 w-12 rounded-md bg-white p-1" />
            <p className="mt-6 text-xs font-bold uppercase tracking-[0.16em] text-white/50">
              Next Legend by HD Sports
            </p>
            <h1 className="mt-4 max-w-sm text-4xl font-semibold leading-tight text-white">
              Intelligence for football decisions that move markets.
            </h1>
          </div>
          <div className="grid grid-cols-2 gap-3">
            {["Player reports", "Market ranking", "Mercato plans", "Scouting briefs"].map((item) => (
              <div key={item} className="rounded-md border border-white/10 bg-white/[0.04] px-3 py-3 text-sm font-semibold text-white">
                {item}
              </div>
            ))}
          </div>
        </section>

        <section className="p-6 md:p-10">
          <div className="space-y-2">
            <p className="nl-kicker">HD Sports workspace</p>
            <h1 className="text-3xl font-extrabold text-slate-950">
              Sign in to Next Legend
            </h1>
            <p className="text-sm leading-6 text-slate-600">
              Access the reports, player rooms and market workflows used by the HD Sports team.
            </p>
          </div>

        <form className="mt-6 space-y-4" onSubmit={handleSubmit}>
          <div className="space-y-2">
            <label htmlFor="login-username" className="text-xs font-bold uppercase tracking-[0.16em] text-slate-500">
              Username
            </label>
            <input
              id="login-username"
              name="username"
              className="nl-field"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              placeholder="Enter username"
              autoComplete="username"
            />
          </div>
          <div className="space-y-2">
            <label htmlFor="login-password" className="text-xs font-bold uppercase tracking-[0.16em] text-slate-500">
              Password
            </label>
            <input
              id="login-password"
              name="password"
              type="password"
              className="nl-field"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="Enter password"
              autoComplete="current-password"
            />
          </div>
          {error ? <p className="text-sm font-semibold text-rose-700">{error}</p> : null}
          <button
            type="submit"
            disabled={loading}
            className="nl-button-primary w-full"
          >
            {loading ? "Signing in..." : "Sign in"}
          </button>
        </form>
        </section>
      </div>
    </main>
  );
}
