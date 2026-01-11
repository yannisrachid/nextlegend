import { useEffect, useState } from "react";
import { postJson, fetchJson } from "@/lib/api";

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchJson("/auth/me")
      .then(() => {
        window.location.href = "/";
      })
      .catch(() => {});
  }, []);

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
      window.location.href = "/";
    } catch (err) {
      setError("Invalid credentials.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-hero-pattern text-slate-100 flex items-center justify-center px-4">
      <div className="w-full max-w-md glass-panel rounded-2xl p-8 border border-white/5">
        <div className="text-center space-y-2">
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
            NextLegend by Your Legend
          </p>
          <h1 className="text-3xl font-semibold text-white">Welcome back</h1>
          <p className="text-slate-300 text-sm">
            Sign in to access the scouting workspace.
          </p>
        </div>

        <form className="mt-6 space-y-4" onSubmit={handleSubmit}>
          <div className="space-y-2">
            <label className="text-xs uppercase tracking-[0.2em] text-slate-400">
              Username
            </label>
            <input
              className="w-full bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              placeholder="Enter username"
              autoComplete="username"
            />
          </div>
          <div className="space-y-2">
            <label className="text-xs uppercase tracking-[0.2em] text-slate-400">
              Password
            </label>
            <input
              type="password"
              className="w-full bg-slate-900/60 border border-slate-700 rounded-md px-3 py-2 text-slate-100"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="Enter password"
              autoComplete="current-password"
            />
          </div>
          {error ? <p className="text-sm text-amber-300">{error}</p> : null}
          <button
            type="submit"
            disabled={loading}
            className="w-full py-2 rounded-md bg-primary text-primary-foreground font-semibold hover:bg-primary/90 transition"
          >
            {loading ? "Signing in..." : "Sign in"}
          </button>
        </form>
      </div>
    </main>
  );
}
