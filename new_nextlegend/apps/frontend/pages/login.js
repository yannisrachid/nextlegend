import { useEffect, useState } from "react";
import { useRouter } from "next/router";
import { postJson } from "@/lib/api";
import { useAuth } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [forgotIdentifier, setForgotIdentifier] = useState("");
  const [resetPassword, setResetPassword] = useState("");
  const [resetConfirmPassword, setResetConfirmPassword] = useState("");
  const [mode, setMode] = useState("login");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const { refreshAuth } = useAuth();
  const resetToken = typeof router.query.reset_token === "string" ? router.query.reset_token : "";

  useEffect(() => {
    if (resetToken) {
      setMode("reset");
    }
  }, [resetToken]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!username.trim() || !password) return;
    setLoading(true);
    setError("");
    setMessage("");
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

  const handleForgotPassword = async (event) => {
    event.preventDefault();
    if (!forgotIdentifier.trim()) return;
    setLoading(true);
    setError("");
    setMessage("");
    try {
      await postJson("/auth/password/forgot", {
        identifier: forgotIdentifier.trim(),
      });
      setMessage("If an account matches this information, a reset link has been sent.");
    } catch (err) {
      console.error(err);
      setError("Unable to start password reset.");
    } finally {
      setLoading(false);
    }
  };

  const handleResetPassword = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setMessage("");
    try {
      if (!resetToken) {
        throw new Error("Missing token");
      }
      if (resetPassword.length < 8) {
        setError("Password must contain at least 8 characters.");
        return;
      }
      if (resetPassword !== resetConfirmPassword) {
        setError("Passwords do not match.");
        return;
      }
      await postJson("/auth/password/reset", {
        token: resetToken,
        new_password: resetPassword,
      });
      setResetPassword("");
      setResetConfirmPassword("");
      setMode("login");
      setMessage("Password updated. You can now sign in.");
      router.replace("/login", undefined, { shallow: true });
    } catch (err) {
      console.error(err);
      setError("Reset link is invalid or expired.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="nl-page flex min-h-screen items-center justify-center px-4 py-10">
      <div className="grid w-full max-w-5xl overflow-hidden rounded-lg border border-white/10 bg-white/[0.045] shadow-[0_32px_100px_rgba(0,0,0,0.45)] backdrop-blur-xl lg:grid-cols-[1.05fr_0.95fr]">
        <section className="hidden min-h-[560px] flex-col justify-between border-r border-white/10 bg-black/40 p-8 text-white lg:flex">
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
            {["Player reports", "Market ranking", "Portfolio rooms", "Scouting briefs"].map((item) => (
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
              {mode === "forgot" ? "Reset your password" : mode === "reset" ? "Choose a new password" : "Sign in to Next Legend"}
            </h1>
            <p className="text-sm leading-6 text-slate-600">
              {mode === "forgot"
                ? "Enter your username or email. If the account exists, a reset link will be sent."
                : mode === "reset"
                  ? "Enter a new password to secure your workspace access."
                  : "Access the reports, player rooms and market workflows used by the HD Sports team."}
            </p>
          </div>

          {mode === "forgot" ? (
            <form className="mt-6 space-y-4" onSubmit={handleForgotPassword}>
              <div className="space-y-2">
                <label htmlFor="forgot-identifier" className="text-xs font-bold uppercase tracking-[0.16em] text-slate-500">
                  Username or email
                </label>
                <input
                  id="forgot-identifier"
                  name="identifier"
                  className="nl-field"
                  value={forgotIdentifier}
                  onChange={(event) => setForgotIdentifier(event.target.value)}
                  placeholder="Enter username or email"
                  autoComplete="username"
                />
              </div>
              {error ? <p className="text-sm font-semibold text-rose-700">{error}</p> : null}
              {message ? <p className="text-sm font-semibold text-[#8CC7A7]">{message}</p> : null}
              <button type="submit" disabled={loading} className="nl-button-primary w-full">
                {loading ? "Sending..." : "Send reset link"}
              </button>
              <button
                type="button"
                className="nl-button-secondary w-full"
                onClick={() => {
                  setMode("login");
                  setError("");
                  setMessage("");
                }}
              >
                Back to sign in
              </button>
            </form>
          ) : mode === "reset" ? (
            <form className="mt-6 space-y-4" onSubmit={handleResetPassword}>
              <input
                className="sr-only"
                tabIndex={-1}
                aria-hidden="true"
                name="username"
                autoComplete="username"
                value={username || ""}
                readOnly
              />
              <div className="space-y-2">
                <label htmlFor="reset-password" className="text-xs font-bold uppercase tracking-[0.16em] text-slate-500">
                  New password
                </label>
                <input
                  id="reset-password"
                  name="new_password"
                  type="password"
                  className="nl-field"
                  value={resetPassword}
                  onChange={(event) => setResetPassword(event.target.value)}
                  placeholder="Minimum 8 characters"
                  autoComplete="new-password"
                />
              </div>
              <div className="space-y-2">
                <label htmlFor="reset-confirm-password" className="text-xs font-bold uppercase tracking-[0.16em] text-slate-500">
                  Confirm password
                </label>
                <input
                  id="reset-confirm-password"
                  name="confirm_password"
                  type="password"
                  className="nl-field"
                  value={resetConfirmPassword}
                  onChange={(event) => setResetConfirmPassword(event.target.value)}
                  placeholder="Confirm new password"
                  autoComplete="new-password"
                />
              </div>
              {error ? <p className="text-sm font-semibold text-rose-700">{error}</p> : null}
              {message ? <p className="text-sm font-semibold text-[#8CC7A7]">{message}</p> : null}
              <button type="submit" disabled={loading} className="nl-button-primary w-full">
                {loading ? "Updating..." : "Update password"}
              </button>
            </form>
          ) : (
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
              {message ? <p className="text-sm font-semibold text-[#8CC7A7]">{message}</p> : null}
              <button
                type="submit"
                disabled={loading}
                className="nl-button-primary w-full"
              >
                {loading ? "Signing in..." : "Sign in"}
              </button>
              <button
                type="button"
                className="w-full text-center text-sm font-semibold text-slate-500 transition hover:text-white"
                onClick={() => {
                  setMode("forgot");
                  setError("");
                  setMessage("");
                }}
              >
                Forgot password?
              </button>
            </form>
          )}
        </section>
      </div>
    </main>
  );
}
