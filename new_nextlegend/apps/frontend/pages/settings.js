import { useEffect, useState } from "react";
import { fetchJson, patchJson, postJson } from "@/lib/api";
import { useAuth } from "@/lib/auth";

const Card = ({ children, className = "" }) => (
  <section className={`surface-panel rounded-lg p-5 ${className}`}>
    {children}
  </section>
);

const Field = ({ label, children, hint }) => (
  <label className="flex flex-col gap-2">
    <span className="text-xs font-bold uppercase tracking-[0.16em] text-slate-500">{label}</span>
    {children}
    {hint ? <span className="text-xs font-semibold text-slate-500">{hint}</span> : null}
  </label>
);

export default function SettingsPage() {
  const { me, refreshAuth } = useAuth();
  const [profile, setProfile] = useState({ display_name: "", email: "" });
  const [passwords, setPasswords] = useState({ current_password: "", new_password: "", confirm_password: "" });
  const [profileStatus, setProfileStatus] = useState("");
  const [passwordStatus, setPasswordStatus] = useState("");
  const [error, setError] = useState("");
  const [savingProfile, setSavingProfile] = useState(false);
  const [savingPassword, setSavingPassword] = useState(false);

  useEffect(() => {
    let active = true;
    fetchJson("/auth/profile")
      .then((data) => {
        if (!active) return;
        setProfile({
          display_name: data.display_name || data.username || "",
          email: data.email || "",
        });
      })
      .catch((err) => {
        console.error(err);
        if (active) setError("Unable to load your settings.");
      });
    return () => {
      active = false;
    };
  }, []);

  const saveProfile = async (event) => {
    event.preventDefault();
    setError("");
    setProfileStatus("");
    setSavingProfile(true);
    try {
      await patchJson("/auth/profile", profile);
      await refreshAuth();
      setProfileStatus("Profile updated.");
    } catch (err) {
      console.error(err);
      setError("Unable to update profile.");
    } finally {
      setSavingProfile(false);
    }
  };

  const changePassword = async (event) => {
    event.preventDefault();
    setError("");
    setPasswordStatus("");
    if (passwords.new_password.length < 8) {
      setError("Password must contain at least 8 characters.");
      return;
    }
    if (passwords.new_password !== passwords.confirm_password) {
      setError("Passwords do not match.");
      return;
    }
    setSavingPassword(true);
    try {
      await postJson("/auth/password/change", {
        current_password: passwords.current_password,
        new_password: passwords.new_password,
      });
      setPasswords({ current_password: "", new_password: "", confirm_password: "" });
      setPasswordStatus("Password updated.");
    } catch (err) {
      console.error(err);
      setError("Unable to update password.");
    } finally {
      setSavingPassword(false);
    }
  };

  return (
    <main className="nl-page px-4 py-8">
      <div className="mx-auto max-w-5xl space-y-6">
        <header className="nl-page-header">
          <p className="nl-kicker">Workspace settings</p>
          <h1 className="mt-2 text-3xl font-semibold tracking-normal text-slate-950 md:text-4xl">
            Account and security
          </h1>
          <p className="mt-2 max-w-2xl text-slate-600">
            Manage your profile information and keep your Next Legend password up to date.
          </p>
        </header>

        {error ? <Card className="border-rose-400/35 text-sm font-semibold text-rose-200">{error}</Card> : null}

        <div className="grid gap-6 lg:grid-cols-[0.82fr_1.18fr]">
          <Card>
            <p className="nl-kicker">Signed in</p>
            <div className="mt-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-md border border-[#3A8967]/35 bg-[#2F7D5C]/18 text-sm font-semibold text-[#DDF3E8]">
                {(me?.display_name || me?.username || "HD").slice(0, 2).toUpperCase()}
              </div>
              <div className="min-w-0">
                <p className="truncate text-base font-semibold text-white">{me?.display_name || me?.username || "Workspace user"}</p>
                <p className="truncate text-sm text-slate-500">{me?.email || "No email configured"}</p>
              </div>
            </div>
            <dl className="mt-5 space-y-3 text-sm">
              <div className="flex items-center justify-between gap-4 border-t border-white/10 pt-3">
                <dt className="text-slate-500">Username</dt>
                <dd className="font-semibold text-white">{me?.username || "-"}</dd>
              </div>
              <div className="flex items-center justify-between gap-4 border-t border-white/10 pt-3">
                <dt className="text-slate-500">Role</dt>
                <dd className="font-semibold capitalize text-white">{me?.role || "user"}</dd>
              </div>
            </dl>
          </Card>

          <div className="space-y-6">
            <Card>
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="nl-kicker">Profile</p>
                  <h2 className="mt-1 text-lg font-semibold text-white">Personal information</h2>
                </div>
                {profileStatus ? <span className="rounded-md border border-[#3A8967]/35 bg-[#2F7D5C]/15 px-3 py-1.5 text-xs font-semibold text-[#8CC7A7]">{profileStatus}</span> : null}
              </div>
              <form className="mt-5 grid gap-4 md:grid-cols-2" onSubmit={saveProfile}>
                <Field label="Display name">
                  <input
                    id="settings-display-name"
                    name="display_name"
                    className="nl-field"
                    value={profile.display_name}
                    onChange={(event) => setProfile((prev) => ({ ...prev, display_name: event.target.value }))}
                    autoComplete="name"
                  />
                </Field>
                <Field label="Email">
                  <input
                    id="settings-email"
                    name="email"
                    className="nl-field"
                    type="email"
                    value={profile.email}
                    onChange={(event) => setProfile((prev) => ({ ...prev, email: event.target.value }))}
                    autoComplete="email"
                  />
                </Field>
                <div className="md:col-span-2">
                  <button type="submit" className="nl-button-primary" disabled={savingProfile}>
                    {savingProfile ? "Saving..." : "Save profile"}
                  </button>
                </div>
              </form>
            </Card>

            <Card>
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="nl-kicker">Security</p>
                  <h2 className="mt-1 text-lg font-semibold text-white">Change password</h2>
                </div>
                {passwordStatus ? <span className="rounded-md border border-[#3A8967]/35 bg-[#2F7D5C]/15 px-3 py-1.5 text-xs font-semibold text-[#8CC7A7]">{passwordStatus}</span> : null}
              </div>
              <form className="mt-5 grid gap-4" onSubmit={changePassword}>
                <input
                  className="sr-only"
                  tabIndex={-1}
                  aria-hidden="true"
                  name="username"
                  autoComplete="username"
                  value={me?.username || ""}
                  readOnly
                />
                <Field label="Current password">
                  <input
                    id="settings-current-password"
                    name="current_password"
                    className="nl-field"
                    type="password"
                    value={passwords.current_password}
                    onChange={(event) => setPasswords((prev) => ({ ...prev, current_password: event.target.value }))}
                    autoComplete="current-password"
                    required
                  />
                </Field>
                <div className="grid gap-4 md:grid-cols-2">
                  <Field label="New password" hint="Minimum 8 characters.">
                    <input
                      id="settings-new-password"
                      name="new_password"
                      className="nl-field"
                      type="password"
                      value={passwords.new_password}
                      onChange={(event) => setPasswords((prev) => ({ ...prev, new_password: event.target.value }))}
                      autoComplete="new-password"
                      required
                    />
                  </Field>
                  <Field label="Confirm password">
                    <input
                      id="settings-confirm-password"
                      name="confirm_password"
                      className="nl-field"
                      type="password"
                      value={passwords.confirm_password}
                      onChange={(event) => setPasswords((prev) => ({ ...prev, confirm_password: event.target.value }))}
                      autoComplete="new-password"
                      required
                    />
                  </Field>
                </div>
                <div>
                  <button type="submit" className="nl-button-primary" disabled={savingPassword}>
                    {savingPassword ? "Updating..." : "Update password"}
                  </button>
                </div>
              </form>
            </Card>
          </div>
        </div>
      </div>
    </main>
  );
}
