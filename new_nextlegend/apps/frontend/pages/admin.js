import { useEffect, useState } from "react";
import { deleteJson, fetchJson, patchJson, postJson } from "@/lib/api";

const Card = ({ children, className = "" }) => (
  <div className={`glass-panel rounded-xl border border-white/5 p-4 ${className}`}>
    {children}
  </div>
);

const Field = ({ label, children }) => (
  <label className="flex flex-col gap-2 text-sm text-slate-200">
    <span className="text-xs uppercase tracking-[0.2em] text-slate-400">
      {label}
    </span>
    {children}
  </label>
);

const formatDate = (value) => {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString();
};

export default function AdminPage() {
  const [me, setMe] = useState(null);
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [creating, setCreating] = useState({
    username: "",
    display_name: "",
    email: "",
    password: "",
    role: "user",
  });
  const [editing, setEditing] = useState(null);

  const isAdmin = me?.role === "admin";

  const loadUsers = async () => {
    setLoading(true);
    setError("");
    try {
      const data = await fetchJson("/admin/users");
      setUsers(data.items || []);
    } catch (err) {
      console.error(err);
      setError("Unable to load users.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
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
  }, []);

  useEffect(() => {
    if (isAdmin) {
      loadUsers();
    }
  }, [isAdmin]);

  const handleCreate = async (event) => {
    event.preventDefault();
    setError("");
    setSuccess("");
    try {
      await postJson("/admin/users", creating);
      setCreating({
        username: "",
        display_name: "",
        email: "",
        password: "",
        role: "user",
      });
      setSuccess("User created.");
      await loadUsers();
    } catch (err) {
      console.error(err);
      setError("Failed to create user.");
    }
  };

  const handleImport = async () => {
    setError("");
    setSuccess("");
    try {
      const result = await postJson("/admin/users/import");
      setSuccess(`Imported ${result.imported || 0} users from credentials.`);
      await loadUsers();
    } catch (err) {
      console.error(err);
      setError("Failed to import users.");
    }
  };

  const handleSave = async (event) => {
    event.preventDefault();
    if (!editing) return;
    setError("");
    setSuccess("");
    try {
      const payload = {
        display_name: editing.display_name,
        email: editing.email,
        role: editing.role,
      };
      if (editing.password) {
        payload.password = editing.password;
      }
      await patchJson(`/admin/users/${editing.username}`, payload);
      setEditing(null);
      setSuccess("User updated.");
      await loadUsers();
    } catch (err) {
      console.error(err);
      setError("Failed to update user.");
    }
  };

  const handleDelete = async (username) => {
    const confirmDelete = window.confirm(`Delete ${username}?`);
    if (!confirmDelete) return;
    setError("");
    setSuccess("");
    try {
      await deleteJson(`/admin/users/${username}`);
      setSuccess("User deleted.");
      await loadUsers();
    } catch (err) {
      console.error(err);
      setError("Failed to delete user.");
    }
  };

  if (!me) {
    return (
      <main className="nl-page px-4 py-12">
        <div className="max-w-4xl mx-auto">
          <Card>Loading profile…</Card>
        </div>
      </main>
    );
  }

  if (!isAdmin) {
    return (
      <main className="nl-page px-4 py-12">
        <div className="max-w-4xl mx-auto">
          <Card>
            <h1 className="text-xl font-semibold text-white">HD Sports access control</h1>
            <p className="text-sm text-slate-300 mt-2">Access denied.</p>
          </Card>
        </div>
      </main>
    );
  }

  return (
    <main className="nl-page px-4 py-12">
      <div className="max-w-6xl mx-auto space-y-8">
        <header className="space-y-2">
          <h1 className="text-2xl font-semibold text-white">Team access control</h1>
          <p className="text-sm text-slate-400">
            Manage users and keep credentials in sync.
          </p>
        </header>

        {error ? (
          <Card className="border-red-500/40 text-red-200">{error}</Card>
        ) : null}
        {success ? (
          <Card className="border-emerald-500/40 text-emerald-200">{success}</Card>
        ) : null}

        <Card className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white">Users</h2>
            <button
              type="button"
              onClick={handleImport}
              className="rounded-md border border-white/10 px-4 py-2 text-sm text-slate-100 hover:border-white/20"
            >
              Import credentials.toml
            </button>
          </div>
          {loading ? (
            <p className="text-sm text-slate-400">Loading users…</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead className="text-xs uppercase tracking-[0.2em] text-slate-500">
                  <tr>
                    <th className="py-2 text-left">User</th>
                    <th className="py-2 text-left">Email</th>
                    <th className="py-2 text-left">Role</th>
                    <th className="py-2 text-left">Last login</th>
                    <th className="py-2 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {users.map((user) => (
                    <tr key={user.username}>
                      <td className="py-3">
                        <div className="text-white font-medium">{user.username}</div>
                        <div className="text-xs text-slate-400">{user.display_name || "—"}</div>
                      </td>
                      <td className="py-3">{user.email || "—"}</td>
                      <td className="py-3 capitalize">{user.role || "user"}</td>
                      <td className="py-3">{formatDate(user.last_login)}</td>
                      <td className="py-3 text-right space-x-2">
                        <button
                          type="button"
                          onClick={() =>
                            setEditing({
                              ...user,
                              password: "",
                            })
                          }
                          className="text-xs text-slate-200 hover:text-white"
                        >
                          Edit
                        </button>
                        <button
                          type="button"
                          onClick={() => handleDelete(user.username)}
                          className="text-xs text-red-400 hover:text-red-200"
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card>
          <h2 className="text-lg font-semibold text-white">Create user</h2>
          <form onSubmit={handleCreate} className="mt-4 grid gap-4 md:grid-cols-2">
            <Field label="Username">
              <input
                className="rounded-md border border-slate-700 bg-slate-900/60 px-3 py-2 text-slate-100"
                value={creating.username}
                onChange={(event) =>
                  setCreating((prev) => ({ ...prev, username: event.target.value }))
                }
                required
              />
            </Field>
            <Field label="Display name">
              <input
                className="rounded-md border border-slate-700 bg-slate-900/60 px-3 py-2 text-slate-100"
                value={creating.display_name}
                onChange={(event) =>
                  setCreating((prev) => ({
                    ...prev,
                    display_name: event.target.value,
                  }))
                }
              />
            </Field>
            <Field label="Email">
              <input
                type="email"
                className="rounded-md border border-slate-700 bg-slate-900/60 px-3 py-2 text-slate-100"
                value={creating.email}
                onChange={(event) =>
                  setCreating((prev) => ({ ...prev, email: event.target.value }))
                }
              />
            </Field>
            <Field label="Role">
              <select
                className="rounded-md border border-slate-700 bg-slate-900/60 px-3 py-2 text-slate-100"
                value={creating.role}
                onChange={(event) =>
                  setCreating((prev) => ({ ...prev, role: event.target.value }))
                }
              >
                <option value="user">User</option>
                <option value="admin">Admin</option>
              </select>
            </Field>
            <Field label="Password">
              <input
                type="password"
                className="rounded-md border border-slate-700 bg-slate-900/60 px-3 py-2 text-slate-100"
                value={creating.password}
                onChange={(event) =>
                  setCreating((prev) => ({ ...prev, password: event.target.value }))
                }
                required
              />
            </Field>
            <div className="flex items-end">
              <button
                type="submit"
                className="rounded-md bg-primary px-4 py-2 text-sm font-semibold text-slate-950"
              >
                Create
              </button>
            </div>
          </form>
        </Card>

        {editing ? (
          <Card>
            <h2 className="text-lg font-semibold text-white">
              Edit user: {editing.username}
            </h2>
            <form onSubmit={handleSave} className="mt-4 grid gap-4 md:grid-cols-2">
              <Field label="Display name">
                <input
                  className="rounded-md border border-slate-700 bg-slate-900/60 px-3 py-2 text-slate-100"
                  value={editing.display_name || ""}
                  onChange={(event) =>
                    setEditing((prev) => ({
                      ...prev,
                      display_name: event.target.value,
                    }))
                  }
                />
              </Field>
              <Field label="Email">
                <input
                  type="email"
                  className="rounded-md border border-slate-700 bg-slate-900/60 px-3 py-2 text-slate-100"
                  value={editing.email || ""}
                  onChange={(event) =>
                    setEditing((prev) => ({ ...prev, email: event.target.value }))
                  }
                />
              </Field>
              <Field label="Role">
                <select
                  className="rounded-md border border-slate-700 bg-slate-900/60 px-3 py-2 text-slate-100"
                  value={editing.role || "user"}
                  onChange={(event) =>
                    setEditing((prev) => ({ ...prev, role: event.target.value }))
                  }
                >
                  <option value="user">User</option>
                  <option value="admin">Admin</option>
                </select>
              </Field>
              <Field label="Reset password">
                <input
                  type="password"
                  className="rounded-md border border-slate-700 bg-slate-900/60 px-3 py-2 text-slate-100"
                  value={editing.password || ""}
                  onChange={(event) =>
                    setEditing((prev) => ({ ...prev, password: event.target.value }))
                  }
                  placeholder="Leave blank to keep"
                />
              </Field>
              <div className="flex items-end gap-3">
                <button
                  type="submit"
                  className="rounded-md bg-primary px-4 py-2 text-sm font-semibold text-slate-950"
                >
                  Save
                </button>
                <button
                  type="button"
                  className="rounded-md border border-white/10 px-4 py-2 text-sm text-slate-200"
                  onClick={() => setEditing(null)}
                >
                  Cancel
                </button>
              </div>
            </form>
          </Card>
        ) : null}
      </div>
    </main>
  );
}
