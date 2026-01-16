import { createContext, useContext } from "react";

export const AuthContext = createContext({
  me: null,
  status: "loading",
  refreshAuth: async () => null,
});

export const useAuth = () => useContext(AuthContext);
