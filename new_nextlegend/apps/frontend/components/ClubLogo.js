import { useEffect, useState } from "react";
import { loadClubLogoData, resolveClubLogoUrl } from "@/lib/clubLogos";

const initials = (name) =>
  String(name || "")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("") || "FC";

export default function ClubLogo({
  name,
  className = "h-8 w-8",
  imageClassName = "",
  fallbackClassName = "",
}) {
  const [logoUrl, setLogoUrl] = useState("");
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let active = true;
    setFailed(false);
    loadClubLogoData().then((data) => {
      if (!active) return;
      setLogoUrl(resolveClubLogoUrl(name, data));
    });
    return () => {
      active = false;
    };
  }, [name]);

  const baseClass = `${className} club-logo-surface shrink-0 overflow-hidden rounded-md border border-white/10 bg-white`;
  if (logoUrl && !failed) {
    return (
      <span className={baseClass}>
        <img
          src={logoUrl}
          alt={name ? `${name} logo` : "Club logo"}
          className={`h-full w-full object-contain p-1 ${imageClassName}`}
          loading="lazy"
          onError={() => setFailed(true)}
        />
      </span>
    );
  }

  return (
    <span className={`${baseClass} club-logo-fallback flex items-center justify-center bg-white/[0.06] text-[10px] font-black text-white/60 ${fallbackClassName}`}>
      {initials(name)}
    </span>
  );
}
