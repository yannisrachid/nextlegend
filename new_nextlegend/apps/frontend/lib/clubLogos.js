let clubLogoDataPromise = null;
let clubLogoData = null;

export const normalizeClubName = (value) =>
  String(value || "")
    .trim()
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/&/g, " and ")
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

const clubNameVariants = (value) => {
  const normalized = normalizeClubName(value);
  if (!normalized) return [];
  const variants = new Set([normalized]);
  const stopWords = new Set(["fc", "cf", "sc", "afc", "ac", "as", "fk", "sk", "cd", "sd", "ud", "club", "football", "futbol", "soccer"]);
  const words = normalized.split(" ").filter(Boolean);
  const stripped = words.filter((word) => !stopWords.has(word));
  if (stripped.length && stripped.length !== words.length) variants.add(stripped.join(" "));
  const expanded = words.map((word) => ({ utd: "united", st: "saint" }[word] || word));
  if (expanded.join(" ") !== normalized) variants.add(expanded.join(" "));
  return Array.from(variants);
};

export const loadClubLogoData = async () => {
  if (clubLogoData) return clubLogoData;
  if (!clubLogoDataPromise) {
    clubLogoDataPromise = fetch("/club-logos.json")
      .then((response) => (response.ok ? response.json() : { aliases: {} }))
      .then((data) => {
        clubLogoData = data || { aliases: {} };
        return clubLogoData;
      })
      .catch(() => ({ aliases: {} }));
  }
  return clubLogoDataPromise;
};

export const resolveClubLogoUrl = (clubName, data = clubLogoData) => {
  const aliases = data?.aliases || {};
  for (const variant of clubNameVariants(clubName)) {
    if (aliases[variant]) return aliases[variant];
  }
  return "";
};
