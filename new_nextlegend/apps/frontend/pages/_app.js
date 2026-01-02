import "@/styles/globals.css";
import Link from "next/link";
import { useRouter } from "next/router";

const NAV_ITEMS = [
  { href: "/", label: "Home" },
  { href: "/ranking", label: "Ranking" },
  { href: "/report", label: "Report" },
  { href: "/comparison", label: "Comparison" },
  { href: "/projection", label: "Projection" },
  { href: "/stats-research", label: "Stats Research" },
  { href: "/vizualisation", label: "Vizualisation" },
];

export default function App({ Component, pageProps }) {
  const router = useRouter();

  return (
    <>
      <header className="sticky top-0 z-40 border-b border-white/5 bg-slate-900/80 backdrop-blur">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-primary text-lg font-semibold text-glow">
              NextLegend
            </span>
            <span className="text-xs uppercase tracking-[0.3em] text-slate-500">
              v2
            </span>
          </div>
          <nav className="flex items-center gap-4">
            {NAV_ITEMS.map((item) => {
              const isActive = router.pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`text-sm font-medium ${
                    isActive
                      ? "text-primary"
                      : "text-slate-300 hover:text-white"
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>
      </header>
      <Component {...pageProps} />
    </>
  );
}
