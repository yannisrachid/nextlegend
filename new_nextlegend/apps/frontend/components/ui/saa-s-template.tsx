import React from "react";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "secondary" | "ghost" | "gradient";
  size?: "default" | "sm" | "lg";
  children: React.ReactNode;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = "default", size = "default", className = "", children, ...props }, ref) => {
    const baseStyles =
      "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/40 disabled:pointer-events-none disabled:opacity-50";

    const variants = {
      default: "bg-white text-[#050706] hover:bg-gray-100",
      secondary: "bg-gray-800 text-white hover:bg-gray-700",
      ghost: "text-white hover:bg-white/10",
      gradient: "bg-gradient-to-b from-white via-white/95 to-white/60 text-[#050706] hover:scale-[1.02] active:scale-[0.98]",
    };

    const sizes = {
      default: "h-10 px-4 py-2 text-sm",
      sm: "h-10 px-5 text-sm",
      lg: "h-12 px-8 text-base",
    };

    return (
      <button ref={ref} className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`} {...props}>
        {children}
      </button>
    );
  }
);

Button.displayName = "Button";

const ArrowRight = ({ className = "", size = 16 }: { className?: string; size?: number }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
    aria-hidden="true"
  >
    <path d="M5 12h14" />
    <path d="m12 5 7 7-7 7" />
  </svg>
);

const DashboardPreview = React.memo(() => (
  <div className="w-full rounded-lg border border-white/10 bg-white/[0.03] p-3 shadow-2xl">
    <div className="rounded-md border border-white/10 bg-black/70 p-4">
      <div className="flex items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.16em] text-white/50">Next Legend</p>
          <h2 className="mt-2 text-xl font-semibold text-white">Scouting command room</h2>
        </div>
        <div className="rounded-md bg-white px-3 py-2 text-sm font-semibold text-[#050706]">Live</div>
      </div>
      <div className="mt-4 grid gap-3 md:grid-cols-3">
        {[
          ["Player level", "84", "Validated profiles"],
          ["Market fit", "91", "Club needs matched"],
          ["Risk", "Low", "Availability checked"],
        ].map(([label, value, desc]) => (
          <div key={label} className="rounded-md border border-white/10 bg-white/[0.04] p-4">
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-white/50">{label}</p>
            <p className="mt-3 text-3xl font-semibold text-white">{value}</p>
            <p className="mt-2 text-sm text-white/50">{desc}</p>
          </div>
        ))}
      </div>
      <div className="mt-3 grid gap-3 md:grid-cols-[1.2fr_0.8fr]">
        <div className="rounded-md border border-white/10 bg-white/[0.04] p-4">
          <div className="flex h-36 items-end gap-2">
            {[42, 68, 56, 78, 64, 88, 72, 92].map((height, index) => (
              <span
                key={index}
                className="flex-1 rounded-t bg-gradient-to-t from-teal-500 to-white"
                style={{ height: `${height}%` }}
              />
            ))}
          </div>
        </div>
        <div className="rounded-md border border-white/10 bg-white/[0.04] p-4">
          {["Minutes", "League strength", "Team strength", "Position fit"].map((item, index) => (
            <div key={item} className="flex items-center justify-between border-b border-white/10 py-2 last:border-b-0">
              <span className="text-sm text-white/60">{item}</span>
              <span className="text-sm font-semibold text-white">{[94, 82, 79, 88][index]}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  </div>
));

DashboardPreview.displayName = "DashboardPreview";

const Navigation = React.memo(() => (
  <header className="fixed top-0 z-50 w-full border-b border-white/10 bg-black/80 backdrop-blur-md">
    <nav className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
      <div className="flex items-center gap-3 text-xl font-semibold text-white">
        <img src="/logo_nl.png" alt="Next Legend" className="h-8 w-8 rounded-md bg-white p-1" />
        Next Legend
      </div>
      <div className="hidden items-center gap-8 md:flex">
        <a href="/hd-players" className="text-sm text-white/60 transition-colors hover:text-white">
          HD Players
        </a>
        <a href="/scouting-lab" className="text-sm text-white/60 transition-colors hover:text-white">
          Scouting
        </a>
      </div>
      <Button type="button" variant="default" size="sm" className="hidden md:inline-flex">
        Open HQ
      </Button>
    </nav>
  </header>
));

Navigation.displayName = "Navigation";

const Hero = React.memo(() => (
  <section className="relative flex min-h-screen flex-col items-center justify-start px-6 pb-20 pt-28 md:pt-32">
    <aside className="mb-8 inline-flex max-w-full flex-wrap items-center justify-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-4 py-2 backdrop-blur-sm">
      <span className="whitespace-nowrap text-center text-xs text-white/50">Global UX refresh for Next Legend</span>
      <a href="/scouting-lab" className="flex items-center gap-1 whitespace-nowrap text-xs text-white/50 transition-all hover:text-white">
        Open scouting
        <ArrowRight size={12} />
      </a>
    </aside>

    <h1 className="mb-6 max-w-3xl px-6 text-center text-4xl font-medium leading-tight text-white md:text-5xl lg:text-6xl">
      Football intelligence, packaged for faster decisions.
    </h1>

    <p className="mb-10 max-w-2xl px-6 text-center text-sm leading-6 text-white/50 md:text-base">
      A focused operating system for HD Sports: portfolio rooms, mercato execution, scouting models and decision briefs.
    </p>

    <div className="relative z-10 mb-16 flex items-center gap-4">
      <Button type="button" variant="gradient" size="lg" aria-label="Open Next Legend HQ">
        Get started
      </Button>
    </div>

    <div className="w-full max-w-5xl pb-20">
      <DashboardPreview />
    </div>
  </section>
));

Hero.displayName = "Hero";

export default function Component() {
  return (
    <main className="min-h-screen bg-black text-white">
      <Navigation />
      <Hero />
    </main>
  );
}
