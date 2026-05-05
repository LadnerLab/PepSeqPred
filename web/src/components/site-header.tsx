import Link from "next/link";
import { assetPath } from "@/lib/format";

const navItems = [
  { href: "/", label: "Overview" },
  { href: "/results/comparison", label: "Model Comparison" },
  { href: "/results/validation", label: "PhaseB Validation" },
  { href: "/results/external-cocci", label: "External Cocci" },
  { href: "/reproducibility", label: "Reproducibility" }
];

export function SiteHeader(): JSX.Element {
  return (
    <header className="site-header">
      <div className="wrap topbar">
        <Link className="brand" href="/">
          <picture>
            <source
              media="(prefers-color-scheme: dark)"
              srcSet={assetPath("/assets/brand/PepSeqPred_logo_white.png")}
            />
            <source
              media="(prefers-color-scheme: light)"
              srcSet={assetPath("/assets/brand/PepSeqPred_logo_black.png")}
            />
            <img
              src={assetPath("/assets/brand/PepSeqPred_logo_black.png")}
              alt="PepSeqPred logo"
              className="brand-logo"
            />
          </picture>
        </Link>
        <nav className="nav">
          {navItems.map((item) => (
            <Link key={item.href} href={item.href}>
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
