import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "QUARRY | Causal decision intelligence",
  description: "Intervention analysis grounded in explicit causal assumptions.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
