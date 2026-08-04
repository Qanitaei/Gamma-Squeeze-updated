import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Gamma Squeeze Platform",
  description: "Institutional gamma-squeeze forecasting dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
