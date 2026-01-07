import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Signup Research - Research New Users",
  description: "AI-powered research for new user signups using Linkup web search",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
