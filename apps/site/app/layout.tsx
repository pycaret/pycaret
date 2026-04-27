/**
 * Root layout for pycaret.org.
 *
 * Sets up the typography (Inter + JetBrains Mono via google-fonts CSS),
 * meta tags, and the persistent header / footer that wrap every page
 * (landing, docs, reference, blog).
 */
import type { Metadata } from 'next';
import './globals.css';
import { SiteHeader } from '@/components/SiteHeader';
import { SiteFooter } from '@/components/SiteFooter';

export const metadata: Metadata = {
  metadataBase: new URL('https://pycaret.org'),
  title: {
    default: 'PyCaret — Low-code machine learning for Python',
    template: '%s · PyCaret',
  },
  description:
    'PyCaret is an open-source, low-code machine learning library that automates the ML workflow. Set up an experiment, compare models, deploy a pipeline — in under 20 lines of code.',
  keywords: [
    'pycaret',
    'machine learning',
    'automl',
    'python',
    'data science',
    'low-code',
    'mlops',
  ],
  authors: [{ name: 'PyCaret contributors', url: 'https://github.com/pycaret/pycaret' }],
  creator: 'Moez Ali',
  openGraph: {
    // OG image comes from the file-based `app/opengraph-image.tsx`
    // convention; Next.js prerenders it to PNG at build time and
    // wires the og:image / twitter:image meta tags automatically.
    type: 'website',
    locale: 'en_US',
    url: 'https://pycaret.org',
    siteName: 'PyCaret',
    title: 'PyCaret — Low-code machine learning for Python',
    description:
      'Open-source AutoML library and dashboard. Set up, compare, deploy — in 20 lines.',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'PyCaret',
    description: 'Low-code ML for Python.',
  },
  icons: {
    icon: { url: '/favicon.svg', type: 'image/svg+xml' },
  },
  alternates: {
    canonical: '/',
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link
          rel="preconnect"
          href="https://fonts.googleapis.com"
          crossOrigin="anonymous"
        />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen flex flex-col bg-white text-ink-800 antialiased">
        <SiteHeader />
        <main className="flex-1">{children}</main>
        <SiteFooter />
      </body>
    </html>
  );
}
