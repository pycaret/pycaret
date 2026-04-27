/**
 * Site-wide Open Graph image, prerendered to a static PNG at build time.
 *
 * Next.js's file-based convention picks this up automatically: every
 * page inherits this image as its OG / Twitter card unless it ships
 * its own `opengraph-image.tsx` next to its page.
 *
 * Static export prerenders the ImageResponse to `/opengraph-image.png`
 * at build time — no runtime image generation in production.
 */
import fs from 'node:fs';
import path from 'node:path';

import { ImageResponse } from 'next/og';

// Static export requires explicit force-static on metadata routes.
export const dynamic = 'force-static';

export const alt = 'PyCaret — Low-code machine learning for Python';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

// Read the real wordmark from /public at build time and inline as base64
// so Satori (which renders the OG card) has it available without a
// network fetch.
const LOGO_DATA_URI = (() => {
  const file = path.join(process.cwd(), 'public', 'logo.png');
  const buf = fs.readFileSync(file);
  return `data:image/png;base64,${buf.toString('base64')}`;
})();

export default async function OgImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          padding: '80px',
          background:
            'linear-gradient(135deg, #0B1220 0%, #1B2B4A 55%, #3457B0 100%)',
          fontFamily: 'sans-serif',
          color: '#fff',
        }}
      >
        {/* Top: real wordmark on a white plate so the dark logo
            stays legible against the navy gradient. Satori doesn't
            support CSS filters, so we use a contrasting plate
            instead of inverting the PNG. */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            background: '#ffffff',
            borderRadius: 18,
            padding: '18px 28px',
            boxShadow: '0 6px 24px rgba(0,0,0,0.18)',
          }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={LOGO_DATA_URI} alt="PyCaret" width={360} height={52} />
        </div>

        {/* Middle: headline */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          <div
            style={{
              fontSize: 78,
              fontWeight: 700,
              lineHeight: 1.05,
              letterSpacing: '-0.02em',
              maxWidth: 1000,
            }}
          >
            Low-code machine learning for Python
          </div>
          <div
            style={{
              fontSize: 32,
              color: '#A9C0EC',
              fontWeight: 400,
              maxWidth: 1000,
            }}
          >
            Set up, compare, deploy — in 20 lines.
          </div>
        </div>

        {/* Bottom: domain + version stripe */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            fontSize: 28,
            color: '#cdd6e6',
          }}
        >
          <div style={{ display: 'flex' }}>pycaret.org</div>
          <div
            style={{
              display: 'flex',
              padding: '10px 22px',
              borderRadius: 999,
              background: 'rgba(91, 141, 239, 0.18)',
              border: '1px solid rgba(91, 141, 239, 0.45)',
              color: '#fff',
              fontWeight: 500,
            }}
          >
            v4.0 · sklearn-native + React dashboard
          </div>
        </div>
      </div>
    ),
    { ...size },
  );
}
