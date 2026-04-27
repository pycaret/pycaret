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
import { ImageResponse } from 'next/og';

// Static export requires explicit force-static on metadata routes.
export const dynamic = 'force-static';

export const alt = 'PyCaret — Low-code machine learning for Python';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

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
        {/* Top: brand mark */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
          <div
            style={{
              width: 96,
              height: 96,
              borderRadius: 22,
              background: 'linear-gradient(135deg, #5B8DEF 0%, #3457B0 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 60,
              fontWeight: 700,
              color: '#fff',
            }}
          >
            P
          </div>
          <div
            style={{
              fontSize: 56,
              fontWeight: 700,
              letterSpacing: '-0.02em',
            }}
          >
            PyCaret
          </div>
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
