import type { Config } from 'tailwindcss';

/**
 * Tailwind tokens for pycaret.org.
 *
 * Palette mirrors the dashboard (`apps/web/src/index.css`) so docs +
 * landing share a brand. Inter for sans, JetBrains Mono for code.
 */
const config: Config = {
  content: ['./app/**/*.{ts,tsx,mdx}', './components/**/*.{ts,tsx}', './content/**/*.{md,mdx}'],
  theme: {
    extend: {
      colors: {
        ink: {
          50: '#F8FAFC',
          100: '#F1F5F9',
          200: '#E2E8F0',
          300: '#CBD5E1',
          400: '#94A3B8',
          500: '#64748B',
          600: '#475569',
          700: '#334155',
          800: '#1E293B',
          900: '#0F172A',
          950: '#020617',
        },
        accent: {
          50: '#EFF4FE',
          100: '#DDE7FD',
          200: '#BCD0FB',
          300: '#92B3F7',
          400: '#658FF2',
          500: '#5B8DEF',
          600: '#3F6FD9',
          700: '#3457B0',
          800: '#1F3A85',
          900: '#0F1F4D',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      typography: () => ({
        DEFAULT: {
          css: {
            maxWidth: 'none',
          },
        },
      }),
    },
  },
  plugins: [],
};

export default config;
