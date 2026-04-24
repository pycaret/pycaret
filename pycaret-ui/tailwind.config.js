/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class', // dark-mode first; we toggle by class on <html>
  theme: {
    extend: {
      // Slate-leaning palette — analytical, not corporate. Dark-mode-first.
      colors: {
        ink: {
          // Darks (page bg, surfaces)
          950: '#0a0b0d',
          900: '#111317',
          800: '#181a1f',
          700: '#23262d',
          // Foregrounds
          200: '#d1d5db',
          100: '#e5e7eb',
          50: '#f3f4f6',
        },
        accent: {
          // Signal color — deliberately not blue. Teal = results.
          400: '#2dd4bf',
          500: '#14b8a6',
          600: '#0d9488',
        },
        success: { 500: '#10b981' },
        danger: { 500: '#ef4444', 600: '#dc2626' },
        warn: { 500: '#f59e0b' },
      },
      fontFamily: {
        sans: [
          'Inter',
          'ui-sans-serif',
          'system-ui',
          '-apple-system',
          'Segoe UI',
          'Roboto',
          'sans-serif',
        ],
        mono: [
          'JetBrains Mono',
          'ui-monospace',
          'SFMono-Regular',
          'Menlo',
          'Consolas',
          'monospace',
        ],
      },
      maxWidth: {
        // Single-column forms want a narrow reading measure.
        form: '32rem',
      },
    },
  },
  plugins: [],
};
