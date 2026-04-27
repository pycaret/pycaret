/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  // Light theme by default. We keep the `class` strategy in case we
  // re-introduce a dark toggle later, but no `dark` class is set on
  // <html>, so dark variants are inert today.
  darkMode: 'class',
  theme: {
    extend: {
      // Neutral grayscale, ascending. Modeled after Tailwind's `zinc`
      // scale — analytical, calm, plays well with one accent.
      //
      // Convention: low number = light surface, high number = dark text.
      colors: {
        ink: {
          50:  '#fafafa', // page background (off-white)
          100: '#f4f4f5', // sidebar / subtle surfaces / hover backgrounds
          200: '#e4e4e7', // borders, dividers
          300: '#d4d4d8', // stronger borders, disabled outlines
          400: '#a1a1aa', // placeholder, very muted text
          500: '#71717a', // muted body text, hint text
          600: '#52525b', // secondary text, icons
          700: '#3f3f46', // default body text
          800: '#27272a', // headings, strong text
          900: '#18181b', // sharpest text (rare; usually 800 is enough)
          950: '#09090b', // logo / max contrast brand surfaces
        },
        // Brand accent — teal. The same hue used in the docs site so the
        // marketing surface and the app feel like one product.
        accent: {
          50:  '#f0fdfa',
          100: '#ccfbf1',
          200: '#99f6e4',
          300: '#5eead4',
          400: '#2dd4bf',
          500: '#14b8a6', // primary accent
          600: '#0d9488', // hover / pressed
          700: '#0f766e',
          800: '#115e59',
          900: '#134e4a',
        },
        success: {
          50:  '#ecfdf5',
          500: '#10b981',
          600: '#059669',
        },
        danger: {
          50:  '#fef2f2',
          500: '#ef4444',
          600: '#dc2626',
        },
        warn: {
          50:  '#fffbeb',
          500: '#f59e0b',
          600: '#d97706',
        },
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
        form: '32rem',
      },
      boxShadow: {
        // Subtle elevation for cards, popovers. No heavy shadows.
        'soft-1': '0 1px 2px 0 rgb(0 0 0 / 0.04)',
        'soft-2': '0 1px 3px 0 rgb(0 0 0 / 0.06), 0 1px 2px -1px rgb(0 0 0 / 0.06)',
        'soft-3': '0 4px 6px -1px rgb(0 0 0 / 0.06), 0 2px 4px -2px rgb(0 0 0 / 0.06)',
      },
    },
  },
  plugins: [],
};
