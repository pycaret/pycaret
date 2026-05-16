import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

/**
 * Vite config. Dev server proxies /api and /ws to the FastAPI backend so the
 * frontend can assume same-origin (no CORS headaches locally).
 *
 * In production the UI is a static build served by nginx; the API lives behind
 * the same origin via reverse proxy (see docker/docker-compose.prod.yml).
 */
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  server: {
    // PyCaret owns the 30xx/80xx +20 block locally (resumly: 3000/8000,
    // halani: 3011/8005, pycaret: 3020/8020) so all three projects can
    // run side-by-side without `kill -9`. See repo README for the map.
    port: 3020,
    proxy: {
      // ws:true is required so the dev server forwards the WebSocket
      // upgrade for /api/v1/runs/:id/events/ws — without it the upgrade
      // 404s, the EventStream shows "closed · 0 events", and the user
      // never sees the live engine event log even though the run is
      // emitting events fine to the DB.
      '/api': { target: 'http://127.0.0.1:8021', changeOrigin: true, ws: true },
      '/ws': { target: 'ws://127.0.0.1:8021', ws: true, changeOrigin: true },
      '/healthz': { target: 'http://127.0.0.1:8021', changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    target: 'es2022',
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./vitest.setup.ts'],
    css: false,
  },
});
