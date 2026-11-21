import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// The built page is written straight into the Python package, so FastAPI
// serves it and `pip install` ships it. `npm run dev` instead serves the page
// on port 5173 with instant reloads and forwards API calls to the Python
// server, which must be running (`presetfit serve`).
export default defineConfig({
  plugins: [react()],
  base: './',
  build: {
    outDir: '../src/presetfit/api/static',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
  },
})
