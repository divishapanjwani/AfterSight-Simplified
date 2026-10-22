/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Port 5200 keeps AfterSight clear of other local projects; the API runs on 8020.
// AFTERSIGHT_API_URL lets the end-to-end check point the proxy at a test backend.
const apiUrl = process.env.AFTERSIGHT_API_URL ?? 'http://localhost:8020'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5200,
    strictPort: true,
    proxy: { '/api': apiUrl },
  },
  preview: {
    port: 5200,
    proxy: { '/api': apiUrl },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
  },
})
