/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Port 5200 keeps AfterSight clear of other local projects; the API runs on 8020.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5200,
    strictPort: true,
    proxy: { '/api': 'http://localhost:8020' },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
  },
})
