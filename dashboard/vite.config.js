import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    host: '0.0.0.0',
    proxy: {
      // Phase 1 — Prediction / Feature Engine
      '/api/v1/predict': { target: 'http://localhost:8004', changeOrigin: true },
      '/api/v1/features': { target: 'http://localhost:8003', changeOrigin: true },
      '/api/v1/alerts': { target: 'http://localhost:8005', changeOrigin: true },
      '/api/v1/acn': { target: 'http://localhost:8006', changeOrigin: true },
      // Phase 2 — New services
      '/api/v1/causal': { target: 'http://localhost:8007', changeOrigin: true },
      '/api/v1/ledger': { target: 'http://localhost:8008', changeOrigin: true },
      '/api/v1/chorus': { target: 'http://localhost:8009', changeOrigin: true },
      '/api/v1/fl': { target: 'http://localhost:8010', changeOrigin: true },
      '/api/v1/evacuation': { target: 'http://localhost:8011', changeOrigin: true },
      '/api/v1/mirror': { target: 'http://localhost:8012', changeOrigin: true },
      // Fallback
      '/api': { target: 'http://localhost:8004', changeOrigin: true },
    },
  },
})
