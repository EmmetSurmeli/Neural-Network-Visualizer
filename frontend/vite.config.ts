import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
export default defineConfig(({mode}) => {
  const env = loadEnv(mode, process.cwd(), 'VITE_');
  // Vercel's deployment environment overrides any accidentally copied production file.
  const environment = process.env.VERCEL_ENV || env.VITE_APP_ENV || 'development';
  return {plugins: [react(), tailwindcss()], define: {'import.meta.env.VITE_APP_ENV': JSON.stringify(environment)}, server: {port: 5173, strictPort: true, proxy: {'/api': {target: 'http://127.0.0.1:8000', rewrite: p => p.replace(/^\/api/, '')}}}};
});
