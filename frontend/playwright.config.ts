import {defineConfig, devices} from '@playwright/test';
export default defineConfig({
  testDir: './tests/browser', fullyParallel: false, workers: 1, retries: process.env.CI ? 1 : 0,
  timeout: 60000, use: {baseURL: 'http://127.0.0.1:4173', trace: 'retain-on-failure'},
  projects: [{name: 'desktop', use: {...devices['Desktop Chrome']}}, {name: 'mobile', use: {...devices['iPhone 13'], defaultBrowserType: 'chromium'}}],
  webServer: [
    {command: '../.venv/bin/python -m uvicorn backend.main:app --app-dir .. --host 127.0.0.1 --port 8001 --no-access-log', url: 'http://127.0.0.1:8001/health', reuseExistingServer: false, env: {ALLOWED_ORIGINS: 'http://127.0.0.1:4173', APP_ENV: 'production'}, timeout: 120000},
    {command: 'npm run build && npm run preview -- --port 4173', url: 'http://127.0.0.1:4173', reuseExistingServer: false, env: {VITE_API_URL: 'http://127.0.0.1:8001', VITE_APP_ENV: 'preview'}, timeout: 120000},
  ],
});
