import { defineConfig, devices } from '@playwright/test'

const deployed = process.env.TEST_BASE_URL
const channel = process.env.PLAYWRIGHT_CHANNEL

export default defineConfig({
  testDir: './tests/browser',
  use: {
    baseURL: deployed ?? 'http://127.0.0.1:4173/penalty-trajectory-solver/',
    channel,
    launchOptions: { args: ['--enable-unsafe-swiftshader'] },
  },
  webServer: deployed ? undefined : {
    command: 'npm run preview -- --host 127.0.0.1 --strictPort',
    url: 'http://127.0.0.1:4173/penalty-trajectory-solver/',
    reuseExistingServer: !process.env.CI,
  },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'] } },
    { name: 'mobile', use: { ...devices['Pixel 7'] } },
  ],
})
