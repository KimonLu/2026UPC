import { expect, test } from '@playwright/test'

test('refits the problem observations and plots the new goal', async ({ page }) => {
  await page.goto('./')
  await expect(page.getByRole('heading', { name: 'Three-Point Prediction' })).toBeVisible()
  await expect(page.getByTestId('goal-result')).toHaveText('(2.604, 1.962) m')
  await expect(page.getByTestId('goal-marker')).toBeVisible()
  await expect(page.locator('#scene canvas')).toBeVisible()
  await expect(page.getByLabel('x at 0.05 s')).toHaveValue('1.5')

  await page.getByLabel('Wind x').fill('5')
  await expect(page.getByTestId('goal-marker')).toHaveCount(0)
  await page.getByRole('button', { name: 'Calculate' }).click()
  await expect(page.getByTestId('goal-result')).toHaveText('(2.621, 1.970) m')
  await expect(page.getByTestId('goal-marker')).toBeVisible()
})

test('opens and refreshes forward simulation, then shows first contact', async ({ page }) => {
  await page.goto('./#/forward')
  await page.reload()
  await expect(page.getByRole('heading', { name: 'Forward Simulation' })).toBeVisible()
  await expect(page.getByTestId('goal-result')).toHaveText('(2.604, 1.962) m')
  await page.getByLabel('Wind x').fill('5')
  await page.getByRole('button', { name: 'Calculate' }).click()
  await expect(page.getByTestId('goal-result')).not.toHaveText('(2.604, 1.962) m')
  await expect(page.getByTestId('goal-marker')).toBeVisible()
  await expect(page.getByLabel('Initial velocity x')).toHaveValue('30.41543629298755')

  await page.getByLabel('Initial velocity z').fill('0')
  await page.getByRole('button', { name: 'Calculate' }).click()
  await expect(page.getByRole('status')).toHaveText('Ground contact')
  await expect(page.getByTestId('goal-marker')).toHaveCount(0)
  await page.getByRole('link', { name: 'Three-Point Prediction' }).click()
  await expect(page.getByRole('heading', { name: 'Three-Point Prediction' })).toBeVisible()
})

test('draws the sampled region and clears it for exact observations', async ({ page }) => {
  await page.goto('./')
  await expect(page.getByTestId('prediction-region')).toBeVisible()
  await expect(page.getByTestId('sample-counts')).toContainText('1956 / 2000')
  await page.getByLabel('Position noise').selectOption('0')
  await page.getByRole('button', { name: 'Calculate' }).click()
  await expect(page.getByTestId('goal-marker')).toBeVisible()
  await expect(page.getByTestId('prediction-region')).toHaveCount(0)
  await expect(page.getByTestId('sample-counts')).toBeHidden()
})

test('moves the 3D camera and restores its default view', async ({ page }, testInfo) => {
  await page.goto('./')
  await expect(page.getByTestId('goal-marker')).toBeVisible()
  const canvas = page.locator('#scene canvas')
  const original = await canvas.screenshot()
  const box = await canvas.boundingBox()
  expect(box).not.toBeNull()
  if (!box) return
  const x = box.x + box.width / 2
  const y = box.y + box.height / 2
  if (testInfo.project.name === 'mobile') {
    const input = await page.context().newCDPSession(page)
    const points = (radius: number) => [{ x: x - radius, y }, { x: x + radius, y }]
    await input.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: points(28) })
    await input.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: points(60) })
    await input.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] })
    await input.detach()
  } else {
    await page.mouse.move(x, y)
    await page.mouse.down()
    await page.mouse.move(x + 80, y + 30, { steps: 8 })
    await page.mouse.up()
  }
  expect((await canvas.screenshot()).equals(original)).toBe(false)
  await page.getByRole('button', { name: 'Reset view' }).click()
  await expect.poll(async () => (await canvas.screenshot()).equals(original)).toBe(true)
})
