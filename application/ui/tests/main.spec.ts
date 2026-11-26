import { expect, test } from '@anomalib-studio/test-fixtures';

test.describe('Anomalib Studio', () => {
    test('Allows users to inspect', async ({ page }) => {
        await page.goto('/', { waitUntil: 'domcontentloaded' });

        await expect(page.getByText(/Anomalib Studio/i)).toBeVisible();
        await expect(page.getByRole('heading', { name: 'Normal images', exact: true })).toBeVisible();
        await expect(page.getByRole('button', { name: 'Train model', exact: true })).toBeDisabled();
    });

    test('Uploads images serially and finishes after a connection reset', async ({ page }) => {
        await page.goto('/', { waitUntil: 'domcontentloaded' });
        const uploadButton = page.getByRole('button', { name: 'Upload images', exact: true });
        await expect(uploadButton).toBeEnabled();

        let requests = 0;
        let inFlight = 0;
        let peakInFlight = 0;
        await page.route('**/api/projects/*/images', async (route) => {
            if (route.request().method() !== 'POST') {
                await route.fallback();
                return;
            }

            const requestNumber = ++requests;
            inFlight++;
            peakInFlight = Math.max(peakInFlight, inFlight);
            await new Promise((resolve) => setTimeout(resolve, 150));
            inFlight--;

            if (requestNumber === 2) {
                await route.abort('connectionreset');
            } else {
                await route.fulfill({ status: 201, json: {} });
            }
        });

        const files = ['first.png', 'second.png', 'third.png'].map((name) => ({
            name,
            mimeType: 'image/png',
            buffer: Buffer.from('image'),
        }));

        await page.locator('input[type="file"]').setInputFiles(files);

        await expect(page.getByText('Upload complete (1 of 3 failed)')).toBeVisible();
        await expect(uploadButton).toBeEnabled();
        expect(requests).toBe(3);
        expect(peakInFlight).toBe(1);
    });
});
