// Test extension on chess.com/play/online
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const EXT = path.join(__dirname, 'extension');
const PROFILE = path.join(__dirname, '.test-profile-online');
const OUT = path.join(__dirname, 'test-screens-online');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  try {
    fs.mkdirSync(OUT, { recursive: true });
    console.log('📦 Extension path:', EXT);
    console.log('🌐 Starting browser...');

    const ctx = await chromium.launchPersistentContext(PROFILE, {
      headless: false,
      viewport: { width: 1400, height: 900 },
      args: [`--disable-extensions-except=${EXT}`, `--load-extension=${EXT}`],
    });

    const page = ctx.pages()[0] || (await ctx.newPage());
    console.log('📄 Navigating to chess.com/play/online...');
    await page.goto('https://www.chess.com/play/online', { waitUntil: 'domcontentloaded', timeout: 30000 });

    await sleep(3000);
    console.log('📸 Taking screenshot...');
    await page.screenshot({ path: path.join(OUT, 'online_1_loaded.png') });

    // Check for test div
    const testDiv = await page.locator('div:has-text("MAGICIAN LOADED")').count();
    console.log(`✓ Test div found: ${testDiv > 0 ? 'YES ✅' : 'NO ❌'}`);

    // Check for console logs
    const logs = [];
    page.on('console', msg => {
      if (msg.text().includes('[MAGICIAN') || msg.text().includes('Magician')) {
        console.log('📝 Console:', msg.text());
        logs.push(msg.text());
      }
    });

    await sleep(3000);
    await page.screenshot({ path: path.join(OUT, 'online_2_after_3s.png') });

    // Try to access panel elements
    const panel = await page.locator('#mg-toggle').count();
    console.log(`🎛️  Panel button found: ${panel > 0 ? 'YES ✅' : 'NO ❌'}`);

    const status = await page.locator('#mg-status').textContent().catch(() => null);
    console.log(`📊 Panel status: ${status || 'NOT FOUND'}`);

    console.log('\n✨ Test complete! Check screenshots in:', OUT);

    await ctx.close();
  } catch (e) {
    console.error('❌ Error:', e.message);
    process.exit(1);
  }
})();
