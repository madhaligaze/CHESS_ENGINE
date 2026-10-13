// Test extension on chess.com/play/online using Microsoft Edge
const { execSync } = require('child_process');
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const EXT = path.join(__dirname, 'extension');
const PROFILE = path.join(__dirname, '.test-profile-edge');
const OUT = path.join(__dirname, 'test-screens-edge');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  try {
    fs.mkdirSync(OUT, { recursive: true });
    console.log('📦 Extension path:', EXT);
    console.log('🌐 Starting Microsoft Edge...');

    // Find Edge executable
    const edgePath = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
    if (!fs.existsSync(edgePath)) {
      console.error('❌ Microsoft Edge not found at:', edgePath);
      process.exit(1);
    }

    const ctx = await chromium.launchPersistentContext(PROFILE, {
      executablePath: edgePath,
      headless: false,
      viewport: { width: 1400, height: 900 },
      args: [`--disable-extensions-except=${EXT}`, `--load-extension=${EXT}`],
    });

    console.log('✓ Edge started');
    const page = ctx.pages()[0] || (await ctx.newPage());

    console.log('📄 Navigating to chess.com/play/online...');
    await page.goto('https://www.chess.com/play/online', { waitUntil: 'domcontentloaded', timeout: 30000 });

    await sleep(2000);
    console.log('📸 Taking screenshot...');
    await page.screenshot({ path: path.join(OUT, 'edge_1_loaded.png') });

    // Check for test div
    const testDiv = await page.locator('div:has-text("MAGICIAN LOADED")').count();
    console.log(`✓ Test div found: ${testDiv > 0 ? 'YES ✅' : 'NO ❌'}`);

    // Check for panel
    const panel = await page.locator('#mg-toggle').count();
    console.log(`🎛️  Panel button found: ${panel > 0 ? 'YES ✅' : 'NO ❌'}`);

    const status = await page.locator('#mg-status').textContent().catch(() => null);
    console.log(`📊 Panel status: ${status || 'NOT FOUND'}`);

    // Get all console messages
    const messages = [];
    page.on('console', msg => {
      console.log('📝', msg.text());
      messages.push(msg.text());
    });

    await sleep(3000);
    await page.screenshot({ path: path.join(OUT, 'edge_2_after_3s.png') });

    console.log('\n✨ Test complete! Check screenshots in:', OUT);
    console.log(`📊 Total console messages: ${messages.length}`);

    await ctx.close();
  } catch (e) {
    console.error('❌ Error:', e.message);
    process.exit(1);
  }
})();
