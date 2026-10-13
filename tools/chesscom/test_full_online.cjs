// Full test: extension on /play/online, panel visible, can move pieces
const { execSync } = require('child_process');
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const EXT = path.join(__dirname, 'extension');
const PROFILE = path.join(__dirname, '.test-full-online');
const OUT = path.join(__dirname, 'test-full-online');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  try {
    fs.mkdirSync(OUT, { recursive: true });
    console.log('🚀 Starting full test: Extension on /play/online');
    console.log('📦 Extension:', EXT);

    const edgePath = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
    const ctx = await chromium.launchPersistentContext(PROFILE, {
      executablePath: edgePath,
      headless: false,
      viewport: { width: 1400, height: 900 },
      args: [`--disable-extensions-except=${EXT}`, `--load-extension=${EXT}`],
    });

    const page = ctx.pages()[0] || (await ctx.newPage());

    // Step 1: Navigate to /play/online
    console.log('\n📄 Step 1: Navigating to https://www.chess.com/play/online...');
    await page.goto('https://www.chess.com/play/online', { waitUntil: 'domcontentloaded', timeout: 30000 });
    await sleep(2000);

    // Step 2: Check if panel exists
    console.log('🔍 Step 2: Checking for Magician panel...');
    const panelVisible = await page.locator('div:has-text("♞ Magician")').count();
    console.log(`   ${panelVisible > 0 ? '✅' : '❌'} Panel visible: ${panelVisible > 0}`);

    const toggleBtn = await page.locator('#mg-toggle').count();
    console.log(`   ${toggleBtn > 0 ? '✅' : '❌'} Toggle button found: ${toggleBtn > 0}`);

    const status = await page.locator('#mg-status').textContent().catch(() => null);
    console.log(`   📊 Status text: "${status}"`);

    // Step 3: Take screenshot of panel
    console.log('📸 Taking screenshot of page with panel...');
    await page.screenshot({ path: path.join(OUT, '1_panel_visible.png'), fullPage: false });

    // Step 4: Start a game (click "Start Game" button)
    console.log('\n🎮 Step 4: Starting game...');
    try {
      const startBtn = await page.locator('button:has-text("Start Game")').first();
      if (await startBtn.count()) {
        console.log('   ▶ Clicking "Start Game"...');
        await startBtn.click({ timeout: 5000 });
        await sleep(3000);
      }
    } catch (e) {
      console.log('   ⚠️  Could not click Start Game:', e.message);
    }

    // Step 5: Check board exists
    console.log('\n🎯 Step 5: Checking chess board...');
    const board = await page.locator('wc-chess-board').count();
    console.log(`   ${board > 0 ? '✅' : '❌'} Chess board found: ${board > 0}`);

    if (board > 0) {
      const pieces = await page.locator('wc-chess-board .piece').count();
      console.log(`   ${pieces > 0 ? '✅' : '❌'} Pieces found: ${pieces}`);
    }

    await page.screenshot({ path: path.join(OUT, '2_game_started.png'), fullPage: false });

    // Step 6: Verify extension can access page
    console.log('\n🔗 Step 6: Checking extension page access...');
    const bodyTagExists = await page.evaluate(() => !!document.body);
    console.log(`   ${bodyTagExists ? '✅' : '❌'} document.body exists: ${bodyTagExists}`);

    // Step 7: Check if panel is still visible after game starts
    console.log('\n✨ Step 7: Final panel check...');
    const panelStillVis = await page.locator('#mg-toggle').isVisible();
    console.log(`   ${panelStillVis ? '✅' : '❌'} Panel still visible: ${panelStillVis}`);

    // Step 8: Get final screenshot
    console.log('📸 Taking final screenshot...');
    await page.screenshot({ path: path.join(OUT, '3_final_state.png'), fullPage: false });

    console.log('\n✅ Test complete!');
    console.log(`📁 Screenshots saved to: ${OUT}`);
    console.log(`   - 1_panel_visible.png`);
    console.log(`   - 2_game_started.png`);
    console.log(`   - 3_final_state.png`);

    // Check files exist
    const files = fs.readdirSync(OUT).filter(f => f.endsWith('.png'));
    console.log(`\n📊 Generated ${files.length} screenshots`);

    await ctx.close();
  } catch (e) {
    console.error('❌ Error:', e.message);
    process.exit(1);
  }
})();
