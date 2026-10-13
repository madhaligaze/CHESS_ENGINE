// Developer check of the extension: Chromium with the extension loaded, a guest game against
// the default Chess.com bot, Magician switched on through its panel. Needs `magician play`
// running. Screenshots go to tools/chesscom/test-screens.
//   node tools/chesscom/test_extension.cjs [seconds]
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const EXT = path.join(__dirname, 'extension');
const PROFILE = path.join(__dirname, '.test-profile');
const OUT = path.join(__dirname, 'test-screens');
const seconds = Number(process.argv[2] || 60);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const ctx = await chromium.launchPersistentContext(PROFILE, {
    headless: false,
    viewport: { width: 1400, height: 900 },
    args: [`--disable-extensions-except=${EXT}`, `--load-extension=${EXT}`],
  });
  const page = ctx.pages()[0] || (await ctx.newPage());
  await page.goto('https://www.chess.com/play/computer', { waitUntil: 'domcontentloaded' });
  await sleep(5000);
  const clickText = async (re) => {
    const loc = page.locator('button').filter({ hasText: re }).first();
    if (await loc.count()) await loc.click({ timeout: 5000 }).catch(() => {});
  };
  await clickText(/^\s*Start\s*$/i);
  await sleep(1000);
  await clickText(/^\s*Play\s*$/i);
  await sleep(3000);
  await page.screenshot({ path: path.join(OUT, '1_started.png') });
  await page.locator('#mg-toggle').click();
  for (let t = 0; t < seconds; t += 10) {
    await sleep(10000);
    const status = await page.locator('#mg-status').innerText().catch(() => '');
    console.log(`${t + 10}s: ${status}`);
    await page.screenshot({ path: path.join(OUT, `2_after_${t + 10}s.png`) });
  }
  await ctx.close();
})();
