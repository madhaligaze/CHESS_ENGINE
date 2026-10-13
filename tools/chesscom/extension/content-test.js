// Super simple test - if you see a red box in the corner, the extension loads
const testDiv = document.createElement('div');
testDiv.textContent = '✓ MAGICIAN LOADED';
testDiv.style.cssText = 'position:fixed;top:10px;right:10px;padding:10px;background:red;color:white;z-index:99999;font-weight:bold;border-radius:5px';
document.body.appendChild(testDiv);
console.log('[MAGICIAN TEST] Extension loaded!');
