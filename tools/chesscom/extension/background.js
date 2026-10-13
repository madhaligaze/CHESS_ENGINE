// Talks to the local Magician app and performs clicks on the board for the content script.
// Clicks go through the debugger protocol, so the page receives them exactly like mouse
// clicks (Edge shows a "debugging this browser" bar while Magician is attached).

let port = null;

async function findPort() {
  for (let p = 8765; p < 8785; p++) {
    try {
      const r = await fetch(`http://127.0.0.1:${p}/api/state`);
      if (r.ok) {
        port = p;
        return p;
      }
    } catch (e) {
      // not this one
    }
  }
  return null;
}

const attached = new Set();

function attach(tabId) {
  if (attached.has(tabId)) return Promise.resolve();
  return new Promise((resolve, reject) => {
    chrome.debugger.attach({ tabId }, '1.3', () => {
      const e = chrome.runtime.lastError;
      if (e && !/already attached/i.test(e.message)) return reject(new Error(e.message));
      attached.add(tabId);
      resolve();
    });
  });
}

chrome.debugger.onDetach.addListener((source) => attached.delete(source.tabId));

function send(tabId, method, params) {
  return new Promise((resolve) => chrome.debugger.sendCommand({ tabId }, method, params, resolve));
}

async function click(tabId, x, y) {
  await attach(tabId);
  const base = { x, y, button: 'left', clickCount: 1 };
  await send(tabId, 'Input.dispatchMouseEvent', { type: 'mouseMoved', x, y });
  await send(tabId, 'Input.dispatchMouseEvent', { ...base, type: 'mousePressed' });
  await send(tabId, 'Input.dispatchMouseEvent', { ...base, type: 'mouseReleased' });
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    try {
      if (msg.type === 'api') {
        if (!port && !(await findPort())) {
          sendResponse({ ok: false, message: 'Приложение Magician не запущено: откройте ярлык Magician' });
          return;
        }
        const qs = new URLSearchParams(msg.params || {}).toString();
        try {
          const r = await fetch(`http://127.0.0.1:${port}${msg.path}?${qs}`, { method: 'POST' });
          sendResponse(await r.json());
        } catch (e) {
          port = null;
          sendResponse({ ok: false, message: 'Приложение Magician не отвечает' });
        }
      } else if (msg.type === 'click') {
        await click(sender.tab.id, msg.x, msg.y);
        sendResponse({ ok: true });
      } else if (msg.type === 'release') {
        if (attached.has(sender.tab.id)) chrome.debugger.detach({ tabId: sender.tab.id }, () => void chrome.runtime.lastError);
        sendResponse({ ok: true });
      }
    } catch (e) {
      sendResponse({ ok: false, message: String(e) });
    }
  })();
  return true;
});
