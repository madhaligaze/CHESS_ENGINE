// Magician panel on Chess.com. Reads the board, asks the local Magician app for a move when it
// is Magician's turn and plays it. Chess.com switches pages without reloading, so the script
// follows the address. The panel is shown on the Play page (hub), on chess.com/play/computer
// (bots) and chess.com/play/online (real players), playing on both. Leaving these pages stops
// it, and on every other page the panel is hidden.
(() => {
  const S = { on: false, busy: false, lastBoard: '', lastAsk: 0, ended: false, style: 'Human' };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // 'bots': game against a computer bot, 'online': game against a real player, 'hub': the Play page, '': any other page.
  function route() {
    const p = location.pathname.replace(/\/+$/, '');
    return p.startsWith('/play/computer') ? 'bots' : p.startsWith('/play/online') ? 'online' : p === '/play' ? 'hub' : '';
  }

  // ---------------------------------------------------------------- panel
  const panel = document.createElement('div');
  panel.style.cssText =
    'position:fixed;right:16px;bottom:16px;z-index:2147483647;width:270px;padding:12px 14px;border-radius:10px;' +
    'background:#262421;color:#eee;font:13px/1.4 system-ui,sans-serif;box-shadow:0 6px 24px rgba(0,0,0,.45)';
  panel.innerHTML = `
    <div style="font-weight:700;font-size:15px;margin-bottom:8px">♞ Magician</div>
    <label style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">Стиль
      <select id="mg-style" style="width:150px;padding:3px;border-radius:5px">
        <option value="Human">Сильный человек</option><option value="Tal">Таль</option>
        <option value="Carlsen">Карлсен</option><option value="Engine">Чистый движок</option>
      </select></label>
    <button id="mg-toggle" style="width:100%;padding:8px;border:0;border-radius:6px;background:#81b64c;color:#fff;font-weight:700;cursor:pointer">▶ Играть за меня</button>
    <div id="mg-status" style="margin-top:8px;color:#bbb;min-height:36px"></div>`;
  panel.style.display = 'none';
  document.body.appendChild(panel);
  const $ = (id) => panel.querySelector(id);
  const setStatus = (t) => { $('#mg-status').textContent = t; };
  chrome.storage.local.get(['style'], (v) => { if (v.style) { S.style = v.style; $('#mg-style').value = v.style; } });
  $('#mg-style').addEventListener('change', (e) => { S.style = e.target.value; chrome.storage.local.set({ style: S.style }); });

  function setOn(on) {
    S.on = on;
    S.lastBoard = '';
    $('#mg-toggle').textContent = on ? '■ Остановить' : '▶ Играть за меня';
    $('#mg-toggle').style.background = on ? '#b33430' : '#81b64c';
    setStatus(on ? 'Magician играет…' : 'Остановлено.');
    if (!on) chrome.runtime.sendMessage({ type: 'release' });
  }

  // On the Play page the button opens the bots page, in place when the page has a link to it.
  function openBots() {
    const link = document.querySelector('a[href*="/play/computer"]');
    if (link) link.click();
    else location.assign('/play/computer');
  }

  $('#mg-toggle').addEventListener('click', () => {
    if (where === 'hub') openBots();
    else if (where === 'bots' || where === 'online') setOn(!S.on);
  });

  // Shows, hides and relabels the panel when the address changes.
  let where = null;
  function follow() {
    if (!panel.isConnected) document.body.appendChild(panel);
    const now = route();
    if (now === where) return;
    where = now;
    if (S.on) setOn(false);
    panel.style.display = now ? '' : 'none';
    if (now === 'hub') {
      $('#mg-toggle').textContent = '▶ Открыть Play Bots';
      setStatus('Magician играет только с ботами Chess.com. Откройте Play Bots, выберите бота и нажмите Play.');
    } else if (now === 'bots') {
      $('#mg-toggle').textContent = '▶ Играть за меня';
      setStatus('Выберите бота и нажмите Play, затем эту кнопку.');
    } else if (now === 'online') {
      $('#mg-toggle').textContent = '▶ Играть за меня';
      setStatus('Нажмите кнопку чтобы Magician собрал данные из партии.');
    }
  }

  // ---------------------------------------------------------------- board
  function readBoard() {
    const board = document.querySelector('wc-chess-board');
    if (!board) return null;
    const r = board.getBoundingClientRect();
    const placement = {};
    for (const el of board.querySelectorAll('.piece')) {
      if (el.classList.contains('dragging')) continue;
      const pc = [...el.classList].find((c) => /^[wb][pnbrqk]$/.test(c));
      const sq = [...el.classList].find((c) => /^square-\d\d$/.test(c));
      if (pc && sq) placement[sq.slice(7)] = pc;
    }
    return { flipped: board.classList.contains('flipped'), rect: r, placement };
  }

  function placementFen(p) {
    const rows = [];
    for (let rank = 8; rank >= 1; rank--) {
      let row = '', empty = 0;
      for (let file = 1; file <= 8; file++) {
        const pc = p[`${file}${rank}`];
        if (!pc) { empty++; continue; }
        if (empty) { row += empty; empty = 0; }
        row += pc[0] === 'w' ? pc[1].toUpperCase() : pc[1];
      }
      rows.push(row + (empty || ''));
    }
    return rows.join('/');
  }

  function center(info, sq) {
    const f = 'abcdefgh'.indexOf(sq[0]), r = Number(sq[1]) - 1;
    const size = info.rect.width / 8;
    const col = info.flipped ? 7 - f : f;
    const row = info.flipped ? r : 7 - r;
    return { x: info.rect.left + (col + 0.5) * size, y: info.rect.top + (row + 0.5) * size };
  }

  function clockMs(el) {
    if (!el) return '';
    const m = el.innerText.trim().match(/(?:(\d+):)?(\d+)(?:\.(\d))?/);
    if (!m) return '';
    return ((Number(m[1] || 0) * 60 + Number(m[2])) * 1000 + Number(m[3] || 0) * 100).toString();
  }

  function clocks(flipped) {
    const bottom = document.querySelector('.clock-bottom'), top = document.querySelector('.clock-top');
    const b = clockMs(bottom), t = clockMs(top);
    return flipped ? { w: t, b } : { w: b, b: t };
  }

  function opponentName() {
    const el = document.querySelector('.player-component.player-top .cc-user-username-component, .player-top [data-cy="user-tagline-username"], .player-top .user-username-component');
    return el ? el.innerText.trim() : '';
  }

  function gameOverText() {
    const m = document.querySelector('.game-over-modal-content, .game-over-header-component, [class*="game-over-modal"]');
    return m && m.innerText ? m.innerText.trim().replace(/\s+/g, ' ') : '';
  }

  function resultFrom(text, ourWhite) {
    const t = text.toLowerCase();
    if (/you won|вы выиграли|победа/.test(t)) return ourWhite ? '1-0' : '0-1';
    if (/you lost|вы проиграли|поражение/.test(t)) return ourWhite ? '0-1' : '1-0';
    if (/draw|stalemate|ничья|пат/.test(t)) return '1/2-1/2';
    if (/white won|белые выиграли/.test(t)) return '1-0';
    if (/black won|чёрные выиграли|черные выиграли/.test(t)) return '0-1';
    return '*';
  }

  const api = (path, params) => chrome.runtime.sendMessage({ type: 'api', path, params });
  const click = (p) => chrome.runtime.sendMessage({ type: 'click', x: p.x, y: p.y });

  async function play(uci, info) {
    await click(center(info, uci.slice(0, 2)));
    await sleep(90);
    await click(center(info, uci.slice(2, 4)));
    if (uci.length > 4) {
      await sleep(350);
      const color = info.flipped ? 'b' : 'w';
      const piece = document.querySelector(`.promotion-window .${color}${uci[4]}, .promotion-piece.${color}${uci[4]}`);
      if (piece) {
        const r = piece.getBoundingClientRect();
        await click({ x: r.left + r.width / 2, y: r.top + r.height / 2 });
      }
    }
  }

  async function tick() {
    follow();
    if (!S.on || S.busy || (where !== 'bots' && where !== 'online')) return;
    const info = readBoard();
    if (!info) return;
    const over = gameOverText();
    if (over) {
      if (!S.ended) {
        S.ended = true;
        await api('/api/bridge/end', { result: resultFrom(over, !info.flipped), reason: over.slice(0, 80) });
        setStatus('Партия окончена: ' + over.slice(0, 80) + '. Запись в CHESS_ENGINE\\games');
      }
      return;
    }
    S.ended = false;
    const fen = placementFen(info.placement);
    if (fen === S.lastBoard && Date.now() - S.lastAsk < 2500) return;
    S.busy = true;
    S.lastBoard = fen;
    S.lastAsk = Date.now();
    try {
      const c = clocks(info.flipped);
      const res = await api('/api/bridge/observe', {
        board: fen, flipped: info.flipped ? 1 : 0, style: S.style, opponent: opponentName(), wclock: c.w, bclock: c.b,
      });
      if (!res || !res.ok) { setStatus((res && res.message) || 'Приложение Magician не отвечает'); return; }
      setStatus(res.status || '');
      if (res.move && S.on && (route() === 'bots' || route() === 'online')) await play(res.move, readBoard() || info);
    } finally {
      S.busy = false;
    }
  }

  follow();
  setInterval(tick, 250);
})();
