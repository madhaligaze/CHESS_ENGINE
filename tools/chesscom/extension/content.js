// Magician panel on Chess.com. Reads the board, asks the local Magician app for a move when it
// is Magician's turn and plays it. Chess.com switches pages without reloading, so the script
// follows the address. The panel is shown on the Play page (hub), on chess.com/play/computer
// (bots), chess.com/play/online (real players) and chess.com/game/<id> (a started game), playing
// on the game pages. Moving between the lobby and a game keeps Magician playing; leaving the game
// area stops it, and on every other page the panel is hidden.
(() => {
  const HOST_ID = 'magician-chess-host';
  // A copy left behind by a reloaded extension (or a double injection) must not stay on the page.
  document.getElementById(HOST_ID)?.remove();
  const alive = () => { try { return !!chrome.runtime?.id; } catch (e) { return false; } };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const S = { on: false, busy: false, lastBoard: '', lastAsk: 0, ended: false, style: 'Human' };

  // 'bots': game against a computer bot, 'online': game against a real player, 'game': a started game
  // (Chess.com moves to /game/<id> once the game begins), 'hub': the Play page, '': any other page.
  function route() {
    const p = location.pathname.replace(/\/+$/, '');
    return p.startsWith('/play/computer') ? 'bots' : p.startsWith('/play/online') ? 'online' :
      p.startsWith('/game/') ? 'game' : p === '/play' ? 'hub' : '';
  }
  const playable = (r) => r === 'bots' || r === 'online' || r === 'game';

  // ---------------------------------------------------------------- panel
  // The panel lives in a shadow root so that Chess.com styles and re-renders cannot break it.
  const host = document.createElement('div');
  host.id = HOST_ID;
  host.style.cssText = 'position:fixed;right:16px;bottom:16px;z-index:2147483647;display:none';
  const panel = document.createElement('div');
  panel.style.cssText =
    'width:270px;padding:12px 14px;border-radius:10px;box-sizing:border-box;' +
    'background:#262421;color:#eee;font:13px/1.4 system-ui,sans-serif;box-shadow:0 6px 24px rgba(0,0,0,.45)';
  panel.innerHTML = `
    <div id="mg-title" style="font-weight:700;font-size:15px;margin-bottom:8px">♞ Magician</div>
    <label style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">Стиль
      <select id="mg-style" style="width:150px;padding:3px;border-radius:5px">
        <option value="Human">Сильный человек</option><option value="Tal">Таль</option>
        <option value="Carlsen">Карлсен</option><option value="Engine">Чистый движок</option>
      </select></label>
    <button id="mg-toggle" style="width:100%;padding:8px;border:0;border-radius:6px;background:#81b64c;color:#fff;font-weight:700;cursor:pointer">▶ Играть за меня</button>
    <div id="mg-status" style="margin-top:8px;color:#bbb;min-height:36px"></div>`;
  host.attachShadow({ mode: 'open' }).appendChild(panel);

  // Keeps the panel attached: Chess.com may replace <body> children while it switches pages.
  function mount() {
    const root = document.body || document.documentElement;
    if (root && host.parentNode !== root) root.appendChild(host);
  }
  mount();
  const $ = (id) => panel.querySelector(id);
  const setStatus = (t) => { $('#mg-status').textContent = t; };
  chrome.storage.local.get(['style'], (v) => { if (v.style) { S.style = v.style; $('#mg-style').value = v.style; } });
  $('#mg-style').addEventListener('change', (e) => { S.style = e.target.value; chrome.storage.local.set({ style: S.style }); });

  // The title bar drags the panel, and the place is kept between visits. The panel stays on screen.
  function place(x, y) {
    const w = host.offsetWidth || 270, h = host.offsetHeight || 120;
    host.style.left = Math.min(Math.max(0, x), window.innerWidth - w) + 'px';
    host.style.top = Math.min(Math.max(0, y), window.innerHeight - h) + 'px';
    host.style.right = 'auto';
    host.style.bottom = 'auto';
  }
  chrome.storage.local.get(['pos'], (v) => { if (v.pos) place(v.pos.x, v.pos.y); });
  const title = $('#mg-title');
  title.style.cursor = 'move';
  title.style.userSelect = 'none';
  title.addEventListener('pointerdown', (e) => {
    const r = host.getBoundingClientRect();
    const dx = e.clientX - r.left, dy = e.clientY - r.top;
    const move = (ev) => place(ev.clientX - dx, ev.clientY - dy);
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      const at = host.getBoundingClientRect();
      chrome.storage.local.set({ pos: { x: at.left, y: at.top } });
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  });

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
    else if (playable(where)) setOn(!S.on);
  });

  // Shows, hides and relabels the panel when the address changes.
  let where = null;
  function follow() {
    mount();
    const now = route();
    host.style.display = now ? '' : 'none';
    if (now === where) return;
    const prev = where;
    where = now;
    // Moving from the lobby into a game keeps Magician playing; leaving the game area stops it.
    if (S.on) {
      if (playable(prev) && playable(now)) return;
      setOn(false);
    }
    if (now === 'hub') {
      $('#mg-toggle').textContent = '▶ Открыть Play Bots';
      setStatus('Magician играет только с ботами Chess.com. Откройте Play Bots, выберите бота и нажмите Play.');
    } else if (now === 'bots') {
      $('#mg-toggle').textContent = '▶ Играть за меня';
      setStatus('Выберите бота и нажмите Play, затем эту кнопку.');
    } else if (now === 'online') {
      $('#mg-toggle').textContent = '▶ Играть за меня';
      setStatus('Нажмите кнопку чтобы Magician собрал данные из партии.');
    } else if (now === 'game') {
      $('#mg-toggle').textContent = '▶ Играть за меня';
      setStatus('Партия идёт. Нажмите кнопку, чтобы Magician играл за вас.');
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

  // A game is on the board when the address is /game/<id> or the Abort/Resign buttons are shown.
  // The lobby (/play/online before Start Game) shows the start position too, and it must not be
  // taken for the start of a game: then Magician would wait for the opponent's first move.
  function abortVisible() {
    return [...document.querySelectorAll('button')].some((b) => /^(Abort|Resign|Прервать|Сдаться)$/i.test((b.innerText || '').trim()));
  }
  function inGame() {
    return route() === 'game' || abortVisible();
  }

  // Chess.com shows "Your move ..." under our player while it is our turn. Only leaf elements count,
  // and the chat and move list are skipped so that messages typed there cannot change the answer.
  function ourTurn() {
    return [...document.querySelectorAll('body *')].some((el) =>
      el.children.length === 0 && /^(Your move|Ваш ход)\b/i.test((el.textContent || '').trim()) &&
      !el.closest('[class*="chat"], [class*="move-list"], [class*="vertical-move"]'));
  }

  function opponentName() {
    const el = document.querySelector('.player-component.player-top .cc-user-username-component, .player-top [data-cy="user-tagline-username"], .player-top .user-username-component');
    return el ? el.innerText.trim() : '';
  }

  // Only a visible game-over box counts: Chess.com can leave one from an earlier game in the page,
  // and reading it as the end of the current game would stop Magician at once.
  function gameOverText() {
    for (const m of document.querySelectorAll('.game-over-modal-content, .game-over-header-component, [class*="game-over-modal"]')) {
      if (m.getClientRects().length && m.innerText) return m.innerText.trim().replace(/\s+/g, ' ');
    }
    return '';
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
    if (!S.on || S.busy || !playable(where)) return;
    if (!inGame()) return;
    const info = readBoard();
    if (!info) return;
    // While Abort/Resign is on the page the game is still going, whatever an old game-over box says.
    const over = abortVisible() ? '' : gameOverText();
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
        ourTurn: ourTurn() ? 1 : 0,
      });
      if (!res || !res.ok) { setStatus((res && res.message) || 'Приложение Magician не отвечает'); return; }
      setStatus(res.status || '');
      if (res.move && S.on && playable(route())) await play(res.move, readBoard() || info);
    } finally {
      S.busy = false;
    }
  }

  // Chess.com changes the address without reloading, so the address is also watched on a timer and
  // on history events. An orphaned copy (extension reloaded) removes itself.
  const timers = [];
  function stop() {
    timers.forEach(clearInterval);
    observer.disconnect();
    host.remove();
  }
  const guarded = (fn) => () => {
    if (!alive()) return stop();
    try { const r = fn(); if (r && r.catch) r.catch((e) => console.warn('[Magician]', e)); } catch (e) { console.warn('[Magician]', e); }
  };
  const observer = new MutationObserver(guarded(follow));
  observer.observe(document.documentElement, { childList: true, subtree: true });
  for (const ev of ['popstate', 'hashchange', 'pageshow']) window.addEventListener(ev, guarded(follow));
  timers.push(setInterval(guarded(tick), 250));
  timers.push(setInterval(guarded(follow), 1000));
  follow();
})();
