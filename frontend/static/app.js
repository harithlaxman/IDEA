/* Client state + SSE streaming.
 *
 * htmx swaps the sidebar/modal fragments; this script owns the query submit →
 * SSE stream → message append flow, which hx-sse handles awkwardly.
 * window.dwCurrentSessionId holds the active session id (null = new/unsaved).
 */

(function () {
  const form = document.getElementById('message-form');
  const messageInput = document.getElementById('message-input');
  const submitBtn = document.getElementById('submit-btn');
  const mainContent = document.getElementById('main-content');

  // --- Context meter -------------------------------------------------------
  // Tokens the agent's context carries, out of the model's window. The server
  // sends an exact provider count straight after each model call and an
  // estimate in between, so a `~` marks the numbers that will be corrected.
  const contextMeter = document.getElementById('context-meter');

  function compactTokens(n) {
    return n >= 1000 ? `${Math.round(n / 1000)}K` : String(n);
  }

  function setContextMeter(usage) {
    if (!contextMeter) return;
    if (!usage || !usage.window || !usage.tokens) {
      contextMeter.hidden = true;
      contextMeter.textContent = '';
      stampContextUsage(null);
      return;
    }
    const pct = Math.round((usage.tokens / usage.window) * 100);
    contextMeter.textContent =
      `${usage.exact ? '' : '~'}${usage.tokens.toLocaleString()} / ` +
      `${compactTokens(usage.window)} \u00b7 ${pct}%`;
    contextMeter.classList.toggle('warn', pct >= 80);
    contextMeter.hidden = false;
    stampContextUsage(usage);
  }

  // Write the live number back onto the marker fragments/main.html rendered.
  // Every turn ends with a sessionChanged that re-swaps the sidebar, and that
  // swap runs syncContextMeterFromDom — without this the stale fragment value
  // would clobber the number the stream just delivered.
  function stampContextUsage(usage) {
    const el = document.getElementById('context-usage');
    if (!el) return;
    if (!usage) {
      delete el.dataset.tokens;
      delete el.dataset.window;
      delete el.dataset.exact;
      return;
    }
    el.dataset.tokens = String(usage.tokens);
    el.dataset.window = String(usage.window);
    el.dataset.exact = usage.exact ? '1' : '';
  }

  // A reloaded or swapped-in session carries its meter as data attributes;
  // pages.frag_session reads them off the checkpoint. A *missing* marker is no
  // opinion rather than a reset: runTurn builds a new session's message list by
  // hand, without one, and the sidebar re-swaps on every sessionChanged — so
  // clearing here would wipe the number the stream had just delivered.
  function syncContextMeterFromDom() {
    const el = document.getElementById('context-usage');
    if (!el) return;
    if (!el.dataset.tokens) {
      setContextMeter(null);
      return;
    }
    setContextMeter({
      tokens: Number(el.dataset.tokens),
      window: Number(el.dataset.window),
      exact: !!el.dataset.exact,
    });
  }

  // --- Reports buttons enabled state --------------------------------------
  window.dwEnableReports = function () {
    document.getElementById('rate-btn').removeAttribute('disabled');
  };

  function disableReports() {
    document.getElementById('rate-btn').setAttribute('disabled', '');
  }


  // --- Logout -------------------------------------------------------------
  document.getElementById('logout-link')?.addEventListener('click', async (e) => {
    e.preventDefault();
    await fetch('/auth/logout', { method: 'POST' });
    window.location.href = '/login';
  });

  // --- Modal helpers -------------------------------------------------------
  window.dwCloseModal = function () {
    const backdrop = document.getElementById('modal-backdrop');
    backdrop.innerHTML = '';
  };

  // Fix rate button to target the current session id
  document.addEventListener('click', (e) => {
    const sid = window.dwCurrentSessionId;
    if (!sid) return;
    if (e.target.closest('#rate-btn')) {
      e.preventDefault();
      fetchAndShowModal(`/frag/rate/${sid}`);
    }
  });

  async function fetchAndShowModal(url) {
    const resp = await fetch(url, { headers: { 'Accept': 'text/html' } });
    if (!resp.ok) return;
    const html = await resp.text();
    document.getElementById('modal-backdrop').innerHTML = html;
  }

  // --- New Analysis --------------------------------------------------------
  document.getElementById('new-analysis-btn').addEventListener('click', () => {
    window.dwCurrentSessionId = null;
    disableReports();
    mainContent.innerHTML = document.getElementById('welcome-tpl').innerHTML;
    // A fresh analysis is never mid-clarification, whatever the last session was.
    clarifyPending = false;
    setClarifyMode(false);
    setContextMeter(null);
    messageInput.focus();
  });

  // --- Markdown ------------------------------------------------------------
  // A message body is drawn live and again by fragments/main.html on reload, so
  // both paths carry the raw markdown as element text and hand the element
  // here — Jinja autoescaping and textContent put the same inert text in the
  // DOM. marked passes raw HTML through, so DOMPurify is not optional.
  if (window.marked) window.marked.setOptions({ gfm: true, breaks: true });

  function renderMarkdownIn(root) {
    if (!window.marked || !window.DOMPurify) return; // leave the raw text as-is
    if (!root || !root.querySelectorAll) return;
    const els = [...root.querySelectorAll('[data-md]:not([data-md-done])')];
    if (root.matches && root.matches('[data-md]:not([data-md-done])')) els.push(root);
    for (const el of els) {
      el.innerHTML = window.DOMPurify.sanitize(window.marked.parse(el.textContent));
      el.classList.add('markdown-body');
      el.dataset.mdDone = '1';
    }
  }

  // History fragments arrive through htmx, not through this file.
  document.body.addEventListener('htmx:afterSwap', (e) => {
    renderMarkdownIn(e.target);
    syncClarifyModeFromDom();
    syncContextMeterFromDom();
  });

  // Restore clarify mode from the swapped-in fragment, not from memory: a
  // reload or session switch leaves `clarifyPending` stale, while
  // pages.frag_session stamps the checkpoint's truth on #message-list.
  function syncClarifyModeFromDom() {
    const list = document.getElementById('message-list');
    const paused = !!(list && list.dataset.clarifyPending);
    const confirming = !!(list && list.dataset.confirmPending);
    // Only on a real change — setClarifyMode focuses the composer, and the
    // sidebar re-swaps on every sessionChanged.
    if (paused !== clarifyPending) {
      clarifyPending = paused;
      setClarifyMode(paused);
    }
    if (confirming !== confirmPending) {
      confirmPending = confirming;
      setConfirmMode(confirming);
    }
  }

  // --- Tool log ------------------------------------------------------------
  // This and fragments/main.html render the same <details class="tool-log">
  // markup — keep them in step, or a bubble changes shape on reload.

  let liveLog = null; // tool log for the in-flight turn, until a bubble adopts it
  // The row the current text segment streams into, cached with its seq. Deltas
  // append to a live Text node because `textContent +=` recopies the whole
  // answer per token, which is O(n²).
  let textSeq = null;
  let textNode = null;

  // Trailing rows a streaming log shows before the user asks for the rest. A
  // long trace would otherwise push the composer off-screen.
  const TOOL_WINDOW = 5;

  function createToolLog() {
    const log = document.createElement('details');
    log.className = 'tool-log';
    // .windowed and the reveal button are live-only: both are gone by the time
    // a bubble adopts the log.
    log.innerHTML =
      '<summary></summary>' +
      '<ol class="tool-rows windowed"></ol>' +
      '<button type="button" class="tool-log-more" hidden></button>' +
      '<div class="thinking-row" hidden></div>';
    // A click mid-stream is the user taking over, so it suppresses the
    // automatic collapse on final. `data-auto` marks our own changes so their
    // toggle event isn't mistaken for one.
    log.addEventListener('toggle', () => {
      if (log.dataset.auto) {
        delete log.dataset.auto;
        return;
      }
      log.dataset.userToggled = '1';
      // Opening by hand asks for the whole trace, not just its tail.
      if (log.open) revealAllRows(log);
    });
    log.querySelector('.tool-log-more').addEventListener('click', (e) => {
      // Outside <summary>, but stop the event anyway so a future move inside
      // one can't toggle the log shut on the same click.
      e.preventDefault();
      revealAllRows(log);
    });
    setOpen(log, true);
    setSummary(log);
    return log;
  }

  function setOpen(log, open) {
    if (log.open === open) return;
    log.dataset.auto = '1';
    log.open = open;
  }

  // Drop the window for the rest of the turn. One-way, so a full trace doesn't
  // re-collapse under the user as later rows arrive.
  function revealAllRows(log) {
    log.querySelector('.tool-rows').classList.remove('windowed');
    setSummary(log);
  }

  function setSummary(log) {
    const rows = log.querySelector('.tool-rows');
    const n = rows.children.length;
    // The label counts tool calls only, so it matches main.html's on reload.
    // The window is about height, so it counts every row.
    const tools = rows.querySelectorAll('.tool-row:not(.tool-text)').length;
    // Before the first call, "0 tool calls" would label the opening paragraph.
    log.querySelector('summary').textContent =
      tools === 0 ? 'Working…' : `${tools} tool call${tools === 1 ? '' : 's'}`;
    const more = log.querySelector('.tool-log-more');
    if (!more) return; // already adopted by a bubble
    const hiddenCount = rows.classList.contains('windowed')
      ? Math.max(0, n - TOOL_WINDOW)
      : 0;
    more.hidden = hiddenCount === 0;
    more.textContent = `⋯ show all ${n}`;
  }

  function findRow(log, runId) {
    return [...log.querySelector('.tool-rows').children].find(
      (li) => li.dataset.runId === runId
    );
  }

  // Drop a log that never got adopted by a bubble (stream errored out).
  function discardLiveLog() {
    if (liveLog && liveLog.parentElement) liveLog.remove();
    liveLog = null;
  }
  // True while a clarification awaits an answer; keeps the submit handler's
  // finally from discarding the live log the prompt lives in.
  let clarifyPending = false;

  // Mirrors the .awaiting-answer div main.html renders on a paused session's
  // last bubble, so the marker survives a reload unchanged.
  function markAwaiting(asstBubble) {
    const el = document.createElement('div');
    el.className = 'awaiting-answer';
    el.textContent = 'Awaiting your answer';
    asstBubble.appendChild(el);
  }

  // True while a proposed preference awaits Approve/Reject. Unlike a
  // clarification, the answer is a click, so the composer is closed rather
  // than repurposed — a typed message can't reach a graph that is waiting on a
  // verdict, and letting one be typed would silently drop it.
  let confirmPending = false;

  function setClarifyMode(on) {
    const input = document.getElementById('message-input');
    const btn = document.getElementById('submit-btn');
    if (on) {
      input.placeholder = 'Type your clarification…';
      btn.textContent = 'Send';
      input.focus();
    } else {
      input.placeholder = "Enter your message (e.g., 'Explore high-level distributions')";
      btn.textContent = 'Run';
    }
  }

  function setConfirmMode(on) {
    const input = document.getElementById('message-input');
    const btn = document.getElementById('submit-btn');
    input.disabled = on;
    btn.disabled = on;
    input.placeholder = on
      ? 'Approve or reject the preference above to continue…'
      : "Enter your message (e.g., 'Explore high-level distributions')";
  }

  // Mirrors main.html's .confirm-actions so a live confirm and a reloaded one
  // are the same markup. Clicks are caught by the delegated listener below.
  function appendConfirmActions(asstBubble) {
    const row = document.createElement('div');
    row.className = 'confirm-actions';
    for (const [decision, label, cls] of [
      ['approve', 'Approve', 'btn-primary'],
      ['reject', 'Reject', 'btn-secondary'],
    ]) {
      const b = document.createElement('button');
      b.className = `btn ${cls}`;
      b.dataset.decision = decision;
      b.textContent = label;
      row.appendChild(b);
    }
    asstBubble.appendChild(row);
  }

  // Delegated, so buttons htmx swapped in on a reload work without rebinding.
  document.addEventListener('click', async (e) => {
    const btn = e.target.closest('.confirm-actions button[data-decision]');
    if (!btn) return;
    e.preventDefault();
    const row = btn.closest('.confirm-actions');
    row.querySelectorAll('button').forEach((b) => (b.disabled = true));
    try {
      await runTurn(null, { decision: btn.dataset.decision });
      row.remove();
    } catch (err) {
      row.querySelectorAll('button').forEach((b) => (b.disabled = false));
      showNotice(`Error: ${err.message}`);
    }
  });

  // --- Submit + SSE streaming ---------------------------------------------
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = messageInput.value.trim();
    if (!message) return;

    messageInput.value = '';
    submitBtn.disabled = true;

    try {
      await runTurn(message, { resume: clarifyPending });
    } catch (err) {
      showNotice(`Error: ${err.message}`);
    } finally {
      // A turn that ended on a confirm closed the composer on purpose; don't
      // reopen it here.
      submitBtn.disabled = confirmPending;
      if (!clarifyPending) discardLiveLog();
    }
  });

  async function ensureSession(message) {
    if (window.dwCurrentSessionId) return window.dwCurrentSessionId;
    const resp = await fetch('/sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: message.slice(0, 60) || 'New Analysis' }),
    });
    if (!resp.ok) throw new Error('Failed to create session');
    const sess = await resp.json();
    window.dwCurrentSessionId = sess.id;
    window.dwEnableReports();
    // Refresh sidebar
    document.body.dispatchEvent(new Event('sessionChanged'));
    return sess.id;
  }

  // One turn, whether it starts a conversation or resumes an interrupted one.
  // The two differ only in how they reach a session and a message list; the
  // bubbles, live log, and SSE loop are identical.
  // `decision` answers an approval gate — a resumed turn whose input is a
  // verdict, not text. The server still reads the pause off the checkpoint;
  // all the client contributes is which button was pressed.
  const DECISION_LABELS = { approve: 'Approved.', reject: 'Declined.' };

  async function runTurn(message, { resume = false, decision = null } = {}) {
    let messageList = document.getElementById('message-list');
    let sessionId;

    if (decision) {
      if (!messageList) return;
      sessionId = window.dwCurrentSessionId;
      confirmPending = false;
      setConfirmMode(false);
      messageList.removeAttribute('data-confirm-pending');
      message = DECISION_LABELS[decision] || 'Declined.';
    } else if (resume) {
      if (!messageList) return;
      sessionId = window.dwCurrentSessionId;
      // Leave clarify mode first: the resumed turn emits ordinary events and
      // takes the normal handleEvent path.
      clarifyPending = false;
      setClarifyMode(false);
      messageList.removeAttribute('data-clarify-pending');
      messageList.querySelectorAll('.awaiting-answer').forEach((el) => el.remove());
    } else {
      sessionId = await ensureSession(message);
      if (!messageList) {
        mainContent.innerHTML = '';
        messageList = document.createElement('div');
        messageList.className = 'message-list';
        messageList.id = 'message-list';
        messageList.dataset.sessionId = sessionId;
        mainContent.appendChild(messageList);
      }
      const oldNotice = document.getElementById('pending-notice');
      if (oldNotice) oldNotice.innerHTML = '';
    }

    appendTurnBubbles(messageList, message);

    // One endpoint for both: only the graph's input differs, and the server
    // picks that from the checkpoint — so a tab that forgot it was
    // mid-clarification still sends the right thing.
    const resp = await fetch(`/sessions/${sessionId}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(decision ? { decision } : { message }),
    });
    if (!resp.ok) {
      throw new Error(`Query failed: ${resp.status}`);
    }
    await readSSE(resp, messageList);
  }

  // User bubble (right) then assistant bubble (left), which hosts the live tool
  // log and gets the answer body appended after it on final. Row and bubble are
  // stashed on the log so handleEvent can reach them after nulling liveLog.
  function appendTurnBubbles(messageList, message) {
    const userRow = document.createElement('div');
    userRow.className = 'msg msg-user';
    const userBubble = document.createElement('div');
    userBubble.className = 'bubble bubble-user';
    userBubble.textContent = message;
    userRow.appendChild(userBubble);
    messageList.appendChild(userRow);

    const asstRow = document.createElement('div');
    asstRow.className = 'msg msg-assistant';
    const asstBubble = document.createElement('div');
    asstBubble.className = 'bubble bubble-assistant';
    asstRow.appendChild(asstBubble);
    messageList.appendChild(asstRow);

    liveLog = createToolLog();
    liveLog._asstRow = asstRow;
    liveLog._asstBubble = asstBubble;
    asstBubble.appendChild(liveLog);
    showThinking(liveLog, 'thinking…');
    textSeq = null;
    textNode = null;
  }

  // SSE via fetch + readable stream (EventSource doesn't support POST).
  async function readSSE(resp, messageList) {
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Parse SSE frames (separated by \n\n)
      let idx;
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const { event, data } = parseSSE(frame);
        handleEvent(event, data, messageList);
      }
    }
  }

  // Trimming each data line is safe only because every payload is one line of
  // JSON — json.dumps escapes newlines inside a delta. A raw delta would break it.
  function parseSSE(frame) {
    let event = 'message';
    let data = '';
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data += line.slice(5).trim();
    }
    try { data = JSON.parse(data); } catch { /* leave as string */ }
    return { event, data };
  }

  function showThinking(log, text) {
    const row = log.querySelector('.thinking-row');
    row.textContent = text;
    row.hidden = false;
  }

  function hideThinking(log) {
    log.querySelector('.thinking-row').hidden = true;
  }

  // Strip the live-only chrome so the adopted markup matches what
  // fragments/main.html renders from the persisted timeline.
  function finalizeLog(log) {
    log.querySelector('.thinking-row').remove();
    log.querySelector('.tool-log-more').remove();
    log.querySelector('.tool-rows').classList.remove('windowed');
    // Drop an empty log, but keep a prose-only one: that is what a failed turn
    // has instead of an answer. main.html gates on the same thing (`entries`).
    if (log.querySelectorAll('.tool-row').length === 0) {
      log.remove();
      return;
    }
    // "Working…" is live-only, so settle on the two strings main.html renders.
    const tools = log.querySelectorAll('.tool-row:not(.tool-text)').length;
    log.querySelector('summary').textContent =
      tools === 0 ? 'Details' : `${tools} tool call${tools === 1 ? '' : 's'}`;
    if (!log.dataset.userToggled) setOpen(log, false);
  }

  // Built as an element, not interpolated markup: the answer has to reach the
  // DOM as text for renderMarkdownIn. An error is a server string, not markdown.
  function appendBody(asstBubble, { text, error }) {
    const body = document.createElement('div');
    if (error) {
      body.className = 'message-error';
      body.textContent = error;
    } else {
      body.className = 'message-text';
      body.dataset.md = '';
      body.textContent = text || '';
    }
    asstBubble.appendChild(body);
    renderMarkdownIn(asstBubble);
  }

  function handleEvent(event, data, messageList) {
    if (event === 'usage') {
      setContextMeter(data);
      return;
    }
    if (!liveLog) return;

    if (event === 'status') {
      showThinking(liveLog, data.text || 'working…');
    } else if (event === 'text') {
      // `seq` is the entry's index in the timeline, so deltas land in the row
      // their own segment opened even after a tool call interrupts.
      hideThinking(liveLog);
      if (textSeq !== data.seq || !textNode) {
        const rows = liveLog.querySelector('.tool-rows');
        let li = rows.querySelector(`li[data-seq="${data.seq}"]`);
        if (!li) {
          li = document.createElement('li');
          li.className = 'tool-row tool-text';
          li.dataset.seq = String(data.seq);
          rows.appendChild(li);
          setSummary(liveLog);
        }
        textNode = li.firstChild || li.appendChild(document.createTextNode(''));
        textSeq = data.seq;
      }
      textNode.appendData(data.delta || '');
    } else if (event === 'tool') {
      if (data.phase === 'start') {
        hideThinking(liveLog); // the tool row replaces it until the tool returns
        const li = document.createElement('li');
        li.className = 'tool-row pending';
        li.dataset.runId = data.run_id || '';
        li.textContent = data.name || 'tool';
        liveLog.querySelector('.tool-rows').appendChild(li);
        setSummary(liveLog);
      } else if (data.phase === 'end') {
        const li = findRow(liveLog, data.run_id || '');
        if (li) li.className = `tool-row ${data.ok === false ? 'error' : 'done'}`;
      }
    } else if (event === 'clarify') {
      // Render the question as the message body (the same path a final takes),
      // then put the input bar into clarify mode. Presentation only — the
      // server reads the pending interrupt from the checkpoint either way.
      clarifyPending = true;
      hideThinking(liveLog);

      const log = liveLog;
      liveLog = null;  // freed for the resumed turn's fresh log
      finalizeLog(log);
      appendBody(log._asstBubble, { text: data.question });
      markAwaiting(log._asstBubble);
      // The marker main.html stamps from the checkpoint, set live too: the
      // sessionChanged below makes syncClarifyModeFromDom read it back, and an
      // unmarked list would clear the mode just set.
      if (messageList) messageList.dataset.clarifyPending = '1';
      // A clarification is a persisted turn, so it needs an id for the rate
      // modal to target.
      log._asstRow.id = `message-${data.message_id}`;

      setClarifyMode(true);
      document.body.dispatchEvent(new Event('sessionChanged'));
    } else if (event === 'confirm') {
      // The other pause shape. Renders like a clarify — the proposal is the
      // message body — but the answer is a click, so the composer closes and
      // Approve/Reject take its place.
      confirmPending = true;
      hideThinking(liveLog);

      const log = liveLog;
      liveLog = null; // freed for the resumed turn's fresh log
      finalizeLog(log);
      appendBody(log._asstBubble, {
        text: `Save this preference?\n\n"${data.preference || ''}"`,
      });
      appendConfirmActions(log._asstBubble);
      // Set live too, so the sessionChanged below doesn't have
      // syncClarifyModeFromDom read an unmarked list and clear the mode.
      if (messageList) messageList.dataset.confirmPending = '1';
      log._asstRow.id = `message-${data.message_id}`;

      setConfirmMode(true);
      document.body.dispatchEvent(new Event('sessionChanged'));
    } else if (event === 'final') {
      const log = liveLog;
      liveLog = null;
      clarifyPending = false;
      setClarifyMode(false);
      confirmPending = false;
      setConfirmMode(false);
      if (messageList) {
        messageList.removeAttribute('data-clarify-pending');
        messageList.removeAttribute('data-confirm-pending');
      }
      const asstRow = log._asstRow;
      // On a success the trailing text entry is the answer, and it renders as
      // the body below — so drop its live row rather than show it twice. A
      // failed turn keeps its prose. Same rule as Message.log_entries.
      const timeline = data.timeline || [];
      if (!data.error && timeline.length && timeline[timeline.length - 1].kind === 'text') {
        const answerRow = log.querySelector(`li[data-seq="${timeline.length - 1}"]`);
        if (answerRow) answerRow.remove();
      }
      finalizeLog(log);
      // Answer body lands after the (kept) tool log in the assistant bubble.
      appendBody(log._asstBubble, { text: data.text, error: data.error });
      asstRow.id = `message-${data.message_id}`;
      // A failed turn keeps its bubble, with the error as the body — as
      // main.html draws it — so a refresh doesn't grow a new reply.
      if (data.failed) {
        showNotice("Sorry—couldn't complete that request. Try rephrasing your message.");
      }
      // Refresh sidebar (title/message count may have changed)
      document.body.dispatchEvent(new Event('sessionChanged'));
    }
  }
  function showNotice(msg) {
    let notice = document.getElementById('pending-notice');
    if (!notice) {
      notice = document.createElement('div');
      notice.id = 'pending-notice';
      mainContent.appendChild(notice);
    }
    notice.innerHTML = `<div class="pending-notice">${escapeHtml(msg)}</div>`;
  }

  function escapeHtml(s) {
    return String(s || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
})();
