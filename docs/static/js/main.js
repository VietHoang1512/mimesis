/* MIMESIS project page: charts and interactions. Vanilla JS, no dependencies.
 * All data comes from static/js/data.js. Charts are inline SVG styled through
 * CSS classes, so the light/dark theme switch recolors them without a redraw.
 * Every chart has a table view, and every value shown on hover is also in it.
 */
(() => {
  "use strict";
  const D = window.MIMESIS_DATA;
  if (!D) return;

  const SVGNS = "http://www.w3.org/2000/svg";
  const FONT = '13px system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  // ------------------------------------------------------------------ DOM
  function setAttrs(node, attrs) {
    if (!attrs) return;
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "text") node.textContent = v;
      else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : String(v));
    }
  }
  function append(node, kids) {
    for (const k of kids.flat(Infinity)) {
      if (k === null || k === undefined || k === false) continue;
      node.append(k instanceof Node ? k : String(k)); // strings become text nodes, never HTML
    }
    return node;
  }
  const el = (tag, attrs, ...kids) => { const n = document.createElement(tag); setAttrs(n, attrs); return append(n, kids); };
  const sv = (tag, attrs, ...kids) => { const n = document.createElementNS(SVGNS, tag); setAttrs(n, attrs); return append(n, kids); };

  const ctx = document.createElement("canvas").getContext("2d");
  function textWidth(str, font = FONT) { ctx.font = font; return ctx.measureText(str).width; }
  function fitText(str, maxW, font = FONT) {
    if (textWidth(str, font) <= maxW) return str;
    let s = str;
    while (s.length > 1 && textWidth(s + "…", font) > maxW) s = s.slice(0, -1);
    return s + "…";
  }

  const MINUS = "−";
  const fmt = (v, d = 1) => (v < 0 ? MINUS : "") + Math.abs(v).toFixed(d);
  const fmtSigned = (v, d = 1) => (v > 0 ? "+" : v < 0 ? MINUS : "") + Math.abs(v).toFixed(d);
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

  function niceStep(span, count) {
    const raw = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const n = raw / mag;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * mag;
  }
  function niceDomain(lo, hi, count) {
    const step = niceStep(hi - lo || 1, count);
    const a = Math.floor(lo / step + 1e-9) * step;
    const b = Math.ceil(hi / step - 1e-9) * step;
    const ticks = [];
    for (let v = a; v <= b + 1e-9; v += step) ticks.push(Math.round(v * 1e6) / 1e6);
    return { lo: a, hi: b, ticks };
  }

  // Horizontal bar: square at the baseline x0, 4px rounded corners at the data end x1.
  function hbarPath(x0, x1, y, h) {
    const w = Math.abs(x1 - x0);
    if (w < 0.5) return null;
    const r = Math.min(4, w, h / 2);
    const s = x1 >= x0 ? 1 : -1;
    return `M${x0},${y}H${x1 - s * r}Q${x1},${y} ${x1},${y + r}V${y + h - r}Q${x1},${y + h} ${x1 - s * r},${y + h}H${x0}Z`;
  }
  // Column: square at the baseline yBase, rounded at the top yTop.
  function vbarPath(x, w, yBase, yTop) {
    const h = yBase - yTop;
    if (h < 0.5) return null;
    const r = Math.min(4, h, w / 2);
    return `M${x},${yBase}V${yTop + r}Q${x},${yTop} ${x + r},${yTop}H${x + w - r}Q${x + w},${yTop} ${x + w},${yTop + r}V${yBase}Z`;
  }

  // Render `draw(width)` into `container` and redraw when its width changes.
  function mount(container, draw) {
    let lastW = 0;
    let raf = 0;
    const render = (force) => {
      raf = 0;
      const w = Math.floor(container.clientWidth);
      if (!w || (!force && w === lastW)) return;
      lastW = w;
      container.replaceChildren(draw(w));
    };
    new ResizeObserver(() => { if (!raf) raf = requestAnimationFrame(() => render(false)); }).observe(container);
    render(false);
    return { redraw: () => render(true) };
  }

  const TYPE_LABEL = {
    ours: "MIMESIS (ours)",
    frontier: "Frontier API model",
    released: "Released user simulator",
    pretrained: "Pretrained model"
  };

  // -------------------------------------------------------------- tooltip
  const tip = $("#viz-tip");
  function placeTip(target, ev) {
    const pad = 14;
    const tw = tip.offsetWidth;
    const th = tip.offsetHeight;
    let x;
    let y;
    if (ev && typeof ev.clientX === "number" && (ev.clientX || ev.clientY)) {
      x = ev.clientX + pad;
      y = ev.clientY + pad;
      if (x + tw > window.innerWidth - 8) x = ev.clientX - tw - pad;
      if (y + th > window.innerHeight - 8) y = ev.clientY - th - pad;
    } else {
      const r = target.getBoundingClientRect();
      x = r.left + r.width / 2 - tw / 2;
      y = r.top - th - 8;
      if (y < 8) y = r.bottom + 8;
    }
    tip.style.left = Math.max(8, Math.min(x, window.innerWidth - tw - 8)) + "px";
    tip.style.top = Math.max(8, y) + "px";
  }
  function showTip(target, ev, t) {
    tip.replaceChildren();
    if (t.title) tip.append(el("div", { class: "viz-tip-title", text: t.title }));
    for (const r of t.rows) {
      tip.append(el("div", { class: "viz-tip-row" },
        el("span", { class: "viz-tip-key " + (r.key || ""), "aria-hidden": "true" }),
        el("strong", { text: r.value }),
        el("span", { class: "l", text: r.label || "" })));
    }
    if (t.note) tip.append(el("div", { class: "viz-tip-note", text: t.note }));
    tip.hidden = false;
    placeTip(target, ev);
  }
  function hideTip() { tip.hidden = true; }
  function bindTip(node, getTip) {
    node.addEventListener("pointerenter", (e) => showTip(node, e, getTip()));
    node.addEventListener("pointermove", (e) => placeTip(node, e));
    node.addEventListener("pointerleave", hideTip);
    node.addEventListener("focus", () => showTip(node, null, getTip()));
    node.addEventListener("blur", hideTip);
  }
  window.addEventListener("scroll", hideTip, { passive: true });

  // ---------------------------------------------------------- table views
  const tableRenderers = {};
  function renderTable(container, caption, head, rows) {
    const table = el("table", null,
      caption ? el("caption", { text: caption }) : null,
      el("thead", null, el("tr", null, head.map((h) => el("th", { scope: "col", text: h })))),
      el("tbody", null, rows.map((r) => el("tr", { class: r.hl ? "hl" : null },
        r.cells.map((c, i) => (i === 0 ? el("th", { scope: "row", text: c }) : el("td", { class: /^[−+-]?\d/.test(c) ? null : "txt", text: c })))))));
    container.replaceChildren(table);
  }
  $$(".table-toggle").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.target;
      const on = btn.getAttribute("aria-pressed") !== "true";
      btn.setAttribute("aria-pressed", String(on));
      $("#" + key + "-table").hidden = !on;
      $("#" + key + "-chart").hidden = on;
      $$(".legend, .axis-notes", btn.closest(".chart-card")).forEach((n) => { n.hidden = on; });
      if (on && tableRenderers[key]) tableRenderers[key]();
    });
  });
  const tableOpen = (key) => $(`.table-toggle[data-target="${key}"]`).getAttribute("aria-pressed") === "true";

  // ---------------------------------------------------------- chat excerpts
  // A transcript is a list of segments; each segment is a list of turns
  // {r: "a" | "u", t: text, th?: thought (string or paragraphs), who?: label}.
  // Segments are separated by an ellipsis that marks omitted turns.
  function chatTurn(turn, userLabel) {
    const user = turn.r === "u";
    const thought = turn.th ? (Array.isArray(turn.th) ? turn.th : [turn.th]) : null;
    return el("li", { class: "chat-turn " + (user ? "chat-user" : "chat-agent") },
      el("span", { class: "chat-who", text: turn.who || (user ? userLabel : "Agent") }),
      user && thought ? el("p", { class: "chat-thought" },
        el("span", { class: "lab", text: "Thought, hidden from the agent" }),
        thought.map((para) => el("span", { class: "para", text: para }))) : null,
      el("p", { class: "chat-say", text: turn.t }));
  }
  function chatList(segments, userLabel, label, id) {
    const ol = el("ol", { class: "chat", "aria-label": label, id });
    segments.forEach((seg, i) => {
      if (i) ol.append(el("li", { class: "chat-gap", text: "· · ·", "aria-label": "Turns omitted" }));
      seg.forEach((turn) => ol.append(chatTurn(turn, userLabel)));
    });
    return ol;
  }

  // ---------------------------------------------------- 01 tau-bench chart
  (function tau() {
    const T = D.tauSuccess;
    const rows = T.rows;
    const draw = (W) => {
      const rowH = 30;
      const barH = 16;
      const top = 26;
      const labelW = Math.min(W * 0.34, Math.max(...rows.map((r) => textWidth(r.name, "600 " + FONT))) + 12);
      const valPad = 46;
      const x0 = labelW + valPad;
      const x1 = W - valPad;
      const lim = 25;
      const x = (v) => x0 + ((v + lim) / (2 * lim)) * (x1 - x0);
      const bottom = top + rows.length * rowH;
      const H = bottom + 26;
      const svg = sv("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "Deviation of each simulator from the real-user success rate on tau-bench" });
      const grid = sv("g", { class: "grid", "aria-hidden": "true" });
      const tauTicks = x1 - x0 < 300 ? [-20, 0, 20] : [-20, -10, 0, 10, 20];
      for (const t of tauTicks) {
        grid.append(sv("line", { x1: x(t), x2: x(t), y1: top - 6, y2: bottom, class: t === 0 ? "axis-line" : null }));
        svg.append(sv("text", { x: x(t), y: bottom + 18, "text-anchor": "middle", class: "tick", "aria-hidden": "true", text: t === 0 ? "0" : fmtSigned(t, 0) }));
      }
      svg.prepend(grid);
      svg.append(sv("text", { x: x(0), y: 12, "text-anchor": "middle", class: "ref-lbl", "aria-hidden": "true", text: `Real users · ${T.human}%` }));
      rows.forEach((r, i) => {
        const y = top + i * rowH;
        const ours = r.type === "ours";
        const g = sv("g", { class: "row" });
        g.append(sv("rect", { class: "row-bg", x: 0, y: y + 1, width: W, height: rowH - 2, rx: 6 }));
        g.append(sv("text", { x: labelW, y: y + rowH / 2, dy: "0.35em", "text-anchor": "end", class: "lbl" + (ours ? " hl" : ""), "aria-hidden": "true", text: r.name }));
        const d = hbarPath(x(0), x(r.delta), y + (rowH - barH) / 2, barH);
        if (d) g.append(sv("path", { d, class: "bar " + (ours ? "m-accent" : "m-gray") }));
        const pos = r.delta >= 0;
        g.append(sv("text", { x: x(r.delta) + (pos ? 6 : -6), y: y + rowH / 2, dy: "0.35em", "text-anchor": pos ? "start" : "end", class: "val" + (ours ? " hl" : ""), "aria-hidden": "true", text: fmtSigned(r.delta) }));
        const hit = sv("rect", { class: "hit", x: 0, y, width: W, height: rowH, tabindex: 0, role: "img", "aria-label": `${r.name}: ${r.rate}% success, ${fmtSigned(r.delta)} points from real users` });
        bindTip(hit, () => ({
          title: r.name,
          rows: [
            { value: r.rate.toFixed(1) + "%", label: "task success", key: ours ? "k-accent" : "k-gray" },
            { value: fmtSigned(r.delta), label: `points vs real users (${T.human}%)` }
          ],
          note: TYPE_LABEL[r.type]
        }));
        g.append(hit);
        svg.append(g);
      });
      return svg;
    };
    mount($("#tau-chart"), draw);
    tableRenderers.tau = () => renderTable($("#tau-table"), "GPT-5.5 agent on τ-bench with each simulator as the user",
      ["Simulated user", "Success rate (%)", "Deviation (points)"],
      [{ cells: ["Real users", T.human.toFixed(1), "0.0"] }].concat(rows.map((r) => ({
        hl: r.type === "ours", cells: [r.name, r.rate.toFixed(1), fmtSigned(r.delta)]
      }))));
  })();

  // ------------------------------------------------- 03 behavior explorer
  (function behaviors() {
    const B = D.behaviors;
    const total = B.reduce((a, b) => a + b.count, 0);
    const rows = B.slice().sort((a, b) => b.count - a.count);
    let selected = rows[0].id;
    const card = $("#beh-detail");
    const pct = (b) => (100 * b.count) / total;

    function renderCard(b) {
      card.replaceChildren(
        el("span", { class: "beh-id", text: b.id }),
        el("h3", { text: b.name }),
        el("p", { class: "beh-count", text: `${b.count} of ${total} annotated instances (${pct(b).toFixed(1)}%)` }),
        el("p", { class: "beh-desc", text: b.desc }),
        el("ul", { class: "quotes", "aria-label": "Examples from ThoughtTrace" }, b.ex.map(([kind, text]) =>
          el("li", { class: "quote" + (kind === "t" ? " thought" : "") },
            el("span", { class: "quote-tag", text: kind === "t" ? "Self-reported thought" : "User message" }),
            text))),
        el("p", { class: "beh-foot", text: "Excerpts from ThoughtTrace, in the users' original wording." })
      );
    }

    const S = D.behaviorSims;
    const simBox = $("#beh-sim");
    let simFull = null; // full conversations, fetched on first request
    function renderSim(b) {
      if (!simBox) return;
      const segs = S && S.byId[b.id];
      simBox.hidden = !segs;
      if (!segs) return;
      const holder = el("div");
      const excerpt = () => chatList(segs, S.model, `${S.model} acting out ${b.name}, excerpt`, "beh-sim-chat");
      holder.append(excerpt());
      const btn = el("button", { class: "chat-toggle", type: "button", "aria-expanded": "false", "aria-controls": "beh-sim-chat", text: "Show the full conversation" });
      let full = false;
      btn.addEventListener("click", async () => {
        if (full) {
          holder.replaceChildren(excerpt());
          full = false;
          btn.textContent = "Show the full conversation";
          btn.setAttribute("aria-expanded", "false");
          return;
        }
        btn.disabled = true;
        try {
          if (!simFull) {
            const res = await fetch(S.full);
            if (!res.ok) throw new Error(String(res.status));
            simFull = await res.json();
          }
          const f = simFull[b.id];
          holder.replaceChildren(
            el("p", { class: "sim-instruction" }, el("strong", { text: `Instruction to ${S.model}: ` }), f.instruction),
            chatList([f.turns], S.model, `${S.model} acting out ${b.name}, full conversation`, "beh-sim-chat"));
          full = true;
          btn.textContent = "Show the excerpt only";
          btn.setAttribute("aria-expanded", "true");
        } catch (err) {
          btn.textContent = "The conversation could not be loaded";
        } finally {
          btn.disabled = false;
        }
      });
      simBox.replaceChildren(
        el("h3", null, el("span", { class: "beh-id", text: b.id }), `${S.model} acting out ${b.name.toLowerCase()}`),
        el("p", { class: "sim-note", text: `An ABCD customer-support chat in which ${S.model} plays the customer and was instructed to show this behavior.` }),
        holder, btn);
    }

    function select(id, refocus) {
      selected = id;
      $$("#beh-chart .row").forEach((g) => {
        const on = g.dataset.id === id;
        g.classList.toggle("is-sel", on);
        $(".lbl", g).classList.toggle("hl", on);
        $(".hit", g).setAttribute("aria-pressed", String(on));
      });
      renderCard(B.find((b) => b.id === id));
      renderSim(B.find((b) => b.id === id));
      if (refocus) { const h = $(`#beh-chart .row[data-id="${id}"] .hit`); if (h) h.focus(); }
    }

    const draw = (W) => {
      const rowH = 28;
      const barH = 14;
      const top = 4;
      const valW = 50;
      const maxLabel = Math.max(...rows.map((b) => textWidth(b.name)));
      const labelW = Math.min(W * 0.46, maxLabel + 12);
      const x0 = labelW + 8;
      const x1 = W - valW;
      const max = 45;
      const x = (v) => x0 + (v / max) * (x1 - x0);
      const bottom = top + rows.length * rowH;
      const H = bottom + 24;
      const svg = sv("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "Share of each behavior in 700 annotated ThoughtTrace instances" });
      const grid = sv("g", { class: "grid", "aria-hidden": "true" });
      for (const t of (x1 - x0 < 260 ? [0, 20, 40] : [0, 10, 20, 30, 40])) {
        grid.append(sv("line", { x1: x(t), x2: x(t), y1: top, y2: bottom, class: t === 0 ? "axis-line" : null }));
        svg.append(sv("text", { x: x(t), y: bottom + 17, "text-anchor": "middle", class: "tick", "aria-hidden": "true", text: t + "%" }));
      }
      svg.append(grid);
      rows.forEach((b, i) => {
        const y = top + i * rowH;
        const on = b.id === selected;
        const g = sv("g", { class: "row" + (on ? " is-sel" : ""), "data-id": b.id });
        g.append(sv("rect", { class: "row-bg", x: 0, y: y + 1, width: W, height: rowH - 2, rx: 6 }));
        g.append(sv("text", { x: labelW, y: y + rowH / 2, dy: "0.35em", "text-anchor": "end", class: "lbl" + (on ? " hl" : ""), "aria-hidden": "true", text: fitText(b.name, labelW - 4) }));
        const d = hbarPath(x(0), x(pct(b)), y + (rowH - barH) / 2, barH);
        if (d) g.append(sv("path", { d, class: "bar m-accent" }));
        g.append(sv("text", { x: x(pct(b)) + 6, y: y + rowH / 2, dy: "0.35em", class: "val", "aria-hidden": "true", text: pct(b).toFixed(1) + "%" }));
        const hit = sv("rect", {
          class: "hit", x: 0, y, width: W, height: rowH, tabindex: 0, role: "button", "aria-pressed": String(on),
          "aria-label": `${b.id} ${b.name}: ${b.count} instances, ${pct(b).toFixed(1)} percent. Show examples.`
        });
        bindTip(hit, () => ({ title: `${b.id} · ${b.name}`, rows: [{ value: b.count + " of " + total, label: `instances (${pct(b).toFixed(1)}%)`, key: "k-accent" }] }));
        hit.addEventListener("click", () => select(b.id, false));
        hit.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(b.id, true); }
        });
        g.append(hit);
        svg.append(g);
      });
      return svg;
    };
    mount($("#beh-chart"), draw);
    renderCard(rows[0]);
    renderSim(rows[0]);
    tableRenderers.beh = () => renderTable($("#beh-table"), "Realistic behaviors in 700 annotated ThoughtTrace instances",
      ["Behavior", "ID", "Instances", "Share (%)"],
      rows.map((b) => ({ cells: [b.name, b.id, String(b.count), pct(b).toFixed(1)] })));
  })();

  // ------------------------------------------------------------------ tabs
  function initTabs(root, onSelect) {
    const tabs = $$('[role="tab"]', root);
    function activate(tab, focus) {
      tabs.forEach((t) => {
        const on = t === tab;
        t.setAttribute("aria-selected", String(on));
        t.tabIndex = on ? 0 : -1;
        const panel = document.getElementById(t.getAttribute("aria-controls"));
        if (panel && tabs.some((o) => o !== t && o.getAttribute("aria-controls") === t.getAttribute("aria-controls"))) {
          if (on) panel.setAttribute("aria-labelledby", t.id); // shared panel
        } else if (panel) {
          panel.hidden = !on;
        }
      });
      if (focus) tab.focus();
      if (onSelect) onSelect(tab);
    }
    tabs.forEach((t, i) => {
      t.addEventListener("click", () => activate(t, false));
      t.addEventListener("keydown", (e) => {
        let j = null;
        if (e.key === "ArrowRight") j = (i + 1) % tabs.length;
        else if (e.key === "ArrowLeft") j = (i - 1 + tabs.length) % tabs.length;
        else if (e.key === "Home") j = 0;
        else if (e.key === "End") j = tabs.length - 1;
        if (j !== null) { e.preventDefault(); activate(tabs[j], true); }
      });
    });
  }

  // -------------------------------------------- 04 simulator benchmarks
  (function simulator() {
    const SUB = {
      soul: "Simulation capability on the five SOUL-Index axes: the Overall score. Higher is better.",
      realusersim: "Trajectory-level behavioral fidelity on RealUserSim PT3: the Fidelity Index. Higher is better.",
      tauusi: "Behavioral alignment and task-outcome calibration on τ-bench: USI₅. Higher is better; the line marks the human inter-annotator score.",
      simarena: "Message similarity and Turing distance on SimulatorArena: the Turing distance. Lower is better.",
      prism: "Behavioral alignment on open-domain PRISM conversations: the mean of D1–D4. Higher is better."
    };
    const TITLE = { soul: "SOUL-Index", realusersim: "RealUserSim PT3", tauusi: "τ-USI", simarena: "SimulatorArena", prism: "PRISM" };
    let key = "soul";
    const digitsFor = (k, col) => (k === "tauusi" ? (col === 4 ? 3 : 2) : k === "simarena" ? (col < 2 ? 2 : 1) : 1);

    function sortedRows(k) {
      const B = D.simulator[k];
      const mi = B.columns.indexOf(B.main);
      const lower = B.better === "lower";
      return B.rows.map((r) => ({ ...r, value: r.v[mi] }))
        .sort((a, b) => (lower ? a.value - b.value : b.value - a.value));
    }

    const draw = (W) => {
      const B = D.simulator[key];
      const rows = sortedRows(key);
      const mi = B.columns.indexOf(B.main);
      const digits = digitsFor(key, mi);
      const max = key === "simarena" ? 50 : 100;
      const rowH = 28;
      const barH = 16;
      const top = B.reference ? 24 : 6;
      const valW = 54;
      const labelW = Math.min(W * 0.36, Math.max(...rows.map((r) => textWidth(r.name, "600 " + FONT))) + 12);
      const x0 = labelW + 8;
      const x1 = W - valW;
      const x = (v) => x0 + (v / max) * (x1 - x0);
      const bottom = top + rows.length * rowH;
      const H = bottom + 24;
      const svg = sv("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": `${TITLE[key]}: ${B.metric} for each simulator` });
      const grid = sv("g", { class: "grid", "aria-hidden": "true" });
      const narrow = x1 - x0 < 300;
      const ticks = key === "simarena" ? (narrow ? [0, 25, 50] : [0, 10, 20, 30, 40, 50]) : (narrow ? [0, 50, 100] : [0, 20, 40, 60, 80, 100]);
      for (const t of ticks) {
        grid.append(sv("line", { x1: x(t), x2: x(t), y1: top - 2, y2: bottom, class: t === 0 ? "axis-line" : null }));
        svg.append(sv("text", { x: x(t), y: bottom + 17, "text-anchor": "middle", class: "tick", "aria-hidden": "true", text: String(t) }));
      }
      svg.append(grid);
      if (B.reference) {
        const rx = x(B.reference.value);
        svg.append(sv("line", { x1: rx, x2: rx, y1: top - 6, y2: bottom, class: "ref-line", "aria-hidden": "true" }));
        svg.append(sv("text", { x: rx, y: 11, "text-anchor": "end", class: "ref-lbl", "aria-hidden": "true", text: B.reference.label }));
      }
      rows.forEach((r, i) => {
        const y = top + i * rowH;
        const ours = r.type === "ours";
        const g = sv("g", { class: "row" });
        g.append(sv("rect", { class: "row-bg", x: 0, y: y + 1, width: W, height: rowH - 2, rx: 6 }));
        g.append(sv("text", { x: labelW, y: y + rowH / 2, dy: "0.35em", "text-anchor": "end", class: "lbl" + (ours ? " hl" : ""), "aria-hidden": "true", text: r.name }));
        const d = hbarPath(x(0), x(r.value), y + (rowH - barH) / 2, barH);
        if (d) g.append(sv("path", { d, class: "bar " + (ours ? "m-accent" : "m-gray") }));
        g.append(sv("text", { x: x(r.value) + 6, y: y + rowH / 2, dy: "0.35em", class: "val" + (ours ? " hl" : ""), "aria-hidden": "true", text: r.value.toFixed(digits) }));
        const hit = sv("rect", { class: "hit", x: 0, y, width: W, height: rowH, tabindex: 0, role: "img", "aria-label": `${r.name}: ${B.metric} ${r.value.toFixed(digits)}` });
        bindTip(hit, () => ({
          title: r.name,
          rows: [{ value: r.value.toFixed(digits), label: B.metric, key: ours ? "k-accent" : "k-gray" }],
          note: TYPE_LABEL[r.type] + (B.better === "lower" ? " · lower is better" : "")
        }));
        g.append(hit);
        svg.append(g);
      });
      return svg;
    };

    function renderSimTable() {
      const B = D.simulator[key];
      renderTable($("#sim-table"), `${TITLE[key]}, all reported columns`, ["Simulator", "Type"].concat(B.columns),
        sortedRows(key).map((r) => ({
          hl: r.type === "ours",
          cells: [r.name, TYPE_LABEL[r.type]].concat(r.v.map((v, ci) => v.toFixed(digitsFor(key, ci))))
        })).concat(key === "tauusi" ? [{ cells: ["Human (inter-annotator)", "Reference", "87.35", "97.94", "88.00", "93.53", "0.081", "91.75"] }] : []));
      if (key === "soul") $("#sim-table").append(soulPerDatasetTable());
    }

    function soulPerDataset() { return D.soulPerDataset; }
    function soulPerDatasetTable() {
      const S = soulPerDataset();
      const ours = new Set(["MIMESIS-9B", "MIMESIS-4B"]);
      const AXIS = { CONV: "Conversational simulation", SS: "Social simulation", COG: "Cognition", ROLE: "Role play", EVAL: "Evaluation and judgment" };
      const body = el("tbody");
      let axis = null;
      for (const r of S.rows) {
        if (r.axis !== axis) {
          axis = r.axis;
          body.append(el("tr", { class: "group" }, el("th", { scope: "colgroup", colspan: String(S.models.length + 1), text: `${r.axis} · ${AXIS[r.axis]}` })));
        }
        body.append(el("tr", null, el("th", { scope: "row", text: r.dataset }),
          r.v.map((v, i) => el("td", { class: ours.has(S.models[i]) ? "hl-col" : null, text: v.toFixed(1) + ((S.flags || {})[r.dataset + "|" + S.models[i]] || "") }))));
      }
      body.append(el("tr", { class: "hl" }, el("th", { scope: "row", text: "Overall" }),
        S.overall.v.map((v, i) => el("td", { class: ours.has(S.models[i]) ? "hl-col" : null, text: v.toFixed(1) }))));
      return el("div", { class: "subtable" },
        el("table", { class: "wide-table" },
          el("caption", { text: "Detailed SOUL results, grouped by conversational interaction, social simulation, cognition, role play, and evaluation/judgment. Means over three independently seeded evaluation runs; standard errors are in the paper. † and ‡ reproduce the marks in the paper\u2019s table." }),
          el("thead", null, el("tr", null, el("th", { scope: "col", text: "Dataset" }),
            S.models.map((m) => el("th", { scope: "col", class: ours.has(m) ? "hl-col" : null, text: m })))),
          body));
    }
    tableRenderers.sim = renderSimTable;

    function update() {
      $("#sim-title").textContent = TITLE[key];
      $("#sim-sub").textContent = SUB[key];
      $("#sim-note").textContent = D.simulator[key].note;
      chart.redraw();
      if (tableOpen("sim")) renderSimTable();
    }
    const chart = mount($("#sim-chart"), draw);
    initTabs($("#simulator [data-tabs]"), (tab) => { key = tab.dataset.bench; update(); });
    update();
  })();

  // -------------------------------------------- 04b pairwise realism (PRISM)
  (function pairwise() {
    const P = D.pairwise;
    const ns = new Set(P.notSignificant.map(([j, b]) => j + "|" + b));
    const rowH = 26;
    const barH = 16;
    const headH = 26;
    const axisH = 22;
    const labelW = Math.max(...P.baselines.map((b) => textWidth(b))) + 12;

    function labels(svg) {
      P.baselines.forEach((b, i) => svg.append(sv("text", {
        x: labelW - 8, y: headH + i * rowH + rowH / 2, dy: "0.35em", "text-anchor": "end", class: "lbl", "aria-hidden": "true", text: b
      })));
    }
    function panel(svg, judge, x0, w) {
      const pad = 34;
      const px0 = x0 + pad;
      const px1 = x0 + w - pad;
      const x = (v) => px0 + ((v + 100) / 200) * (px1 - px0);
      const bottom = headH + P.baselines.length * rowH;
      svg.append(sv("text", { x: x(0), y: 13, "text-anchor": "middle", class: "panel-title", "aria-hidden": "true", text: "Judge: " + judge }));
      const grid = sv("g", { class: "grid", "aria-hidden": "true" });
      for (const t of [-100, -50, 0, 50, 100]) {
        grid.append(sv("line", { x1: x(t), x2: x(t), y1: headH - 4, y2: bottom, class: t === 0 ? "axis-line" : null }));
        svg.append(sv("text", { x: x(t), y: bottom + 16, "text-anchor": "middle", class: "tick", "aria-hidden": "true", text: String(Math.abs(t)) }));
      }
      svg.append(grid);
      P.wtl[judge].forEach(([win, tie, loss], i) => {
        const base = P.baselines[i];
        const yRow = headH + i * rowH;
        const y = yRow + (rowH - barH) / 2;
        const g = sv("g", { class: "row" });
        g.append(sv("rect", { class: "row-bg", x: x0, y: yRow + 1, width: w, height: rowH - 2, rx: 6 }));
        const tl = x(-tie / 2);
        const tr = x(tie / 2);
        if (tr - tl > 2) g.append(sv("rect", { class: "bar m-tie", x: tl + 1, y, width: tr - tl - 2, height: barH }));
        const dl = hbarPath(tl - 1, x(-(tie / 2 + loss)), y, barH);
        if (dl) g.append(sv("path", { d: dl, class: "bar m-loss" }));
        const dw = hbarPath(tr + 1, x(tie / 2 + win), y, barH);
        if (dw) g.append(sv("path", { d: dw, class: "bar m-win" }));
        const mid = yRow + rowH / 2;
        g.append(sv("text", { x: x(-(tie / 2 + loss)) - 5, y: mid, dy: "0.35em", "text-anchor": "end", class: "val", "aria-hidden": "true", text: loss.toFixed(1) }));
        const tieTxt = tie.toFixed(1);
        if (tr - tl - 2 >= textWidth(tieTxt) + 6) {
          g.append(sv("text", { x: (tl + tr) / 2, y: mid, dy: "0.35em", "text-anchor": "middle", class: "val on-tie", "aria-hidden": "true", text: tieTxt }));
        }
        const winTxt = win.toFixed(1);
        const wx = x(tie / 2 + win) + 5;
        g.append(sv("text", { x: wx, y: mid, dy: "0.35em", class: "val hl", "aria-hidden": "true", text: winTxt }));
        const isNs = ns.has(judge + "|" + base);
        const hit = sv("rect", {
          class: "hit", x: x0, y: yRow, width: w, height: rowH, tabindex: 0, role: "img",
          "aria-label": `${judge} judge, MIMESIS versus ${base}: MIMESIS wins ${winTxt}%, ties ${tieTxt}%, ${base} wins ${loss.toFixed(1)}%${isNs ? ", nonsignificant" : ""}`
        });
        bindTip(hit, () => ({
          title: `MIMESIS vs ${base}`,
          rows: [
            { value: winTxt + "%", label: "MIMESIS wins", key: "k-win" },
            { value: tieTxt + "%", label: "ties", key: "k-tie" },
            { value: loss.toFixed(1) + "%", label: `${base} wins`, key: "k-loss" }
          ],
          note: `Judge: ${judge}${isNs ? " · marked nonsignificant in the paper" : ""}`
        }));
        g.append(hit);
        svg.append(g);
      });
    }
    const draw = (W) => {
      const H = headH + P.baselines.length * rowH + axisH;
      if (W >= 860) {
        const gap = 14;
        const pw = (W - labelW - gap * 2) / 3;
        const svg = sv("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "Pairwise next-turn realism of MIMESIS against eight simulators under three judges" });
        labels(svg);
        P.judges.forEach((j, k) => panel(svg, j, labelW + k * (pw + gap), pw));
        return svg;
      }
      const wrap = el("div", { class: "pair-stack" });
      P.judges.forEach((j) => {
        const svg = sv("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": `Pairwise next-turn realism under the ${j} judge` });
        labels(svg);
        panel(svg, j, labelW, W - labelW);
        wrap.append(svg);
      });
      return wrap;
    };
    mount($("#pair-chart"), draw);
    tableRenderers.pair = () => renderTable($("#pair-table"), "Pairwise next-turn realism on PRISM (percent of comparisons)",
      ["Judge", "Baseline", "MIMESIS wins (%)", "Ties (%)", "Baseline wins (%)", "Significant"],
      P.judges.flatMap((j) => P.wtl[j].map(([w, t, l], i) => ({
        cells: [j, P.baselines[i], w.toFixed(1), t.toFixed(1), l.toFixed(1), ns.has(j + "|" + P.baselines[i]) ? "no" : "yes"]
      }))));
  })();

  // ------------------------------------------------------ 05 agent explorer
  (function agents() {
    const A = D.agent;
    const SERIES = {
      base: { label: "Base (Qwen3-8B before RL)", cls: "m-base1", key: "k-base1", hollow: "circle" },
      infopo: { label: "InfoPO w. MIMESIS-9B", cls: "m-base2", key: "k-base2", hollow: "diamond" },
      userrl: { label: "GRPO w. GPT-5.5 (UserRL)", cls: "m-ord1", key: "k-ord1" },
      userrlp: { label: "GRPO w. MIMESIS-9B (UserRL+)", cls: "m-ord2", key: "k-ord2" },
      csd: { label: "CSD w. MIMESIS-9B", cls: "m-ord3", key: "k-ord3" }
    };
    const MAIN = ["userrl", "userrlp", "csd"];
    const state = { view: "overview", user: 0, baselines: false };
    const keys = () => (state.baselines ? ["base", "infopo"] : []).concat(MAIN);
    const legendOrder = () => ["csd", "userrlp", "userrl"].concat(state.baselines ? ["infopo", "base"] : []);
    const userMean = (k) => mean(A.users.map((u) => u.scores[k].avg));

    function renderLegend() {
      const dots = state.view === "overview";
      $("#agent-legend").replaceChildren(...legendOrder().map((k) => {
        const s = SERIES[k];
        const cls = ["key", s.key, dots ? "" : "bar-key", dots && s.hollow ? "hollow" : "", dots && s.hollow === "diamond" ? "diamond" : ""].join(" ");
        return el("li", null, el("span", { class: cls, "aria-hidden": "true" }), s.label);
      }));
    }

    function tipRows(getVal) {
      return legendOrder().filter((k) => getVal(k) != null).map((k) => ({ value: getVal(k).toFixed(2), label: SERIES[k].label, key: SERIES[k].key }));
    }

    function marker(k, cx, cy) {
      const s = SERIES[k];
      if (s.hollow === "diamond") {
        const r = 5.5;
        return sv("path", { d: `M${cx},${cy - r}L${cx + r},${cy}L${cx},${cy + r}L${cx - r},${cy}Z`, class: `dot dot-hollow ${s.cls}`, "aria-hidden": "true" });
      }
      if (s.hollow) return sv("circle", { cx, cy, r: 4.5, class: `dot dot-hollow ${s.cls}`, "aria-hidden": "true" });
      return sv("circle", { cx, cy, r: 5.5, class: `dot ${s.cls}`, "aria-hidden": "true" });
    }

    function drawOverview(W) {
      const ks = keys();
      const groups = [];
      for (const u of A.users) {
        if (!groups.length || groups[groups.length - 1].name !== u.group) groups.push({ name: u.group, users: [] });
        groups[groups.length - 1].users.push(u);
      }
      const all = A.users.flatMap((u) => ks.map((k) => u.scores[k].avg));
      const dom = niceDomain(Math.min(...all) - 0.5, Math.max(...all) + 0.5, 6);
      const rowH = 30;
      const headH = 26;
      const meanH = 48;
      const labelW = Math.min(W * 0.34, Math.max(...A.users.map((u) => textWidth(u.name)), textWidth("Mean of 9 users", "600 " + FONT)) + 12);
      const x0 = labelW + 18;
      const x1 = W - 16;
      const x = (v) => x0 + ((v - dom.lo) / (dom.hi - dom.lo)) * (x1 - x0);
      const plotTop = 6;
      let y = plotTop;
      const layout = [];
      groups.forEach((g) => {
        layout.push({ head: g.name, y });
        y += headH;
        g.users.forEach((u) => { layout.push({ user: u, y }); y += rowH; });
      });
      const meanY = y + 6;
      const bottom = meanY + meanH;
      const H = bottom + 26;
      const svg = sv("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "Average agent score under each evaluation user, by training condition" });
      const grid = sv("g", { class: "grid", "aria-hidden": "true" });
      for (const t of dom.ticks) {
        grid.append(sv("line", { x1: x(t), x2: x(t), y1: plotTop, y2: bottom }));
        svg.append(sv("text", { x: x(t), y: bottom + 18, "text-anchor": "middle", class: "tick", "aria-hidden": "true", text: String(t) }));
      }
      svg.append(grid);

      function drawRow(name, vals, yy, h, isMean, groupName) {
        const g = sv("g", { class: "row" });
        g.append(sv("rect", { class: "row-bg", x: 0, y: yy + 1, width: W, height: h - 2, rx: 6 }));
        const cy = isMean ? yy + h - 16 : yy + h / 2;
        g.append(sv("text", { x: labelW, y: cy, dy: "0.35em", "text-anchor": "end", class: "lbl" + (isMean ? " hl" : ""), "aria-hidden": "true", text: name }));
        const mainVals = MAIN.map((k) => vals[k]);
        g.append(sv("line", { x1: x(Math.min(...mainVals)), x2: x(Math.max(...mainVals)), y1: cy, y2: cy, class: "conn", "aria-hidden": "true" }));
        ks.filter((k) => SERIES[k].hollow && vals[k] != null).forEach((k) => g.append(marker(k, x(vals[k]), cy)));
        MAIN.forEach((k) => g.append(marker(k, x(vals[k]), cy)));
        if (isMean) {
          // Direct labels on the mean row only; drop below the dot if two would collide.
          let lastRight = -Infinity;
          MAIN.slice().sort((a, b) => vals[a] - vals[b]).forEach((k) => {
            const txt = vals[k].toFixed(2);
            const w = textWidth(txt);
            const cx = x(vals[k]);
            const below = cx - w / 2 < lastRight + 4;
            g.append(sv("text", { x: cx, y: below ? cy + 20 : cy - 12, "text-anchor": "middle", class: "val hl", "aria-hidden": "true", text: txt }));
            if (!below) lastRight = cx + w / 2;
          });
        }
        const hit = sv("rect", {
          class: "hit", x: 0, y: yy, width: W, height: h, tabindex: 0, role: "img",
          "aria-label": `${name}: ` + legendOrder().filter((k) => vals[k] != null).map((k) => `${SERIES[k].label} ${vals[k].toFixed(2)}`).join("; ")
        });
        bindTip(hit, () => ({ title: name, rows: tipRows((k) => vals[k]), note: groupName || "Unweighted mean over the nine evaluation users" }));
        g.append(hit);
        svg.append(g);
      }

      layout.forEach((it) => {
        if (it.head) {
          svg.append(sv("text", { x: 2, y: it.y + headH / 2 + 2, dy: "0.35em", class: "grp-head", "aria-hidden": "true", text: it.head.toUpperCase() + (it.head.endsWith("API") ? " USERS" : "S") }));
        } else {
          const u = it.user;
          drawRow(u.name, Object.fromEntries(ks.map((k) => [k, u.scores[k].avg])), it.y, rowH, false, u.group);
        }
      });
      svg.append(sv("line", { x1: 0, x2: W, y1: meanY - 3, y2: meanY - 3, class: "sep", "aria-hidden": "true" }));
      drawRow("Mean of 9 users", Object.fromEntries(MAIN.map((k) => [k, userMean(k)])), meanY, meanH, true, null);
      return svg;
    }

    function drawEnv(W) {
      const u = A.users[state.user];
      const ks = keys();
      const cols = W >= 860 ? 4 : W >= 460 ? 2 : 1;
      const gap = 12;
      const panelW = (W - gap * (cols - 1)) / cols;
      const wrap = el("div", { class: "multiples", style: `grid-template-columns: repeat(${cols}, minmax(0, 1fr))` });
      A.envs.forEach((env, ei) => {
        const vals = Object.fromEntries(ks.map((k) => [k, u.scores[k].env[ei]]));
        const vmax = Math.max(...ks.map((k) => vals[k]));
        const dom = niceDomain(0, vmax > 0 ? vmax : 10, 3);
        const w = Math.max(120, panelW - 22);
        const h = 150;
        const padL = 30;
        const padT = 18;
        const padB = 6;
        const plotW = w - padL - 4;
        const yb = h - padB;
        const y = (v) => yb - (v / dom.hi) * (yb - padT);
        const n = ks.length;
        const colW = Math.min(24, Math.floor((plotW * 0.72 - (n - 1) * 2) / n));
        const groupW = n * colW + (n - 1) * 2;
        const gx = padL + (plotW - groupW) / 2;
        const svg = sv("svg", { width: w, height: h, viewBox: `0 0 ${w} ${h}`, role: "img", "aria-hidden": "true" });
        const grid = sv("g", { class: "grid" });
        for (const t of dom.ticks) {
          grid.append(sv("line", { x1: padL, x2: w - 2, y1: y(t), y2: y(t), class: t === 0 ? "axis-line" : null }));
          svg.append(sv("text", { x: padL - 6, y: y(t), dy: "0.35em", "text-anchor": "end", class: "tick", text: String(t) }));
        }
        svg.prepend(grid);
        const g = sv("g", { class: "colg" });
        let best = null;
        ks.forEach((k, i) => {
          const cx = gx + i * (colW + 2);
          const d = vbarPath(cx, colW, yb, y(vals[k]));
          if (d) g.append(sv("path", { d, class: "col " + SERIES[k].cls }));
          if (best === null || vals[k] > vals[best.k]) best = { k, cx };
        });
        if (vmax > 0) {
          g.append(sv("text", { x: best.cx + colW / 2, y: y(vals[best.k]) - 5, "text-anchor": "middle", class: "val hl", text: vals[best.k].toFixed(2) }));
        } else {
          g.append(sv("text", { x: padL + plotW / 2, y: padT + 34, "text-anchor": "middle", class: "empty-note", text: "User model refused" }));
          g.append(sv("text", { x: padL + plotW / 2, y: padT + 50, "text-anchor": "middle", class: "empty-note", text: "this role: all zero" }));
        }
        svg.append(g);
        const panel = el("div", {
          class: "mini", tabindex: 0, role: "img",
          "aria-label": `${env.key}${env.heldOut ? " (held out)" : ""}, evaluated with ${u.name}: ` + legendOrder().map((k) => `${SERIES[k].label} ${vals[k].toFixed(2)}`).join("; ")
        },
          el("h4", null, el("span", { text: env.key + "Gym" }), env.heldOut ? el("span", { class: "tag", text: "held out" }) : null),
          svg);
        bindTip(panel, () => ({ title: `${env.key}Gym · ${u.name}`, rows: tipRows((k) => vals[k]), note: env.heldOut ? "Held out from agent training" : "Used in agent training" }));
        wrap.append(panel);
      });
      return wrap;
    }

    const draw = (W) => (state.view === "overview" ? drawOverview(W) : drawEnv(W));
    const chart = mount($("#agent-chart"), draw);

    function renderAgentTable() {
      const lo = legendOrder();
      if (state.view === "overview") {
        renderTable($("#agent-table"), "Mean task score across eight environments for each evaluation user, as reported in the paper",
          ["Evaluation user"].concat(lo.map((k) => SERIES[k].label)),
          A.users.map((u) => ({ cells: [u.name].concat(lo.map((k) => u.scores[k].avg.toFixed(2))) }))
            .concat([{ hl: true, cells: ["Mean of 9 users"].concat(lo.map((k) => (MAIN.includes(k) ? userMean(k).toFixed(2) : "—"))) }]));
      } else {
        const u = A.users[state.user];
        renderTable($("#agent-table"), `Scores by environment, evaluated with ${u.name}`,
          ["Training condition"].concat(A.envs.map((e) => e.key + (e.heldOut ? " (held out)" : "")), ["Avg."]),
          lo.map((k) => ({ hl: k === "csd", cells: [SERIES[k].label].concat(u.scores[k].env.map((v) => v.toFixed(2)), [u.scores[k].avg.toFixed(2)]) })));
      }
    }
    tableRenderers.agent = renderAgentTable;

    function update() {
      const overview = state.view === "overview";
      $("#agent-user-wrap").hidden = overview;
      $("#agent-title").textContent = overview ? "Mean task score across eight environments, per evaluation user" : `Scores by environment, evaluated with ${A.users[state.user].name}`;
      $("#agent-sub").textContent = overview
        ? "None of the nine evaluation user models is used during agent training. Each dot is one training condition; the gray line spans the three main conditions."
        : "Each panel has its own scale. The tallest column in each panel is labeled; hover or open the table for the rest.";
      renderLegend();
      chart.redraw();
      if (tableOpen("agent")) renderAgentTable();
    }

    const sel = $("#agent-user");
    const optgroups = {};
    A.users.forEach((u, i) => {
      const label = u.group === "Frontier API" ? "Frontier API users" : "Released simulators";
      if (!optgroups[label]) { optgroups[label] = el("optgroup", { label }); sel.append(optgroups[label]); }
      optgroups[label].append(el("option", { value: String(i), text: u.name }));
    });
    sel.addEventListener("change", () => { state.user = Number(sel.value); update(); });
    $$('input[name="agent-view"]').forEach((r) => r.addEventListener("change", () => { if (r.checked) { state.view = r.value; update(); } }));
    $("#agent-baselines").addEventListener("change", (e) => { state.baselines = e.target.checked; update(); });
    update();
  })();

  // ------------------------------------- 06 CSD versus GRPO training curves
  (function curves() {
    const C = D.csdCurves;
    if (!C || !$("#curve-chart")) return;
    const ARMS = [
      { k: "csd", label: "CSD", line: "ln ln-csd", pt: "pt pt-csd", key: "k-ord3" },
      { k: "grpo", label: "GRPO", line: "ln ln-grpo", pt: "pt pt-grpo", key: "k-ord2" }
    ];
    const state = { run: 0, train: 99, val: 19 };
    const f3 = (v) => v.toFixed(3);
    const lineKey = (cls) => sv("svg", { class: "line-key", width: 24, height: 10, viewBox: "0 0 24 10", "aria-hidden": "true" },
      sv("line", { x1: 2, y1: 5, x2: 22, y2: 5, class: cls }));
    $("#curve-legend").replaceChildren(...ARMS.map((a) => el("li", null, lineKey(a.line), a.label)));

    // One panel: lines over training steps 0..100, a crosshair, and a keyboard-movable readout.
    function panel(W, spec) {
      const H = spec.h;
      const m = { l: 40, r: 50, t: 30, b: 42 };
      const x0 = m.l, x1 = W - m.r, yb = H - m.b, yt = m.t;
      const ydom = niceDomain(0, spec.yHi, 5);
      const X = (v) => x0 + (v / 100) * (x1 - x0);
      const Y = (v) => yb - ((v - ydom.lo) / (ydom.hi - ydom.lo)) * (yb - yt);
      const st = ydom.ticks.length > 1 ? ydom.ticks[1] - ydom.ticks[0] : 1;
      const whole = (v) => Math.abs(v - Math.round(v)) < 1e-6;
      const dec = whole(st) ? 0 : whole(st * 10) ? 1 : 2; // 0.25 steps need two decimals
      const svg = sv("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": spec.aria });
      const grid = sv("g", { class: "grid", "aria-hidden": "true" });
      for (const t of ydom.ticks) {
        grid.append(sv("line", { x1: x0, x2: x1, y1: Y(t), y2: Y(t), class: t === 0 ? "axis-line" : null }));
        svg.append(sv("text", { x: x0 - 8, y: Y(t), dy: "0.35em", "text-anchor": "end", class: "tick", "aria-hidden": "true", text: t === 0 ? "0" : t.toFixed(dec) }));
      }
      for (const t of (x1 - x0 < 300 ? [0, 50, 100] : [0, 20, 40, 60, 80, 100])) {
        svg.append(sv("text", { x: X(t), y: yb + 17, "text-anchor": "middle", class: "tick", "aria-hidden": "true", text: String(t) }));
      }
      svg.prepend(grid);
      svg.append(sv("text", { x: 0, y: 14, class: "panel-title", "aria-hidden": "true", text: spec.title }));
      svg.append(sv("text", { x: (x0 + x1) / 2, y: H - 6, "text-anchor": "middle", class: "axis-title", "aria-hidden": "true", text: "Training step" }));
      const path = (xs, ys) => xs.map((xv, i) => `${i ? "L" : "M"}${X(xv).toFixed(1)},${Y(ys[i]).toFixed(1)}`).join("");
      const order = ARMS.slice().reverse(); // GRPO underneath, CSD on top
      const g = sv("g", { "aria-hidden": "true" });
      order.forEach((a) => g.append(sv("path", { d: path(spec.xs, spec.ys[a.k]), class: a.line })));
      if (spec.points) order.forEach((a) => spec.xs.forEach((xv, i) => g.append(sv("circle", { cx: X(xv), cy: Y(spec.ys[a.k][i]), r: 4, class: a.pt }))));
      // Direct labels at the line ends. Labels that would overlap are spread apart and
      // tied back to their line ends with short leader lines.
      const last = spec.xs.length - 1;
      const xe = X(spec.xs[last]);
      const ends = ARMS.map((a) => { const y = Y(spec.ys[a.k][last]); return { a, y0: y, y }; });
      if (Math.abs(ends[0].y - ends[1].y) < 14) {
        const mid = (ends[0].y0 + ends[1].y0) / 2;
        const up = ends[0].y0 <= ends[1].y0 ? 0 : 1;
        ends[up].y = mid - 7; ends[1 - up].y = mid + 7;
      }
      const clear = spec.points ? 5 : 3; // start the leader past the end marker
      ends.forEach((e) => {
        const moved = Math.abs(e.y - e.y0) > 1.5;
        if (moved) g.append(sv("path", { d: `M${(xe + clear).toFixed(1)},${e.y0.toFixed(1)}L${(xe + clear + 5).toFixed(1)},${e.y.toFixed(1)}h3`, class: "leader" }));
        g.append(sv("text", { x: xe + (moved ? clear + 10 : 8), y: e.y, dy: "0.35em", class: "end-lbl", text: e.a.label }));
      });
      svg.append(g);

      const xh = sv("line", { class: "xhair", x1: 0, x2: 0, y1: yt, y2: yb, visibility: "hidden", "aria-hidden": "true" });
      const marks = ARMS.map((a) => sv("circle", { r: 5, class: a.pt, visibility: "hidden", "aria-hidden": "true" }));
      svg.append(xh, ...marks);
      const hit = sv("rect", { class: "plot-hit", x: x0, y: yt - 6, width: x1 - x0, height: yb - yt + 12, tabindex: 0, role: "img", "aria-label": spec.aria + ". Use the left and right arrow keys to read values." });
      svg.append(hit);
      function show(i, ev) {
        const cx = X(spec.xs[i]);
        xh.setAttribute("x1", cx); xh.setAttribute("x2", cx); xh.setAttribute("visibility", "visible");
        ARMS.forEach((a, j) => { marks[j].setAttribute("cx", cx); marks[j].setAttribute("cy", Y(spec.ys[a.k][i])); marks[j].setAttribute("visibility", "visible"); });
        let at = ev;
        if (!at) { // keyboard: anchor the readout to the crosshair
          const r = svg.getBoundingClientRect();
          at = { clientX: r.left + cx * (r.width / W), clientY: r.top + Math.min(...ARMS.map((a) => Y(spec.ys[a.k][i]))) * (r.height / H) };
        }
        showTip(hit, at, spec.tip(i));
      }
      function hide() { xh.setAttribute("visibility", "hidden"); marks.forEach((n) => n.setAttribute("visibility", "hidden")); hideTip(); }
      function nearest(clientX) {
        const r = svg.getBoundingClientRect();
        const v = (((clientX - r.left) * (W / r.width) - x0) / (x1 - x0)) * 100;
        let best = 0;
        spec.xs.forEach((xv, i) => { if (Math.abs(xv - v) < Math.abs(spec.xs[best] - v)) best = i; });
        return best;
      }
      hit.addEventListener("pointermove", (e) => { const i = nearest(e.clientX); spec.setIndex(i); show(i, e); });
      hit.addEventListener("pointerleave", hide);
      hit.addEventListener("focus", () => show(spec.getIndex(), null));
      hit.addEventListener("blur", hide);
      hit.addEventListener("keydown", (e) => {
        let i = spec.getIndex();
        if (e.key === "ArrowRight") i = Math.min(spec.xs.length - 1, i + 1);
        else if (e.key === "ArrowLeft") i = Math.max(0, i - 1);
        else if (e.key === "Home") i = 0;
        else if (e.key === "End") i = spec.xs.length - 1;
        else return;
        e.preventDefault();
        spec.setIndex(i);
        show(i, null);
      });
      return svg;
    }

    function draw(W) {
      const run = C.runs[state.run];
      const T = run.train;
      const V = run.val;
      const two = W >= 720;
      const pw = two ? Math.floor((W - 24) / 2) : W;
      const h = two ? 252 : Math.max(240, Math.min(300, Math.round(W * 0.64))); // stacked panels keep about 3:2
      const steps = T.csd.map((_, i) => i + 1);
      const wrap = el("div", { class: "curve-panels", style: `grid-template-columns: ${two ? "repeat(2, minmax(0, 1fr))" : "minmax(0, 1fr)"}` });
      wrap.append(panel(pw, {
        h,
        title: "Training reward",
        aria: `Training reward over 100 steps with ${run.sim} as the training simulator. At step 100 it is ${f3(T.csd[99])} for CSD and ${f3(T.grpo[99])} for GRPO`,
        xs: steps, ys: { csd: T.csd, grpo: T.grpo },
        yHi: Math.max(...T.csd, ...T.grpo),
        getIndex: () => state.train, setIndex: (i) => { state.train = i; },
        tip: (i) => ({
          title: `Step ${i + 1} · training reward`,
          rows: ARMS.map((a) => ({ value: f3(T[a.k][i]), label: a.label, key: a.key }))
        })
      }));
      wrap.append(panel(pw, {
        h,
        title: "Validation reward",
        aria: `Validation reward every 5 steps with ${run.sim} as the training simulator. At step ${V.steps[V.steps.length - 1]} it is ${f3(V.csd[V.csd.length - 1])} for CSD and ${f3(V.grpo[V.grpo.length - 1])} for GRPO`,
        xs: V.steps, ys: { csd: V.csd, grpo: V.grpo }, points: true,
        yHi: Math.max(...V.csd, ...V.grpo),
        getIndex: () => state.val, setIndex: (i) => { state.val = i; },
        tip: (i) => ({
          title: `Step ${V.steps[i]} · validation reward`,
          rows: ARMS.map((a) => ({ value: f3(V[a.k][i]), label: a.label, key: a.key })),
          note: "Validation runs every 5 steps."
        })
      }));
      return wrap;
    }

    const chart = mount($("#curve-chart"), draw);
    function update() {
      const run = C.runs[state.run];
      state.train = run.train.csd.length - 1;
      state.val = run.val.steps.length - 1;
      $("#curve-sub").textContent = `Training and validation rewards over 100 optimization steps using the same ${run.sim} training simulator. CSD and GRPO therefore differ only in the coaching objective.`;
      $("#curve-note").textContent = run.note;
      chart.redraw();
    }
    initTabs($("#csd [data-tabs]"), (tab) => { state.run = Number(tab.dataset.run); update(); });
    update();
  })();

  // ------------------------------------------------------------ 07 examples
  initTabs($("#examples [data-tabs]"));

  (function guess() {
    const P = D.prism;
    $("#prism-context").replaceChildren(el("span", { class: "who", text: "Assistant" }), ...P.context.map((t) => el("p", { text: t })));
    const order = [3, 0, 4, 1, 5, 2]; // fixed shuffle: the real user is not first
    const letters = "ABCDEF";
    const box = $("#prism-options");
    const result = $("#prism-result");
    const buttons = order.map((idx, i) => {
      const r = P.responses[idx];
      const who = el("span", { class: "guess-who", hidden: true });
      const btn = el("button", { class: "guess", type: "button", "aria-label": `Reply ${letters[i]}: ${r.text}` },
        el("span", { class: "guess-letter", "aria-hidden": "true", text: letters[i] }),
        el("span", null, el("span", { text: r.text }), who));
      btn.addEventListener("click", () => reveal(i));
      return { btn, who, r };
    });
    box.replaceChildren(...buttons.map((b) => b.btn));

    function reveal(chosen) {
      buttons.forEach((b, i) => {
        b.who.textContent = b.r.who + (b.r.real ? " · the recorded reply" : "");
        b.who.hidden = false;
        b.btn.classList.toggle("is-real", !!b.r.real);
        b.btn.classList.toggle("is-ours", !!b.r.ours);
        b.btn.classList.toggle("is-chosen", i === chosen);
        b.btn.disabled = true;
        b.btn.setAttribute("aria-label", `Reply ${letters[i]} was written by ${b.r.who}: ${b.r.text}`);
      });
      const realIdx = buttons.findIndex((b) => b.r.real);
      const right = buttons[chosen].r.real;
      const reset = el("button", { class: "guess-reset", type: "button", text: "Try again" });
      reset.addEventListener("click", () => {
        buttons.forEach((b, i) => {
          b.who.hidden = true;
          b.btn.disabled = false;
          b.btn.classList.remove("is-real", "is-ours", "is-chosen");
          b.btn.setAttribute("aria-label", `Reply ${letters[i]}: ${b.r.text}`);
        });
        result.replaceChildren();
        buttons[0].btn.focus();
      });
      result.replaceChildren(
        el("strong", { text: right ? "Correct. " : `Not this time: the recorded user wrote reply ${letters[realIdx]}. ` }),
        "The recorded user accepts the explanation and invites further conversation. MIMESIS similarly acknowledges the feedback without questioning the corrections, whereas all three frontier simulators focus on inconsistencies in the assistant's explanation. Qwen3.5-9B also expresses appreciation, then introduces a request for a follow-up email example. Producing a helpful or critical response is not sufficient to reproduce a user's next turn.",
        reset);
      reset.focus();
    }
  })();

  (function retail() {
    const R = D.retail;
    $("#retail-instruction").replaceChildren(
      "The user is reluctant to disclose personal information and wants to replace a desk lamp with the cheapest available option. ",
      el("strong", { text: "Instruction: " }), R.instruction,
      " Only user turns are shown. For simulators that expose a thought, the labeled text is that generated thought, and only the public utterance reaches the agent.");
    const pills = $("#retail-pills");
    const out = $("#retail-transcript");
    let current = 0;
    const btns = R.transcripts.map((t, i) => {
      const b = el("button", { class: "pill", type: "button", role: "radio", "aria-checked": String(i === 0), tabindex: i === 0 ? 0 : -1, text: t.who });
      b.addEventListener("click", () => choose(i, false));
      b.addEventListener("keydown", (e) => {
        const n = R.transcripts.length;
        let j = null;
        if (e.key === "ArrowRight" || e.key === "ArrowDown") j = (i + 1) % n;
        if (e.key === "ArrowLeft" || e.key === "ArrowUp") j = (i - 1 + n) % n;
        if (j !== null) { e.preventDefault(); choose(j, true); }
      });
      return b;
    });
    pills.replaceChildren(...btns);

    function choose(i, focus) {
      current = i;
      btns.forEach((b, j) => { b.setAttribute("aria-checked", String(j === i)); b.tabIndex = j === i ? 0 : -1; });
      if (focus) btns[i].focus();
      const t = R.transcripts[i];
      out.replaceChildren(
        el("ol", { "aria-label": `${t.who} transcript` }, t.turns.map((turn, k) => el("li", { class: "turn" },
          el("span", { class: "turn-n", text: "T" + (k + 1) }),
          el("div", null,
            turn.t ? el("p", { class: "turn-thought" }, el("span", { class: "lab", text: "Thought, hidden from the agent" }), turn.t) : null,
            el("div", { class: "turn-say", text: turn.u }))))),
        t.more ? el("p", { class: "transcript-more", text: "… " + t.more }) : null,
        el("p", { class: "transcript-note", text: t.note }));
    }
    choose(current, false);
  })();

  (function travel() {
    const T = D.travel;
    const out = $("#travel-transcript");
    if (!T || !out) return;
    const later = [];
    T.exchanges.forEach((x) => { later.push({ r: "a", t: x.agent }); later.push({ r: "u", t: x.user, th: x.thought }); });
    out.replaceChildren(chatList([[{ r: "u", t: T.opening, who: "User's opening request" }], later, []], "MIMESIS-9B", "TravelGym transcript excerpt"));
  })();

  // ------------------------------------------------------------- utilities
  $$(".copy").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const code = document.getElementById(btn.dataset.copy);
      const status = $(".copy-status", btn.parentElement);
      try {
        await navigator.clipboard.writeText(code.textContent);
        status.textContent = "Copied";
      } catch (err) {
        const range = document.createRange();
        range.selectNodeContents(code);
        const s = window.getSelection();
        s.removeAllRanges();
        s.addRange(range);
        status.textContent = "Selected: press Ctrl+C or ⌘C";
      }
      setTimeout(() => { status.textContent = ""; }, 2400);
    });
  });

  (function theme() {
    const root = document.documentElement;
    const btn = $(".theme-toggle");
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const effective = () => root.dataset.theme || (mq.matches ? "dark" : "light");
    const sync = () => btn.setAttribute("aria-label", effective() === "dark" ? "Switch to light theme" : "Switch to dark theme");
    btn.addEventListener("click", () => {
      const next = effective() === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      try { localStorage.setItem("mimesis-theme", next); } catch (err) { /* private mode */ }
      sync();
    });
    if (mq.addEventListener) mq.addEventListener("change", sync);
    sync();
  })();

  (function navSpy() {
    const links = $$(".site-nav a");
    const byId = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        links.forEach((a) => a.removeAttribute("aria-current"));
        const a = byId.get(e.target.id);
        if (a) a.setAttribute("aria-current", "location");
      });
    }, { rootMargin: "-40% 0px -55% 0px" });
    $$("main section[id]").forEach((s) => io.observe(s));
  })();
})();
