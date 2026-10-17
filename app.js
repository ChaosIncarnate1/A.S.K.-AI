const $ = s => document.querySelector(s);
let token = localStorage.getItem("cp_token"), me = null, chatId = null, mode = "login";

async function api(path, opts = {}) {
  const r = await fetch("/api" + path, {
    method: opts.method || "GET",
    headers: { "Content-Type": "application/json", ...(token ? { Authorization: "Bearer " + token } : {}) },
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || "Something went wrong. Try again.");
  return d;
}

/* ---------- auth ---------- */
function setMode(m) {
  mode = m;
  $("#nameRow").classList.toggle("hidden", m === "login");
  $("#authBtn").textContent = m === "login" ? "Log in" : "Create account";
  $("#authSwap").textContent = m === "login" ? "New here? Create an account" : "Have an account? Log in";
  $("#password").autocomplete = m === "login" ? "current-password" : "new-password";
  $("#authErr").textContent = "";
}
$("#authSwap").onclick = () => setMode(mode === "login" ? "register" : "login");
$("#authForm").onsubmit = async e => {
  e.preventDefault();
  try {
    const body = { email: $("#email").value, password: $("#password").value };
    if (mode === "register") body.name = $("#name").value || "Debater";
    const d = await api("/" + mode, { method: "POST", body });
    token = d.token; me = d.user; localStorage.setItem("cp_token", token); enter();
  } catch (err) { $("#authErr").textContent = err.message; }
};
$("#logout").onclick = () => { localStorage.removeItem("cp_token"); token = me = chatId = null; location.hash = ""; location.reload(); };

async function boot() {
  setMode("login");
  if (token) { try { me = await api("/me"); return enter(); } catch { localStorage.removeItem("cp_token"); } }
  $("#auth").classList.remove("hidden");
}
function enter() {
  $("#auth").classList.add("hidden"); $("#app").classList.remove("hidden");
  $("#who").textContent = me.name;
  $("#usersLink").classList.toggle("hidden", !me.is_admin);
  loadChats(); route();
}

/* ---------- routing: the users page is guarded here and on the server ---------- */
function route() {
  const h = location.hash;
  const view = h === "#/search" ? "search" : h === "#/users" && me && me.is_admin ? "users" : "chat";
  for (const v of ["chat", "search", "users"]) $("#" + v + "View").classList.toggle("hidden", v !== view);
  if (view === "search") loadCfg();
  if (view === "users") loadUsers();
}
addEventListener("hashchange", route);

/* ---------- chats ---------- */
const ICON = {
  edit: '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20h4L19 9l-4-4L4 16z"/></svg>',
  del: '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13"/></svg>',
};
async function loadChats() {
  const list = await api("/chats");
  const nav = $("#chatList"); nav.innerHTML = "";
  list.forEach(c => {
    const row = document.createElement("div"); row.className = "row" + (c.id === chatId ? " on" : "");
    const name = document.createElement("button"); name.className = "name"; name.textContent = c.title;
    name.onclick = () => { location.hash = ""; openChat(c.id); };
    const acts = document.createElement("span"); acts.className = "acts";
    acts.append(iconBtn("Rename chat", ICON.edit, () => renameChat(row, name, c)), iconBtn("Delete chat", ICON.del, () => deleteChat(c)));
    row.append(name, acts); nav.append(row);
  });
}
function iconBtn(label, svg, fn) {
  const b = document.createElement("button"); b.className = "ico"; b.title = label; b.setAttribute("aria-label", label);
  b.innerHTML = svg; b.onclick = e => { e.stopPropagation(); fn(); }; return b;
}
function renameChat(row, nameBtn, c) {
  const inp = document.createElement("input"); inp.className = "rename"; inp.value = c.title; inp.maxLength = 60;
  row.classList.add("editing"); nameBtn.replaceWith(inp); inp.focus(); inp.select();
  let done = false;
  const finish = async save => {
    if (done) return; done = true;
    const t = inp.value.trim();
    if (save && t && t !== c.title) { try { await api(`/chats/${c.id}`, { method: "PUT", body: { title: t } }); } catch (err) { alert(err.message); } }
    loadChats();
  };
  inp.onkeydown = e => { if (e.key === "Enter") finish(true); if (e.key === "Escape") finish(false); };
  inp.onblur = () => finish(true);
}
async function deleteChat(c) {
  try { await api(`/chats/${c.id}`, { method: "DELETE" }); } catch (err) { return alert(err.message); }
  if (c.id === chatId) $("#newChat").click(); else loadChats();
}
async function openChat(id) {
  chatId = id; clearStream();
  const msgs = await api(`/chats/${id}/messages`);
  msgs.forEach(m => m.role === "user" ? addUser(m.content) : addBot(m.content, m.citations, false));
  loadChats();
}
function clearStream() {
  $("#stream").innerHTML = ""; document.body.classList.remove("chatting");
}
$("#newChat").onclick = () => {
  chatId = null; location.hash = ""; clearStream();
  $("#stream").innerHTML = '<div id="empty" class="empty"><h2>What\'s your position?</h2><p>Make a claim. I\'ll test it.</p></div>';
  loadChats();
};

$("#brand").onclick = e => { e.preventDefault(); $("#newChat").click(); };

function col(cls) {
  document.body.classList.add("chatting");
  const el = document.createElement("div"); el.className = "col " + cls;
  $("#stream").append(el); $("#stream").scrollTop = 1e9; return el;
}
function addUser(text) { const el = col("user"); const d = document.createElement("div"); d.textContent = text; el.append(d); }
function addBot(text, cites = [], animate = true) {
  const el = col("bot");
  el.innerHTML = '<span class="dot"></span><div style="flex:1"><div class="txt"></div><div class="cites"></div></div>';
  const t = el.querySelector(".txt"), c = el.querySelector(".cites");
  const done = () => cites.forEach(x => { const a = document.createElement("a"); a.href = x.url; a.target = "_blank"; a.rel = "noopener"; a.textContent = x.title; c.append(a); });
  if (!animate || matchMedia("(prefers-reduced-motion:reduce)").matches) { t.textContent = text; return done(); }
  let i = 0;
  (function tick() { i += 3; t.textContent = text.slice(0, i); $("#stream").scrollTop = 1e9; i < text.length ? setTimeout(tick, 16) : done(); })();
}

$("#composer").onsubmit = async e => {
  e.preventDefault();
  const text = $("#input").value.trim(); if (!text) return;
  $("#input").value = ""; $("#input").style.height = "auto";
  $("#empty")?.remove(); addUser(text);
  const wait = col("bot"); wait.innerHTML = '<span class="dot"></span><div class="think"><i></i><i></i></div>';
  try {
    if (!chatId) { chatId = (await api("/chats", { method: "POST", body: { title: text.slice(0, 40) } })).id; loadChats(); }
    const d = await api(`/chats/${chatId}/messages`, { method: "POST", body: { content: text } });
    wait.remove(); addBot(d.reply, d.citations);
  } catch (err) { wait.remove(); addBot(err.message, [], false); }
};
$("#input").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); } });
$("#input").addEventListener("input", e => { e.target.style.height = "auto"; e.target.style.height = e.target.scrollHeight + "px"; });

/* ---------- search settings (everyone reads, admins edit) ---------- */
async function loadCfg() {
  const ro = !me.is_admin;
  ["#boost", "#maxRes", "#blocked"].forEach(s => $(s).disabled = ro);
  $("#saveCfg").classList.toggle("hidden", ro);
  try {
    const c = await api("/search-config");
    $("#boost").value = c.diversity_boost; $("#boostOut").textContent = Math.round(c.diversity_boost * 100) + "%";
    $("#maxRes").value = c.max_results; $("#blocked").value = c.blocked_domains;
    $("#adminMsg").textContent = "";
  } catch (err) { $("#adminMsg").textContent = err.message; }
}
$("#boost").oninput = e => $("#boostOut").textContent = Math.round(e.target.value * 100) + "%";
$("#saveCfg").onclick = async () => {
  try {
    await api("/admin/search-config", { method: "PUT", body: { diversity_boost: +$("#boost").value, max_results: +$("#maxRes").value, blocked_domains: $("#blocked").value } });
    $("#adminMsg").textContent = "Settings saved.";
  } catch (err) { $("#adminMsg").textContent = err.message; }
};

/* ---------- admin: user database ---------- */
async function loadUsers() {
  try {
    const d = await api("/admin/users");
    $("#usersMsg").textContent = "";
    $("#usersBody").innerHTML = "";
    d.users.forEach(u => {
      const tr = document.createElement("tr");
      [u.name, u.email, u.is_admin ? "Admin" : "User", new Date(u.created_at * 1000).toLocaleDateString()]
        .forEach(v => { const td = document.createElement("td"); td.textContent = v; tr.append(td); });
      $("#usersBody").append(tr);
    });
  } catch (err) { $("#usersMsg").textContent = err.message; }
}

/* ---------- live background: two streams of points that drift toward each other and braid ---------- */
(function () {
  const cv = $("#bg"), g = cv.getContext("2d"); let w, h, pts = [];
  const still = matchMedia("(prefers-reduced-motion:reduce)").matches;
  function size() { w = cv.width = innerWidth; h = cv.height = innerHeight; pts = Array.from({ length: Math.min(110, w / 12) }, (_, i) => ({ s: i % 2, x: Math.random() * w, y: Math.random() * h, p: Math.random() * 6.28, v: .25 + Math.random() * .5 })); }
  function frame(t) {
    g.clearRect(0, 0, w, h);
    for (const q of pts) {
      q.x += (q.s ? -q.v : q.v); if (q.x > w + 10) q.x = -10; if (q.x < -10) q.x = w + 10;
      const mid = h / 2 + Math.sin(q.x / 140 + t / 2400 + q.s * 3.14) * h * .22 * (1 - Math.abs(q.x / w - .5) * 1.4);
      q.y += (mid + Math.sin(q.p + t / 900) * 40 - q.y) * .01;
      g.fillStyle = q.s ? "rgba(232,163,61,.55)" : "rgba(45,91,227,.5)";
      g.beginPath(); g.arc(q.x, q.y, 2.6, 0, 6.28); g.fill();
    }
    if (!still) requestAnimationFrame(frame);
  }
  addEventListener("resize", size); size(); requestAnimationFrame(frame);
})();

boot();
