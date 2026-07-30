'use strict';

const { app, BrowserWindow, shell, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const http = require('http');

const isDev = !app.isPackaged;
const BACKEND_PORT = process.env.PORT || 3001;
const BACKEND_URL = `http://localhost:${BACKEND_PORT}`;

let backendProc = null;
let mainWindow = null;

// Resolve the backend entry + cwd for dev vs packaged.
function backendPaths() {
  if (isDev) {
    const root = path.resolve(__dirname, '../../..'); // app/out/main -> repo root
    return { script: path.join(root, 'src', 'backend', 'server.js'), cwd: root };
  }
  // Packaged: backend bundled under resources/backend (see electron-builder cfg).
  const base = path.join(process.resourcesPath, 'backend');
  return { script: path.join(base, 'src', 'backend', 'server.js'), cwd: base };
}

// Packaged: user data must NOT live under resources/ (the backend cwd) — an
// NSIS update reinstalls into a wiped install dir, deleting the DB, its sync
// sidecars, and settings.json. Point the backend at Electron's per-user data
// dir instead, migrating anything an older build left in resources/backend/data
// (one-time copy; the filename scenario.db must not change — see keystore.js).
// Dev keeps the repo-relative default; an explicit LOCAL_DB_PATH always wins.
function packagedDataEnv(cwd) {
  if (isDev || process.env.LOCAL_DB_PATH) return {};
  const dataDir = path.join(app.getPath('userData'), 'data');
  const newDb = path.join(dataDir, 'scenario.db');
  const oldDir = path.join(cwd, 'data');
  try {
    fs.mkdirSync(dataDir, { recursive: true });
    if (!fs.existsSync(newDb) && fs.existsSync(oldDir)) {
      for (const f of fs.readdirSync(oldDir)) {
        const dst = path.join(dataDir, f);
        if (!fs.existsSync(dst)) fs.copyFileSync(path.join(oldDir, f), dst);
      }
      console.log(`[main] migrated legacy data dir ${oldDir} -> ${dataDir}`);
    }
  } catch (e) {
    console.error('[main] data dir migration failed:', e.message);
  }
  return { LOCAL_DB_PATH: newDb };
}

// Last lines the backend wrote before dying. If it refuses to start (e.g. it
// will not open the database unencrypted), the window would otherwise just say
// "backend offline — is Ollama running?", which points at the wrong thing.
let backendStderr = '';

function startBackend() {
  const { script, cwd } = backendPaths();
  backendProc = spawn(process.execPath, [script], {
    cwd,
    env: { ...process.env, ...packagedDataEnv(cwd), PORT: String(BACKEND_PORT), ELECTRON_RUN_AS_NODE: '1' },
    stdio: 'pipe',
  });
  backendProc.stdout.on('data', (d) => process.stdout.write(`[backend] ${d}`));
  backendProc.stderr.on('data', (d) => {
    process.stderr.write(`[backend] ${d}`);
    backendStderr = (backendStderr + d).slice(-4000); // keep the tail only
  });
  backendProc.on('exit', (code) => console.log(`[backend] exited (${code})`));
}

// Poll /health until the backend answers (or time out).
function waitForBackend(timeoutMs = 20000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      const req = http.get(`${BACKEND_URL}/health`, (res) => {
        res.resume();
        if (res.statusCode === 200) return resolve();
        retry();
      });
      req.on('error', retry);
      req.setTimeout(1500, () => { req.destroy(); retry(); });
    };
    const retry = () => {
      if (Date.now() - start > timeoutMs) return reject(new Error('backend health timeout'));
      setTimeout(tick, 400);
    };
    tick();
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 940,
    minHeight: 640,
    backgroundColor: '#0a0a0f',
    show: false,
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      // Explicit rather than inherited: renderers are sandboxed by default on
      // Electron 20+, but stating it here keeps a future webPreferences edit
      // from silently dropping it. The preload only requires 'electron', which
      // stays available under the sandbox.
      sandbox: true,
      webviewTag: false,
    },
  });

  mainWindow.once('ready-to-show', () => mainWindow.show());

  // The app never needs camera/mic/geolocation/etc. Electron grants permission
  // requests by default — deny them all so a compromised renderer can't ask.
  mainWindow.webContents.session.setPermissionRequestHandler((_wc, _permission, cb) => cb(false));

  // Open external links in the system browser — but ONLY safe web schemes, so a
  // compromised renderer can't trigger file://, javascript:, or app-launch URLs.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    try {
      const u = new URL(url);
      if (u.protocol === 'https:' || u.protocol === 'http:') shell.openExternal(url);
    } catch { /* malformed url — ignore */ }
    return { action: 'deny' };
  });

  // Block the main window from navigating away from the app (defense against a
  // compromised renderer redirecting to a phishing/remote page).
  const allowedNav = (url) => {
    if (isDev && process.env.ELECTRON_RENDERER_URL && url.startsWith(process.env.ELECTRON_RENDERER_URL)) return true;
    return url.startsWith('file://');
  };
  mainWindow.webContents.on('will-navigate', (e, url) => {
    if (!allowedNav(url)) e.preventDefault();
  });
  mainWindow.webContents.on('will-attach-webview', (e) => e.preventDefault());

  if (isDev && process.env.ELECTRON_RENDERER_URL) {
    mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL);
  } else {
    mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
  }
}

// The preload asks for the backend URL synchronously at load, so the renderer
// never has to guess the port from its own environment (see preload/index.js).
ipcMain.on('app:backend-url', (e) => { e.returnValue = BACKEND_URL; });

// Renderer asks for a relaunch after saving sync settings (they apply at
// backend boot). Goes through the normal quit path, so before-quit still
// flushes + kills the backend first.
ipcMain.handle('app:relaunch', () => {
  app.relaunch();
  app.quit();
});

app.whenReady().then(async () => {
  startBackend();
  try {
    await waitForBackend();
  } catch (e) {
    console.error('Backend did not become ready:', e.message);
    // The backend exited instead of listening. That is a hard startup failure
    // with a real cause (most importantly: it refused to open the database
    // unencrypted), and it is worth showing verbatim — the in-app status pill
    // can only say "offline", which sends people to look at Ollama.
    if (backendProc && backendProc.exitCode !== null) {
      const detail = backendStderr.trim().split('\n').slice(-6).join('\n');
      dialog.showErrorBox(
        'Vessel could not start',
        'The Vessel backend stopped during startup.\n\n' +
        (detail || 'No error output was captured.'),
      );
    }
  }
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// Flush + kill the backend child on quit so it doesn't linger. On Windows,
// child.kill() is TerminateProcess — the backend's SIGTERM handler (final cloud
// sync) never runs — so ask it to flush over HTTP first, then kill.
let quitting = false;
app.on('before-quit', (e) => {
  if (quitting) return;
  quitting = true;
  if (!backendProc || backendProc.killed) return;
  e.preventDefault();

  const finish = () => {
    try { if (backendProc && !backendProc.killed) backendProc.kill(); } catch { /* best effort */ }
    app.exit(0);
  };
  const req = http.request(
    { host: '127.0.0.1', port: BACKEND_PORT, path: '/shutdown', method: 'POST', headers: { 'x-vessel-shutdown': '1' }, timeout: 3000 },
    (res) => { res.resume(); res.on('end', finish); },
  );
  req.on('error', finish);
  req.on('timeout', () => { req.destroy(); finish(); });
  req.end();
});
