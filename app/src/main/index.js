'use strict';

const { app, BrowserWindow, shell } = require('electron');
const path = require('path');
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

function startBackend() {
  const { script, cwd } = backendPaths();
  backendProc = spawn(process.execPath, [script], {
    cwd,
    env: { ...process.env, PORT: String(BACKEND_PORT), ELECTRON_RUN_AS_NODE: '1' },
    stdio: 'pipe',
  });
  backendProc.stdout.on('data', (d) => process.stdout.write(`[backend] ${d}`));
  backendProc.stderr.on('data', (d) => process.stderr.write(`[backend] ${d}`));
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

app.whenReady().then(async () => {
  startBackend();
  try {
    await waitForBackend();
  } catch (e) {
    console.error('Backend did not become ready:', e.message);
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
