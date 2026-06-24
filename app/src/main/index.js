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

  // Open external links in the system browser, not the app window.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

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

// Kill the backend child on quit so it doesn't linger.
app.on('before-quit', () => {
  if (backendProc && !backendProc.killed) {
    try { backendProc.kill(); } catch { /* best effort */ }
  }
});
