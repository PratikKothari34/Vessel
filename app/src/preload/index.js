'use strict';

const { contextBridge, ipcRenderer } = require('electron');

// The backend port comes from the MAIN process, which owns it: main computes
// BACKEND_PORT, spawns the backend with it, and talks to /shutdown on it.
// Reading process.env.PORT here instead would be a second, independent guess —
// the preload is sandboxed, so its process.env is the renderer's, and whether
// that mirrors main's environment is not something to rely on. ipcRenderer
// .sendSync resolves it before the renderer's first fetch, so there is no
// window where the URL is wrong.
const backendUrl = ipcRenderer.sendSync('app:backend-url');

// Expose only the backend URL and a relaunch trigger to the renderer.
// Everything else goes over HTTP to the local backend (loopback-only).
// relaunch is safe to expose: worst case a compromised renderer restarts
// the app, which it could already do by other means (e.g. crashing itself).
contextBridge.exposeInMainWorld('scenario', {
  backendUrl,
  relaunch: () => ipcRenderer.invoke('app:relaunch'),
});
