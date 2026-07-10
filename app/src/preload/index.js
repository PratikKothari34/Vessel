'use strict';

const { contextBridge, ipcRenderer } = require('electron');

const BACKEND_PORT = process.env.PORT || 3001;

// Expose only the backend URL and a relaunch trigger to the renderer.
// Everything else goes over HTTP to the local backend (loopback-only).
// relaunch is safe to expose: worst case a compromised renderer restarts
// the app, which it could already do by other means (e.g. crashing itself).
contextBridge.exposeInMainWorld('scenario', {
  backendUrl: `http://localhost:${BACKEND_PORT}`,
  relaunch: () => ipcRenderer.invoke('app:relaunch'),
});
