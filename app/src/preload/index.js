'use strict';

const { contextBridge } = require('electron');

const BACKEND_PORT = process.env.PORT || 3001;

// Expose only the backend URL to the renderer. Everything else goes over HTTP
// to the local backend (loopback-only), so no privileged IPC surface is needed.
contextBridge.exposeInMainWorld('scenario', {
  backendUrl: `http://localhost:${BACKEND_PORT}`,
});
