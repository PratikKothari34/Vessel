// Dev launcher. Clears ELECTRON_RUN_AS_NODE before starting electron-vite —
// if that var is set in the shell, Electron runs the main process as plain Node
// and `require('electron')` returns undefined (app never opens a window). We set
// it ONLY for the backend child (see src/main/index.js), never for the app.
import { spawn } from 'child_process';

const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;

const bin = process.platform === 'win32' ? 'electron-vite.cmd' : 'electron-vite';
const child = spawn(bin, ['dev'], { stdio: 'inherit', env, shell: true });
child.on('exit', (code) => process.exit(code ?? 0));
