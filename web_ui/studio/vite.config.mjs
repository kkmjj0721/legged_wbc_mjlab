import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

const directory = path.dirname(fileURLToPath(import.meta.url));
const upstream = path.resolve(directory, process.env.STUDIO_UPSTREAM || 'vendor');
export default defineConfig({
  root: directory,
  base: '/studio/',
  publicDir: path.join(directory, "public"),
  plugins: [react(), tailwindcss(), {
    name: 'migration-preview-viewport',
    enforce: 'pre',
    resolveId(source, importer) {
      // Add local camera tools without changing the pinned upstream sources.
      if (source === './RobotCanvasViewport' && importer === path.join(upstream, 'src/lib/components/RobotCanvas.tsx')) {
        return path.join(directory, 'src/RobotCanvasViewport.tsx');
      }
    },
  }, {
    name: 'record-upstream-sources',
    transform(code, id) {
      if (!id.startsWith(upstream + '/src/')) return;
      return code.replaceAll('/fonts/', '/studio/fonts/').replaceAll('/potsdamer_platz_1k.hdr', '/studio/potsdamer_platz_1k.hdr');
    },
    generateBundle() {
      if (!process.env.STUDIO_SOURCE_LIST) return;
      fs.writeFileSync(process.env.STUDIO_SOURCE_LIST, JSON.stringify(
        [...this.getModuleIds()].filter(id => id.startsWith(upstream + '/src/')).map(id => id.split('?')[0]), null, 2));
    },
  }],
  resolve: {
    alias: [
      {find: '@urdf-studio/react-robot-canvas', replacement: path.join(upstream, 'src/lib/index.ts')},
      {find: '@urdf-studio/robot-runtime/parser', replacement: path.join(upstream, 'src/lib/robot-parser/dispatch.ts')},
      {find: '@', replacement: path.join(upstream, 'src')},
    ],
    dedupe: ['react', 'react-dom', 'three', '@react-three/fiber'],
  },
  worker: {format: 'es'},
  build: {outDir: 'dist', emptyOutDir: true, sourcemap: false, chunkSizeWarningLimit: 3500},
});
