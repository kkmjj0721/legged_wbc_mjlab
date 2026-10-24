import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
const directory=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const manifest=JSON.parse(fs.readFileSync(path.join(directory,'vendor/UPSTREAM.json'),'utf8'));
for(const file of manifest.unmodified_sources){
 const content=fs.readFileSync(path.join(directory,'vendor',file.path));
 if(crypto.createHash('sha256').update(content).digest('hex')!==file.sha256)throw Error(`Upstream source differs from pinned manifest: ${file.path}`);
}
console.log(`Verified URDF Studio ${manifest.commit}: ${manifest.unmodified_sources.length} source files`);
