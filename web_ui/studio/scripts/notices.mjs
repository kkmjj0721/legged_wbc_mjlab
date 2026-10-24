import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const directory=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const lock=JSON.parse(fs.readFileSync(path.join(directory,'package-lock.json'),'utf8'));
const blocks=[`Robot migration local preview\n\nURDF Studio: https://github.com/OpenLegged/URDF-Studio\nPinned commit 23b96e2f3d5b1b68ad9629925d4479e9847d35a5\nApache-2.0\n\n${fs.readFileSync(path.join(directory,'vendor/LICENSE'),'utf8')}`];
for(const [relative,entry] of Object.entries(lock.packages)){
 if(!relative || entry.dev)continue;
 const moduleDirectory=path.join(directory,relative);
 if(!fs.existsSync(moduleDirectory))continue;
 const candidates=fs.readdirSync(moduleDirectory).filter(name=>/^(licen[cs]e|copying|notice)(\.|$)/i.test(name)&&fs.statSync(path.join(moduleDirectory,name)).isFile());
 const manifestPath=path.join(moduleDirectory,'package.json');
 const manifest=fs.existsSync(manifestPath)?JSON.parse(fs.readFileSync(manifestPath,'utf8')):{};
 blocks.push(`${manifest.name||relative} ${entry.version||''}\n${manifest.license||entry.license||'See upstream distribution'}\n`+candidates.map(name=>fs.readFileSync(path.join(moduleDirectory,name),'utf8')).join('\n'));
}
for(const file of ['JetBrainsMono-LICENSE.txt','FiraCode-LICENSE.txt'])blocks.push(fs.readFileSync(path.join(directory,'public/fonts',file),'utf8'));
blocks.push('Studio HDRI: potsdamer_platz_1k.hdr, Poly Haven, CC0. https://polyhaven.com/a/potsdamer_platz and https://polyhaven.com/license');
fs.writeFileSync(path.join(directory,'dist/THIRD_PARTY_NOTICES.txt'),blocks.join('\n\n'+'='.repeat(72)+'\n\n'));
