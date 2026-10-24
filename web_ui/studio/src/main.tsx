import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { invalidate } from '@react-three/fiber';
import * as THREE from 'three';
import { RobotCanvas } from '@urdf-studio/react-robot-canvas';
import { parseRobotDefinitionAsync } from '@urdf-studio/robot-runtime/parser';
import { ViewportControls, type CameraAction } from './viewportControls';
import './style.css';

const token = new URLSearchParams(location.search).get('token') || '';
const endpoint = `/api/preview/studio/${encodeURIComponent(token)}`;
async function json(path: string, signal?: AbortSignal) {
  const response = await fetch(`${endpoint}/${path}`, {signal, cache: 'no-store'});
  const value = await response.json();
  if (!response.ok) throw new Error(value.message || '预览已关闭，请重新打开');
  return value;
}
class RenderError extends React.Component<React.PropsWithChildren, {message: string}> {
  state = {message: ''};
  static getDerivedStateFromError(error: Error) { return {message: error.message}; }
  render() { return this.state.message ? <div className="error">3D 显示失败：{this.state.message}</div> : this.props.children; }
}

function disposeMarkers(group: THREE.Group | null) {
  group?.removeFromParent();
  group?.traverse((item: any) => { item.geometry?.dispose(); if (item.material) { item.material.map?.dispose(); item.material.dispose(); } });
}

function App() {
  const [model, setModel] = useState<any>(null);
  const [pose, setPose] = useState<any>(null);
  const [error, setError] = useState('');
  const [ready, setReady] = useState(false);
  const [robot, setRobot] = useState<THREE.Object3D | null>(null);
  const [mode, setMode] = useState<'rotate' | 'pan'>('rotate');
  const [action, setAction] = useState<{type: CameraAction; revision: number} | null>(null);
  const [resolution, setResolution] = useState(0);
  const [options, setOptions] = useState({showVisual: true, showCollision: false, showWorldAxes: true, showJointAxes: false, showCenterOfMass: false, showInertia: false, feet: true});
  const robotRef = useRef<any>(null);
  const poseRef = useRef<any>(null);
  const markersRef = useRef<THREE.Group | null>(null);
  const feetRef = useRef(true);
  feetRef.current = options.feet;

  const applyPose = useCallback(() => {
    const robot = robotRef.current;
    const value = poseRef.current;
    if (!robot || !value) return;
    if (value.pos) robot.position.fromArray(value.pos);
    if (value.rot) robot.quaternion.set(value.rot[1], value.rot[2], value.rot[3], value.rot[0]);
    for (const [name, angle] of Object.entries(value.joint_positions || {})) robot.joints?.[name]?.setJointValue?.(angle);
    robot.updateMatrixWorld(true);
    const group = markersRef.current;
    if (group) {
      if (!group.parent && robot.parent) robot.parent.add(group);
      group.visible = feetRef.current;
      for (const child of group.children) {
        const point = value.feet?.[child.userData.site];
        if (point) child.position.fromArray(point);
      }
      group.updateMatrixWorld(true);
    }
    invalidate();
  }, []);

  const loaded = useCallback((robot: any) => {
    disposeMarkers(markersRef.current);
    markersRef.current = null;
    robotRef.current = robot;
    const group = new THREE.Group();
    group.name = 'migration-foot-markers';
    const colors = [0xb1633a, 0x5b8067, 0x59769a, 0xb28c37];
    Object.keys(poseRef.current?.feet || {}).forEach((site, index) => {
      const point = new THREE.Group();
      point.userData.site = site;
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.008, 24, 16), new THREE.MeshBasicMaterial({color: colors[index % 4], depthTest: false}));
      mesh.renderOrder = 999;
      point.add(mesh);
      const canvas = document.createElement('canvas'); canvas.width = 512; canvas.height = 128;
      const context = canvas.getContext('2d')!;
      context.fillStyle = '#3e3026'; context.font = 'bold 48px sans-serif'; context.textAlign = 'center'; context.fillText(site, 256, 76);
      const texture = new THREE.CanvasTexture(canvas);
      const label = new THREE.Sprite(new THREE.SpriteMaterial({map: texture, depthTest: false, transparent: true}));
      label.position.z = 0.025; label.scale.set(0.16, 0.04, 1); label.renderOrder = 1000;
      point.add(label); group.add(point);
    });
    robot.parent?.add(group);
    markersRef.current = group;
    applyPose();
    setRobot(robot);
    setReady(true);
    // Available locally for inspection and geometry/pose acceptance checks.
    (window as any).__migrationStudio = {renderer: 'URDF Studio RobotCanvas', robot, pose: poseRef.current};
  }, [applyPose]);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    let previousRevision = -1;
    async function poll() {
      try {
        const value = await json('state', controller.signal);
        if (value.revision !== previousRevision) {
          previousRevision = value.revision; poseRef.current = value.pose; setPose(value.pose);
        }
        applyPose();
        if ((window as any).__migrationStudio) (window as any).__migrationStudio.pose = value.pose;
        timer = setTimeout(poll, document.hidden ? 1500 : 300);
      } catch (failure: any) { if (!controller.signal.aborted) setError(failure.message); }
    }
    (async () => {
      try {
        const data = await json('model', controller.signal);
        const initial = await json('state', controller.signal);
        previousRevision = initial.revision; poseRef.current = initial.pose; setPose(initial.pose);
        const parsed = await parseRobotDefinitionAsync(data.source.content, data.source.sourceFilePath, {assets: data.assets, sourcePath: data.source.sourceFilePath, signal: controller.signal});
        if (parsed.status !== 'ready') throw new Error(parsed.message || 'URDF Studio 无法解析当前模型');
        if (controller.signal.aborted) return;
        setModel({...data, robotData: parsed.robotData});
        poll();
      } catch (failure: any) { if (!controller.signal.aborted) setError(failure.message); }
    })();
    return () => { controller.abort(); clearTimeout(timer); disposeMarkers(markersRef.current); markersRef.current = null; robotRef.current = null; delete (window as any).__migrationStudio; };
  }, [applyPose]);
  useEffect(applyPose, [pose, options.feet, applyPose]);
  const moveCamera = (type: CameraAction) => setAction(previous => ({type, revision: (previous?.revision || 0) + 1}));
  return <main>
    <div className="toolbar" role="toolbar" aria-label="3D 预览工具">
      <div className="toolgroup camera-tools">
        <button type="button" aria-pressed={mode === 'rotate'} onClick={()=>setMode('rotate')} title="左键拖动旋转；右键平移">旋转</button>
        <button type="button" aria-pressed={mode === 'pan'} onClick={()=>setMode('pan')} title="左键拖动平移">平移</button>
        <button type="button" onClick={()=>moveCamera('zoomIn')} disabled={!ready} title="放大" aria-label="放大">＋</button>
        <button type="button" onClick={()=>moveCamera('zoomOut')} disabled={!ready} title="缩小" aria-label="缩小">−</button>
        <button type="button" onClick={()=>moveCamera('fit')} disabled={!ready}>适应视图</button>
        <button type="button" onClick={()=>moveCamera('reset')} disabled={!ready} title="恢复等距视角并完整显示机器人">复位</button>
        <select aria-label="对齐视角" value="" disabled={!ready} onChange={event=>moveCamera(event.target.value as CameraAction)}>
          <option value="" disabled>对齐视角</option><option value="front">前视 · +X</option><option value="side">侧视 · −Y</option><option value="top">俯视 · +Z</option>
        </select>
      </div>
      <div className="toolgroup display-tools">{Object.entries({showVisual:'外观',showCollision:'碰撞',showWorldAxes:'坐标',showJointAxes:'关节轴',showCenterOfMass:'质心',showInertia:'惯量',feet:'足端'}).map(([key,label])=><label key={key}><input type="checkbox" checked={(options as any)[key]} onChange={event=>setOptions(previous=>({...previous,[key]:event.target.checked}))}/>{label}</label>)}</div>
    </div>
    <section className="viewport" aria-label="机器人初始姿态画布">{model && <RenderError><ViewportControls.Provider value={{mode, showWorldAxes: options.showWorldAxes, robot, action, onResolution:setResolution}}><RobotCanvas source={model.source} robotData={model.robotData} assets={model.assets} jointAngles={pose?.joint_positions || {}} onRobotLoaded={loaded} lang="zh" theme="light" enableJointInteraction={false} display={{...options, renderQuality:'high', showMjcfWorldLink:false}} /></ViewportControls.Provider></RenderError>}
    {!ready && !error && <div className="status">正在本地加载模型与网格…</div>}{error && <div className="error" role="alert">{error}</div>}</section>
    <footer><span>左键{mode === 'pan' ? '平移' : '旋转'} · 右键平移 · 滚轮缩放 <span className="fk-note">· MuJoCo 姿态同步</span></span><span className="quality">{resolution ? `高清 ${resolution.toFixed(2).replace(/0$/, '').replace(/\.$/, '')}×` : '高清预览'}</span><a href="https://github.com/OpenLegged/URDF-Studio" target="_blank" rel="noreferrer">URDF Studio · Apache-2.0</a></footer>
  </main>;
}
createRoot(document.getElementById('root')!).render(<App/>);
