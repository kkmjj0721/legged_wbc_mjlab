"use strict";

// MigrationSpec is the only editable state. Forms, preview and generator share it.
(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const legs = ["FL", "FR", "RL", "RR"];
  const legLabels = {FL:"左前腿", FR:"右前腿", RL:"左后腿", RR:"右后腿"};
  const roles = ["hip", "thigh", "calf"];
  const taskMeta = {
    ppo:{label:"PPO",sub:"CLASSIC LOCOMOTION",description:"标准速度跟踪训练。独立的观测、奖励与策略配置。",suffix:""},
    him:{label:"HIM + PPO",sub:"HISTORY INFORMATION ESTIMATOR",description:"通过历史观测估计隐状态，沿用完整 HIM 训练与导出契约。",suffix:"-HIM"},
    him_symmetry:{label:"HIM Symmetry",sub:"SYMMETRY AUGMENTATION",description:"在 HIM 上加入左右对称增强，校验新机器人的镜像映射。",suffix:"-HIM-Symmetry"},
    him_lpacrl:{label:"HIM LP-ACRL",sub:"LEARNING PROGRESS CURRICULUM",description:"按学习进展调整任务采样，保留现有默认、Compact 及历史课程。",suffix:"-HIM-LPACRL"},
  };
  const groupMeta = {
    environment:{label:"时序与环境",sub:"仿真频率、环境数量与回合设置",open:true},
    init_state:{label:"机体初始状态",sub:"世界坐标位置与 wxyz 四元数",open:true},
    control:{label:"控制与执行器延迟",sub:"动作缩放、控制降采样与执行器时序",open:true},
    commands:{label:"速度指令",sub:"速度范围、重采样时间与站立比例",open:true},
    terrain:{label:"地形",sub:"任务地形及地形生成参数",open:false},
    curriculum:{label:"课程设置",sub:"课程开关与进度参数",open:false},
    reset:{label:"Reset 随机化",sub:"机体与关节重置范围",open:false},
    randomization:{label:"动力学随机化",sub:"质量、摩擦、扰动与 IMU 偏差",open:false},
    observations:{label:"观测与延迟",sub:"观测延迟独立于执行器延迟，噪声按观测项配置",open:false},
    rewards:{label:"奖励项",sub:"奖励权重及每项参数",open:false},
    learning:{label:"网络与训练",sub:"Policy、PPO、Estimator 与训练预算",open:false},
  };
  const fieldLabels = {
    num_envs:"环境数量",env_spacing:"环境间距",episode_length_s:"回合长度",sim_dt:"仿真步长",dt:"仿真步长",decimation:"控制降采样",pos:"机体初始位置 [x, y, z]",rot:"机体初始姿态 [w, x, y, z]",stiffness:"PD 刚度 Kp",damping:"PD 阻尼 Kd",effort_limit:"力矩限幅",armature:"转子惯量 armature",friction:"被动摩擦",passive_damping:"被动阻尼",action_scale:"动作缩放",hip_reduction:"髋关节动作系数",action_clip:"动作裁剪",delay_min_lag:"最小延迟",delay_max_lag:"最大延迟",delay_hold_prob:"延迟保持概率",delay_update_period:"延迟更新周期",enabled:"启用",curriculum:"启用课程",max_curriculum:"课程上限",resampling_time:"指令重采样间隔",heading_command:"启用航向指令",rel_standing_envs:"站立环境比例",rel_forward_envs:"前进环境比例",rel_heading_envs:"航向环境比例",heading_control_stiffness:"航向控制刚度",lin_vel_x:"前进速度 vx",lin_vel_y:"横向速度 vy",ang_vel_yaw:"转向速度 wz",heading:"航向角",link_mass_range:"腿部质量系数范围",base_mass_range:"机体质量系数范围",KpKd_factor_range:"PD 系数随机范围",com_displacement_range:"质心偏移范围",motor_offset_range:"电机零点偏差范围",motor_strength_range:"电机强度系数范围",effort_limit_range:"力矩上限系数范围",joint_friction_range:"关节摩擦系数范围",joint_armature_range:"关节惯量系数范围",joint_damping_range:"被动阻尼系数范围",imu_gyro_bias_range:"陀螺仪偏差范围",imu_mount_angle_range:"IMU 安装角范围",external_force_range:"外力范围",external_force_duration_s:"外力持续时间",external_force_cooldown_s:"外力冷却时间",foot_contact_time_constant_range:"足端接触时间常数",enable_corruption:"启用观测噪声",weight:"权重",n_min:"噪声下限",n_max:"噪声上限",init_noise_std:"初始动作噪声",actor_hidden_dims:"Actor 隐藏层",critic_hidden_dims:"Critic 隐藏层",activation:"激活函数",obs_normalization:"观测归一化",encoder_hidden_dims:"Encoder 隐藏层",target_hidden_dims:"Target 隐藏层",num_prototype:"原型数量",temperature:"温度",sinkhorn_eps:"Sinkhorn ε",sinkhorn_iters:"Sinkhorn 迭代次数",value_loss_coef:"价值损失系数",use_clipped_value_loss:"启用价值裁剪",clip_param:"PPO 裁剪系数",entropy_coef:"熵奖励系数",num_learning_epochs:"学习轮数",num_mini_batches:"Mini-batch 数量",learning_rate:"学习率",schedule:"学习率计划",gamma:"折扣因子 γ",lam:"GAE λ",desired_kl:"目标 KL",max_grad_norm:"梯度范数上限",estimator_learning_rate:"Estimator 学习率",estimator_max_grad_norm:"Estimator 梯度上限",num_steps_per_env:"每次更新采样步数",max_iterations:"最大训练更新次数",save_interval:"检查点保存间隔",experiment_name:"实验目录名称",seed:"随机种子",horizontal_scale:"水平分辨率",vertical_scale:"高度分辨率",border_size:"边界宽度",num_rows:"地形行数",num_cols:"地形列数",size:"地形尺寸",difficulty_range:"难度范围",max_init_terrain_level:"初始地形等级上限",proportion:"采样比例",noise_range:"噪声范围",position_range:"关节位置偏移范围",velocity_range:"速度偏移范围",x:"X 偏移",y:"Y 偏移",z:"Z 偏移",roll:"Roll 偏移",pitch:"Pitch 偏移",yaw:"Yaw 偏移",symmetry_loss_coefficient:"对称损失系数",use_data_augmentation:"启用对称数据增强",use_mirror_loss:"启用镜像损失",std:"标准差",mean:"均值",min:"最小值",max:"最大值",
  };
  const unitFor = (path) => {
    const key = pathSegments(path).at(-1);
    if (["num_envs","num_rows","num_cols"].includes(key)) return "个";
    if (["episode_length_s","sim_dt","dt","resampling_time","external_force_duration_s","external_force_cooldown_s","foot_contact_time_constant_range"].includes(key)) return "s";
    if (["env_spacing","com_displacement_range","horizontal_scale","vertical_scale","border_size","size","pos","x","y","z"].includes(key)) return "m";
    if (["lin_vel_x","lin_vel_y"].includes(key)) return "m/s";
    if (["ang_vel_yaw","imu_gyro_bias_range"].includes(key)) return "rad/s";
    if (["heading","roll","pitch","yaw","imu_mount_angle_range","motor_offset_range"].includes(key)) return "rad";
    if (key === "external_force_range") return "N";
    if (["delay_min_lag","delay_max_lag","delay_update_period","decimation"].includes(key)) return "步";
    if (["max_iterations","save_interval","sinkhorn_iters","num_learning_epochs"].includes(key)) return "次";
    return "";
  };
  const fixedKeys = new Set(["num_actions","num_action","num_observations","num_privileged_obs","num_one_step_obs","history_size","history_term_dims","history_order","num_obs_history","num_obs_one_step","num_obs","estimator_velocity_slice","estimator_target_slices","observation_layout"]);
  const jointPaths = new Set(["control.stiffness","control.damping","control.effort_limit","control.armature","control.friction","control.passive_damping","control.action_scale","control.action_clip","init_state.default_joint"]);
  const state = {
    step:0, scope:"common", importMode:"path", templates:[], inspection:null, spec:{version:1,model:{},robot:{slug:"",prefix:""},bindings:{base_body:"",joints:[],feet:[],base_geoms:[],leg_geoms:[]},parameters:{},tasks:{him:{terrains:["Rough"],parameters:{}}},confirmed:false},
    importDefaults:null, plan:null, planRevision:-1, revision:0, busy:false, previewOpen:false, poseTimer:null, poseInFlight:false, posePending:false, generated:false, generatedRevision:-1, generatedTaskIds:[], job:null, jobTimer:null, currentFile:null, draftId:null, invalidFields:new Set(), templatesError:false,
  };

  function merge(base, overlay) {
    if (overlay === undefined) return clone(base);
    if (base && overlay && !Array.isArray(base) && !Array.isArray(overlay) && typeof base === "object" && typeof overlay === "object") {
      const result = clone(base);
      for (const [key,value] of Object.entries(overlay)) result[key] = key in result ? merge(result[key],value) : clone(value);
      return result;
    }
    return clone(overlay);
  }
  // Dots in MJCF names, reward regexes and terrain labels belong to the key.
  // Encode each UI path segment; the saved MigrationSpec always keeps raw keys.
  function encodeSegment(key) { return encodeURIComponent(key).replace(/\./g,"%2E"); }
  function pathSegments(path) { return path.split(".").map((key)=>decodeURIComponent(key)); }
  function displayPath(path) { return pathSegments(path).join("."); }
  function get(object,path) { return pathSegments(path).reduce((value,key) => value?.[key],object); }
  function set(object,path,value) {
    const keys=pathSegments(path); let current=object;
    keys.slice(0,-1).forEach((key) => { if (!current[key] || typeof current[key] !== "object") current[key]={}; current=current[key]; });
    current[keys.at(-1)]=value;
  }
  function remove(object,path) {
    const keys=pathSegments(path); const parent=keys.slice(0,-1).reduce((o,k)=>o?.[k],object);
    if (parent) delete parent[keys.at(-1)];
  }
  function template(key) { return state.templates.find((item)=>(item.key || item.id)===key); }
  function firstTemplate() { return template(Object.keys(state.spec.tasks)[0]) || state.templates[0]; }
  function baseDefaults(scope=state.scope) { return (scope === "common" ? firstTemplate() : template(scope))?.defaults || {}; }
  function effective(scope=state.scope) { return merge(merge(baseDefaults(scope),state.spec.parameters),scope === "common" ? {} : state.spec.tasks[scope]?.parameters || {}); }
  function targetParameters() { return state.scope === "common" ? state.spec.parameters : state.spec.tasks[state.scope].parameters; }
  function sourceBadge(path) {
    const own=get(targetParameters(),path);
    if (own !== undefined) {
      const imported=state.scope === "common" ? get(state.importDefaults?.parameters,path) : undefined;
      const modelSourced=state.scope === "common" && imported !== undefined && JSON.stringify(own)===JSON.stringify(imported) && (path.startsWith("init_state") || path.startsWith("control"));
      const initialGuess=modelSourced&&path.startsWith("init_state")&&state.inspection?.initial_source!=="keyframe";
      return `<em class="source-badge ${initialGuess?"template":modelSourced?"model":"manual"}">${initialGuess?"模板起点":modelSourced?"模型读取":"人工修改"}</em>`;
    }
    if (state.scope !== "common" && get(state.spec.parameters,path)!==undefined) return '<em class="source-badge manual">公共覆盖</em>';
    return '<em class="source-badge template">模板默认</em>';
  }
  function changed({pose=false}={}) {
    state.revision+=1;state.spec.confirmed=false;$("#confirm-review").checked=false;
    state.plan=null;state.currentFile=null;
    $("#file-review").hidden=true;$("#diff-section").hidden=true;
    $("#plan-summary").textContent="配置已更新，请重新生成文件预览";
    $("#plan-issues").replaceChildren();
    if (pose && state.previewOpen) schedulePose();
    updateSummary();updateActions();
  }
  function toast(message,type="") {
    const node=$("#toast");node.textContent=message;node.className=`toast ${type}`;node.hidden=false;
    clearTimeout(toast.timer);toast.timer=setTimeout(()=>{node.hidden=true;},5000);
  }
  function displayError(error) {
    const message=error?.message || String(error);
    $("#global-error").textContent=message;$("#global-error").hidden=false;
    toast(message,"error");
  }
  function clearError() { $("#global-error").hidden=true; }
  async function api(path,options={}) {
    const init={...options};
    if (init.body && !(init.body instanceof FormData)) {init.headers={"Content-Type":"application/json",...init.headers};init.body=JSON.stringify(init.body);}
    const response=await fetch(path,init);
    const contentType=response.headers.get("content-type") || "";
    const data=contentType.includes("json") ? await response.json() : {message:await response.text()};
    if (!response.ok) {
      const details=data.issues?.map(issueText).join("\n");
      throw new Error([data.message || data.error || `请求失败 (${response.status})`,details].filter(Boolean).join("\n"));
    }
    return data;
  }
  async function busy(label,action) {
    if (state.busy) return;
    state.busy=true;clearError();$("#busy-text").textContent=label;$("#busy-overlay").hidden=false;updateActions();
    try { return await action(); } catch(error) { displayError(error);return null; }
    finally {state.busy=false;$("#busy-overlay").hidden=true;updateActions();}
  }
  function taskIds() {
    const prefix=state.spec.robot.prefix || "MyBot";
    return Object.entries(state.spec.tasks).flatMap(([key,item])=>{
      const ids=(item.terrains||[]).flatMap((terrain)=>{
        const base=`${prefix}${taskMeta[key]?.suffix || ""}-${terrain}`;
        if(key!=="him_lpacrl"||terrain!=="Rough")return [base];
        return [base,...(template(key)?.presets||[]).filter((preset)=>preset.suffix).map((preset)=>`${base}-${preset.suffix}`)];
      });
      return ids;
    });
  }
  function localIssues(step=5) {
    const issues=[];
    if (!state.spec.model?.id || !state.inspection) issues.push("请先导入并解析机器人模型");
    if (step>=1 && (!Object.keys(state.spec.tasks).length||Object.values(state.spec.tasks).some((task)=>!task.terrains?.length))) issues.push("至少选择一个算法，且每个所选算法至少选择一种地形");
    if (step>=2) {
      if (!/^[A-Za-z][A-Za-z0-9_-]*$/.test(state.spec.robot.prefix||"")) issues.push("任务前缀须以英文字母开头，仅包含字母、数字、下划线和短横线");
      if (!/^[a-z][a-z0-9_]*$/.test(state.spec.robot.slug||"")) issues.push("目录标识须以小写字母开头，仅包含小写字母、数字和下划线");
      if (!state.spec.bindings.base_body) issues.push("请确认机体 base_body");
      const joints=state.spec.bindings.joints||[];
      if (joints.length!==12 || joints.some((joint)=>!joint.name) || new Set(joints.map((joint)=>joint.name)).size!==12) issues.push("12 个动作必须绑定 12 个不同的模型关节");
      if (state.inspection?.joints?.length!==12) issues.push("第一版需要恰好 12 个驱动关节的四足模型");
      if (state.spec.bindings.feet.length!==4 || state.spec.bindings.feet.some((foot)=>!foot.body||!foot.geom)) issues.push("四个足端分别需要明确的 body 和接触 geom 绑定");
      if (state.spec.bindings.feet.length===4&&new Set(state.spec.bindings.feet.map((foot)=>foot.geom)).size!==4) issues.push("四个足端需要使用不同的接触 geom");
      if (!state.spec.bindings.base_geoms?.length) issues.push("请选择至少一个机体碰撞 geom");
    }
    if (step>=3) {
      if (state.invalidFields.size) issues.push(`还有 ${state.invalidFields.size} 项参数格式需要修正`);
      const parameters=effective("common");
      for (const joint of state.spec.bindings.joints) {
        const value=get(parameters,"init_state.default_joint")?.[joint.name];
        const limits=state.inspection?.joints.find((item)=>item.name===joint.name)?.range;
        if (typeof value === "number" && limits?.length===2 && (value<limits[0]-1e-8 || value>limits[1]+1e-8)) issues.push(`${joint.name} 初始角度 ${value} 超出模型限位 [${limits.join(", ")}]`);
        const clip=get(parameters,"control.action_clip")?.[joint.name];
        if(clip&&(!Array.isArray(clip)||clip.length!==2||!clip.every(Number.isFinite)||clip[0]>=clip[1]))issues.push(`${joint.name} 动作裁剪需要两个有限数值，且下限小于上限`);
      }
    }
    return issues;
  }
  function updateSummary() {
    $("#summary-robot").textContent=state.spec.robot.prefix || "新机器人";
    $("#summary-model").textContent=state.spec.model.xml?.split("/").at(-1) || "等待导入模型";
    $("#summary-joints").innerHTML=`${state.inspection?.joints?.length ?? "—"} <small>/ 12</small>`;
    const ids=state.plan?.task_ids || taskIds();
    $("#summary-count").textContent=Object.keys(state.spec.tasks).length ? ids.length : 0;
    $("#summary-tasks").innerHTML=Object.keys(state.spec.tasks).length ? ids.map((id)=>`<span class="task-chip">${esc(id)}</span>`).join("") : '<span class="muted">选择算法后显示任务名称</span>';
    const checks=[
      {done:!!state.inspection,text:state.inspection?"模型与资源已解析":"导入模型与全部资源"},
      {done:localIssues(2).length===0,text:localIssues(2).length===0?"机器人与动作映射已填写":"确认关节、足端及机器人标识"},
      {done:state.spec.confirmed && !!state.plan,text:state.spec.confirmed?"文件与配置已确认":"检查站姿并审阅生成文件"},
    ];
    $("#pending-count").textContent=checks.filter((check)=>!check.done).length;
    $("#pending-items").innerHTML=checks.map((check)=>`<li class="${check.done?"done":""}">${esc(check.text)}</li>`).join("");
    $$(".step-link").forEach((node,index)=>{
      let complete=false;
      if(index===0)complete=!!state.inspection;
      if(index===1)complete=!!state.inspection&&Object.keys(state.spec.tasks).length>0;
      if(index===2)complete=localIssues(2).length===0;
      if(index===3)complete=state.previewOpen&&localIssues(3).length===0;
      if(index===4)complete=!!state.plan&&state.spec.confirmed;
      if(index===5)complete=state.generated;
      node.classList.toggle("complete",complete);$(".step-tick",node).textContent=complete?"✓":"";
    });
  }
  function updateActions() {
    const running=!!state.job && ["running","queued"].includes(state.job.status);
    ["import-button","save-draft","load-draft","refresh-plan","start-preview","reset-config","reset-joints","apply-batch"].forEach((id)=>{$(`#${id}`).disabled=state.busy;});
    $("#import-button").disabled=state.busy||running;$("#load-draft").disabled=state.busy||running;
    $("#generate-button").disabled=state.busy||running||!state.plan||!state.spec.confirmed||state.generatedRevision===state.revision;
    ["validate-cpu","validate-simulation"].forEach((id)=>{$(`#${id}`).disabled=state.busy||running||!state.generated;});
    $("#stop-job").hidden=!running;$("#stop-job").disabled=state.busy;
    $("#previous-step").disabled=state.step===0||state.busy;$("#next-step").hidden=state.step===5;$("#next-step").disabled=state.busy;
    $("#start-preview").disabled=state.busy||!state.inspection;
  }
  function showStep(index) {
    state.step=Math.max(0,Math.min(5,index));clearError();
    $$(".step-panel").forEach((node,i)=>{node.hidden=i!==state.step;node.classList.toggle("active",i===state.step);});
    $$(".step-link").forEach((node,i)=>{node.classList.toggle("active",i===state.step);node.setAttribute("aria-current",i===state.step?"step":"false");});
    $("#step-progress").textContent=`第 ${state.step+1} 步 / 共 6 步`;
    const labels=["选择训练任务 →","确认机器人映射 →","配置与姿态预览 →","审阅生成文件 →","生成与验证 →"];
    $("#next-step").textContent=labels[state.step] || "";
    if(state.step===3) {renderConfigTabs();renderParameters();renderJointControls();}
    updateActions();
    if(state.step===4 && state.inspection && !state.plan && localIssues(3).length===0) buildPlan();
    $("#main-content").scrollIntoView({behavior:"smooth",block:"start"});
  }
  function selectOptions(items,value,placeholder="请选择",extra="") {
    const names=items.map((item)=>typeof item==="string"?item:item.name).filter(Boolean);
    const unique=[...new Set(names)];
    return `<option value="">${esc(placeholder)}</option>`+unique.map((name)=>`<option value="${esc(name)}" ${name===value?"selected":""}>${esc(name)}${extra}</option>`).join("");
  }
  function renderImport() {
    if(!state.inspection) {$("#import-result").hidden=true;return;}
    const inspection=state.inspection;
    $("#import-result").hidden=false;
    $("#import-result").innerHTML=`<div class="card"><div class="card-top"><div><h3>模型已解析</h3><p>继续确认动作映射与新任务的机器人标识。</p></div><span class="tag">${inspection.joints?.length===12?"12 关节已识别":"需要检查关节数"}</span></div><div class="import-stats">${[[inspection.joints?.length,"驱动关节"],[inspection.bodies?.length,"机体 body"],[inspection.geoms?.length,"几何 geom"],[inspection.sites?.length,"定位 site"]].map(([value,label])=>`<div class="import-stat"><strong>${value??0}</strong><span>${label}</span></div>`).join("")}</div><div class="import-path">${esc(state.spec.model.root)} / ${esc(state.spec.model.xml)}</div>${inspection.files?.length?`<details class="subsection"><summary>模型与资源依赖 <span>${inspection.files.length} files</span></summary><ul class="dependency-list">${inspection.files.map((file)=>`<li>${esc(typeof file==="string"?file:file.path||file.name||JSON.stringify(file))}</li>`).join("")}</ul></details>`:""}</div>${renderIssues(inspection.issues||[])}`;
  }
  function defaultTask(key) { return {terrains:["Rough"],parameters:{}}; }
  function presetMarkup(item) {
    const groups=new Map();
    for(const preset of item.presets||[]){if(!groups.has(preset.group))groups.set(preset.group,[]);groups.get(preset.group).push(preset);}
    if(!groups.size)return '<div class="muted tiny">默认课程、Compact 与历史对照将完整生成。</div>';
    return Array.from(groups.entries()).map(([group,presets])=>`<details class="preset-group" ${group==="默认"?"open":""}><summary>${esc(group)}<span>${presets.length} 个预设</span></summary><div>${presets.map((preset)=>`<span class="preset-item">${esc(preset.label)}</span>`).join("")}</div></details>`).join("");
  }
  function renderTasks() {
    const templates=state.templates.length?state.templates:Object.entries(taskMeta).map(([key,meta])=>({key,...meta}));
    $("#task-cards").innerHTML=templates.map((item)=>{
      const key=item.key||item.id;const meta=taskMeta[key]||{};const selected=!!state.spec.tasks[key];
      const source=item.source || `src/tasks/locomotion/ri_4438${key==="ppo"?"":`_${key}`}`;
      return `<article class="task-card ${selected?"selected":""}" data-task="${esc(key)}"><label class="task-card-label"><input type="checkbox" data-toggle-task="${esc(key)}" ${selected?"checked":""}><div><h3>${esc(item.label||meta.label||key)}</h3><div class="task-en">${esc(meta.sub||"")}</div></div></label><p>${esc(meta.description||item.description||"")}</p><div class="terrain-options">${["Flat","Rough"].map((terrain)=>`<label><input type="checkbox" data-terrain="${terrain}" data-task="${key}" ${state.spec.tasks[key]?.terrains.includes(terrain)?"checked":""} ${selected?"":"disabled"}>${terrain==="Flat"?"Flat · 平地":"Rough · 复杂地形"}</label>`).join("")}</div>${key==="him_lpacrl"?`<div class="curriculum-choice"><label>Rough 包含全部课程预设</label>${presetMarkup(item)}</div>`:""}<div class="template-source">${esc(source)}</div></article>`;
    }).join("");
    $("#contract-info").innerHTML='<span class="hint-icon" aria-hidden="true">i</span><div><strong>HIM 固定契约 · phase47_v1</strong><p>actor / critic、estimator、历史顺序和 ONNX 接口按现有实现保留。镜像模板会额外校验配对与几何。</p><div class="contract-grid"><span><strong>47</strong>单帧 actor 观测</span><span><strong>6 / 282</strong>历史帧 / 历史维度</span><span><strong>12</strong>动作维度</span></div></div>';
  }
  function ensureBindings() {
    const bindings=state.spec.bindings;
    const sourceJoints=bindings.joints||[];
    bindings.joints=legs.flatMap((leg)=>roles.map((role)=>{
      const known=sourceJoints.find((joint)=>joint.leg===leg&&joint.role===role);
      return known || {name:"",leg,role,sign:role==="hip"?-1:1};
    }));
    bindings.feet=legs.map((leg)=>bindings.feet?.find((foot)=>foot.leg===leg) || {leg,site:"",body:"",geom:"",position:[0,0,0]});
    bindings.base_geoms ||= [];bindings.leg_geoms ||= [];
  }
  function renderBindings() {
    ensureBindings();const inspection=state.inspection || {joints:[],bodies:[],sites:[],geoms:[]};
    $("#robot-prefix").value=state.spec.robot.prefix||"";$("#robot-slug").value=state.spec.robot.slug||"";
    $("#base-body").innerHTML=selectOptions(inspection.bodies||[],state.spec.bindings.base_body,"请选择机体 body");
    $("#joint-count").textContent=`${inspection.joints?.length||0} 个模型关节`;
    $("#joint-mapping").innerHTML=state.spec.bindings.joints.map((joint,index)=>{
      const modelJoint=inspection.joints?.find((j)=>j.name===joint.name);
      return `<tr><td>${String(index).padStart(2,"0")}</td><td><span class="joint-label">${joint.leg} · ${joint.role}<small>${legLabels[joint.leg]}</small></span></td><td><select data-joint-map="${index}" aria-label="${joint.leg} ${joint.role} 关节">${selectOptions(inspection.joints||[],joint.name,"选择关节")}</select></td><td><select data-joint-sign="${index}" aria-label="${joint.leg} ${joint.role} 镜像符号"><option value="1" ${joint.sign===1?"selected":""}>+1</option><option value="-1" ${joint.sign===-1?"selected":""}>−1</option></select></td><td class="joint-limits">${modelJoint?.range?.map((v)=>Number(v).toFixed(3)).join(" ~ ") || "—"}</td></tr>`;
    }).join("");
    $("#feet-mapping").innerHTML=state.spec.bindings.feet.map((foot,index)=>`<div class="foot-binding"><div class="foot-binding-heading"><span class="leg-tag">${foot.leg}</span>${legLabels[foot.leg]}</div><div class="foot-fields"><label class="field"><span>已有 site <em>可选</em></span><select data-foot="${index}" data-field="site">${selectOptions(inspection.sites||[],foot.site,"新增 site / 不使用已有")}</select></label><label class="field"><span>足端 body</span><select data-foot="${index}" data-field="body">${selectOptions(inspection.bodies||[],foot.body,"选择足端 body")}</select></label><label class="field"><span>接触 geom</span><select data-foot="${index}" data-field="geom">${selectOptions(inspection.geoms||[],foot.geom,"选择接触 geom")}</select></label><label class="field"><span>Site 局部位置 <em>x / y / z · m</em></span><span class="vector-inputs">${[0,1,2].map((axis)=>`<input type="number" step="0.001" value="${Number(foot.position?.[axis]||0)}" data-foot-position="${index}" data-axis="${axis}" aria-label="${foot.leg} site ${["x","y","z"][axis]}">`).join("")}</span></label></div></div>`).join("");
    ["base","leg"].forEach((kind)=>{
      $(`#${kind}-geoms`).innerHTML=(inspection.geoms||[]).map((geom)=>{const name=typeof geom==="string"?geom:geom.name;return name?`<option value="${esc(name)}" ${state.spec.bindings[`${kind}_geoms`].includes(name)?"selected":""}>${esc(name)}</option>`:"";}).join("");
    });
  }
  function jointValue(parameters,key,joint,fallback=0) {
    const value=get(parameters,`control.${key}`);
    if(typeof value==="number")return value;
    if(value===null||value===undefined){
      if(["armature","friction","passive_damping"].includes(key)){
        const inspected=state.inspection?.joints.find((item)=>item.name===joint.name);
        return inspected?.[key] ?? null;
      }
    }
    return value?.[joint.name] ?? value?.[joint.role] ?? fallback;
  }
  function jointRange(joint) {return state.inspection?.joints.find((item)=>item.name===joint.name)?.range || [-Math.PI,Math.PI];}
  function jointInitial(parameters,joint) {return get(parameters,"init_state.default_joint")?.[joint.name] ?? 0;}
  function setJointValue(key,joint,value) {
    const target=targetParameters();
    if(key==="initial") {
      target.init_state ||= {};target.init_state.default_joint ||= {};
      target.init_state.default_joint[joint.name]=value;
    } else if(["clip_min","clip_max"].includes(key)) {
      target.control ||= {};const original=get(effective(),"control.action_clip")||{};
      const range=Array.isArray(original?.[joint.name])?clone(original[joint.name]):[-10,10];range[key==="clip_min"?0:1]=value;
      if(typeof target.control.action_clip!=="object" || Array.isArray(target.control.action_clip))target.control.action_clip={};
      target.control.action_clip[joint.name]=range;
    } else {
      target.control ||= {};
      const current=target.control[key];
      if(typeof current!=="object"||Array.isArray(current)||!current) {
        const parameters=effective();target.control[key]=Object.fromEntries(state.spec.bindings.joints.filter((j)=>j.name).map((j)=>[j.name,jointValue(parameters,key,j,0)]).filter(([,v])=>v!==null));
      }
      target.control[key][joint.name]=value;
    }
  }
  function renderJointControls() {
    const parameters=effective();
    $("#joint-control-scope").textContent=`${state.scope==="common"?"公共配置":taskMeta[state.scope]?.label||state.scope} · PD、力矩限幅、动作与站姿在此页作用范围内生效。`;
    $("#joint-controls").innerHTML=state.spec.bindings.joints.map((joint,index)=>{
      if(!joint.name)return `<tr><td colspan="5" class="muted">${joint.leg} ${joint.role} 尚未绑定模型关节</td></tr>`;
      const range=jointRange(joint);const initial=jointInitial(parameters,joint);
      const clips=get(parameters,"control.action_clip");const clip=Array.isArray(clips?.[joint.name])?clips[joint.name]:[-10,10];
      return `<tr><td><div class="joint-label">${esc(joint.name)} <span class="source-badge model">模型读取</span></div><small class="muted tiny">初始角度 · ${state.inspection?.initial_source==="keyframe"?"模型 keyframe 起点":"模板推测起点，请检查"}</small><div class="pose-control"><input type="range" min="${range[0]}" max="${range[1]}" step="0.001" value="${initial}" data-pose-slider="${index}" aria-label="${esc(joint.name)} 初始角度滑块"><input type="number" step="0.001" value="${initial}" min="${range[0]}" max="${range[1]}" data-joint-control="initial" data-joint="${index}" aria-label="${esc(joint.name)} 初始角度"></div><details class="control-advanced"><summary>执行器与动作缩放</summary><div class="mini-control-fields">${[["armature","惯量 kg·m²",0.008234],["friction","摩擦 N·m",0.01],["passive_damping","被动阻尼",0],["action_scale","缩放 rad",0.5]].map(([key,label,fallback])=>`<label>${label} ${get(parameters,`control.${key}`)==null?"· 模型读取，编辑后覆盖":"· 配置值"}<input type="number" step="any" min="0" value="${jointValue(parameters,key,joint,fallback)??""}" placeholder="继承 XML" data-joint-control="${key}" data-joint="${index}" aria-label="${esc(joint.name)} ${label}"></label>`).join("")}</div></details></td>${[["stiffness",32.5065],["damping",2.0694],["effort_limit",10]].map(([key,fallback])=>`<td><input type="number" step="any" min="0" value="${jointValue(parameters,key,joint,fallback)??""}" placeholder="继承 XML" data-joint-control="${key}" data-joint="${index}" aria-label="${esc(joint.name)} ${key}"></td>`).join("")}<td><div class="clip-inputs"><input type="number" step="any" value="${clip[0]}" data-joint-control="clip_min" data-joint="${index}" aria-label="${esc(joint.name)} 动作下限"><input type="number" step="any" value="${clip[1]}" data-joint-control="clip_max" data-joint="${index}" aria-label="${esc(joint.name)} 动作上限"></div><small class="muted tiny">模型限位起点</small></td></tr>`;
    }).join("");
  }
  function renderConfigTabs() {
    if(state.scope!=="common"&&!state.spec.tasks[state.scope])state.scope="common";
    $("#config-tabs").innerHTML=[{key:"common",label:"公共配置"},...Object.keys(state.spec.tasks).map((key)=>({key,label:taskMeta[key]?.label||key}))].map((item)=>`<button role="tab" aria-selected="${item.key===state.scope}" data-config-scope="${item.key}" class="${item.key===state.scope?"active":""}">${esc(item.label)}</button>`).join("");
    $("#config-scope-note").textContent=state.scope==="common"?"公共修改将应用于全部所选任务。未修改的字段使用各任务原有默认值。3D 预览随当前配置页切换。":`${taskMeta[state.scope]?.label || state.scope} 的专属覆盖；未修改的字段继承公共配置与本模板默认值。`;
  }
  function fieldMetadata(path) {
    const fields=(state.scope==="common"?firstTemplate():template(state.scope))?.fields;
    const rawPath=displayPath(path);
    if(Array.isArray(fields))return fields.find((field)=>field.path===path)||fields.find((field)=>field.path===rawPath)||{};
    return fields?.[path]||fields?.[rawPath]||{};
  }
  function numericConstraints(key,path,metadata) {
    const result={};
    if(metadata.min!==undefined)result.min=metadata.min;
    if(metadata.max!==undefined)result.max=metadata.max;
    const integers=metadata.type==="integer"||/^(num_|max_iterations|save_interval|seed$|decimation$|delay_(min_lag|max_lag|update_period)$|sinkhorn_iters$)/.test(key);
    result.step=metadata.step??(integers?1:"any");
    if(integers&&!key.startsWith("delay_")&&key!=="seed")result.min ??= 1;
    if(["delay_min_lag","delay_max_lag","delay_update_period","init_noise_std","entropy_coef","max_init_terrain_level"].includes(key))result.min ??= 0;
    if(["delay_hold_prob","rel_standing_envs","rel_forward_envs","rel_heading_envs","gamma","lam","clip_param"].includes(key)){result.min ??= 0;result.max ??= 1;}
    if(["sim_dt","dt","episode_length_s","learning_rate","estimator_learning_rate","env_spacing","desired_kl","max_grad_norm","estimator_max_grad_norm"].includes(key))result.min ??= 0.000000001;
    return Object.entries(result).map(([name,value])=>`${name}="${value}"`).join(" ");
  }
  function renderField(path,value) {
    const key=pathSegments(path).at(-1);const metadata=fieldMetadata(path);
    const readonly=fixedKeys.has(key)||metadata.readonly;
    const label=metadata.label&&metadata.label!==key?metadata.label:fieldLabels[key]||key;
    const unit=metadata.unit||unitFor(path);const title=`${label}${unit?` / ${unit}`:""}`;
    const attr=`data-param="${esc(path)}" ${readonly?"disabled":""} aria-label="${esc(title)}"`;
    let input="",full=false;
    let options=metadata.options;
    if(!options && key==="activation")options=["elu","relu","selu","crelu","lrelu","tanh","sigmoid"];
    if(!options && key==="schedule")options=["adaptive","fixed"];
    if(options)input=`<select ${attr} data-type="${typeof value}">${options.map((option)=>{const v=typeof option==="object"?option.value:option;const l=typeof option==="object"?option.label:option;return `<option value="${esc(v)}" ${v===value?"selected":""}>${esc(l)}</option>`;}).join("")}</select>`;
    else if(typeof value==="boolean")input=`<select ${attr} data-type="boolean"><option value="true" ${value?"selected":""}>开启 · true</option><option value="false" ${!value?"selected":""}>关闭 · false</option></select>`;
    else if(typeof value==="number")input=`<input type="number" ${attr} data-type="number" ${numericConstraints(key,path,metadata)} value="${value}" required>`;
    else if(typeof value==="string")input=`<input type="text" ${attr} data-type="string" value="${esc(value)}">`;
    else if(Array.isArray(value)&&value.every((v)=>typeof v==="number"))input=`<input type="text" ${attr} data-type="number-array" value="${value.join(", ")}" placeholder="以逗号分隔，如 0, 0, 0.3">`;
    else if(Array.isArray(value)&&value.every((v)=>typeof v==="string"))input=`<input type="text" ${attr} data-type="string-array" value="${esc(value.join(", "))}" placeholder="以逗号分隔">`;
    else {full=true;input=`<textarea ${attr} data-type="json" class="json-input">${esc(JSON.stringify(value,null,2))}</textarea>`;}
    return `<label class="field ${full?"full":""}"><span><span class="field-label-content">${esc(label)}${unit?`<span class="field-unit">${esc(unit)}</span>`:""}<span class="parameter-key">${esc(displayPath(path))}</span></span>${readonly?'<em class="source-badge template">固定契约</em>':sourceBadge(path)}</span>${input}${metadata.description?`<small>${esc(metadata.description)}</small>`:""}</label>`;
  }
  function flattenFields(object,prefix="") {
    return Object.entries(object||{}).flatMap(([key,value])=>{
      const segment=encodeSegment(key);const path=prefix?`${prefix}.${segment}`:segment;
      if(jointPaths.has(path))return [];
      if(value && typeof value==="object"&&!Array.isArray(value)&&Object.keys(value).length)return flattenFields(value,path);
      return [{path,value}];
    });
  }
  function renderParameters() {
    const parameters=effective();const groupNames=[...Object.keys(groupMeta).filter((key)=>parameters[key]!==undefined),...Object.keys(parameters).filter((key)=>!groupMeta[key])];
    if(!groupNames.length) {$("#parameter-groups").innerHTML='<div class="card config-search-note">正在读取当前任务模板参数…</div>';return;}
    $("#parameter-groups").innerHTML=groupNames.map((group)=>{
      const meta=groupMeta[group]||{label:fieldLabels[group]||group,sub:group,open:false};
      const fields=flattenFields({[group]:parameters[group]});
      if(!fields.length)return "";
      const nestedGroups=new Map();fields.forEach((field)=>{const parts=pathSegments(field.path);const subgroup=parts.length>2?parts[1]:"";if(!nestedGroups.has(subgroup))nestedGroups.set(subgroup,[]);nestedGroups.get(subgroup).push(field);});
      const contents=Array.from(nestedGroups.entries()).map(([subgroup,items])=>{
        const content=`<div class="form-grid two">${items.map(({path,value})=>renderField(path,value)).join("")}</div>`;
        if(!subgroup||meta.open)return content;
        const sublabel={policy:"策略网络 Policy",algorithm:"PPO 与 Estimator",runner:"训练预算 Runner",noise:"观测噪声",base_offset:"机体偏移",joint_offset:"关节偏移",ranges:"指令范围"}[subgroup]||subgroup;
        return `<details class="subsection"><summary>${esc(sublabel)} <span>${items.length} 项参数</span></summary>${content}</details>`;
      }).join("");
      return meta.open?`<div class="card parameter-card"><div class="card-top"><div><h3>${esc(meta.label)}</h3><p>${esc(meta.sub)}</p></div></div>${contents}</div>`:`<details class="card parameter-card"><summary>${esc(meta.label)} <span>${fields.length} 项高级参数</span></summary><p class="scope-note">${esc(meta.sub)}</p>${contents}</details>`;
    }).join("");
  }
  function parseParameter(input) {
    const type=input.dataset.type;
    if(type==="number") {if(!input.value.trim()||!Number.isFinite(Number(input.value)))throw Error("请输入有效数值");if(!input.checkValidity())throw Error(input.validationMessage);return Number(input.value);}
    if(type==="boolean")return input.value==="true";
    if(type==="number-array") {
      const values=input.value.trim().replace(/^\[|\]$/g,"").split(/[,，]/).map((v)=>v.trim());
      if(!values.length||values.some((v)=>!v||!Number.isFinite(Number(v))))throw Error("请用逗号分隔有效数值");
      const numbers=values.map(Number);const path=input.dataset.param;
      if(path==="init_state.pos"&&numbers.length!==3)throw Error("机体位置需要 x、y、z 三个数值");
      if(path==="init_state.rot"&&(numbers.length!==4||Math.abs(numbers.reduce((sum,v)=>sum+v*v,0)-1)>1e-4))throw Error("初始姿态需要单位四元数，顺序为 w、x、y、z");
      if(path.endsWith("hidden_dims")&&numbers.some((v)=>!Number.isInteger(v)||v<=0))throw Error("网络隐藏层需要正整数维度");
      return numbers;
    }
    if(type==="string-array")return input.value.split(/[,，]/).map((v)=>v.trim()).filter(Boolean);
    if(type==="json")try{return JSON.parse(input.value);}catch(_error){throw Error("请输入有效 JSON 值");}
    return input.value;
  }
  function updateFieldError(input,message) {
    const field=input.closest(".field");const key=`${state.scope}:${input.dataset.param}`;
    field?.classList.toggle("invalid",!!message);
    if(message){state.invalidFields.add(key);input.setCustomValidity(message);}
    else {state.invalidFields.delete(key);input.setCustomValidity("");}
    let note=field?.querySelector(".validation-message");
    if(message&&!note){note=document.createElement("small");note.className="validation-message";field?.append(note);}
    if(note){note.textContent=message||"";note.hidden=!message;}
  }
  function issueText(issue) {return typeof issue==="string"?issue:issue.message||issue.detail||JSON.stringify(issue);}
  function renderIssues(issues) {
    if(!issues?.length)return "";
    return issues.map((issue)=>{const level=typeof issue==="object"?(issue.severity||issue.level||"warning"):"warning";return `<div class="notice ${["error","fatal"].includes(level)?"error":"warning"}">${esc(issueText(issue))}</div>`;}).join("");
  }
  function hasBlockingIssues(issues=[]) {return issues.some((issue)=>typeof issue==="object"&&["error","fatal"].includes(issue.severity||issue.level));}
  async function buildPlan() {
    const issues=localIssues(3);if(issues.length){displayError(new Error(issues.join("\n")));return;}
    return busy("正在构建任务文件预览…",async()=>{
      const revision=state.revision;const result=await api("/api/plan",{method:"POST",body:{spec:state.spec}});
      if(revision!==state.revision)throw Error("配置已在预览期间修改，请重新生成预览");
      state.plan=result;state.planRevision=revision;
      $("#plan-summary").textContent=`${result.files.length} 个文件 · ${result.task_ids?.length||0} 个注册任务 · ${formatBytes(result.files.reduce((sum,file)=>sum+(file.size||0),0))}`;
      $("#plan-issues").innerHTML=renderIssues(result.issues);
      $("#file-review").hidden=false;$("#diff-section").hidden=false;
      const fileGroups=new Map();result.files.forEach((file,index)=>{const path=file.path.split("/");const group=path.slice(0,Math.min(4,path.length-1)).join("/");if(!fileGroups.has(group))fileGroups.set(group,[]);fileGroups.get(group).push({file,index});});
      $("#file-tree").innerHTML=Array.from(fileGroups.entries()).map(([group,files])=>`<div class="file-folder">${esc(group)}/</div>${files.map(({file,index})=>`<button class="file-button" data-file-index="${index}" title="${esc(file.path)}">${esc(file.path.slice(group.length+1))}<span class="file-kind">${formatBytes(file.size||0)}</span></button>`).join("")}`).join("");
      $("#config-diff").textContent=JSON.stringify({parameters:state.spec.parameters,tasks:state.spec.tasks,template_sources:result.manifest?.templates || result.manifest?.template_sources || result.manifest},null,2);
      if(result.files.length)selectFile(0);
      if(hasBlockingIssues(result.issues)){state.spec.confirmed=false;$("#confirm-review").checked=false;}
      updateSummary();toast("文件预览已就绪，可以逐个审阅","success");
    });
  }
  function formatBytes(bytes) {return bytes<1024?`${bytes} B`:bytes<1024*1024?`${(bytes/1024).toFixed(1)} KB`:`${(bytes/1024/1024).toFixed(1)} MB`;}
  function selectFile(index) {
    const file=state.plan?.files[index];if(!file)return;state.currentFile=index;
    $("#current-file").textContent=file.path;
    $("#file-content").textContent=file.kind==="binary"||file.content===null||file.content===undefined?`二进制模型资源 · ${formatBytes(file.size||0)}\n\n生成时保留原始内容与目录结构。`:file.content;
    $$("[data-file-index]").forEach((node)=>node.classList.toggle("active",Number(node.dataset.fileIndex)===index));
    $("#copy-file").disabled=file.kind==="binary"||file.content===null||file.content===undefined;
  }
  async function openPreview() {
    const issues=localIssues(2);if(issues.length){displayError(new Error(issues.join("\n")));return;}
    return busy("正在启动本地 3D 预览…",async()=>{
      const previewSpec=clone(state.spec);previewSpec.parameters=effective();
      previewSpec.parameters.init_state.default_joint=Object.fromEntries(state.spec.bindings.joints.filter((j)=>j.name).map((j)=>[j.name,jointInitial(previewSpec.parameters,j)]));
      const result=await api("/api/preview",{method:"POST",body:{spec:previewSpec}});
      if(!result.url)throw Error("预览服务未返回可用地址");
      const url=new URL(result.url,window.location.origin);
      if(!["http:","https:"].includes(url.protocol))throw Error("预览服务返回了无效地址");
      $("#viewer-frame").src=url.href;$("#viewer-frame").hidden=false;$(".viewer-empty").hidden=true;
      state.previewOpen=true;$("#start-preview").textContent="重新加载预览";showPoseInfo(result.pose);updateSummary();
    });
  }
  function schedulePose() {clearTimeout(state.poseTimer);state.poseTimer=setTimeout(sendPose,140);}
  async function sendPose() {
    if(!state.previewOpen)return;
    if(state.poseInFlight){state.posePending=true;return;}
    state.poseInFlight=true;
    try {
      const parameters=effective();const joint_positions=Object.fromEntries(state.spec.bindings.joints.filter((joint)=>joint.name).map((joint)=>[joint.name,jointInitial(parameters,joint)]));
      const result=await api("/api/preview/pose",{method:"POST",body:{joint_positions,pos:parameters.init_state?.pos||[0,0,0.3],rot:parameters.init_state?.rot||[1,0,0,0]}});
      showPoseInfo(result.pose||result);
    }catch(error){$("#preview-info").textContent=`姿态更新失败：${error.message}`;}
    finally{state.poseInFlight=false;if(state.posePending){state.posePending=false;schedulePose();}}
  }
  function showPoseInfo(pose) {
    if(!pose?.feet){$("#preview-info").textContent="姿态已更新。请旋转检查足端、站姿和模型方向。";return;}
    const feet=Array.isArray(pose.feet)?pose.feet:Object.entries(pose.feet).map(([leg,position])=>({leg,position}));
    const display=feet.map((foot)=>{const position=foot.position||foot.pos||foot.world_position;return `${foot.leg||foot.name||"足端"} [${Array.isArray(position)?position.map((v)=>Number(v).toFixed(3)).join(", "):"—"}]`;});
    $("#preview-info").textContent=`足端世界坐标 / m：${display.join(" · ")}`;
  }
  async function runGenerate() {
    const issues=localIssues(3);
    if(issues.length){displayError(new Error(issues.join("\n")));return;}
    if(!state.plan||state.planRevision!==state.revision||!state.spec.confirmed){displayError(new Error("请先审阅当前配置的文件预览，并勾选确认"));return;}
    if(hasBlockingIssues(state.plan.issues)){displayError(new Error("文件预览中仍有阻止生成的问题，请先修正"));return;}
    await busy("正在提交生成任务…",async()=>{
      const result=await api("/api/generate",{method:"POST",body:{spec:state.spec}});
      $("#generate-title").textContent="正在生成并检查任务";$("#generate-description").textContent="文件暂存、检查与写入过程记录在下方日志中。";
      $("#validation-card").hidden=false;
      if(result.job_id){state.job={id:result.job_id,status:result.status||"queued",mode:"generate"};state.generationRevision=state.revision;pollJob();}
      else {completeGeneration(result);}
    });
  }
  function completeGeneration(result) {
    state.generated=true;state.generatedRevision=state.generationRevision??state.revision;
    state.generatedTaskIds=result.task_ids||result.manifest?.task_ids||state.plan?.task_ids||taskIds();
    $("#generate-title").textContent="新任务已生成";$("#generate-description").textContent="独立任务文件已写入仓库，生成清单与验证记录可追溯。";
    $("#generate-button").textContent="任务已生成 ✓";
    const paths=result.paths||result.directories||result.output_dirs||[];
    $("#generate-result").innerHTML=`<div class="notice success">已生成 ${state.generatedTaskIds.length} 个注册任务。${result.message?`\n${esc(result.message)}`:""}</div>${paths.length?`<ul class="result-paths">${paths.map((path)=>`<li>${esc(path)}</li>`).join("")}</ul>`:""}`;
    $("#validation-card").hidden=false;$("#commands-card").hidden=false;
    $("#validate-task").innerHTML=state.generatedTaskIds.map((id)=>`<option value="${esc(id)}">${esc(id)}</option>`).join("");
    renderCommands();updateSummary();updateActions();toast("任务已生成，CPU 检查结果见验证日志","success");
  }
  function renderCommands() {
    const ids=state.generatedTaskIds;
    $("#task-commands").innerHTML=ids.map((id,index)=>{
      const train=`uv run --no-sync python -m scripts.train ${id} --env.scene.num-envs 1024 --agent.seed 42`;
      const play=`uv run --no-sync python -m scripts.play ${id} --agent zero --num-envs 1 --viewer viser`;
      return `<details ${index===0?"open":""} class="subsection"><summary>${esc(id)}</summary><div class="command-item"><div class="command-label"><span>训练</span><button class="button quiet small" data-copy-command="${esc(train)}">复制</button></div><pre class="command-code">${esc(train)}</pre></div><div class="command-item"><div class="command-label"><span>零动作回放 · 检查环境</span><button class="button quiet small" data-copy-command="${esc(play)}">复制</button></div><pre class="command-code">${esc(play)}</pre></div></details>`;
    }).join("");
  }
  async function runValidation(mode) {
    const task_id=$("#validate-task").value;if(!task_id)return;
    await busy("正在提交验证任务…",async()=>{
      const result=await api("/api/validate",{method:"POST",body:{task_id,mode}});
      state.job={id:result.job_id||result.id,status:result.status||"queued",mode};$("#job-log").textContent="验证已排队…";pollJob();
    });
  }
  async function pollJob() {
    clearTimeout(state.jobTimer);if(!state.job?.id)return;
    try {
      const job=await api(`/api/jobs/${encodeURIComponent(state.job.id)}`);
      const previousMode=state.job.mode;state.job={...state.job,...job};
      const labels={queued:"已排队",running:"运行中",passed:"通过",failed:"失败",stopped:"已停止",skipped:"未运行"};
      $("#job-status").textContent=`${previousMode==="generate"?"生成与 CPU 检查":previousMode==="simulation"?"GPU 短仿真":"CPU 配置检查"} · ${labels[job.status]||job.status}${job.error?` · ${job.error}`:""}`;
      if(job.log) {const log=$("#job-log");const bottom=log.scrollTop+log.clientHeight>=log.scrollHeight-40;log.textContent=job.log;if(bottom)log.scrollTop=log.scrollHeight;}
      if(["queued","running"].includes(job.status)){state.jobTimer=setTimeout(pollJob,1200);}
      else if(previousMode==="generate"&&job.status==="passed")completeGeneration(job.result||job);
      else if(job.status==="failed"){
        if(previousMode==="generate"){$("#generate-title").textContent="生成未完成";$("#generate-description").textContent="请查看日志定位问题，修正后可重新预览并生成。";}
        toast(job.error||"任务未通过，请查看验证日志","error");
      }
      else if(job.status==="skipped")toast(job.error||job.result?.message||"本次验证未运行，请查看日志");
    }catch(error){$("#job-status").textContent=`获取任务状态失败：${error.message}`;state.jobTimer=setTimeout(pollJob,3000);}
    updateActions();
  }
  async function saveDraft() {
    if(!state.inspection){toast("请先导入模型，再保存迁移草稿");return;}
    if(state.invalidFields.size){displayError(new Error("请先修正参数格式，再保存当前草稿"));return;}
    await busy("正在保存草稿…",async()=>{
      const result=await api("/api/drafts",{method:"POST",body:{spec:state.spec,name:state.spec.robot.prefix||"未命名机器人",...(state.draftId?{id:state.draftId}:{})}});
      state.draftId=result.id;toast("草稿已保存在本项目中","success");
    });
  }
  async function listDrafts() {
    await busy("正在读取草稿…",async()=>{
      const result=await api("/api/drafts");const drafts=result.drafts||[];
      $("#draft-list").innerHTML=drafts.length?drafts.map((draft)=>`<div class="draft-row"><div><strong>${esc(draft.name||draft.id)}</strong><small>${esc(draft.updated_at||draft.updated||"")}</small></div><button class="button secondary small" data-load-draft="${esc(draft.id)}">恢复 →</button></div>`).join(""):'<div class="empty-list">还没有保存的草稿。<br>导入模型后，可以随时保存当前配置。</div>';
      $("#draft-dialog").showModal();
    });
  }
  async function restoreDraft(id) {
    $("#draft-dialog").close();
    await busy("正在恢复模型与草稿…",async()=>{
      const draft=await api(`/api/drafts/${encodeURIComponent(id)}`);
      const spec=draft.spec;
      // A fresh inspection also restores the server's active model after restart.
      const imported=await api("/api/import",{method:"POST",body:{root:spec.model.root,xml:spec.model.xml}});
      const importedId=imported.model?.id||imported.spec?.model?.id;
      state.spec=clone(spec);state.spec.model={...state.spec.model,...imported.model,id:importedId||state.spec.model.id};
      state.inspection=imported.inspection;state.importDefaults=clone(imported.spec||state.spec);state.draftId=id;
      state.spec.version ||= 1;state.spec.parameters ||= {};state.spec.tasks ||= {};state.spec.confirmed=false;
      resetGeneration();state.previewOpen=false;state.invalidFields.clear();resetPreview();changed();renderAll();showStep(2);toast("已恢复草稿，请重新审阅生成结果","success");
    });
  }
  function resetPreview() {$("#viewer-frame").hidden=true;$("#viewer-frame").src="about:blank";$(".viewer-empty").hidden=false;$("#start-preview").textContent="打开 3D 预览";}
  function resetGeneration() {
    state.generated=false;state.generatedRevision=-1;state.generatedTaskIds=[];state.generationRevision=undefined;
    clearTimeout(state.jobTimer);state.job=null;
    $("#generate-title").textContent="准备生成独立训练任务";$("#generate-description").textContent="目标目录已存在时将拒绝覆盖。文件在检查通过后写入仓库。";
    $("#generate-button").innerHTML='生成任务并检查 <span aria-hidden="true">→</span>';
    $("#generate-result").replaceChildren();$("#validation-card").hidden=true;$("#commands-card").hidden=true;
  }
  async function importModel(event) {
    event.preventDefault();
    if(state.job&&["queued","running"].includes(state.job.status)){toast("请先等待当前验证完成，或停止验证");return;}
    await busy("正在解析机器人模型与资源…",async()=>{
      let body;
      if(state.importMode==="zip") {
        const file=$("#model-zip").files[0];if(!file)throw Error("请选择包含 XML 和资源的 ZIP 文件");
        body=new FormData();body.append("file",file);if($("#zip-xml").value.trim())body.append("xml",$("#zip-xml").value.trim());
      }else{
        const root=$("#model-root").value.trim();if(!root)throw Error("请填写模型目录");
        body={root,xml:$("#model-xml").value.trim()};
      }
      const result=await api("/api/import",{method:"POST",body});
      if(!result.inspection||!result.spec)throw Error("模型解析未返回完整的配置与检查结果");
      state.spec=result.spec;state.spec.model={...state.spec.model,...result.model};state.spec.parameters ||= {};state.spec.tasks ||= {him:defaultTask("him")};
      if(!Object.keys(state.spec.tasks).length)state.spec.tasks={him:defaultTask("him")};
      state.spec.robot ||= {slug:"",prefix:""};state.spec.bindings ||= {joints:[],feet:[]};
      state.inspection=result.inspection;state.importDefaults=clone(state.spec);state.draftId=null;state.previewOpen=false;state.invalidFields.clear();resetGeneration();resetPreview();changed();renderAll();
      $("#import-result").scrollIntoView({behavior:"smooth",block:"nearest"});
      toast("模型已解析，请继续选择训练任务","success");
    });
  }
  async function copyText(text) {
    try {await navigator.clipboard.writeText(text);toast("已复制到剪贴板","success");}
    catch(_error){const area=document.createElement("textarea");area.value=text;area.style.position="fixed";area.style.opacity="0";document.body.append(area);area.select();const ok=document.execCommand("copy");area.remove();toast(ok?"已复制到剪贴板":"浏览器暂不允许复制，请手动选择文本",ok?"success":"");}
  }
  function renderAll() {renderImport();renderTasks();renderBindings();renderConfigTabs();renderParameters();renderJointControls();updateSummary();updateActions();}

  // Event delegation keeps dynamically generated forms tied to MigrationSpec.
  $("#step-nav").addEventListener("click",(event)=>{const button=event.target.closest("[data-step]");if(button&&!state.busy)showStep(Number(button.dataset.step));});
  $("#previous-step").addEventListener("click",()=>showStep(state.step-1));
  $("#next-step").addEventListener("click",()=>{const issues=localIssues(Math.min(state.step,3));if(issues.length){displayError(new Error(issues.join("\n")));return;}if(state.step===4&&(!state.plan||!state.spec.confirmed)){displayError(new Error("请审阅文件并勾选确认后继续"));return;}showStep(state.step+1);});
  $("#import-form").addEventListener("submit",importModel);
  function switchImport(mode) {state.importMode=mode;$("#path-fields").hidden=mode!=="path";$("#zip-fields").hidden=mode!=="zip";$("#model-root").required=mode==="path";["path","zip"].forEach((key)=>{$(`#${key}-tab`).classList.toggle("active",key===mode);$(`#${key}-tab`).setAttribute("aria-selected",key===mode);});}
  $("#path-tab").addEventListener("click",()=>switchImport("path"));$("#zip-tab").addEventListener("click",()=>switchImport("zip"));
  $("#model-zip").addEventListener("change",()=>{$("#zip-name").textContent=$("#model-zip").files[0]?.name||"保留 XML 与 assets 的相对目录结构";});
  ["dragenter","dragover"].forEach((type)=>$("#upload-zone").addEventListener(type,(event)=>{event.preventDefault();$("#upload-zone").classList.add("dragover");}));
  ["dragleave","drop"].forEach((type)=>$("#upload-zone").addEventListener(type,(event)=>{event.preventDefault();$("#upload-zone").classList.remove("dragover");if(type==="drop"&&event.dataTransfer.files.length){const file=event.dataTransfer.files[0];if(!file.name.toLowerCase().endsWith(".zip")){toast("请选择 .zip 模型资源包","error");return;}$("#model-zip").files=event.dataTransfer.files;$("#zip-name").textContent=file.name;}}));
  $("#task-cards").addEventListener("change",(event)=>{
    const input=event.target;
    if(input.dataset.toggleTask){const key=input.dataset.toggleTask;if(input.checked)state.spec.tasks[key]=defaultTask(key);else delete state.spec.tasks[key];changed();renderTasks();renderConfigTabs();renderParameters();renderJointControls();}
    if(input.dataset.terrain){const task=state.spec.tasks[input.dataset.task];if(!task)return;task.terrains=["Flat","Rough"].filter((terrain)=>terrain===input.dataset.terrain?input.checked:task.terrains.includes(terrain));changed();}
  });
  ["prefix","slug"].forEach((key)=>$(`#robot-${key}`).addEventListener("input",(event)=>{state.spec.robot[key]=event.target.value.trim();changed();}));
  $("#base-body").addEventListener("change",(event)=>{state.spec.bindings.base_body=event.target.value;changed();});
  $("#joint-mapping").addEventListener("change",(event)=>{
    const input=event.target;
    if(input.dataset.jointMap!==undefined){const joint=state.spec.bindings.joints[Number(input.dataset.jointMap)];joint.name=input.value;changed();renderBindings();renderJointControls();}
    if(input.dataset.jointSign!==undefined){state.spec.bindings.joints[Number(input.dataset.jointSign)].sign=Number(input.value);changed();}
  });
  $("#feet-mapping").addEventListener("change",(event)=>{
    const input=event.target;
    if(input.dataset.foot!==undefined){
      const foot=state.spec.bindings.feet[Number(input.dataset.foot)];foot[input.dataset.field]=input.value;
      if(input.dataset.field==="site"&&input.value){const site=state.inspection?.site_details?.find((s)=>s.name===input.value);if(site){foot.body=site.body;foot.position=clone(site.position);}}
      changed();renderBindings();
    }
    if(input.dataset.footPosition!==undefined){
      const key=`foot:${input.dataset.footPosition}:${input.dataset.axis}`;
      if(!input.value||!input.checkValidity()){state.invalidFields.add(key);input.reportValidity();return;}
      state.invalidFields.delete(key);const foot=state.spec.bindings.feet[Number(input.dataset.footPosition)];foot.position ||= [0,0,0];foot.position[Number(input.dataset.axis)]=Number(input.value);changed();
    }
  });
  ["base","leg"].forEach((kind)=>$(`#${kind}-geoms`).addEventListener("change",(event)=>{state.spec.bindings[`${kind}_geoms`]=Array.from(event.target.selectedOptions).map((option)=>option.value);changed();}));
  $("#config-tabs").addEventListener("click",(event)=>{const button=event.target.closest("[data-config-scope]");if(button){state.scope=button.dataset.configScope;renderConfigTabs();renderParameters();renderJointControls();if(state.previewOpen)schedulePose();}});
  $("#parameter-groups").addEventListener("change",(event)=>{
    const input=event.target;if(!input.dataset.param)return;
    try{input.setCustomValidity("");const value=parseParameter(input);updateFieldError(input,null);set(targetParameters(),input.dataset.param,value);changed({pose:input.dataset.param.startsWith("init_state")});const badge=input.closest(".field").querySelector(".source-badge");if(badge){badge.textContent="人工修改";badge.className="source-badge manual";}if(input.dataset.param.startsWith("control"))renderJointControls();}
    catch(error){updateFieldError(input,error.message);toast(error.message,"error");updateSummary();}
  });
  function applyJointInput(input,fromSlider=false) {
    const index=Number(fromSlider?input.dataset.poseSlider:input.dataset.joint);const key=fromSlider?"initial":input.dataset.jointControl;const joint=state.spec.bindings.joints[index];
    if(input.value===""||!Number.isFinite(Number(input.value))||!input.checkValidity()){input.reportValidity();state.invalidFields.add(`joint:${state.scope}:${index}:${key}`);updateActions();return;}
    state.invalidFields.delete(`joint:${state.scope}:${index}:${key}`);setJointValue(key,joint,Number(input.value));
    if(key==="initial"){
      const peer=fromSlider?$(`[data-joint-control="initial"][data-joint="${index}"]`):$(`[data-pose-slider="${index}"]`);if(peer)peer.value=input.value;
    }
    changed({pose:key==="initial"});
  }
  $("#joint-controls").addEventListener("change",(event)=>{if(event.target.dataset.jointControl)applyJointInput(event.target);});
  $("#joint-controls").addEventListener("input",(event)=>{if(event.target.dataset.poseSlider!==undefined)applyJointInput(event.target,true);});
  $("#apply-batch").addEventListener("click",()=>{
    const input=$("#batch-value");if(input.value===""||!Number.isFinite(Number(input.value))){toast("请填写批量设置的数值","error");return;}
    const key=$("#batch-property").value;const role=$("#batch-role").value;const value=Number(input.value);
    if(key!=="initial"&&value<0){toast("该控制参数需要非负数值","error");return;}
    const matching=state.spec.bindings.joints.filter((joint)=>joint.role===role&&joint.name);
    if(!matching.length){toast("请先完成关节映射");return;}
    if(key==="initial"&&matching.some((joint)=>{const [min,max]=jointRange(joint);return value<min||value>max;})){toast("该角度超出至少一个同类关节的模型限位","error");return;}
    matching.forEach((joint)=>setJointValue(key,joint,value));changed({pose:key==="initial"});renderJointControls();toast(`已更新 ${matching.length} 个 ${role} 关节`,"success");
  });
  $("#reset-joints").addEventListener("click",()=>{
    jointPaths.forEach((path)=>{remove(targetParameters(),path);const imported=state.scope==="common"?get(state.importDefaults?.parameters,path):undefined;if(imported!==undefined)set(targetParameters(),path,clone(imported));});
    state.invalidFields.forEach((key)=>{if(key.startsWith(`joint:${state.scope}:`))state.invalidFields.delete(key);});changed({pose:true});renderJointControls();renderParameters();toast("关节配置已恢复默认值");
  });
  $("#reset-config").addEventListener("click",()=>{
    if(state.scope==="common"){
      state.spec.parameters=clone(state.importDefaults?.parameters||{});
    }else state.spec.tasks[state.scope].parameters={};
    state.invalidFields.forEach((key)=>{if(key.startsWith(`${state.scope}:`)||key.startsWith(`joint:${state.scope}:`))state.invalidFields.delete(key);});changed({pose:true});renderParameters();renderJointControls();toast("当前配置页已恢复默认");
  });
  $("#start-preview").addEventListener("click",openPreview);$("#refresh-plan").addEventListener("click",buildPlan);
  $("#file-tree").addEventListener("click",(event)=>{const node=event.target.closest("[data-file-index]");if(node)selectFile(Number(node.dataset.fileIndex));});
  $("#copy-file").addEventListener("click",()=>copyText(state.plan?.files[state.currentFile]?.content||""));
  $("#confirm-review").addEventListener("change",(event)=>{
    if(event.target.checked&&(!state.plan||state.planRevision!==state.revision||hasBlockingIssues(state.plan.issues))){event.target.checked=false;toast("请先生成无阻塞问题的文件预览","error");return;}
    state.spec.confirmed=event.target.checked;updateSummary();updateActions();
  });
  $("#generate-button").addEventListener("click",runGenerate);$("#validate-cpu").addEventListener("click",()=>runValidation("cpu"));$("#validate-simulation").addEventListener("click",()=>runValidation("simulation"));
  $("#stop-job").addEventListener("click",()=>busy("正在停止验证…",async()=>{await api(`/api/jobs/${encodeURIComponent(state.job.id)}/stop`,{method:"POST"});pollJob();}));
  $("#task-commands").addEventListener("click",(event)=>{const node=event.target.closest("[data-copy-command]");if(node)copyText(node.dataset.copyCommand);});
  $("#save-draft").addEventListener("click",saveDraft);$("#load-draft").addEventListener("click",listDrafts);$("#close-drafts").addEventListener("click",()=>$("#draft-dialog").close());
  $("#draft-list").addEventListener("click",(event)=>{const node=event.target.closest("[data-load-draft]");if(node)restoreDraft(node.dataset.loadDraft);});
  $("#draft-dialog").addEventListener("click",(event)=>{if(event.target===$("#draft-dialog")){const r=event.target.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)event.target.close();}});
  window.addEventListener("beforeunload",()=>{clearTimeout(state.poseTimer);clearTimeout(state.jobTimer);});
  window.addEventListener("pagehide",()=>{if(state.previewOpen)navigator.sendBeacon?.("/api/preview/stop");});

  async function initialize() {
    renderAll();updateActions();
    await busy("正在读取 RI-4438 任务模板…",async()=>{
      const result=await api("/api/templates");state.templates=Array.isArray(result)?result:result.templates||[];
      if(!state.templates.length)throw Error("未找到可用的机器人迁移模板");
      const examples=result.examples||[
        {label:"本项目 RI-4438",root:"src/assets/robots/ri_4438",xml:""},
        {label:"本项目 Go2",root:"src/assets/robots/unitree_go2",xml:""},
      ];
      $("#example-models").innerHTML=examples.map((item,index)=>`<button type="button" data-example="${index}">${esc(item.label||item.name)}</button>`).join("");
      $("#example-models").addEventListener("click",(event)=>{const node=event.target.closest("[data-example]");if(!node)return;const item=examples[Number(node.dataset.example)];switchImport("path");$("#model-root").value=item.root;$("#model-xml").value=item.xml||"";$("#model-root").focus();});
      renderAll();
    });
  }
  initialize();
})();
