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
    importDefaults:null, modelXml:null, plan:null, planRevision:-1, revision:0, busy:false, previewOpen:false, poseTimer:null, poseInFlight:false, posePending:false, generated:false, generatedRevision:-1, generatedTaskIds:[], job:null, jobTimer:null, currentFile:null, draftId:null, invalidFields:new Set(), templatesError:false,
    pd:{epoch:0,revision:-1,scope:null,report:null,running:false,method:"model",reason:"",inputSnapshot:null},
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
  function sourceBadge(path,joint=null) {
    const valueAt=(parameters)=>{
      const value=get(parameters,path);
      return joint&&value&&typeof value==="object"&&!Array.isArray(value) ? value[joint.name]??value[joint.role] : value;
    };
    const own=valueAt(targetParameters());
    const inherited=state.scope!=="common"&&own===undefined;
    const value=inherited?valueAt(state.spec.parameters):own;
    if(value===undefined)return '<em class="source-badge template">模板默认</em>';
    const pdApplied=state.spec.pd_analysis?.scopes?.[inherited?"common":state.scope]?.applied;
    const pdValue=get(pdApplied?.proposal,path);
    if(joint&&pdValue?.[joint.name]!==undefined&&value===pdValue[joint.name])return '<em class="source-badge model">PD 计算建议</em>';
    if(state.scope==="common"||inherited){
      const imported=valueAt(state.importDefaults?.parameters);
      if(imported!==undefined&&JSON.stringify(value)===JSON.stringify(imported)){
        const source=state.inspection?.parameter_sources?.[displayPath(path)];
        const modelSourced=source==="model"||path==="control.action_clip"||(path.startsWith("init_state")&&state.inspection?.initial_source==="keyframe");
        if(modelSourced)return '<em class="source-badge model">模型读取</em>';
        if(path.startsWith("init_state"))return '<em class="source-badge template">模板起点</em>';
        return '<em class="source-badge template">模板默认</em>';
      }
    }
    return `<em class="source-badge manual">${inherited?"公共覆盖":"人工修改"}</em>`;
  }
  function changed({pose=false}={}) {
    invalidatePd("配置或分析输入已修改，请重新计算候选。");
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
    const sourceFile=state.spec.model.source_file||state.spec.model.xml;
    $("#summary-model").textContent=sourceFile ? `${sourceFile.split("/").at(-1)}${state.spec.model.source_format==="urdf"?" → XML":""}` : "等待导入模型";
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
    $$("[data-show-model-xml]").forEach((button)=>{button.disabled=state.busy;});
    updatePdActions();
  }
  function showStep(index) {
    state.step=Math.max(0,Math.min(5,index));clearError();
    $$(".step-panel").forEach((node,i)=>{node.hidden=i!==state.step;node.classList.toggle("active",i===state.step);});
    $$(".step-link").forEach((node,i)=>{node.classList.toggle("active",i===state.step);node.setAttribute("aria-current",i===state.step?"step":"false");});
    $("#step-progress").textContent=`第 ${state.step+1} 步 / 共 6 步`;
    const labels=["选择训练任务 →","确认机器人映射 →","配置与姿态预览 →","审阅生成文件 →","生成与验证 →"];
    $("#next-step").textContent=labels[state.step] || "";
    if(state.step===3) {renderConfigTabs();renderParameters();renderJointControls();renderPd();}
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
    const inspection=state.inspection;const model=state.spec.model;
    const converted=model.source_format==="urdf";
    $("#import-result").hidden=false;
    $("#import-result").innerHTML=`<div class="card"><div class="card-top"><div><h3>${converted?"URDF 已转换并解析":"模型已解析"}</h3><p>继续确认动作映射与新任务的机器人标识。</p></div><span class="tag">${converted?"URDF → MJCF XML":"MJCF XML"} · ${inspection.joints?.length===12?"12 关节已识别":"需要检查关节数"}</span></div><div class="import-stats">${[[inspection.joints?.length,"驱动关节"],[inspection.bodies?.length,"机体 body"],[inspection.geoms?.length,"几何 geom"],[inspection.sites?.length,"定位 site"]].map(([value,label])=>`<div class="import-stat"><strong>${value??0}</strong><span>${label}</span></div>`).join("")}</div><div class="import-path">${esc(model.root)} / ${esc(model.xml)}</div>${inspection.files?.length?`<details class="subsection"><summary>模型与资源依赖 <span>${inspection.files.length} files</span></summary><ul class="dependency-list">${inspection.files.map((file)=>`<li>${esc(typeof file==="string"?file:file.path||file.name||JSON.stringify(file))}</li>`).join("")}</ul></details>`:""}</div>${converted?renderConversion():""}${renderIssues(inspection.issues||[])}`;
  }
  function reportItems(value) {
    if(value===undefined||value===null)return "";
    const items=Array.isArray(value)?value:[value];
    return items.map((item)=>`<li>${esc(typeof item==="string"?item:issueText(item))}</li>`).join("");
  }
  function renderConversion() {
    const model=state.spec.model;const report=model.conversion||state.inspection?.conversion||{};
    const source=model.source_file||report.source_file||"见转换报告";
    const siteSourceLabels={named_imu_link:"按具名 IMU link 放置",base_origin_candidate:"基座原点候选，待确认",named_foot_link_candidate:"按具名足端 link 放置，待确认"};
    const generated=report.generated||report.generated_items||report.additions||report.generated_sites?.map((site)=>`site ${site.name} → ${site.body}，局部位置 [${site.position?.join(", ")||""}] m（${siteSourceLabels[site.source]||site.source||"待确认"}）`);
    const preserved=report.preserved||report.sources||report.source_properties;
    const summary=report.summary;
    const warnings=report.warnings||[];
    return `<div class="card conversion-card"><div class="card-top"><div><h3>URDF 转换报告</h3><p>XML 按 Go2 / RI-4438 的结构分组，可在生成任务前检查完整代码。</p></div><span class="tag">原文件保留</span></div><dl class="conversion-paths"><div><dt>输入 URDF</dt><dd>${esc(source)}</dd></div><div><dt>生成 MJCF</dt><dd>${esc(report.xml||model.xml)}</dd></div></dl>${summary?`<ul class="conversion-list">${reportItems(summary)}</ul>`:""}${preserved?`<h4>模型来源与保留内容</h4><ul class="conversion-list">${reportItems(preserved)}</ul>`:""}${generated?`<h4>转换时补充的内容</h4><ul class="conversion-list">${reportItems(generated)}</ul>`:""}${renderIssues(warnings)}<p class="field-note">URDF 的惯量、关节限位等仅按转换报告注明的来源使用；PD、执行器延迟和初始站姿仍需人工确认。转换结果不代表这些实机参数已标定。</p><details class="subsection"><summary>完整转换记录 <span>CONVERSION REPORT</span></summary><pre class="json-pre">${esc(JSON.stringify(report,null,2))}</pre></details><div class="inline-actions"><span class="muted tiny">转换 XML 与原 URDF 均会随模型保留</span><button type="button" class="button secondary small" data-show-model-xml>查看转换后的 XML 源码</button></div><div id="model-xml-review" class="model-xml-review" ${state.modelXml?"":"hidden"}><div class="code-heading"><span id="model-xml-path">${esc(state.modelXml?.path||model.xml)}</span><button type="button" class="button quiet small" data-copy-model-xml>复制 XML</button></div><pre id="model-xml-content" tabindex="0">${esc(state.modelXml?.xml||"")}</pre></div></div>`;
  }
  async function showModelXml() {
    await busy("正在读取转换后的 MJCF XML…",async()=>{
      if(!state.modelXml) {
        const result=await api("/api/model-xml",{method:"POST",body:{model:state.spec.model}});
        if(typeof result.xml!=="string")throw Error("未返回可查看的 XML 源码");
        state.modelXml={xml:result.xml,path:result.path||state.spec.model.xml};
      }
      $("#model-xml-content").textContent=state.modelXml.xml;$("#model-xml-path").textContent=state.modelXml.path;
      $("#model-xml-review").hidden=false;$("#model-xml-review").scrollIntoView({behavior:"smooth",block:"nearest"});
    });
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
      return `<tr><td><div class="joint-label">${esc(joint.name)} <span class="source-badge model">模型读取</span></div><small class="muted tiny">初始角度 · ${state.inspection?.initial_source==="keyframe"?"模型 keyframe 起点":"模板推测起点，请检查"}</small><div class="pose-control"><input type="range" min="${range[0]}" max="${range[1]}" step="0.001" value="${initial}" data-pose-slider="${index}" aria-label="${esc(joint.name)} 初始角度滑块"><input type="number" step="0.001" value="${initial}" min="${range[0]}" max="${range[1]}" data-joint-control="initial" data-joint="${index}" aria-label="${esc(joint.name)} 初始角度"></div><details class="control-advanced"><summary>执行器与动作缩放</summary><div class="mini-control-fields">${[["armature","惯量 kg·m²",0.008234],["friction","摩擦 N·m",0.01],["passive_damping","被动阻尼",0],["action_scale","缩放 rad",0.5]].map(([key,label,fallback])=>`<label>${label} ${get(parameters,`control.${key}`)==null?"· 模型读取，编辑后覆盖":"· 配置值"}<input type="number" step="any" min="0" value="${jointValue(parameters,key,joint,fallback)??""}" placeholder="继承 XML" data-joint-control="${key}" data-joint="${index}" aria-label="${esc(joint.name)} ${label}"></label>`).join("")}</div></details></td>${[["stiffness",32.5065],["damping",2.0694],["effort_limit",10]].map(([key,fallback])=>`<td><input type="number" step="any" min="0" value="${jointValue(parameters,key,joint,fallback)??""}" placeholder="继承 XML" data-joint-control="${key}" data-joint="${index}" aria-label="${esc(joint.name)} ${key}"><small class="joint-source" data-control-source>${sourceBadge(`control.${key}`,joint)}</small></td>`).join("")}<td><div class="clip-inputs"><input type="number" step="any" value="${clip[0]}" data-joint-control="clip_min" data-joint="${index}" aria-label="${esc(joint.name)} 动作下限"><input type="number" step="any" value="${clip[1]}" data-joint-control="clip_max" data-joint="${index}" aria-label="${esc(joint.name)} 动作上限"></div><small class="muted tiny">模型限位起点</small></td></tr>`;
    }).join("");
  }
  function renderConfigTabs() {
    if(state.scope!=="common"&&!state.spec.tasks[state.scope])state.scope="common";
    $("#config-tabs").innerHTML=[{key:"common",label:"公共配置"},...Object.keys(state.spec.tasks).map((key)=>({key,label:taskMeta[key]?.label||key}))].map((item)=>`<button role="tab" aria-selected="${item.key===state.scope}" data-config-scope="${item.key}" class="${item.key===state.scope?"active":""}">${esc(item.label)}</button>`).join("");
    $("#config-scope-note").textContent=state.scope==="common"?"公共修改将应用于全部所选任务。未修改的字段使用各任务原有默认值。3D 预览随当前配置页切换。":`${taskMeta[state.scope]?.label || state.scope} 的专属覆盖；未修改的字段继承公共配置与本模板默认值。`;
  }
  const pdDefaults={frequency_hz:5,damping_ratio:1,pose_delta_rad:0.15,action_amplitude:1,gain_grouping:"auto",joints:{}};
  const pdMethods={model:"完整模型锁定惯量",armature:"电机惯量基线"};
  function pdScopeEntry(create=false) {
    if(create){state.spec.pd_analysis ||= {version:1,scopes:{}};state.spec.pd_analysis.scopes ||= {};state.spec.pd_analysis.scopes[state.scope] ||= {};}
    return state.spec.pd_analysis?.scopes?.[state.scope];
  }
  function pdOptions() {return merge(pdDefaults,pdScopeEntry()?.options||{});}
  function pdFresh() {return !!state.pd.report && state.pd.report===pdScopeEntry()?.report && state.pd.scope===state.scope && state.pd.revision===state.revision;}
  function invalidatePd(reason) {
    state.pd.epoch+=1;state.pd.revision=-1;state.pd.reason=reason;
    for(const entry of Object.values(state.spec.pd_analysis?.scopes||{})){if(entry.report)entry.report_status="needs_recompute";}
    renderPdResults();
  }
  function pdNumber(value,digits=4) {
    if(typeof value!=="number"||!Number.isFinite(value))return "未提供";
    if(value!==0&&Math.abs(value)<0.001)return value.toExponential(3);
    return Number(value.toFixed(digits)).toString();
  }
  function pdRatio(value) {return typeof value==="number"&&Number.isFinite(value)?`${pdNumber(value*100,1)}%`:"未提供";}
  function pdRange(value) {return Array.isArray(value)?value.map((v)=>pdNumber(v,2)).join("–"):"未提供";}
  function pdScopeLabel() {return state.scope==="common"?"公共配置":taskMeta[state.scope]?.label||state.scope;}
  function renderPd() {
    const options=pdOptions();
    $("#pd-scope").textContent=`作用范围：${pdScopeLabel()}。${state.scope==="common"?"候选以首个所选模板的公共有效参数计算；应用后各任务仍保留自己的专属覆盖。":"计算与应用均使用当前任务的有效配置。"}`;
    const fields=[
      ["frequency_hz","设计自然频率","Hz",0.1,false,"决定期望响应速度"],
      ["damping_ratio","设计阻尼比 ζ","",0.05,false,"1 为临界阻尼的理想二阶近似"],
      ["pose_delta_rad","采样姿态扰动","rad",0,false,"从初始姿态按关节类型取 ± 扰动"],
      ["action_amplitude","预算动作幅值 ±a","",0,false,"检查给定动作幅值下的比例项力矩"],
      ["hardware_pd_hz","实机内部 PD 频率","Hz",1,true,"不是策略推理或命令发送频率"],
      ["hardware_delay_ms","实机控制链延迟","ms",0,true,"未知时留空，不从仿真时序推断"],
    ];
    const maxima={frequency_hz:100,damping_ratio:10,pose_delta_rad:Math.PI,action_amplitude:1000,hardware_pd_hz:100000,hardware_delay_ms:1000};
    $("#pd-options").innerHTML=fields.map(([key,label,unit,min,optional,note])=>`<label class="field"><span>${label}${unit?`<em>${unit}</em>`:""}</span><input type="number" step="any" min="${min}" max="${maxima[key]}" value="${esc(options[key]??"")}" data-pd-option="${key}" aria-label="${label}" ${optional?'placeholder="未提供"':"required"}><small>${note}</small></label>`).join("")+`<label class="field full"><span>增益分组</span><select data-pd-grouping aria-label="PD 增益分组">${[["auto","自动（对称任务左右共用）"],["individual","逐关节"],["mirror_pairs","左右成对共用"]].map(([key,label])=>`<option value="${key}" ${options.gain_grouping===key?"selected":""}>${label}</option>`).join("")}</select><small>对称任务要求左右增益相同。成对设计使用配对的最大惯量与阻尼需求，并保留各关节真实惯量范围。</small></label>`;
    $("#pd-hardware").innerHTML=state.spec.bindings.joints.filter((joint)=>joint.name).map((joint)=>`<tr><td><span class="joint-label">${esc(joint.name)}<small>${joint.leg} · ${joint.role}</small></span></td>${[["rotor_inertia","转子惯量"],["gear_ratio","减速比"],["continuous_torque","连续力矩"],["velocity_limit","速度上限"]].map(([key,label])=>`<td><input type="number" step="any" min="${key==="rotor_inertia"?0:0.000000000001}" max="${{rotor_inertia:100,gear_ratio:10000,continuous_torque:1000000,velocity_limit:100000}[key]}" placeholder="未提供" value="${esc(options.joints?.[joint.name]?.[key]??"")}" data-pd-joint="${esc(joint.name)}" data-pd-hardware="${key}" aria-label="${esc(joint.name)} ${label}"></td>`).join("")}</tr>`).join("")||'<tr><td colspan="5" class="muted">导入模型并确认关节映射后填写。</td></tr>';
    state.invalidFields.forEach((key)=>{if(key.startsWith(`pd:${state.scope}:`))state.invalidFields.delete(key);});
    renderPdResults();updatePdActions();
  }
  function pdCandidateCell(candidate) {
    if(!candidate||candidate.available===false)return '<span class="muted">不可用</span>';
    return `<strong>${pdNumber(candidate.kp)} / ${pdNumber(candidate.kd)}</strong><small>比例力矩 ${pdNumber(candidate.peak_proportional_torque,3)} N·m<br>限矩占比 <span class="${candidate.effort_ratio>1?"pd-over-budget":""}">${pdRatio(candidate.effort_ratio)}</span></small>`;
  }
  function renderPdResults() {
    const entry=pdScopeEntry();const report=entry?.report;const node=$("#pd-results");
    if(!node)return;
    node.hidden=!report;
    const fresh=pdFresh();
    $("#pd-status").textContent=state.pd.running?"正在从模型计算惯量与力矩预算…":report?(fresh?"候选已计算。请比较结果，再选择方法并审阅应用值。":state.pd.reason||"草稿中的历史报告仅供查阅，请重新计算后应用。"):"导入模型并确认映射后，即可计算候选。";
    if(!report){node.replaceChildren();return;}
    const method=state.pd.method;const proposal=report.proposals?.[method];const control=proposal?.control;const timing=report.timing||{};
    const joints=report.joints||[];const parameters=effective();
    const current=(joint,key)=>jointValue(parameters,key,joint,null);
    const warnings=report.warnings||[];
    node.innerHTML=`${fresh?"":'<div class="notice warning pd-stale">历史计算结果 · 当前配置、作用范围或会话已变化，重新计算前不能应用。</div>'}<div class="pd-summary-grid"><div><span>采样姿态</span><strong>${report.samples?.length??"—"} 组</strong><small>锁定基座与其他关节 · 无接触</small></div><div><span>仿真 / 策略频率</span><strong>${pdNumber(timing.physics_hz,1)} / ${pdNumber(timing.policy_hz,1)} Hz</strong><small>实机 PD：${timing.hardware_pd_hz==null?"未提供":`${pdNumber(timing.hardware_pd_hz,1)} Hz`}</small></div><div><span>设计自然频率 / 阻尼比</span><strong>${pdNumber(report.settings?.frequency_hz,2)} Hz / ${pdNumber(report.settings?.damping_ratio,2)}</strong><small>实机延迟：${timing.hardware_delay_ms==null?"未提供":`${pdNumber(timing.hardware_delay_ms,2)} ms`}</small></div></div><div class="pd-section-heading"><div><h4>两种方法，与当前增益比较</h4><p>KP / KD 单位为 N·m/rad / N·m·s/rad。力矩为初始姿态、零速度下的比例项预算。</p></div><button id="pd-download" class="button secondary small">下载报告 JSON ↓</button></div><div class="table-scroll pd-result-scroll"><table class="data-table pd-comparison-table"><thead><tr><th>关节</th><th>完整模型惯量 kg·m²<br>最小 / 初始 / 最大</th><th>当前 KP / KD</th><th>完整模型 KP / KD</th><th>电机惯量 KP / KD</th></tr></thead><tbody>${joints.map((joint)=>`<tr><td><span class="joint-label">${esc(joint.name)}<small>${esc(joint.leg)} · ${esc(joint.role)}</small></span><small>armature ${pdNumber(joint.armature)}<br>${esc(joint.armature_source||"来源见报告")}</small></td><td>${[joint.inertia?.min,joint.inertia?.nominal,joint.inertia?.max].map((value)=>pdNumber(value)).join(" / ")}<small>有效缩放 ${pdNumber(joint.action_scale)} rad<br>已含 hip 系数</small></td><td>${pdCandidateCell(joint.current)}</td><td>${pdCandidateCell(joint.candidates?.model)}</td><td>${pdCandidateCell(joint.candidates?.armature)}</td></tr>`).join("")}</tbody></table></div><div class="pd-warning-block">${warnings.length?`<details class="subsection" open><summary>分析提示 <span>${warnings.length} 项</span></summary><ul>${warnings.map((warning)=>`<li>${warning.joint?`<strong>${esc(warning.joint)}</strong> · `:""}${warning.method?`${esc(pdMethods[warning.method]||warning.method)} · `:""}${esc(issueText(warning))}</li>`).join("")}</ul></details>`:""}</div><div class="pd-apply-panel"><div class="pd-section-heading"><div><h4>选择候选，审阅应用前后</h4><p>写入 ${esc(pdScopeLabel())} 的 KP、KD 和本次计算所用 armature。${fresh?"":"下表保留历史候选，应用已禁用。"}</p></div><label class="pd-method-select">候选方法<select id="pd-method" aria-label="PD 候选方法">${Object.entries(pdMethods).map(([key,label])=>`<option value="${key}" ${key===method?"selected":""} ${report.proposals?.[key]?"":"disabled"}>${label}${report.proposals?.[key]?"":"（不可用）"}</option>`).join("")}</select></label></div><div class="table-scroll pd-result-scroll"><table class="data-table pd-review-table"><thead><tr><th>关节</th><th>KP<br>现值 → 候选</th><th>KD<br>现值 → 候选</th><th>armature kg·m²<br>现值 → 计算值</th><th>限矩 / 连续力矩占比</th><th>估计自然频率 / 阻尼比范围</th></tr></thead><tbody>${joints.map((joint)=>{const candidate=joint.candidates?.[method];return `<tr><td>${esc(joint.name)}</td><td>${pdNumber(current(joint,"stiffness"))} → <strong>${pdNumber(control?.stiffness?.[joint.name])}</strong></td><td>${pdNumber(current(joint,"damping"))} → <strong>${pdNumber(control?.damping?.[joint.name])}</strong></td><td>${pdNumber(current(joint,"armature"))} → <strong>${pdNumber(control?.armature?.[joint.name])}</strong></td><td><span class="${candidate?.effort_ratio>1?"pd-over-budget":""}">${pdRatio(candidate?.effort_ratio)}</span> / ${pdRatio(candidate?.continuous_ratio)}</td><td>${pdRange(candidate?.frequency_hz_range)} Hz<small>ζ ${pdRange(candidate?.damping_ratio_range)}</small></td></tr>`;}).join("")}</tbody></table></div><p class="field-note">限矩占比只比较静止时的比例项，不含重力、阻尼力矩或速度相关限矩；超过 100% 表示该探针会请求饱和。应用后请用现有姿态预览和有限仿真继续验证。</p><div class="inline-actions"><span class="muted tiny">${entry.applied?`最近应用：${esc(pdMethods[entry.applied.method]||entry.applied.method)} · ${esc(entry.applied.applied_at||"")}`:"动作缩放、动作裁剪、限矩与初始站姿按当前配置继续使用。"}</span><button id="pd-apply" class="button primary" ${fresh&&proposal?"":"disabled"}>应用到${esc(pdScopeLabel())}</button></div></div><details class="subsection"><summary>本次输入、方法说明与参考来源 <span>REPORT DETAILS</span></summary>${(report.method_notes||[]).map((note)=>`<p class="field-note">${esc(issueText(note))}</p>`).join("")}<ul class="pd-references">${(report.references||[]).filter((ref)=>/^https?:\/\//i.test(ref.url||"")).map((ref)=>`<li><a href="${esc(ref.url)}" target="_blank" rel="noopener noreferrer">${esc(ref.title||ref.url)}</a></li>`).join("")}</ul><pre class="json-pre">${esc(JSON.stringify({settings:report.settings,timing:report.timing,samples:report.samples},null,2))}</pre></details>`;
    updatePdActions();
  }
  function updatePdActions() {
    const button=$("#pd-analyze");if(!button)return;
    button.disabled=state.busy||state.pd.running||!state.inspection||state.invalidFields.size>0;
    button.textContent=state.pd.running?"计算中…":"计算 PD 候选 ↗";
    const apply=$("#pd-apply");if(apply)apply.disabled=state.busy||state.pd.running||state.invalidFields.size>0||!pdFresh()||!pdScopeEntry()?.report?.proposals?.[state.pd.method];
  }
  function editPdInput(input) {
    const option=input.dataset.pdOption;const name=input.dataset.pdJoint;const key=option||input.dataset.pdHardware;if(!key)return;
    const invalidKey=`pd:${state.scope}:${name||"global"}:${key}`;
    input.setCustomValidity("");
    const options=pdOptions();const optional=!!name||["hardware_pd_hz","hardware_delay_ms"].includes(key);
    const empty=input.value.trim()==="";
    if((empty&&!optional)||(!empty&&(!Number.isFinite(Number(input.value))||!input.checkValidity()))){
      input.setCustomValidity("请填写符合范围的有限数值");input.classList.add("pd-invalid");state.invalidFields.add(invalidKey);invalidatePd("分析输入尚未填写完整，请修正后重新计算。");updateActions();return;
    }
    state.invalidFields.delete(invalidKey);input.classList.remove("pd-invalid");
    const target=name?(options.joints[name] ||= {}):options;
    if(empty)delete target[key];else target[key]=Number(input.value);
    pdScopeEntry(true).options=options;changed();
  }
  async function analyzePd() {
    const issues=localIssues(3);if(issues.length){displayError(new Error(issues.join("\n")));return;}
    const options=pdOptions();
    for(const [name,hardware] of Object.entries(options.joints)){
      if((hardware.rotor_inertia!=null)!==(hardware.gear_ratio!=null)){displayError(new Error(`${name} 的转子惯量与减速比必须成对填写，或同时留空`));return;}
    }
    pdScopeEntry(true).options=clone(options);changed();invalidatePd("重新计算中，之前的候选暂不可应用。");
    const revision=state.revision,epoch=state.pd.epoch,scope=state.scope;const inputSnapshot={model:clone(state.spec.model),bindings:clone(state.spec.bindings),parameters:effective(),options:clone(options)};
    state.pd.running=true;clearError();renderPdResults();updateActions();
    try{
      const report=await api("/api/pd/analyze",{method:"POST",body:{spec:state.spec,scope,options}});
      if(revision!==state.revision||epoch!==state.pd.epoch||scope!==state.scope){toast("计算期间配置或作用范围已变化，返回结果已丢弃；请重新计算。");return;}
      if(!Array.isArray(report.joints)||!report.proposals?.model)throw Error("PD 分析未返回完整候选与应用参数");
      const entry=pdScopeEntry(true);entry.report=report;entry.report_created_at=new Date().toISOString();entry.report_status="computed_in_session";
      state.pd.report=report;state.pd.revision=revision;state.pd.scope=scope;state.pd.reason="";state.pd.inputSnapshot=inputSnapshot;
      if(!report.proposals[state.pd.method])state.pd.method="model";
      // Analysis metadata changes the generated manifest, so an older file review is invalid.
      state.plan=null;state.spec.confirmed=false;$("#confirm-review").checked=false;$("#file-review").hidden=true;$("#diff-section").hidden=true;$("#plan-issues").replaceChildren();$("#plan-summary").textContent="分析报告已更新，请重新生成文件预览";
      toast("PD 候选已就绪，请查看惯量、力矩预算与应用前后值","success");
    }catch(error){displayError(error);}
    finally{state.pd.running=false;renderPdResults();updateSummary();updateActions();}
  }
  function applyPd() {
    if(!pdFresh()||state.pd.running||state.invalidFields.size){toast("请根据当前有效输入重新计算候选后应用","error");return;}
    const entry=pdScopeEntry();const method=state.pd.method;const proposal=entry.report.proposals?.[method];if(!proposal?.control)return;
    const before=Object.fromEntries(state.spec.bindings.joints.map((joint)=>[joint.name,{kp:jointValue(effective(),"stiffness",joint),kd:jointValue(effective(),"damping",joint),armature:jointValue(effective(),"armature",joint)}]));
    const target=targetParameters();target.control ||= {};
    for(const key of ["stiffness","damping","armature"])target.control[key]=clone(proposal.control[key]);
    entry.applied={method,applied_at:new Date().toISOString(),scope:state.scope,options:clone(entry.options),input_snapshot:clone(state.pd.inputSnapshot),before,proposal:clone(proposal),report:clone(entry.report),hardware_calibrated:false};
    changed();renderJointControls();renderParameters();state.pd.reason="候选已应用到当前配置；报告保留应用依据，继续分析前请重新计算。";renderPdResults();
    toast(`已将 ${pdMethods[method]} 的 KP、KD 与 armature 应用到${pdScopeLabel()}`,"success");
  }
  function downloadPd() {
    const entry=pdScopeEntry();if(!entry?.report)return;
    const payload={version:1,scope:state.scope,robot:clone(state.spec.robot),model:clone(state.spec.model),options:clone(entry.options||{}),report_status:pdFresh()?"computed_in_session":"needs_recompute",report_created_at:entry.report_created_at,report:entry.report,applied:entry.applied||null,hardware_calibrated:false};
    const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:"application/json;charset=utf-8"}));
    const link=document.createElement("a");link.href=url;link.download=`${state.spec.robot.slug||"robot"}_pd_${state.scope}.json`;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
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
      state.inspection=imported.inspection;state.importDefaults=clone(imported.spec||state.spec);state.draftId=id;state.modelXml=null;
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
        const file=$("#model-zip").files[0];if(!file)throw Error("请选择包含 XML / URDF 和资源的 ZIP 文件");
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
      state.inspection=result.inspection;state.importDefaults=clone(state.spec);state.draftId=null;state.modelXml=null;state.previewOpen=false;state.invalidFields.clear();resetGeneration();resetPreview();changed();renderAll();
      $("#import-result").scrollIntoView({behavior:"smooth",block:"nearest"});
      toast(state.spec.model.source_format==="urdf"?"URDF 已转为 XML，请先检查转换报告":"模型已解析，请继续选择训练任务","success");
    });
  }
  async function copyText(text) {
    try {await navigator.clipboard.writeText(text);toast("已复制到剪贴板","success");}
    catch(_error){const area=document.createElement("textarea");area.value=text;area.style.position="fixed";area.style.opacity="0";document.body.append(area);area.select();const ok=document.execCommand("copy");area.remove();toast(ok?"已复制到剪贴板":"浏览器暂不允许复制，请手动选择文本",ok?"success":"");}
  }
  function renderAll() {renderImport();renderTasks();renderBindings();renderConfigTabs();renderParameters();renderJointControls();renderPd();updateSummary();updateActions();}

  // Event delegation keeps dynamically generated forms tied to MigrationSpec.
  $("#step-nav").addEventListener("click",(event)=>{const button=event.target.closest("[data-step]");if(button&&!state.busy)showStep(Number(button.dataset.step));});
  $("#previous-step").addEventListener("click",()=>showStep(state.step-1));
  $("#next-step").addEventListener("click",()=>{const issues=localIssues(Math.min(state.step,3));if(issues.length){displayError(new Error(issues.join("\n")));return;}if(state.step===4&&(!state.plan||!state.spec.confirmed)){displayError(new Error("请审阅文件并勾选确认后继续"));return;}showStep(state.step+1);});
  $("#import-form").addEventListener("submit",importModel);
  $("#import-result").addEventListener("click",(event)=>{
    if(event.target.closest("[data-show-model-xml]"))showModelXml();
    if(event.target.closest("[data-copy-model-xml]")&&state.modelXml)copyText(state.modelXml.xml);
  });
  function switchImport(mode) {state.importMode=mode;$("#path-fields").hidden=mode!=="path";$("#zip-fields").hidden=mode!=="zip";$("#model-root").required=mode==="path";["path","zip"].forEach((key)=>{$(`#${key}-tab`).classList.toggle("active",key===mode);$(`#${key}-tab`).setAttribute("aria-selected",key===mode);});}
  $("#path-tab").addEventListener("click",()=>switchImport("path"));$("#zip-tab").addEventListener("click",()=>switchImport("zip"));
  $("#model-zip").addEventListener("change",()=>{$("#zip-name").textContent=$("#model-zip").files[0]?.name||"保留 XML / URDF 与 assets 的相对目录结构";});
  ["dragenter","dragover"].forEach((type)=>$("#upload-zone").addEventListener(type,(event)=>{event.preventDefault();$("#upload-zone").classList.add("dragover");}));
  ["dragleave","drop"].forEach((type)=>$("#upload-zone").addEventListener(type,(event)=>{event.preventDefault();$("#upload-zone").classList.remove("dragover");if(type==="drop"&&event.dataTransfer.files.length){const file=event.dataTransfer.files[0];if(!file.name.toLowerCase().endsWith(".zip")){toast("请选择 .zip 模型资源包","error");return;}$("#model-zip").files=event.dataTransfer.files;$("#zip-name").textContent=file.name;}}));
  $("#task-cards").addEventListener("change",(event)=>{
    const input=event.target;
    if(input.dataset.toggleTask){const key=input.dataset.toggleTask;if(input.checked)state.spec.tasks[key]=defaultTask(key);else delete state.spec.tasks[key];changed();renderTasks();renderConfigTabs();renderParameters();renderJointControls();renderPd();}
    if(input.dataset.terrain){const task=state.spec.tasks[input.dataset.task];if(!task)return;task.terrains=["Flat","Rough"].filter((terrain)=>terrain===input.dataset.terrain?input.checked:task.terrains.includes(terrain));changed();}
  });
  ["prefix","slug"].forEach((key)=>$(`#robot-${key}`).addEventListener("input",(event)=>{state.spec.robot[key]=event.target.value.trim();changed();}));
  $("#base-body").addEventListener("change",(event)=>{state.spec.bindings.base_body=event.target.value;changed();});
  $("#joint-mapping").addEventListener("change",(event)=>{
    const input=event.target;
    if(input.dataset.jointMap!==undefined){const joint=state.spec.bindings.joints[Number(input.dataset.jointMap)];joint.name=input.value;changed();renderBindings();renderJointControls();renderPd();}
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
  $("#config-tabs").addEventListener("click",(event)=>{const button=event.target.closest("[data-config-scope]");if(button&&button.dataset.configScope!==state.scope){state.scope=button.dataset.configScope;invalidatePd("配置作用范围已切换，请重新计算后应用。");renderConfigTabs();renderParameters();renderJointControls();renderPd();if(state.previewOpen)schedulePose();}});
  $("#pd-design").addEventListener("input",(event)=>{if(event.target.dataset.pdOption||event.target.dataset.pdHardware)editPdInput(event.target);});
  $("#pd-design").addEventListener("click",(event)=>{if(event.target.closest("#pd-download"))downloadPd();if(event.target.closest("#pd-apply"))applyPd();});
  $("#pd-design").addEventListener("change",(event)=>{if(event.target.id==="pd-method"){state.pd.method=event.target.value;renderPdResults();}if(event.target.matches("[data-pd-grouping]")){const options=pdOptions();options.gain_grouping=event.target.value;pdScopeEntry(true).options=options;changed();}});
  $("#pd-analyze").addEventListener("click",analyzePd);
  // An unfinished or invalid edit also invalidates results, before its change event.
  $("#main-content").addEventListener("input",(event)=>{if(event.target.matches("[data-param], [data-joint-control], [data-pose-slider], [data-foot-position], #robot-prefix, #robot-slug")){invalidatePd("配置正在编辑，请完成输入后重新计算候选。");updatePdActions();}});
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
    const badge=input.closest("td")?.querySelector("[data-control-source]");
    if(badge)badge.innerHTML=sourceBadge(`control.${key}`,joint);
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
