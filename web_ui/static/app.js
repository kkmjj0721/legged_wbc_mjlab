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
    lpacrl:{label:"LP-ACRL 任务课程",sub:"任务分组、采样与学习进展参数",open:false},
    reset:{label:"Reset 随机化",sub:"机体与关节重置范围",open:false},
    randomization:{label:"动力学随机化",sub:"质量、摩擦、扰动与 IMU 偏差",open:false},
    observations:{label:"观测与延迟",sub:"观测延迟独立于执行器延迟，噪声按观测项配置",open:false},
    rewards:{label:"奖励权重",sub:"只调整权重；奖励函数参数沿用当前模板，并自动适配机器人绑定",open:false},
    learning:{label:"网络与训练",sub:"Policy、PPO、Estimator 与训练预算",open:false},
  };
  const fieldLabels = {
    mujoco:"MuJoCo 仿真器",integrator:"积分方法",cone:"接触摩擦锥模型",impratio:"摩擦 / 法向约束阻抗比",jacobian:"约束雅可比类型",solver:"约束求解器",iterations:"求解器最大迭代次数",tolerance:"求解器收敛容差",ls_iterations:"线搜索最大迭代次数",ls_tolerance:"线搜索收敛容差",ccd_iterations:"连续碰撞检测最大迭代次数",gravity:"重力 [x, y, z]",disableflags:"关闭的物理选项",enableflags:"启用的物理选项",nconmax:"接触缓冲区上限",njmax:"约束缓冲区上限",contact_sensor_maxmatch:"接触传感器匹配上限",
    num_envs:"环境数量",env_spacing:"环境间距",episode_length_s:"回合长度",sim_dt:"仿真步长",dt:"仿真步长",decimation:"控制降采样",pos:"机体初始位置 [x, y, z]",rot:"机体初始姿态 [w, x, y, z]",stiffness:"PD 刚度 Kp",damping:"PD 阻尼 Kd",effort_limit:"力矩限幅",armature:"自动计算的关节惯量 armature",rotor_inertia:"电机转子惯量",gear_ratio:"关节总传动比 N（默认 1:1）",continuous_torque:"额定连续扭矩 N·m",velocity_limit:"允许最高工作转速",friction:"被动摩擦",passive_damping:"被动阻尼",action_scale:"动作缩放",hip_reduction:"髋关节动作系数",action_clip:"动作裁剪",delay_min_lag:"最小延迟",delay_max_lag:"最大延迟",delay_hold_prob:"延迟保持概率",delay_update_period:"延迟更新周期",enabled:"启用",curriculum:"启用课程",max_curriculum:"课程上限",resampling_time:"指令重采样间隔",heading_command:"启用航向指令",rel_standing_envs:"站立环境比例",rel_forward_envs:"前进环境比例",rel_heading_envs:"航向环境比例",heading_control_stiffness:"航向控制刚度",lin_vel_x:"前进速度 vx",lin_vel_y:"横向速度 vy",ang_vel_yaw:"转向速度 wz",heading:"航向角",link_mass_range:"腿部质量系数范围",base_mass_range:"机体质量系数范围",KpKd_factor_range:"PD 系数随机范围",com_displacement_range:"质心偏移范围",motor_offset_range:"电机零点偏差范围",motor_strength_range:"电机强度系数范围",effort_limit_range:"力矩上限系数范围",joint_friction_range:"关节摩擦系数范围",joint_armature_range:"关节惯量系数范围",joint_damping_range:"被动阻尼系数范围",imu_gyro_bias_range:"陀螺仪偏差范围",imu_mount_angle_range:"IMU 安装角范围",external_force_range:"外力范围",external_force_duration_s:"外力持续时间",external_force_cooldown_s:"外力冷却时间",foot_contact_time_constant_range:"足端接触时间常数",enable_corruption:"启用观测噪声",weight:"权重",n_min:"噪声下限",n_max:"噪声上限",init_noise_std:"初始动作噪声",actor_hidden_dims:"Actor 隐藏层",critic_hidden_dims:"Critic 隐藏层",activation:"激活函数",obs_normalization:"观测归一化",encoder_hidden_dims:"Encoder 隐藏层",target_hidden_dims:"Target 隐藏层",num_prototype:"原型数量",temperature:"温度",sinkhorn_eps:"Sinkhorn ε",sinkhorn_iters:"Sinkhorn 迭代次数",value_loss_coef:"价值损失系数",use_clipped_value_loss:"启用价值裁剪",clip_param:"PPO 裁剪系数",entropy_coef:"熵奖励系数",num_learning_epochs:"学习轮数",num_mini_batches:"Mini-batch 数量",learning_rate:"学习率",schedule:"学习率计划",gamma:"折扣因子 γ",lam:"GAE λ",desired_kl:"目标 KL",max_grad_norm:"梯度范数上限",estimator_learning_rate:"Estimator 学习率",estimator_max_grad_norm:"Estimator 梯度上限",num_steps_per_env:"每次更新采样步数",max_iterations:"最大训练更新次数",save_interval:"检查点保存间隔",experiment_name:"实验目录名称",seed:"随机种子",horizontal_scale:"水平分辨率",vertical_scale:"高度分辨率",border_size:"边界宽度",num_rows:"地形行数",num_cols:"地形列数",size:"地形尺寸",difficulty_range:"难度范围",max_init_terrain_level:"初始地形等级上限",proportion:"采样比例",noise_range:"噪声范围",position_range:"关节位置偏移范围",velocity_range:"速度偏移范围",x:"X 偏移",y:"Y 偏移",z:"Z 偏移",roll:"Roll 偏移",pitch:"Pitch 偏移",yaw:"Yaw 偏移",symmetry_loss_coefficient:"对称损失系数",use_data_augmentation:"启用对称数据增强",use_mirror_loss:"启用镜像损失",std:"标准差",mean:"均值",min:"最小值",max:"最大值",
  };
  const unitFor = (path) => {
    const key = pathSegments(path).at(-1);
    if(key==="gravity")return "m/s²";
    if (["num_envs","num_rows","num_cols"].includes(key)) return "个";
    if (["episode_length_s","sim_dt","dt","resampling_time","external_force_duration_s","external_force_cooldown_s","foot_contact_time_constant_range"].includes(key)) return "s";
    if (["env_spacing","com_displacement_range","horizontal_scale","vertical_scale","border_size","border_width","size","pos","x","y","z"].includes(key)) return "m";
    if (["lin_vel_x","lin_vel_y"].includes(key)) return "m/s";
    if (["ang_vel_yaw","imu_gyro_bias_range"].includes(key)) return "rad/s";
    if (["heading","roll","pitch","yaw","imu_mount_angle_range","motor_offset_range"].includes(key)) return "rad";
    if (key === "external_force_range") return "N";
    if (["delay_min_lag","delay_max_lag","delay_update_period","decimation"].includes(key)) return "步";
    if (["max_iterations","save_interval","sinkhorn_iters","num_learning_epochs","iterations","ls_iterations","ccd_iterations"].includes(key)) return "次";
    return "";
  };
  const fixedKeys = new Set(["num_actions","num_action","num_observations","num_privileged_obs","num_one_step_obs","history_size","history_term_dims","history_order","num_obs_history","num_obs_one_step","num_obs","estimator_velocity_slice","estimator_target_slices","observation_layout"]);
  const jointPaths = new Set(["control.stiffness","control.damping","control.effort_limit","control.armature","control.friction","control.passive_damping","control.action_scale","control.action_clip","control.hip_reduction","init_state.default_joint"]);
  const state = {
    pendingEdits:new Set(), pdUi:{phase:0,mode:"search",inputConfirmed:{}}, step:0, scope:"common", importMode:"path", templates:[], inspection:null, spec:{version:1,model:{},robot:{slug:"",prefix:""},bindings:{base_body:"",joints:[],feet:[],base_geoms:[],leg_geoms:[]},parameters:{},tasks:{him:{terrains:["Rough"],parameters:{}}},confirmed:false},
    importDefaults:null, modelXml:null, plan:null, planRevision:-1, revision:0, busy:false, previewOpen:false, poseFocus:true, terrainAnalysis:null, poseTimer:null, poseInFlight:false, posePending:false, generated:false, generatedRevision:-1, generatedTaskIds:[], job:null, jobTimer:null, currentFile:null, draftId:null, invalidFields:new Set(), templatesError:false,
    pd:{sessions:{},epoch:0,revision:-1,scope:null,report:null,running:false,method:"model",reason:"",inputSnapshot:null},
    pdTest:{sessions:{},epoch:0,revision:-1,scope:null,report:null,starting:false,job:null,timer:null,candidateId:null,jointName:null,direction:"worst",reason:"",inputSnapshot:null},
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
  function motorInertiaInput(joint,scope=state.scope) {
    const name=joint.name,scopes=state.spec.motor_inertia?.scopes;
    const common=scopes?.common?.[name]||(scopes?{}:state.spec.pd_analysis?.scopes?.common?.options?.joints?.[name])||{};
    const local=scope==="common"?{}:(scopes?.[scope]?.[name]||(scopes?{}:state.spec.pd_analysis?.scopes?.[scope]?.options?.joints?.[name])||{});
    const value={rotor_inertia:null,gear_ratio:1};
    for(const row of [common,local])for(const key of ["rotor_inertia","gear_ratio"])if(Object.hasOwn(row,key))value[key]=row[key];
    value.gear_ratio ??= 1;
    return value;
  }
  function ensureMotorInertia() {
    if(state.spec.motor_inertia)return;
    const scopes={common:{}};
    for(const [scope,entry] of Object.entries(state.spec.pd_analysis?.scopes||{})){
      const joints={};
      for(const [name,row] of Object.entries(entry.options?.joints||{})){
        const values=Object.fromEntries(["rotor_inertia","gear_ratio"].filter((key)=>Object.hasOwn(row,key)).map((key)=>[key,row[key]]));
        if(Object.keys(values).length)joints[name]=values;
      }
      if(Object.keys(joints).length)scopes[scope]=joints;
    }
    state.spec.motor_inertia={version:1,scopes};
  }
  function effective(scope=state.scope) {
    const p=merge(merge(baseDefaults(scope),state.spec.parameters),scope === "common" ? {} : state.spec.tasks[scope]?.parameters || {});
    const joints=state.spec.bindings.joints.filter((joint)=>joint.name),derived={};let known=false;
    for(const joint of joints){
      const h=motorInertiaInput(joint,scope);
      if(h.rotor_inertia!=null&&Number.isFinite(h.rotor_inertia)&&h.rotor_inertia>=0&&Number.isFinite(h.gear_ratio)&&h.gear_ratio>0){derived[joint.name]=h.rotor_inertia*h.gear_ratio**2;known=true;}
      else derived[joint.name]=jointValue(p,"armature",joint,null);
    }
    if(known){p.control ||= {};p.control.armature=derived;}
    return p;
  }
  function targetParameters() { return state.scope === "common" ? state.spec.parameters : state.spec.tasks[state.scope].parameters; }
  function sourceBadge(path,joint=null) {
    const valueAt=(parameters)=>{
      const value=get(parameters,path);
      return joint&&value&&typeof value==="object"&&!Array.isArray(value) ? (Object.hasOwn(value,joint.name)?value[joint.name]:value[joint.role]) : value;
    };
    const own=valueAt(targetParameters());
    const inherited=state.scope!=="common"&&own===undefined;
    const value=inherited?valueAt(state.spec.parameters):own;
    if(value==null&&["control.stiffness","control.damping"].includes(path))return '<em class="source-badge pending">待搜索 / 人工填写</em>';
    if(value===undefined)return '<em class="source-badge template">模板默认</em>';
    const pdApplied=state.spec.pd_analysis?.scopes?.[inherited?"common":state.scope]?.applied;
    const pdValue=get(pdApplied?.proposal,path);
    if(joint&&pdValue?.[joint.name]!==undefined&&value===pdValue[joint.name])return `<em class="source-badge model">${pdApplied.method==="response"?"PD 仿真建议":"PD 计算建议"}</em>`;
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
  const workflow = window.MigrationWorkflow;
  const steps = workflow.STEPS;
  const configSteps = steps.slice(0,11);
  const motorKeys = new Set(["effort_limit","armature","friction","passive_damping","action_scale","hip_reduction","action_clip"]);
  function fieldStep(path) {
    const [group,key] = pathSegments(path);
    if(group==="init_state")return 3;
    if(group==="control")return ["stiffness","damping"].includes(key)?6:motorKeys.has(key)?4:5;
    if(group==="environment")return 5;
    if(group==="observations")return key?.includes("delay")?5:8;
    if(["commands","terrain","curriculum","lpacrl"].includes(group))return 7;
    if(["reset","randomization"].includes(group))return 8;
    if(group==="rewards")return 9;
    return 10;
  }
  function parameterLeaves(object,prefix="") {
    return Object.entries(object||{}).flatMap(([key,value])=>{
      const path=prefix?`${prefix}.${encodeSegment(key)}`:encodeSegment(key);
      return !jointPaths.has(path)&&value&&typeof value==="object"&&!Array.isArray(value)&&Object.keys(value).length
        ? parameterLeaves(value,path):[{path,value}];
    });
  }
  function stableModel() {
    const model=state.spec.model;
    return {sha256:model.sha256||null,xml:model.xml||null,source_format:model.source_format||null};
  }
  function pdInputFingerprint(scope=state.scope) {
    const p=effective(scope);
    return workflow.stable({model:stableModel(),bindings:state.spec.bindings,
      algorithms:scope==="common"?Object.keys(state.spec.tasks):[scope],
      control:p.control,init_state:p.init_state,
      environment:{sim_dt:p.environment?.sim_dt,mujoco:p.environment?.mujoco},
      options:pdOptions(scope),motor_curves:state.spec.motor_curves||[]});
  }
  function pdSession(kind,scope=state.scope) {return state[kind].sessions?.[scope];}
  function pdPending(scope=state.scope) {
    return [...(state.pendingEdits||[])].some((key)=>{
      if(!key.startsWith(`${scope}:`))return false;
      const path=key.slice(scope.length+1),step=fieldStep(path);
      return [3,4,6].includes(step)||step===5&&(path.startsWith("control.")||path==="environment.sim_dt"||path.startsWith("environment.mujoco."));
    });
  }
  function stepSnapshots() {
    const model=stableModel(),bindings=state.spec.bindings,taskKeys=Object.keys(state.spec.tasks);
    const taskSelection=taskKeys.map((key)=>[key,state.spec.tasks[key].terrains]);
    const profiles=(index)=>Object.fromEntries(["common",...taskKeys].map((scope)=>[scope,Object.fromEntries(parameterLeaves(effective(scope)).filter((field)=>fieldStep(field.path)===index).map(({path,value})=>[path,value]))]));
    const snapshots={import:{model,budget:state.spec.model.mesh_face_limit},tasks:{model,selection:taskSelection},mapping:{model,bindings,robot:state.spec.robot}};
    for(let index=3;index<=10;index++)snapshots[steps[index].id]={model,profiles:profiles(index)};
    snapshots.pose.bindings=bindings;
    snapshots.motors.joints=bindings.joints;
    snapshots.motors.curves=state.spec.motor_curves||[];
    snapshots.motors.hardware=Object.fromEntries(["common",...taskKeys].map((scope)=>[scope,pdOptions(scope).joints]));
    snapshots.pd.inputs=Object.fromEntries(["common",...taskKeys].map((scope)=>[scope,pdInputFingerprint(scope)]));
    for(const id of ["commands","randomization","rewards"])snapshots[id].selection=taskSelection;
    snapshots.randomization.bindings=bindings;
    snapshots.rewards.bindings=bindings;
    snapshots.review={revision:state.revision,ready:!!state.plan&&state.spec.confirmed};
    snapshots.generate={revision:state.revision,generated:state.generated,task_ids:state.generatedTaskIds};
    return snapshots;
  }
  function workflowRecords() {return workflow.summarize(state.spec,stepSnapshots());}
  function unconfirmedSteps() {return workflowRecords().slice(0,11).filter((item)=>item.status!=="confirmed");}
  function stepSummary(index) {
    const p=effective(),scope=Object.keys(state.spec.tasks).length;
    return [
      `${state.spec.model.source_file||state.spec.model.xml||"模型"} · ${state.inspection?.joints?.length||0} 关节`,
      Object.keys(state.spec.tasks).map((key)=>`${taskMeta[key]?.label||key} ${(state.spec.tasks[key].terrains||[]).join(" / ")}`).join("；"),
      `${state.spec.robot.prefix} · 12 个关节 · ${state.spec.bindings.feet.length} 个足端`,
      `机体 [${(p.init_state?.pos||[]).join(", ")}] m · ${scope} 个任务的站姿`,
      `12 关节电机/动作 · ${(state.spec.motor_curves||[]).filter((curve)=>curve.confirmed&&curve.enabled!==false).length} 条启用曲线`,
      `dt ${p.environment?.sim_dt??"—"} s · decimation ${p.control?.decimation??"—"} · ${p.environment?.num_envs??"—"} 环境`,
      `${scope} 个任务的有效 Kp / Kd 已核对${pdScopeEntry()?.applied?.candidate_label?` · ${pdScopeEntry().applied.candidate_label}`:""}`,
      `${scope} 个任务的速度、地形与课程已核对`,
      `Reset、动力学随机化与观测噪声 · ${scope} 个任务`,
      `${Object.keys(p.rewards||{}).length} 项奖励权重 · 函数参数沿用模板`,
      `训练 ${p.learning?.runner?.max_iterations??"—"} 次更新 · 保存间隔 ${p.learning?.runner?.save_interval??"—"}`,
      `${state.plan?.files?.length||0} 个文件已审阅`,
      `${state.generatedTaskIds.length} 个任务已生成`,
    ][index];
  }
  function pendingFieldStep(key) {
    const raw=key.replace(/^(common|ppo|him|him_symmetry|him_lpacrl):/,"");
    if(key.startsWith("pd:")||key.startsWith("inertia:"))return 4;
    if(key.startsWith("joint:"))return state.step;
    if(key.startsWith("foot:"))return 2;
    if(key.startsWith("motor-curve:"))return 4;
    if(key.startsWith("model:"))return 0;
    return fieldStep(raw);
  }
  function stepIssues(index) {
    const issues=localIssues(Math.min(index,2));
    if(index===0)issues.push(...meshBudgetIssues());
    if(index>=3&&index<=10) {
      const invalid=[...state.invalidFields,...(state.pendingEdits||[])].filter((key)=>pendingFieldStep(key)===index);
      if(invalid.length)issues.push(`本步还有 ${new Set(invalid).size} 项输入未完成或格式不正确，请先完成编辑。`);
    }
    if(index===3)for(const scope of ["common",...Object.keys(state.spec.tasks)])for(const joint of state.spec.bindings.joints){
      const value=jointInitial(effective(scope),joint),range=jointRange(joint);
      if(!Number.isFinite(value)||value<range[0]||value>range[1])issues.push(`${scope} · ${joint.name} 初始角度超出限位`);
    }
    if(index===4)for(const scope of ["common",...Object.keys(state.spec.tasks)])for(const joint of state.spec.bindings.joints){
      const p=effective(scope),clip=p.control?.action_clip?.[joint.name];
      for(const key of ["effort_limit","armature","friction","passive_damping","action_scale"]){const value=jointValue(p,key,joint,null);if(value!=null&&(!Number.isFinite(value)||value<0))issues.push(`${scope} · ${joint.name} 的 ${key} 需要非负有限数值`);}
      if(clip&&(!Array.isArray(clip)||clip.length!==2||!clip.every(Number.isFinite)||clip[0]>=clip[1]))issues.push(`${joint.name} 目标角裁剪范围无效`);
      const h=motorInertiaInput(joint,scope);
      if(h.rotor_inertia!=null&&(!Number.isFinite(h.rotor_inertia)||h.rotor_inertia<0||h.rotor_inertia>100))issues.push(`${joint.name} 电机转子惯量需要为 0–100 的有限数值`);
      if(!Number.isFinite(h.gear_ratio)||h.gear_ratio<=0||h.gear_ratio>10000)issues.push(`${joint.name} 关节总传动比必须为正数（直驱填 1）`);
    }
    if(index===4)for(const scope of ["common",...Object.keys(state.spec.tasks)]){const value=effective(scope).control?.hip_reduction??1;if(!Number.isFinite(value)||value<=0||value>1)issues.push(`${scope} · hip_reduction 需要位于 (0, 1]`);}
    if(index===6)for(const scope of Object.keys(state.spec.tasks)) {
      const p=effective(scope),missing=state.spec.bindings.joints.filter((joint)=>["stiffness","damping"].some((key)=>{const value=jointValue(p,key,joint,null);return !Number.isFinite(value)||value<0;}));
      if(missing.length)issues.push(`${taskMeta[scope]?.label||scope} 仍有 ${missing.length} 个关节需要完整的非负 Kp / Kd；请应用候选或手动填写。`);
    }
    if(index===7)for(const scope of ["common",...Object.keys(state.spec.tasks)])issues.push(...terrainConfigurationIssues(effective(scope)).map((message)=>`${scope} · ${message}`));
    if(index===11){issues.push(...localIssues(4));if(!state.plan||!state.spec.confirmed)issues.push("请生成文件预览并勾选审阅确认。");}
    return [...new Set(issues)];
  }
  function renderWorkflow() {
    const records=workflowRecords(),current=records[state.step],labels={todo:"待确认",confirmed:"已确认",stale:"待复核"};
    if(!$("#step-nav").children.length){
      const groups=[...new Set(steps.map((step)=>step.group))];
      $("#step-nav").innerHTML=groups.map((group)=>`<div class="workflow-group"><h3>${esc(group)}</h3>${steps.filter((step)=>step.group===group).map((step)=>`<button class="step-link" data-step="${step.index}"><span class="step-num">${String(step.index+1).padStart(2,"0")}</span><span class="step-label">${esc(step.title)}<small>${esc(step.hint)}</small></span><span class="step-tick"></span></button>`).join("")}</div>`).join("");
    }
    $$(".step-link").forEach((node)=>{const item=records[Number(node.dataset.step)],active=item.index===state.step;node.classList.toggle("active",active);node.classList.toggle("complete",item.status==="confirmed");node.classList.toggle("needs-review",item.status==="stale");node.setAttribute("aria-current",active?"step":"false");$(".step-tick",node).textContent=item.status==="confirmed"?"已确认":item.status==="stale"?"待复核":"";});
    const count=records.slice(0,11).filter((item)=>item.status==="confirmed").length;
    $("#workflow-confirmed-count").textContent=`${count} / 11 项配置已确认`;
    $("#workflow-step-state").textContent=`${String(state.step+1).padStart(2,"0")} · ${steps[state.step].title} · ${labels[current.status]}`;
    $("#workflow-step-state").className=current.status==="confirmed"?"complete":current.status==="stale"?"needs-review":"";
    $("#workflow-step-description").textContent=current.status==="stale"?`相关输入已修改，请复核本步。上次确认：${current.summary}`:current.status==="confirmed"?`已确认：${current.summary}`:steps[state.step].hint+"。完成后点击底部确认，草稿会自动保存。";
    $("#confirmation-hint").textContent=current.status==="stale"?"保留已有输入，请复核受影响的内容":current.status==="confirmed"?"本步已确认，可继续或回看":"确认本步后自动保存草稿";
    $("#workflow-checklist").innerHTML=records.slice(0,11).map((item)=>`<button class="workflow-checkpoint ${item.status==="confirmed"?"complete":item.status==="stale"?"needs-review":""}" data-workflow-jump="${item.index}"><b>${String(item.index+1).padStart(2,"0")} ${esc(item.title)}</b><span>${labels[item.status]}</span><small>${esc(item.status==="todo"?item.hint:item.summary)}</small></button>`).join("");
    $("#config-context").hidden=state.step<3||state.step>10;
    $("#parameter-search-drawer").hidden=!state.inspection||state.step<3||state.step>10;
    $("#joint-batch").hidden=![3,4,6].includes(state.step);
    if([3,4,6].includes(state.step)){
      const keys=state.step===3?["initial"]:state.step===4?["effort_limit","rotor_inertia","gear_ratio","continuous_torque","velocity_limit","friction","passive_damping","action_scale"]:["stiffness","damping"];
      const selected=$("#batch-property").value;
      if($("#batch-property").dataset.step!==String(state.step)){$("#batch-property").innerHTML=keys.map((key)=>`<option value="${key}">${esc(key==="initial"?"初始角度 rad":fieldLabels[key]||key)}</option>`).join("");$("#batch-property").dataset.step=String(state.step);if(keys.includes(selected))$("#batch-property").value=selected;}
    }
    updateSpeedUnitControls();
    const blockers=records.slice(0,11).filter((item)=>item.status!=="confirmed");
    $("#review-checkpoints").hidden=!blockers.length;
    $("#review-checkpoints").innerHTML=blockers.length?`<strong>生成前还有 ${blockers.length} 步需要确认或复核</strong><div class="pd-input-links">${blockers.map((item)=>`<button class="button quiet small" data-workflow-jump="${item.index}">${item.index+1}. ${esc(item.title)} · ${labels[item.status]}</button>`).join("")}</div>`:"";
    renderPdFlow();
  }
  function generationSpec() {const spec=clone(state.spec);delete spec.workflow;return spec;}
  async function confirmCurrentStep() {
    const issues=stepIssues(state.step);if(issues.length){displayError(new Error(issues.join("\n")));return;}
    if(state.step>=11&&unconfirmedSteps().length){displayError(new Error("请先确认右侧清单中待确认或待复核的配置步骤。"));return;}
    const id=steps[state.step].id;
    workflow.confirm(state.spec,id,stepSnapshots()[id],stepSummary(state.step));
    const next=Math.min(12,state.step+1);workflow.ensure(state.spec).last_step=next;updateSummary();
    const saved=await saveDraft({quiet:true});
    if(saved)showStep(next);
  }
  function renderPdFlow() {
    if(!$("#pd-input-summary"))return;
    const report=pdScopeEntry()?.response_report,selected=pdSelectedCandidate(),p=effective();
    const applied=pdScopeEntry()?.applied;
    const current=pdAppliedMatches();
    const phase=state.pdUi?.phase??0,mode=state.pdUi?.mode||"search";
    if(state.step===6)$("#joint-batch").hidden=mode!=="manual"||phase!==3;
    $$("[data-pd-stage]").forEach((node)=>{node.hidden=Number(node.dataset.pdStage)!==phase;});
    $$(".pd-phase").forEach((node)=>{const index=Number(node.dataset.pdPhase);node.classList.toggle("active",index===phase);node.setAttribute("aria-current",index===phase?"step":"false");node.disabled=index===2&&!report||index===3&&!report&&mode!=="manual";});
    $$("[data-pd-mode]").forEach((node)=>{node.classList.toggle("active",node.dataset.pdMode===mode);node.setAttribute("aria-pressed",String(node.dataset.pdMode===mode));});
    const control=p.control||{},timing=p.environment||{};
    $("#pd-input-summary").innerHTML=`<div><strong>模型与站姿</strong><span>${esc(state.spec.model.name||"待导入")} · ${state.spec.bindings.joints.length} 关节</span></div><div><strong>仿真时序</strong><span>dt ${pdNumber(timing.sim_dt,5)} s · ${esc(timing.mujoco?.integrator||"implicitfast")}</span></div><div><strong>执行器延迟</strong><span>${control.delay_min_lag??0}–${control.delay_max_lag??0} 个物理步</span></div><div><strong>力矩与惯量</strong><span>使用第 05 步已填写的逐关节值，来源可展开核对</span></div>`;
    $("#pd-no-result").hidden=!!report;
    $("#pd-review-next").disabled=!selected||!pdResponseFresh()||pdReferenceOnly(selected)||!selected.summary?.eligible;
    $("#pd-applied-status").textContent=current?`已应用：${applied.candidate_label||pdMethods[applied.method]||"所选候选"} → ${pdScopeLabel()}。请在底部确认本步。`:mode==="manual"?`手动填写 ${pdScopeLabel()} 的 Kp / Kd，然后在底部确认。`:report&&!pdResponseFresh()?"相关测试输入已改变，报告保留供查看；请返回核对输入后重新搜索。":"检查下方候选写入值，点击应用后再确认本步。";
    if(mode==="manual")$("#pd-manual-editor").open=true;
  }
  function setPdPhase(phase) {
    if(phase===2&&!pdScopeEntry()?.response_report)return;
    if(phase<3)state.pdUi.mode="search";
    state.pdUi.phase=phase;renderPdFlow();
    $("#pd-design").scrollIntoView({behavior:"smooth",block:"start"});
  }
  function pdAppliedMatches() {
    const applied=pdScopeEntry()?.applied;
    return !!applied&&state.spec.bindings.joints.length===12&&state.spec.bindings.joints.every((joint)=>
      ["stiffness","damping","armature"].every((key)=>
        jointValue(effective(),key,joint,null)===jointValue({control:applied.proposal?.control||{}},key,joint,null)));
  }

  function changed({pose=false}={}) {
    if(state.terrainAnalysis){state.terrainAnalysis.stale=true;renderTerrainAnalysis();}
    state.revision+=1;state.spec.confirmed=false;$("#confirm-review").checked=false;
    state.plan=null;state.currentFile=null;
    $("#file-review").hidden=true;$("#diff-section").hidden=true;
    $("#plan-summary").textContent="配置已更新，请重新生成文件预览";$("#plan-issues").replaceChildren();
    invalidatePd("相关测试输入已修改，请核对后重新测试。");
    if(pose&&state.previewOpen)schedulePose();
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
  function meshMaximum() {
    return Math.max(0,Number(state.inspection?.mesh_max_faces)||0,Number(state.spec.model.mesh_max_faces)||0,...(state.inspection?.meshes||[]).map((mesh)=>Number(mesh.faces)||0));
  }
  function ensureMeshBudget() {
    if(state.spec.model.mesh_face_limit===undefined)state.spec.model.mesh_face_limit=Math.max(200000,meshMaximum());
  }
  function meshBudgetIssues() {
    if(!state.inspection)return [];
    const limit=state.spec.model.mesh_face_limit;
    if(!Number.isSafeInteger(limit)||limit<=0)return ["单个网格面数上限需要填写正整数"];
    const maximum=meshMaximum();
    return limit<maximum?[`网格面数预算 ${limit.toLocaleString("zh-CN")} 低于模型最高面数 ${maximum.toLocaleString("zh-CN")}；请调高上限后再生成`]:[];
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
      issues.push(...meshBudgetIssues());
      const parameters=effective("common");
      for (const joint of state.spec.bindings.joints) {
        const value=get(parameters,"init_state.default_joint")?.[joint.name];
        const limits=state.inspection?.joints.find((item)=>item.name===joint.name)?.range;
        if (typeof value === "number" && limits?.length===2 && (value<limits[0]-1e-8 || value>limits[1]+1e-8)) issues.push(`${joint.name} 初始角度 ${value} 超出模型限位 [${limits.join(", ")}]`);
        const clip=get(parameters,"control.action_clip")?.[joint.name];
        if(clip&&(!Array.isArray(clip)||clip.length!==2||!clip.every(Number.isFinite)||clip[0]>=clip[1]))issues.push(`${joint.name} 动作裁剪需要两个有限数值，且下限小于上限`);
      }
    }
    if(step>=4) {
      for(const scope of Object.keys(state.spec.tasks)) {
        const parameters=effective(scope);
        const missing=state.spec.bindings.joints.filter((joint)=>joint.name).filter((joint)=>["stiffness","damping"].some((key)=>!Number.isFinite(jointValue(parameters,key,joint,null))));
        if(missing.length)issues.push(`${taskMeta[scope]?.label||scope} 有 ${missing.length} 个关节尚未设置完整 PD，请先应用搜索候选或手动填写。`);
      }
    }
    return issues;
  }
  function updateSummary() {
    $("#summary-robot").textContent=state.spec.robot.prefix||"新机器人";
    const file=state.spec.model.source_file||state.spec.model.xml;
    $("#summary-model").textContent=file?file.split("/").at(-1):"等待导入模型";
    $("#summary-joints").innerHTML=`${state.inspection?.joints?.length??"—"} <small>/ 12</small>`;
    const ids=state.plan?.task_ids||taskIds();$("#summary-count").textContent=Object.keys(state.spec.tasks).length?ids.length:0;
    $("#summary-tasks").innerHTML=Object.keys(state.spec.tasks).length?ids.map((id)=>`<span class="task-chip">${esc(id)}</span>`).join(""):"选择算法后显示任务名称";
    const pending=workflowRecords().slice(0,11).filter((item)=>item.status!=="confirmed");
    $("#pending-count").textContent=pending.length;
    const stale=pending.filter((item)=>item.status==="stale");
    $("#pending-items").innerHTML=stale.length?`<li>${stale.length} 步受相关输入修改影响，需要复核。已填写的数据继续保留。</li>`:pending.length?`<li>按步骤确认，每次只处理当前部分。${11-pending.length} 步已确认。</li>`:'<li class="done">11 项配置已确认，可以审阅文件。</li>';
    renderOutputLayout();renderWorkflow();
  }
  function updateActions() {
    const running=!!state.job && ["running","queued"].includes(state.job.status);
    ["import-button","save-draft","load-draft","refresh-plan","start-preview","reset-config","reset-joints","apply-batch"].forEach((id)=>{$(`#${id}`).disabled=state.busy;});
    $("#import-button").disabled=state.busy||running;$("#load-draft").disabled=state.busy||running;
    $("#generate-button").disabled=state.busy||running||!state.plan||!state.spec.confirmed||state.generatedRevision===state.revision||unconfirmedSteps().length>0;
    ["validate-cpu","validate-simulation"].forEach((id)=>{$(`#${id}`).disabled=state.busy||running||!state.generated;});
    $("#stop-job").hidden=!running;$("#stop-job").disabled=state.busy;
    $("#previous-step").disabled=state.step===0||state.busy;$("#next-step").hidden=state.step===12;$("#next-step").disabled=state.busy;
    $("#refresh-plan").disabled=state.busy||unconfirmedSteps().length>0;
    $("#start-preview").disabled=state.busy||!state.inspection;
    $("#mesh-face-limit").disabled=state.busy||!state.inspection;
    $("#motor-curve-upload").disabled=state.busy||!state.inspection;
    $$("[data-show-model-xml]").forEach((button)=>{button.disabled=state.busy;});
    updatePdActions();
  }
  function showStep(index) {
    state.step=Math.max(0,Math.min(12,index));document.body.classList.toggle("pose-focus",state.step===3&&state.poseFocus);workflow.ensure(state.spec).last_step=state.step;clearError();
    $$(".step-panel").forEach((node)=>{const active=Number(node.dataset.panel)===state.step;node.hidden=!active;node.classList.toggle("active",active);});
    $("#step-progress").textContent=`第 ${state.step+1} 步 / 共 13 步`;
    $("#next-step").textContent=state.step===11?"确认审阅并继续 →":"确认本步并继续 →";
    renderConfigTabs();renderJointControls();renderPdFlow();updateSummary();updateActions();
    if(state.step===11&&state.inspection&&!state.plan&&!unconfirmedSteps().length&&!localIssues(4).length)buildPlan();
    $(`.step-link[data-step="${state.step}"]`)?.scrollIntoView({block:"nearest",inline:"nearest"});
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
  function renderMeshBudget({keepInput=false}={}) {
    const input=$("#mesh-face-limit");
    const meshes=state.inspection?.meshes||[];
    const maximum=meshMaximum();
    if(state.inspection)ensureMeshBudget();
    if(!keepInput)input.value=state.inspection?state.spec.model.mesh_face_limit:"";
    const report=state.spec.model.conversion||state.inspection?.conversion||{};
    const converted=(report.meshes||[]).filter((mesh)=>mesh.converted&&(mesh.outputs||[]).every((path)=>path.toLowerCase().endsWith(".obj")));
    const objOnly=report.mesh_output_format==="obj";
    const convertedSummary=converted.slice(0,2).map((mesh)=>`${esc(mesh.source||"Mesh")}：${Number(mesh.triangle_count||0).toLocaleString("zh-CN")} 面`).join("；")+(converted.length>2?`；共 ${converted.length} 个转换网格，明细见下方` : "");
    $("#mesh-face-summary").innerHTML=state.inspection?`<span>当前最高面数</span><strong>${maximum.toLocaleString("zh-CN")} <small>三角面 / mesh</small></strong><span>${meshes.length} 个网格${objOnly?" · 输出统一为 OBJ":converted.length?` · ${converted.length} 个网格已转为 OBJ`:""}</span>${converted.length?`<small>转换为 OBJ，保留三角几何。${convertedSummary}</small>`:""}`:"<span>导入模型后显示面数与格式转换信息。</span>";
    const issues=meshBudgetIssues();
    const invalid=state.invalidFields.has("model:mesh_face_limit");
    $("#mesh-face-field").classList.toggle("invalid",invalid||issues.length>0);
    $("#mesh-face-status").textContent=invalid?"请输入正整数面数上限。":issues[0]||(state.inspection?"当前网格在预算内。预算会随草稿和生成清单保存，适用于所有任务。":"导入模型后显示网格统计。");
    $("#mesh-face-status").classList.toggle("mesh-budget-error",invalid||issues.length>0);
    $("#mesh-face-details").hidden=!meshes.length;
    $("#mesh-face-rows").innerHTML=meshes.map((mesh)=>`<tr><td>${esc(mesh.name)}</td><td>${Number(mesh.faces||0).toLocaleString("zh-CN")}</td><td>${Number(mesh.vertices||0).toLocaleString("zh-CN")}</td></tr>`).join("");
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
      if(["stiffness","damping"].includes(key))return null;
      if(["armature","friction","passive_damping"].includes(key)){
        const inspected=state.inspection?.joints.find((item)=>item.name===joint.name);
        return inspected?.[key] ?? null;
      }
    }
    if(value&&typeof value==="object"&&Object.hasOwn(value,joint.name))return value[joint.name];
    return value?.[joint.role] ?? fallback;
  }
  function jointRange(joint) {return state.inspection?.joints.find((item)=>item.name===joint.name)?.range || [-Math.PI,Math.PI];}
  function jointInitial(parameters,joint) {return get(parameters,"init_state.default_joint")?.[joint.name] ?? 0;}
  function setJointValue(key,joint,value) {
    if(["rotor_inertia","gear_ratio"].includes(key)){
      ensureMotorInertia();const scopes=state.spec.motor_inertia.scopes;scopes[state.scope] ||= {};scopes[state.scope][joint.name] ||= {};scopes[state.scope][joint.name][key]=value;return;
    }
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
        const parameters=effective();target.control[key]=Object.fromEntries(state.spec.bindings.joints.filter((j)=>j.name).map((j)=>[j.name,jointValue(parameters,key,j,["stiffness","damping"].includes(key)?null:0)]));
      }
      target.control[key][joint.name]=value;
    }
  }
  const motorFieldHelp={
    effort_limit:{label:"仿真扭矩上限",note:"关节侧：填控制允许的峰值或降额值；若只允许持续额定扭矩，就填额定值。固定限矩，无峰值时长模型。"},
    armature:{label:"关节等效驱动惯量 armature",note:"关节侧：直驱等于转子惯量，减速后为 J转子 × N²。无峰值/额定之分，不填连杆惯量。"},
    passive_damping:{label:"关节被动阻尼",note:"填关节侧黏性阻尼，单位 N·m·s/rad；与第 07 步的 PD 阻尼 Kd 分开。"},
    friction:{label:"关节被动摩擦",note:"填关节侧的被动摩擦扭矩，单位 N·m。"},
    action_scale:{label:"动作缩放",note:"策略动作转换为目标角的缩放量，单位 rad；这里不是电机减速比。"},
    action_clip:{label:"目标角裁剪",note:"填关节目标角的下限、上限，单位 rad；这里不是转速限制。"},
    rotor_inertia:{label:"电机转子惯量（电机轴）",note:"直接填电机转子轴惯量，无峰值/额定之分；关节侧惯量自动按 J转子 × N² 计算，不填连杆惯量。"},
    gear_ratio:{label:"关节总传动比（默认1:1）",note:"默认 1，即 1:1 直驱；有减速时填转子到关节的总比，多级相乘。N = |转子转速 / 关节转速|。"},
    continuous_torque:{label:"额定连续扭矩（关节侧）",note:"填关节侧额定持续扭矩，不填短时峰值。仅作 PD 静态预算，不替换仿真限矩。"},
    velocity_limit:{label:"允许最高工作转速（关节侧）",note:"填关节允许最高工作转速，不填额定转速；可选 rpm / rad/s，系统自动换算。仅检查 PD 是否超速。"},
    reflected_armature:{label:"自动计算的关节惯量",note:"只读：armature = J转子 × N²，PD 测试与生成训练配置立即使用同一结果；不必应用 PD 才生效。"},
  };
  function renderMotorFieldHelp() {
    [["#motor-control-card",["effort_limit","rotor_inertia","gear_ratio","reflected_armature","passive_damping","friction","action_clip"]],["#pd-hardware-details",["continuous_torque","velocity_limit"]]].forEach(([selector,keys])=>{
      const headings=$$(`${selector} thead th`);
      keys.forEach((key,index)=>{
        const heading=headings[index+1];if(!heading)return;
        let note=heading.querySelector("[data-motor-field-help]");
        if(!note){note=document.createElement("small");note.className="motor-column-help";note.dataset.motorFieldHelp=key;note.id=`motor-help-${key}`;heading.append(note);}
        note.textContent=motorFieldHelp[key].note;
      });
    });
  }
  function renderJointControls() {
    const parameters=effective();
    $("#joint-control-scope").textContent=`${pdScopeLabel()} · 这里只填写 PD；站姿、电机和动作分别在第 04、05 步。`;
    function input(joint,index,key,value,min=0) {const help=motorFieldHelp[["clip_min","clip_max"].includes(key)?"action_clip":key];return `<input type="number" step="any" ${min===null?"":`min="${min}"`} value="${value??""}" placeholder="${["stiffness","damping"].includes(key)?"待搜索 / 手动填写":"模型值"}" data-joint-control="${key}" data-joint="${index}" aria-label="${esc(joint.name)} ${esc(help?.label||key)}" ${help?`title="${esc(help.note)}" aria-describedby="motor-help-${["clip_min","clip_max"].includes(key)?"action_clip":key}"`:""}><small data-control-source>${sourceBadge(`control.${key}`,joint)}</small>`;}
    $("#pose-joint-controls").innerHTML=state.spec.bindings.joints.map((joint,index)=>{const range=jointRange(joint),initial=jointInitial(parameters,joint);return `<tr><th>${esc(joint.name||`${joint.leg} ${joint.role}`)}</th><td><div class="pose-control"><input type="range" min="${range[0]}" max="${range[1]}" step="0.001" value="${initial}" data-pose-slider="${index}" aria-label="${esc(joint.name)} 初始角度滑块"><input type="number" min="${range[0]}" max="${range[1]}" step="0.001" value="${initial}" data-joint-control="initial" data-joint="${index}" aria-label="${esc(joint.name)} 初始角度"></div></td><td>${range.join(" / ")}</td></tr>`;}).join("");
    $("#motor-joint-controls").innerHTML=state.spec.bindings.joints.map((joint,index)=>{
      const clip=parameters.control?.action_clip?.[joint.name]||jointRange(joint),h=motorInertiaInput(joint);
      const raw=(key)=>`<input type="number" step="any" min="${key==="gear_ratio"?0.000000000001:0}" max="${key==="gear_ratio"?10000:100}" value="${esc(h[key]??"")}" placeholder="${key==="gear_ratio"?"1（1:1 直驱）":"未知留空"}" data-motor-inertia="${key}" data-joint="${index}" aria-label="${esc(joint.name)} ${esc(motorFieldHelp[key].label)}" title="${esc(motorFieldHelp[key].note)}" aria-describedby="motor-help-${key}">`;
      return `<tr><th>${esc(joint.name||`${joint.leg} ${joint.role}`)}</th><td>${input(joint,index,"effort_limit",jointValue(parameters,"effort_limit",joint,null))}</td><td>${raw("rotor_inertia")}</td><td>${raw("gear_ratio")}</td><td class="motor-inertia-result" data-motor-armature="${esc(joint.name)}" aria-describedby="motor-help-reflected_armature"></td>${["passive_damping","friction"].map((key)=>`<td>${input(joint,index,key,jointValue(parameters,key,joint,null))}</td>`).join("")}<td><div class="clip-inputs">${["clip_min","clip_max"].map((key,i)=>input(joint,index,key,clip[i],null)).join("")}</div></td></tr>`;
    }).join("");
    $("#joint-controls").innerHTML=state.spec.bindings.joints.map((joint,index)=>`<tr><th>${esc(joint.name||`${joint.leg} ${joint.role}`)}</th>${["stiffness","damping"].map((key)=>`<td>${input(joint,index,key,jointValue(parameters,key,joint,null))}</td>`).join("")}</tr>`).join("");
    renderMotorHardwarePreviews();
    renderMotorFieldHelp();renderActionScaling();
  }
  function renderActionScaling() {
    const p=effective(),joints=state.spec.bindings.joints.filter((j)=>j.name),values=joints.map((j)=>jointValue(p,"action_scale",j,null));
    const uniform=values.length?values.every((v)=>v===values[0]):typeof p.control?.action_scale==="number";
    const base=uniform?(values[0]??p.control?.action_scale??0.5):null,hip=p.control?.hip_reduction??1;
    $("#action-scaling-fields").innerHTML=`<div class="form-grid two"><label class="field"><span><span class="field-label-content">基础动作缩放<span class="field-unit">rad</span><span class="parameter-key">control.action_scale</span></span>${sourceBadge("control.action_scale")}</span><input id="action-scale-base" type="number" min="0.000000001" step="any" value="${base??""}" placeholder="逐关节自定义，展开下方查看" aria-label="基础动作缩放"><small>通常填 0.5。统一填写会替换当前作用范围的逐关节基础缩放。</small></label>${renderField("control.hip_reduction",hip)}</div>`;
    $("#action-scaling-summary").innerHTML=uniform?`<strong>最终缩放</strong><span>hip：${pdNumber(base)} × ${pdNumber(hip)} = <b>${pdNumber(base*hip)} rad</b></span><span>thigh / calf：<b>${pdNumber(base)} rad</b></span><small>目标角 = 初始角 + 策略动作 × 最终缩放，再按目标角范围裁剪。hip 系数只乘一次。</small>`:'<strong>当前使用逐关节基础缩放</strong><span>hip 仍额外乘下方设置的髋关节系数；最终数值见逐关节表。</span>';
    $("#action-joint-controls").innerHTML=joints.map((joint)=>{const index=state.spec.bindings.joints.indexOf(joint),value=jointValue(p,"action_scale",joint,null);return `<tr><th>${esc(joint.name)}</th><td><input type="number" min="0.000000001" step="any" value="${value??""}" data-joint-control="action_scale" data-joint="${index}" aria-label="${esc(joint.name)} 基础动作缩放"></td><td>${pdNumber(value*(joint.role==="hip"?hip:1))}</td></tr>`;}).join("");
  }
  function trainingOutputs() {
    return state.plan?.manifest?.training_outputs||Object.keys(state.spec.tasks).map((algorithm)=>({algorithm,experiment_name:`${state.spec.robot.slug}_${algorithm}`,path_template:`logs/rsl_rl/${state.spec.robot.slug}_${algorithm}/<timestamp>[_<run_name>]/`}));
  }
  function renderOutputLayout() {
    const node=$("#output-layout");if(!node)return;const slug=state.spec.robot.slug||"<机器人名>";
    const paths=state.plan?.manifest?.output_layout||[{path:`src/assets/robots/${slug}/`,purpose:"机器人运行 XML、网格与工厂"},{path:`src/config/${slug}/`,purpose:"所选算法配置、必要接口与生成清单"},...Object.keys(state.spec.tasks).map((algorithm)=>({path:`src/tasks/locomotion/${slug}_${algorithm}/`,purpose:taskMeta[algorithm]?.label||algorithm}))];
    node.innerHTML=`<h4>按当前选择生成的目录</h4>${paths.map((item)=>`<div><span>${esc(item.purpose||item.algorithm||"")}</span><code>${esc(item.path)}</code></div>`).join("")}<h4>训练命令的输出位置</h4>${trainingOutputs().map((item)=>`<div><span>${esc(taskMeta[item.algorithm]?.label||item.algorithm)}</span><code>${esc(item.path_template)}</code></div>`).join("")}<p class="field-note">timestamp 是启动训练的时间；可用 --agent.run-name 添加目录后缀。只生成选中的算法。</p>${/_(him|ppo|him_symmetry|him_lpacrl)$/.test(slug)?'<p class="notice warning">目录标识目前带有算法后缀。建议改成机器人名，例如 ri_5036；任务目录会自动附加算法名。</p>':""}`;
  }
  function renderTerrainAnalysis() {
    const node=$("#terrain-analysis-result");if(!node)return;const entry=state.terrainAnalysis;
    if(!entry){node.textContent="查看地图布局不运行仿真；计算实际 geom 会用当前参数在 CPU 构建整张地图。";return;}
    const result=entry.result,s=result.summary||{},g=result.geometry||{};
    node.innerHTML=`${entry.stale?'<p class="notice warning">地形或配置已修改，以下是上次计算结果，请重新计算。</p>':""}<p><strong>${esc(taskMeta[entry.scope]?.label||entry.scope)} · ${esc(entry.terrain)} · seed ${esc(result.seed??42)}</strong></p><div class="terrain-stat-grid"><div><span>实际网格</span><strong>${s.mode==="plane"?"无限平面":`${s.rows??"—"} 行 × ${s.columns??"—"} 列`}</strong></div><div><span>单地块 / m</span><strong>${esc(s.patch_size_m?s.patch_size_m.join(" × "):"—")}</strong></div><div><span>含外围整图 / m</span><strong>${esc(s.mode==="plane"?"无限平面":s.total_size_m?s.total_size_m.join(" × "):"—")}</strong></div><div><span>全局 geom</span><strong>${g.global_geoms??(g.status==="budget_exceeded"?"超出构建预算":"待实际计算")}</strong></div><div><span>机器人 / 地形 geom</span><strong>${g.robot_geoms??"—"} / ${g.terrain_geoms??"—"}</strong></div><div><span>Mesh / Heightfield</span><strong>${g.meshes??"—"} / ${g.heightfields??"—"}</strong></div></div>${(s.sub_terrains||[]).length?`<div class="table-scroll"><table class="data-table"><thead><tr><th>地形类型</th><th>配置权重</th><th>归一化权重</th><th>地图占比</th></tr></thead><tbody>${s.sub_terrains.map((item)=>`<tr><th>${esc(item.label||item.name)}</th><td>${item.weight??"—"}</td><td>${pdNumber(100*(item.normalized_weight??0),1)}%</td><td>${item.layout_share==null?"待实际构建":`${pdNumber(100*item.layout_share,1)}%`}</td></tr>`).join("")}</tbody></table></div>`:""}<p class="field-note">${esc(g.note||result.note||"geom 按实际编译模型统计；配置权重与按列课程的地图占比可能不同。")}${g.status==="budget_exceeded"?`<br>${esc(g.message||g.estimate_note||"请先减少地图行列或提高网格宽度，再实际构建。")}`:""}</p>`;
  }
  async function analyzeTerrain(build) {
    if(!state.inspection){toast("请先导入模型","error");return;}
    if(state.invalidFields.size||state.pendingEdits.size){toast("请先完成当前输入，再计算地形","error");return;}
    const scope=state.scope==="common"?Object.keys(state.spec.tasks)[0]:state.scope,terrain=$("#terrain-analysis-kind").value,revision=state.revision;
    await busy(build?"正在 CPU 构建地图并统计全局 geom…":"正在计算地图布局…",async()=>{const result=await api("/api/terrain/analyze",{method:"POST",body:{spec:generationSpec(),scope,terrain,seed:42,build}});state.terrainAnalysis={result,scope,terrain,stale:revision!==state.revision};renderTerrainAnalysis();});
  }
  function renderConfigTabs() {
    if(state.scope!=="common"&&!state.spec.tasks[state.scope])state.scope="common";
    $("#config-tabs").innerHTML=[{key:"common",label:"公共配置"},...Object.keys(state.spec.tasks).map((key)=>({key,label:taskMeta[key]?.label||key}))].map((item)=>`<button role="tab" aria-selected="${item.key===state.scope}" data-config-scope="${item.key}" class="${item.key===state.scope?"active":""}">${esc(item.label)}</button>`).join("");
    $("#config-scope-note").textContent=state.scope==="common"?"公共修改将应用于全部所选任务。未修改的字段使用各任务原有默认值。3D 预览随当前配置页切换。":`${taskMeta[state.scope]?.label || state.scope} 的专属覆盖；未修改的字段继承公共配置与本模板默认值。`;
  }
  const pdDefaults={frequency_hz:5,damping_ratio:1,pose_delta_rad:0.15,action_amplitude:1,gain_grouping:"auto",joints:{}};
  const pdMethods={model:"完整模型锁定惯量",armature:"电机惯量基线"};
  function pdScopeEntry(create=false,scope=state.scope) {
    if(create){state.spec.pd_analysis ||= {version:1,scopes:{}};state.spec.pd_analysis.scopes ||= {};state.spec.pd_analysis.scopes[scope] ||= {};}
    return state.spec.pd_analysis?.scopes?.[scope];
  }
  function pdOptions(scope=state.scope) {
    const options=merge(pdDefaults,pdScopeEntry(false,scope)?.options||{});
    for(const joint of state.spec.bindings.joints.filter((joint)=>joint.name))options.joints[joint.name]={...(options.joints[joint.name]||{}),...motorInertiaInput(joint,scope)};
    return options;
  }
  function pdFresh() {const session=pdSession("pd");return !!session&&session.report===pdScopeEntry()?.report&&session.fingerprint===pdInputFingerprint()&&!pdPending();}
  function invalidatePd(reason) {
    for(const [scope,entry] of Object.entries(state.spec.pd_analysis?.scopes||{})) {
      if(entry.report)entry.report_status=pdSession("pd",scope)?.fingerprint===pdInputFingerprint(scope)?"computed_in_session":"needs_recompute";
      if(entry.response_report)entry.response_status=pdSession("pdTest",scope)?.fingerprint===pdInputFingerprint(scope)?"computed_in_session":"needs_recompute";
    }
    if(!pdFresh())state.pd.reason=reason;
    if(!pdResponseFresh())state.pdTest.reason=reason;
    renderMotorHardwarePreviews();renderPdInputs();renderPdResults();renderPdResponse();
  }
  function pdNumber(value,digits=4) {
    if(typeof value!=="number"||!Number.isFinite(value))return "未提供";
    if(value!==0&&Math.abs(value)<0.001)return value.toExponential(3);
    return Number(value.toFixed(digits)).toString();
  }
  function pdRatio(value) {return typeof value==="number"&&Number.isFinite(value)?`${pdNumber(value*100,1)}%`:"未提供";}
  function pdRange(value) {return Array.isArray(value)?value.map((v)=>pdNumber(v,2)).join("–"):"未提供";}
  function pdScopeLabel() {return state.scope==="common"?"公共配置":taskMeta[state.scope]?.label||state.scope;}
  function renderMotorHardwarePreviews() {
    const parameters=effective();
    $$("[data-motor-armature]").forEach((node)=>{
      const name=node.dataset.motorArmature,joint=state.spec.bindings.joints.find((item)=>item.name===name);if(!joint)return;
      const h=motorInertiaInput(joint),current=pdControlInput(parameters,"armature",joint).value;
      if(["rotor_inertia","gear_ratio"].some((key)=>state.invalidFields.has(`inertia:${state.scope}:${name}:${key}`))){node.innerHTML='<small>请先修正转子惯量 / 传动比</small>';return;}
      node.innerHTML=`<output>${pdNumber(current,8)}</output><small>${h.rotor_inertia==null?"未提供转子惯量，保留原配置 / 模型假设":`${pdNumber(h.rotor_inertia,8)} × ${pdNumber(h.gear_ratio,5)}²${h.gear_ratio===1?" · 1:1 直驱":""}<br>自动用于 PD 与训练配置`}</small>`;
    });
  }
  function editMotorInertia(input) {
    const key=input.dataset.motorInertia,joint=state.spec.bindings.joints[Number(input.dataset.joint)],invalidKey=`inertia:${state.scope}:${joint.name}:${key}`;
    input.setCustomValidity("");const empty=input.value.trim()==="",value=empty?(key==="gear_ratio"?1:null):Number(input.value);
    if(input.validity.badInput||!empty&&(!Number.isFinite(value)||!input.checkValidity())){input.setCustomValidity("请填写有效转子惯量或正数传动比；直驱为 1");state.invalidFields.add(invalidKey);input.classList.add("pd-invalid");invalidatePd("电机惯量输入无效，请修正。");updateActions();return;}
    state.invalidFields.delete(invalidKey);input.classList.remove("pd-invalid");setJointValue(key,joint,value);changed();renderParameterIndex();
  }
  function resetMotorInertia() {
    if(state.step!==4)return;
    ensureMotorInertia();const imported=state.scope==="common"?state.importDefaults?.motor_inertia?.scopes?.common:undefined;
    if(imported)state.spec.motor_inertia.scopes.common=clone(imported);else delete state.spec.motor_inertia.scopes[state.scope];
    for(const key of state.invalidFields)if(key.startsWith(`inertia:${state.scope}:`))state.invalidFields.delete(key);
  }
  function pdControlInput(parameters,key,joint) {
    const raw=get(parameters,`control.${key}`);
    const value=typeof raw==="object"&&raw!==null ? (Object.hasOwn(raw,joint.name)?raw[joint.name]:raw[joint.role]) : raw;
    if(value==null&&["armature","friction","passive_damping"].includes(key)) {
      return {value:state.inspection?.joints.find((item)=>item.name===joint.name)?.[key]??null,source:'<em class="source-badge model">模型读取</em>'};
    }
    return {value,source:sourceBadge(`control.${key}`,joint)};
  }
  function renderPdInputs() {
    const node=$("#pd-test-inputs");if(!node)return;
    const intro='<h4 id="pd-inputs-title">本次测试使用的输入</h4><p class="field-note">自动测试省去手动挑选 PD 候选；动力学仍依赖导入模型和当前配置，不会自动测出实机惯量。以下显示已生效配置，完成编辑后同步更新。</p>';
    if(!state.inspection){node.innerHTML=`${intro}<p class="muted tiny">导入模型并确认关节映射后，在这里核对实际使用的参数及来源。</p>`;return;}
    const parameters=effective(),options=pdOptions(),joints=state.spec.bindings.joints.filter((joint)=>joint.name);
    let templateArmatures=0,continuousKnown=0,velocityKnown=0;
    const templateArmatureValues=new Set();
    const cell=(key,value,source)=>`<td data-pd-input-key="${key}"><span class="pd-input-value">${pdNumber(value,6)}</span>${source}</td>`;
    const rows=joints.map((joint)=>{
      const hardware=options.joints?.[joint.name]||{};
      const armature=pdControlInput(parameters,"armature",joint);
      const rotorGiven=hardware.rotor_inertia!=null,gearGiven=hardware.gear_ratio!=null;
      const paired=rotorGiven&&gearGiven&&Number.isFinite(hardware.rotor_inertia)&&hardware.rotor_inertia>=0&&Number.isFinite(hardware.gear_ratio)&&hardware.gear_ratio>0;
      if(paired){armature.value=hardware.rotor_inertia*hardware.gear_ratio**2;armature.source='<em class="source-badge manual">电机转子 × N² · 自动生效</em>';}

      if(armature.source.includes("模板默认")){templateArmatures+=1;templateArmatureValues.add(pdNumber(armature.value,6));}
      if(hardware.continuous_torque!=null)continuousKnown+=1;
      if(hardware.velocity_limit!=null)velocityKnown+=1;
      const providedInitial=state.scope==="common"?state.spec.parameters.init_state?.default_joint:merge(state.spec.parameters.init_state?.default_joint||{},state.spec.tasks[state.scope]?.parameters?.init_state?.default_joint||{});
      const initial=Object.hasOwn(providedInitial||{},joint.name)?providedInitial[joint.name]:baseDefaults()?.init_state?.default_joint?.[`${joint.leg}_${joint.role}_joint`];
      const pd=["stiffness","damping"].map((key)=>{const input=pdControlInput(parameters,key,joint);return `<span class="pd-input-pair"><span>${key==="stiffness"?"Kp":"Kd"} ${input.value==null?"未设置":pdNumber(input.value,5)}</span>${input.source}</span>`;}).join("");
      const effort=pdControlInput(parameters,"effort_limit",joint),passive=pdControlInput(parameters,"passive_damping",joint),friction=pdControlInput(parameters,"friction",joint);
      return `<tr data-pd-input-joint="${esc(joint.name)}"><th scope="row"><span class="joint-label">${esc(joint.name)}<small>${esc(joint.leg)} · ${esc(joint.role)}</small></span></th>${cell("armature",armature.value,armature.source)}<td data-pd-input-key="pd">${pd}</td>${cell("effort_limit",effort.value,effort.source)}${cell("passive_damping",passive.value,passive.source)}${cell("friction",friction.value,friction.source)}${cell("initial",initial,sourceBadge("init_state.default_joint",joint))}</tr>`;
    }).join("");
    const masses=Object.values(state.inspection.body_masses||{}).filter((value)=>typeof value==="number"&&Number.isFinite(value)),totalMass=masses.reduce((sum,value)=>sum+value,0);
    const dt=parameters.environment?.sim_dt,control=parameters.control||{},low=control.delay_min_lag??0,high=control.delay_max_lag??0;
    node.innerHTML=`${intro}<div class="pd-input-model"><strong>刚体质量、质心与惯量</strong> <em class="source-badge model">导入模型</em><p>随 XML / URDF 进入 MuJoCo；编译模型共 ${masses.length} 个 body，总质量 ${pdNumber(totalMass,3)} kg。请核对源模型数据是否符合实物。</p><p><code>armature</code> 是额外折算到关节端的电机惯量，与连杆刚体惯量分开。URDF 的 <code>&lt;inertial&gt;</code> 通常描述刚体，不能据此认定已包含电机折算惯量。</p></div>${templateArmatures?`<p id="pd-input-template-warning" class="notice warning">${templateArmatures} / ${joints.length} 个关节的 armature 沿用模板默认值（当前值 ${Array.from(templateArmatureValues).join(" / ")} kg·m²），尚未确认属于当前机器人。可以据此做假设筛选；请在第 05 步填入真实电机转子惯量，由传动比自动折算。</p>`:""}<div class="table-scroll pd-input-scroll"><table class="data-table pd-input-table"><thead><tr><th>关节</th><th>电机折算惯量<br>armature · kg·m²</th><th>当前 PD<br>Kp / Kd</th><th>力矩限幅<br>N·m</th><th>被动阻尼<br>N·m·s/rad</th><th>被动摩擦<br>N·m</th><th>初始角度<br>rad</th></tr></thead><tbody>${rows||'<tr><td colspan="7" class="muted">请先完成关节映射。</td></tr>'}</tbody></table></div><p class="field-note pd-input-timing" data-pd-input-timing>仿真步长 <strong>${pdNumber(dt,6)} s</strong> ${sourceBadge("environment.sim_dt")}；执行器<strong>目标位置</strong>延迟 ${pdNumber(low,0)}–${pdNumber(high,0)} 个物理步（${pdNumber(low*dt*1000,3)}–${pdNumber(high*dt*1000,3)} ms）。延迟保持概率 ${pdNumber(control.delay_hold_prob??0,3)}，更新周期 ${pdNumber(control.delay_update_period??0,0)} 步。这里只延迟位置目标，不延迟 PD 反馈。</p><p class="field-note">关节侧补充参数：额定连续扭矩已填 ${continuousKnown} / ${joints.length}，允许最高工作转速已填 ${velocityKnown} / ${joints.length}；未填项保持未知。连续扭矩仅供高级静态预算比较；转速用于响应超速判定，不钳速、不生成训练限速。已确认的 T–N 离散曲线用于响应测试的速度相关限矩；仅上传图片时保持参考资料。没有温升模型；实机内环频率与实机链路延迟仅记录，不参与本次仿真。</p><div class="pd-input-links"><button type="button" class="button secondary small" data-pd-edit-controls>修改逐关节参数 ↓</button><button type="button" class="button quiet small" data-pd-edit-hardware>补充电机参数 ↓</button></div>`;
  }
  const pendingSpeedValues=new WeakMap();
  function motorSpeedUnit() {return state.spec.workflow?.ui?.motor_speed_unit==="rad/s"?"rad/s":"rpm";}
  function speedFactor(unit=motorSpeedUnit()) {return unit==="rpm"?2*Math.PI/60:1;}
  function speedDisplay(value,unit=motorSpeedUnit()) {return value==null?"":Number((value/speedFactor(unit)).toPrecision(12)).toString();}
  function updateSpeedUnitControls() {
    const unit=motorSpeedUnit();$$("[data-speed-unit]").forEach((input)=>{input.value=unit;});
    $("#batch-speed-unit-field").hidden=state.step!==4||$("#batch-property").value!=="velocity_limit";
    $("#hardware-batch-speed-unit-field").hidden=$("#hardware-batch-field").value!=="velocity_limit";
  }
  function renderHardwareInputs() {
    const options=pdOptions(),unit=motorSpeedUnit();
    $("#hardware-batch-scope").textContent=` 本次批量只更新${pdScopeLabel()}页签内的同类关节。`;
    $("#pd-hardware").innerHTML=state.spec.bindings.joints.filter((joint)=>joint.name).map((joint)=>`<tr><td><span class="joint-label">${esc(joint.name)}<small>${joint.leg} · ${joint.role}</small></span></td>${["continuous_torque","velocity_limit"].map((key)=>{
      const velocity=key==="velocity_limit",value=options.joints?.[joint.name]?.[key],factor=velocity?speedFactor(unit):1;
      return `<td><input type="number" step="any" min="${1e-12/factor}" max="${(velocity?100000:1000000)/factor}" placeholder="未知留空" title="${esc(motorFieldHelp[key].note)}${velocity?` 当前单位：${unit}。`:""}" aria-describedby="motor-help-${key}" value="${esc(velocity?speedDisplay(value,unit):value??"")}" ${velocity?`data-speed-value-unit="${unit}"`:""} data-pd-joint="${esc(joint.name)}" data-pd-hardware="${key}" aria-label="${esc(joint.name)} ${esc(motorFieldHelp[key].label)} ${velocity?unit:"N·m"}"></td>`;
    }).join("")}</tr>`).join("")||'<tr><td colspan="3" class="muted">导入模型并确认关节映射后填写。</td></tr>';
    updateSpeedUnitControls();
  }
  function changeMotorSpeedUnit(input) {
    const before=motorSpeedUnit(),after=input.value;if(!["rpm","rad/s"].includes(after)){updateSpeedUnitControls();return;}if(after===before)return;
    if([...(state.invalidFields||[])].some((key)=>key.startsWith(`pd:${state.scope}:`)&&key.includes(":")&&["continuous_torque","velocity_limit"].some((field)=>key.endsWith(`:${field}`)))){updateSpeedUnitControls();toast("请先修正电机补充参数，再切换单位","error");return;}
    for(const [fieldSelector,valueSelector] of [["#batch-property","#batch-value"],["#hardware-batch-field","#hardware-batch-value"]]){
      const value=$(valueSelector);if($(fieldSelector).value!=="velocity_limit"||value.value.trim()===""||!Number.isFinite(Number(value.value)))continue;
      const cached=pendingSpeedValues.get(value),canonical=cached?.display===value.value&&cached?.unit===before?cached.canonical:Number(value.value)*speedFactor(before);
      value.value=speedDisplay(canonical,after);pendingSpeedValues.set(value,{canonical,unit:after,display:value.value});
    }
    const progress=workflow.ensure(state.spec);progress.ui ||= {};progress.ui.motor_speed_unit=after;
    renderHardwareInputs();renderMotorFieldHelp();
  }
  function applyHardwareBatch(key,role,value) {
    const canonical=key==="velocity_limit"?value*speedFactor():value,high=key==="velocity_limit"?100000:1000000;
    if(!Number.isFinite(canonical)||canonical<=0||canonical>high){toast(`请填写大于 0 且不超过 ${key==="velocity_limit"?speedDisplay(high):high} ${key==="velocity_limit"?motorSpeedUnit():"N·m"} 的数值`,"error");return;}
    const matching=state.spec.bindings.joints.filter((joint)=>joint.role===role&&joint.name);
    if(!matching.length){toast("请先完成关节映射","error");return;}
    const replaced=new Set(matching.map((joint)=>`pd:${state.scope}:${joint.name}:${key}`));
    if([...state.invalidFields].some((field)=>field.startsWith(`pd:${state.scope}:`)&&["continuous_torque","velocity_limit"].some((name)=>field.endsWith(`:${name}`))&&!replaced.has(field))){toast("请先修正其他电机补充参数，再批量应用","error");return;}
    const entry=pdScopeEntry(true);entry.options ||= {};entry.options.joints ||= {};
    for(const joint of matching){entry.options.joints[joint.name] ||= {};entry.options.joints[joint.name][key]=canonical;state.invalidFields.delete(`pd:${state.scope}:${joint.name}:${key}`);}
    changed();renderHardwareInputs();renderMotorFieldHelp();
    toast(`已将${motorFieldHelp[key].label}应用到 ${pdScopeLabel()} 的 ${matching.length} 条腿 ${role} 关节`,"success");
  }
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
    renderHardwareInputs();
    state.invalidFields.forEach((key)=>{if(key.startsWith(`pd:${state.scope}:`))state.invalidFields.delete(key);});
    renderMotorFieldHelp();renderMotorHardwarePreviews();renderPdInputs();renderPdResults();renderPdResponse();renderMotorCurves();updatePdActions();
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
    const testing=pdTestActive();
    button.disabled=state.busy||state.pd.running||testing||!state.inspection||state.invalidFields.size>0;
    button.textContent=state.pd.running?"计算中…":"计算 PD 候选 ↗";
    const apply=$("#pd-apply");if(apply)apply.disabled=state.busy||state.pd.running||testing||state.invalidFields.size>0||!pdFresh()||!pdScopeEntry()?.report?.proposals?.[state.pd.method];
    $("#pd-test").disabled=state.busy||state.pd.running||testing||!state.inspection||state.invalidFields.size>0||state.pdUi.inputConfirmed[state.scope]!==pdInputFingerprint();
    $("#pd-test").textContent=testing?"正在搜索并测试…":"搜索并测试 PD ↗";
    $("#pd-test-stop").hidden=!testing;
    $("#pd-test-stop").disabled=state.pdTest.starting||!state.pdTest.job?.id||!!state.pdTest.job?.stopping;
    const testApply=$("#pd-test-apply");if(testApply)testApply.disabled=state.busy||state.pd.running||testing||state.invalidFields.size>0||!pdResponseFresh()||!pdSelectedCandidate()?.summary?.eligible||pdReferenceOnly(pdSelectedCandidate());
  }
  function editPdInput(input) {
    const option=input.dataset.pdOption;const name=input.dataset.pdJoint;const key=option||input.dataset.pdHardware;if(!key)return;
    const invalidKey=`pd:${state.scope}:${name||"global"}:${key}`;
    input.setCustomValidity("");
    const options=pdOptions();const optional=!!name||["hardware_pd_hz","hardware_delay_ms"].includes(key);
    const empty=input.value.trim()==="";
    if(input.validity.badInput||(empty&&!optional)||(!empty&&(!Number.isFinite(Number(input.value))||!input.checkValidity()))){
      input.setCustomValidity("请填写符合范围的有限数值");input.classList.add("pd-invalid");state.invalidFields.add(invalidKey);invalidatePd("分析输入尚未填写完整，请修正后重新计算。");updateActions();return;
    }
    state.invalidFields.delete(invalidKey);input.classList.remove("pd-invalid");
    const target=name?(options.joints[name] ||= {}):options;
    if(empty)delete target[key];else target[key]=Number(input.value)*(key==="velocity_limit"?speedFactor(input.dataset.speedValueUnit):1);
    pdScopeEntry(true).options=options;changed();
  }
  async function analyzePd() {
    if(state.pd.running||pdTestActive())return;
    const issues=localIssues(3);if(issues.length){displayError(new Error(issues.join("\n")));return;}
    const options=pdOptions();
    pdScopeEntry(true).options=clone(options);changed();invalidatePd("重新计算中，之前的候选暂不可应用。");
    const revision=state.revision,epoch=state.pd.epoch,scope=state.scope,fingerprint=pdInputFingerprint(scope);const inputSnapshot={model:clone(state.spec.model),bindings:clone(state.spec.bindings),parameters:effective(),options:clone(options)};
    state.pd.running=true;clearError();renderPdResults();updateActions();
    try{
      const report=await api("/api/pd/analyze",{method:"POST",body:{spec:state.spec,scope,options}});
      if(fingerprint!==pdInputFingerprint(scope)||scope!=="common"&&!state.spec.tasks[scope]){toast("计算期间配置或作用范围已变化，返回结果已丢弃；请重新计算。");return;}
      if(!Array.isArray(report.joints)||!report.proposals?.model)throw Error("PD 分析未返回完整候选与应用参数");
      const entry=pdScopeEntry(true,scope);entry.report=report;state.pd.sessions[scope]={report,fingerprint,inputSnapshot};entry.report_created_at=new Date().toISOString();entry.report_status="computed_in_session";
      state.pd.report=report;state.pd.revision=revision;state.pd.scope=scope;state.pd.reason="";state.pd.inputSnapshot=inputSnapshot;
      if(!report.proposals[state.pd.method])state.pd.method="model";
      // Analysis metadata changes the generated manifest, so an older file review is invalid.
      state.plan=null;state.spec.confirmed=false;$("#confirm-review").checked=false;$("#file-review").hidden=true;$("#diff-section").hidden=true;$("#plan-issues").replaceChildren();$("#plan-summary").textContent="分析报告已更新，请重新生成文件预览";
      toast("PD 候选已就绪，请查看惯量、力矩预算与应用前后值","success");
    }catch(error){displayError(error);}
    finally{state.pd.running=false;renderPdResults();updateSummary();updateActions();}
  }
  function applyPd() {
    if(!pdFresh()||state.pd.running||pdTestActive()||state.invalidFields.size){toast("请根据当前有效输入重新计算候选后应用","error");return;}
    const entry=pdScopeEntry();const method=state.pd.method;const proposal=entry.report.proposals?.[method];if(!proposal?.control)return;
    const before=Object.fromEntries(state.spec.bindings.joints.map((joint)=>[joint.name,{kp:jointValue(effective(),"stiffness",joint),kd:jointValue(effective(),"damping",joint),armature:jointValue(effective(),"armature",joint)}]));
    const target=targetParameters();target.control ||= {};
    for(const key of ["stiffness","damping","armature"])target.control[key]=clone(proposal.control[key]);
    entry.applied={method,applied_at:new Date().toISOString(),scope:state.scope,options:clone(entry.options),input_snapshot:clone(pdSession("pd")?.inputSnapshot||state.pd.inputSnapshot),before,proposal:clone(proposal),report:clone(entry.report),hardware_calibrated:false};
    if(pdSession("pd"))pdSession("pd").fingerprint=pdInputFingerprint();
    state.pdUi.phase=3;state.pdUi.mode="manual";
    changed();renderJointControls();renderParameters();state.pd.reason="候选已应用到当前配置；报告保留应用依据，继续分析前请重新计算。";renderPdResults();
    toast(`已将 ${pdMethods[method]} 的 KP、KD 与 armature 应用到${pdScopeLabel()}`,"success");
  }
  function downloadPd() {
    const entry=pdScopeEntry();if(!entry?.report)return;
    const payload={version:1,scope:state.scope,robot:clone(state.spec.robot),model:clone(state.spec.model),options:clone(entry.options||{}),report_status:pdFresh()?"computed_in_session":"needs_recompute",report_created_at:entry.report_created_at,report:entry.report,applied:entry.applied||null,hardware_calibrated:false};
    const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:"application/json;charset=utf-8"}));
    const link=document.createElement("a");link.href=url;link.download=`${state.spec.robot.slug||"robot"}_pd_${state.scope}.json`;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function pdTestActive() {return state.pdTest.starting||["queued","running"].includes(state.pdTest.job?.status);}
  function pdResponseFresh() {const session=pdSession("pdTest");return !!session&&session.report===pdScopeEntry()?.response_report&&session.fingerprint===pdInputFingerprint()&&!pdPending();}
  function pdSelectedCandidate() {return pdScopeEntry()?.response_report?.candidates?.find((candidate)=>candidate.id===state.pdTest.candidateId);}
  function pdReferenceOnly(candidate,report=pdScopeEntry()?.response_report) {return !!candidate&&(candidate.reference_only===true||(candidate.id==="current"&&report?.pd_reference?.reference_only===true));}
  function pdDefaultCandidateId(report) {
    const candidates=report?.candidates||[];
    const recommended=candidates.find((candidate)=>candidate.id===report.recommended_id&&!pdReferenceOnly(candidate,report));
    return recommended?.id||candidates.find((candidate)=>candidate.id!=="current"&&!pdReferenceOnly(candidate,report))?.id||null;
  }
  function pdSeconds(value) {return typeof value==="number"&&Number.isFinite(value)?`${pdNumber(value,3)} s`:"未到位";}
  function pdPercent(value) {return typeof value==="number"&&Number.isFinite(value)?`${pdNumber(value,1)}%`:"未提供";}
  function renderPdResponse() {
    const node=$("#pd-test-results");if(!node)return;
    const test=state.pdTest,job=test.job,report=pdScopeEntry()?.response_report;
    const labels={queued:"已排队",running:"测试中",passed:"已完成",failed:"失败",stopped:"已停止",skipped:"未运行"};
    const jobScope=job?.scope==="common"?"公共配置":taskMeta[job?.scope]?.label||job?.scope;
    $("#pd-test-status").textContent=test.starting?"正在提交受控单关节仿真…":job?`${jobScope} · ${labels[job.status]||job.status}${job.stopping?" · 正在停止…":""}${job.discarded?" · 配置或作用范围已变化，返回结果已丢弃，请重新测试。":""}${job.error?` · ${job.error}`:""}`:"导入模型并确认映射后，即可搜索 PD；审阅并应用后才写入配置。";
    $("#pd-test-log-details").hidden=!job;
    if(job){const log=$("#pd-test-log"),bottom=log.scrollTop+log.clientHeight>=log.scrollHeight-40;log.textContent=job.log||"等待测试日志…";if(bottom)log.scrollTop=log.scrollHeight;}
    node.hidden=!report;
    if(!report){node.replaceChildren();$("#pd-application-preview").replaceChildren();$("#pd-application-actions").replaceChildren();updatePdActions();renderPdFlow();return;}
    const fresh=pdResponseFresh(),candidates=Array.isArray(report.candidates)?report.candidates:[];
    if(!candidates.some((candidate)=>candidate.id===test.candidateId))test.candidateId=pdDefaultCandidateId(report);
    const selected=pdSelectedCandidate(),joints=selected?.joints||[];
    if(!joints.some((joint)=>joint.name===test.jointName))test.jointName=joints[0]?.name;
    const joint=joints.find((item)=>item.name===test.jointName),conditions=report.conditions||{},control=selected?.proposal?.control;
    const current=(binding,key)=>jointValue(effective(),key,binding,null);
    const warnings=report.warnings||[];
    node.innerHTML=`${fresh?"":'<div class="notice warning">历史仿真结果 · 当前配置、作用范围或会话已变化，重新测试前不能应用。</div>'}${report.pd_reference?.reference_only?'<div class="notice warning">Kp 20.0 / Kd 0.5 仅在本次测试副本中作为比较基线；该基线没有自动写入配置。它不参与推荐，请选择搜索得到的候选。</div>':""}<div class="pd-section-heading"><div><h4>${report.recommended_id?"候选已比较，请审阅响应与增益":"本次未找到全部关节满足指标的候选"}</h4><p>3 / 5 / 8 Hz × ζ 0.7 / 1 / 1.3，并与已有增益或测试专用基线比较。频率与阻尼比是候选设计目标，下表是仿真实际响应。</p></div><button id="pd-test-download" class="button secondary small">下载测试报告 ↓</button></div><div class="table-scroll pd-result-scroll"><table class="data-table pd-response-table"><thead><tr><th>选择候选 / 设计目标</th><th>通过关节</th><th>最慢到位时间</th><th>最大过冲</th><th>最大饱和时间占比</th><th>最大末端误差</th></tr></thead><tbody>${candidates.map((candidate)=>{const summary=candidate.summary||{};return `<tr class="${candidate.id===test.candidateId?"pd-selected-row":""}"><td><label class="pd-candidate-choice"><input type="radio" name="pd-response-candidate" value="${esc(candidate.id)}" data-pd-test-candidate ${candidate.id===test.candidateId?"checked":""}><span><strong>${esc(candidate.label||candidate.id)}${pdReferenceOnly(candidate,report)?' <em class="source-badge template">仅对照 · 不可应用</em>':""}${candidate.id===report.recommended_id?' <em class="source-badge model">本次推荐</em>':""}</strong><small>${candidate.frequency_hz==null?(report.pd_reference?.baseline_used?"测试专用比较基线 · 未写入配置":"当前配置 · 无指定设计目标"):`${pdNumber(candidate.frequency_hz,1)} Hz / ζ ${pdNumber(candidate.damping_ratio,2)}`}</small></span></label></td><td>${summary.passed_joints??"—"} / ${summary.total_joints??"—"}<small>${pdReferenceOnly(candidate,report)?"仅作测试对照":summary.eligible?"满足本次指标":"请检查未通过关节"}</small></td><td>${pdSeconds(summary.worst_settling_time_s)}</td><td>${pdPercent(summary.max_overshoot_pct)}</td><td>${pdRatio(summary.max_saturation_fraction)}</td><td>${pdNumber(summary.max_final_error_rad,5)} rad</td></tr>`;}).join("")}</tbody></table></div><p class="field-note">到位时间、过冲、饱和占比与末端误差取各关节两个方向的较差结果。推荐只比较本次有限候选与受控工况，应用后仍需检查站立和任务运行。</p><div class="pd-chart-heading"><h4>查看关节响应</h4><label>关节<select id="pd-test-joint" aria-label="响应曲线关节">${joints.map((item)=>`<option value="${esc(item.name)}" ${item.name===test.jointName?"selected":""}>${esc(item.name)} · ${esc(item.leg)} ${esc(item.role)}</option>`).join("")}</select></label><label>方向<select id="pd-test-direction" aria-label="响应曲线测试方向"><option value="worst" ${test.direction==="worst"?"selected":""}>较差方向</option><option value="-1" ${test.direction==="-1"?"selected":""}>负向阶跃</option><option value="1" ${test.direction==="1"?"selected":""}>正向阶跃</option></select></label></div><div class="pd-response-chart">${pdResponseChart(joint,test.direction)}</div><div class="pd-chart-legend"><span class="pd-line-position">关节角度</span><span class="pd-line-target">目标角度</span><span class="pd-line-torque">电机力矩</span><span>选中关节 · ${joint?.metrics?.passed?"通过本次指标":"未通过本次指标"}</span></div><details class="subsection" open><summary>审阅所选候选的实际增益 <span>${esc(selected?.label||"")}</span></summary><p class="field-note">KP：N·m/rad；KD：N·m·s/rad。仅写入下面三项，作用范围为 ${esc(pdScopeLabel())}。</p><div class="table-scroll pd-result-scroll"><table class="data-table pd-response-review"><thead><tr><th>关节</th><th>KP · 现值 → 候选</th><th>KD · 现值 → 候选</th><th>armature kg·m² · 现值 → 候选</th><th>关节测试结果</th></tr></thead><tbody>${joints.map((item)=>{const binding=state.spec.bindings.joints.find((entry)=>entry.name===item.name)||item;return `<tr><td>${esc(item.name)}</td>${["stiffness","damping","armature"].map((key)=>`<td>${pdNumber(current(binding,key))} → <strong>${pdNumber(control?.[key]?.[item.name])}</strong></td>`).join("")}<td>${item.metrics?.passed?"通过":"未通过"}<small>到位 ${pdSeconds(item.metrics?.settling_time_s)} · 过冲 ${pdPercent(item.metrics?.overshoot_pct)}</small></td></tr>`;}).join("")}</tbody></table></div></details>${videoMarkup([...(selected?.videos||[]),...(report.videos||[])])?`<section class="test-videos pd-test-videos"><h4>本次测试视频</h4><p class="field-note">根据实际测得的关节轨迹回放，固定机体与其他关节；没有重力和接触，不代表站立或行走。视频在生成成功后自动删除，测试报告和日志保留。</p>${videoMarkup([...(selected?.videos||[]),...(report.videos||[])])}</section>`:'<p class="field-note">没有可播放的视频，生成任务后会自动清理视频；响应曲线和报告继续保留。</p>'}${warnings.length?`<details class="subsection pd-warning-block"><summary>测试提示 <span>${warnings.length} 项</span></summary><ul>${warnings.map((warning)=>`<li>${esc(issueText(warning))}</li>`).join("")}</ul></details>`:""}<div class="inline-actions pd-response-apply"><span class="muted tiny">${!fresh?"历史结果不能应用，请重新测试。":pdReferenceOnly(selected,report)?"测试基准仅用于对照，不能应用；请选择搜索候选。":!selected?.summary?.eligible?"所选候选未通过全部关节的本次指标，请查看曲线与日志。":"审阅候选后点击应用；不会自动改写配置。"}</span><button id="pd-test-apply" class="button primary">应用此候选到${esc(pdScopeLabel())}</button></div><details class="subsection"><summary>测试工况与说明 <span>REPRODUCIBLE CONDITIONS</span></summary><p class="field-note">CPU · 种子 ${esc(conditions.seed??42)} · 每方向 ${pdNumber(conditions.duration_s,1)} s · 仿真步长 ${pdNumber(conditions.sim_dt,5)} s。固定基座、锁定其他关节，关闭重力和接触；使用配置中的惯量、被动阻尼、摩擦、限矩与执行器延迟。</p>${(report.method_notes||[]).map((note)=>`<p class="field-note">${esc(issueText(note))}</p>`).join("")}<pre class="json-pre">${esc(JSON.stringify(conditions,null,2))}</pre></details>`;
    const organized=window.MigrationPdFlow.organize({root:node,applicationPreview:$("#pd-application-preview"),applicationActions:$("#pd-application-actions"),report,selected,fresh,scopeLabel:pdScopeLabel(),applied:pdAppliedMatches()&&pdScopeEntry()?.applied?.candidate_id===selected?.id});
    const advance=document.createElement("button"),canReview=fresh&&selected?.summary?.eligible&&!pdReferenceOnly(selected);
    advance.type="button";advance.className="button primary small";advance.dataset.pdPhase=canReview?"3":"0";
    advance.textContent=canReview?"审阅并应用所选候选 →":"返回核对测试输入 →";
    organized.summary?.append(advance);
    updatePdActions();renderPdFlow();
  }
  function pdResponseChart(joint,direction) {
    const trial=direction==="worst"?null:joint?.directions?.find((item)=>String(item.direction)===direction);
    const trace=trial?.trace||(direction==="worst"?joint?.trace:null);
    const points=(trace?.time||[]).map((time,index)=>({time,position:trace.position?.[index],target:trace.target?.[index],torque:trace.torque?.[index]})).filter((point)=>Object.values(point).every(Number.isFinite));
    if(!points.length)return '<p class="field-note">该关节没有可用曲线，请查看日志与测试结果。</p>';
    const duration=Math.max(points.at(-1).time,0.001),positions=points.flatMap((point)=>[point.position,point.target]),low=Math.min(...positions),high=Math.max(...positions),padding=Math.max((high-low)*0.15,0.003),min=low-padding,max=high+padding;
    const peak=Math.max(...points.map((point)=>Math.abs(point.torque)),0.001),x=(time)=>62+time/duration*694,y=(value)=>180-(value-min)/(max-min)*144,torqueY=(value)=>274-value/peak*38;
    const path=(key,mapY)=>points.map((point,index)=>`${index?"L":"M"}${x(point.time).toFixed(2)},${mapY(point[key]).toFixed(2)}`).join(" ");
    const grid=[0,0.5,1].map((fraction)=>{const time=duration*fraction,px=x(time);return `<line x1="${px}" y1="30" x2="${px}" y2="316" class="pd-chart-grid"/><text x="${px}" y="337" text-anchor="middle">${pdNumber(time,2)} s</text>`;}).join("");
    return `<svg viewBox="0 0 800 350" role="img" aria-label="${esc(joint.name)} 的关节角度、目标角度与电机力矩随时间曲线"><title>${esc(joint.name)} · ${direction==="worst"?"较差方向":direction==="1"?"正向阶跃":"负向阶跃"}</title>${grid}<text x="14" y="18">角度 / rad</text><text x="14" y="224">力矩 / N·m</text><line x1="62" y1="180" x2="756" y2="180" class="pd-chart-axis"/><line x1="62" y1="274" x2="756" y2="274" class="pd-chart-axis"/><text x="54" y="42" text-anchor="end">${pdNumber(max,3)}</text><text x="54" y="180" text-anchor="end">${pdNumber(min,3)}</text><text x="54" y="242" text-anchor="end">${pdNumber(peak,2)}</text><text x="54" y="312" text-anchor="end">${pdNumber(-peak,2)}</text><path d="${path("target",y)}" class="pd-chart-target"/><path d="${path("position",y)}" class="pd-chart-position"/><path d="${path("torque",torqueY)}" class="pd-chart-torque"/></svg>`;
  }
  async function testPdResponse() {
    if(state.pd.running||pdTestActive())return;
    if(state.pdUi.inputConfirmed[state.scope]!==pdInputFingerprint()){setPdPhase(0);toast("请先核对并确认本次测试输入。","error");return;}
    const issues=localIssues(3);if(issues.length){displayError(new Error(issues.join("\n")));return;}
    const options=pdOptions();
    pdScopeEntry(true).options=clone(options);changed();
    const test=state.pdTest,scope=state.scope,revision=state.revision,epoch=test.epoch,fingerprint=pdInputFingerprint(scope);
    const inputSnapshot={model:clone(state.spec.model),bindings:clone(state.spec.bindings),parameters:effective(),options:clone(options)};
    test.starting=true;test.job=null;clearTimeout(test.timer);clearError();renderPdResponse();updateActions();
    try{
      const job=await api("/api/pd/test",{method:"POST",body:{spec:clone(state.spec),scope,options}});
      if(!job.job_id&&!job.id)throw Error("自动测试未返回任务编号");
      test.job={...job,id:job.job_id||job.id,status:job.status||"queued",scope,revision,epoch,inputSnapshot,fingerprint};
      pollPdResponse();
    }catch(error){displayError(error);}
    finally{test.starting=false;renderPdResponse();updateActions();}
  }
  async function pollPdResponse() {
    const test=state.pdTest;clearTimeout(test.timer);if(!test.job?.id||test.polling)return;
    const id=test.job.id;test.polling=true;
    try{
      const result=await api(`/api/jobs/${encodeURIComponent(id)}`);if(test.job?.id!==id)return;
      test.job={...test.job,...result,id};const job=test.job;
      if(["queued","running"].includes(job.status))test.timer=setTimeout(pollPdResponse,1100);
      else if(job.status==="passed"){
        if(job.fingerprint!==pdInputFingerprint(job.scope)||job.scope!=="common"&&!state.spec.tasks[job.scope]){job.discarded=true;toast("测试期间配置或作用范围已变化，返回结果已丢弃；请重新测试。");}
        else{
          const report=job.result;if(!Array.isArray(report?.candidates)||!report.candidates.length)throw Error("自动测试未返回完整候选与曲线");
          const entry=pdScopeEntry(true,job.scope);entry.response_report=report;test.sessions[job.scope]={report,fingerprint:job.fingerprint,inputSnapshot:job.inputSnapshot};entry.response_created_at=new Date().toISOString();entry.response_status="computed_in_session";
          test.report=report;test.revision=job.revision;test.scope=job.scope;test.inputSnapshot=job.inputSnapshot;test.reason="";test.candidateId=pdDefaultCandidateId(report);test.jointName=null;test.direction="worst";if(job.scope===state.scope)state.pdUi.phase=2;
          state.plan=null;state.spec.confirmed=false;$("#confirm-review").checked=false;$("#file-review").hidden=true;$("#diff-section").hidden=true;$("#plan-issues").replaceChildren();$("#plan-summary").textContent="自动测试报告已更新，请重新生成文件预览";
          toast(report.recommended_id?"PD 响应已比较，请查看曲线与增益，再选择是否应用":"测试已完成，本次没有全部关节满足指标的候选，请查看结果",report.recommended_id?"success":"");
        }
      }else if(job.status==="failed")toast(job.error||"自动测试失败，请查看日志","error");
    }catch(error){
      if(test.job?.id!==id)return;
      test.job.error=error.message;
      if(["queued","running"].includes(test.job.status))test.timer=setTimeout(pollPdResponse,3000);else displayError(error);
    }finally{test.polling=false;renderPdResponse();updateSummary();updateActions();}
  }
  async function stopPdResponse() {
    const test=state.pdTest;if(!test.job?.id||!pdTestActive()||test.job.stopping)return;
    const id=test.job.id;test.job.stopping=true;renderPdResponse();
    try{await api(`/api/jobs/${encodeURIComponent(id)}/stop`,{method:"POST"});if(!test.polling)pollPdResponse();}
    catch(error){displayError(error);}
    finally{if(test.job?.id===id)test.job.stopping=false;renderPdResponse();}
  }
  function applyPdResponse() {
    const candidate=pdSelectedCandidate();
    if(pdReferenceOnly(candidate)){toast("测试基准仅用于对照，不能应用到机器人；请选择搜索候选。","error");return;}
    if(!pdResponseFresh()||pdTestActive()||state.pd.running||state.invalidFields.size||!candidate?.summary?.eligible){toast("请先按当前配置完成测试，再选择通过本次指标的候选","error");return;}
    const control=candidate.proposal?.control;if(!control)return;
    const entry=pdScopeEntry(),before=Object.fromEntries(state.spec.bindings.joints.map((joint)=>[joint.name,{kp:jointValue(effective(),"stiffness",joint),kd:jointValue(effective(),"damping",joint),armature:jointValue(effective(),"armature",joint)}]));
    const target=targetParameters();target.control ||= {};
    for(const key of ["stiffness","damping","armature"])target.control[key]=clone(control[key]);
    entry.applied={method:"response",candidate_id:candidate.id,candidate_label:candidate.label,applied_at:new Date().toISOString(),scope:state.scope,options:clone(entry.options),input_snapshot:clone(pdSession("pdTest")?.inputSnapshot||state.pdTest.inputSnapshot),before,proposal:clone(candidate.proposal),response_summary:clone(candidate.summary),conditions:clone(entry.response_report.conditions),hardware_calibrated:false};
    if(pdSession("pdTest"))pdSession("pdTest").fingerprint=pdInputFingerprint();
    state.pdUi.phase=3;changed();renderJointControls();renderParameters();renderPdResponse();toast(`已将 ${candidate.label} 的 KP、KD 与 armature 应用到${pdScopeLabel()}`,"success");
  }
  function downloadPdResponse() {
    const entry=pdScopeEntry();if(!entry?.response_report)return;
    const payload={version:1,scope:state.scope,robot:clone(state.spec.robot),model:clone(state.spec.model),options:clone(entry.options||{}),report_status:pdResponseFresh()?"computed_in_session":"needs_recompute",report_created_at:entry.response_created_at,report:entry.response_report,applied:entry.applied||null,hardware_calibrated:false};
    const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:"application/json;charset=utf-8"})),link=document.createElement("a");
    link.href=url;link.download=`${state.spec.robot.slug||"robot"}_pd_response_${state.scope}.json`;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function localMediaUrl(value) {
    if(typeof value!=="string"||!value)return null;
    try{const url=new URL(value,window.location.origin);return url.origin===window.location.origin&&["http:","https:"].includes(url.protocol)?url.href:null;}catch(_error){return null;}
  }
  function videoMarkup(videos=[]) {
    return videos.filter((item)=>localMediaUrl(item?.url)).map((item)=>`<figure class="test-video"><figcaption>${esc(item.label||"测试视频")}</figcaption><video controls preload="metadata" playsinline src="${esc(localMediaUrl(item.url))}" ${item.poster&&localMediaUrl(item.poster)?`poster="${esc(localMediaUrl(item.poster))}"`:""}>当前浏览器不支持视频播放。</video><a href="${esc(localMediaUrl(item.url))}" target="_blank" rel="noopener">打开视频 ↗</a></figure>`).join("");
  }
  function renderMotorCurves() {
    const list=$("#motor-curves-list");if(!list)return;
    const curves=state.spec.motor_curves||[],joints=state.spec.bindings.joints.filter((joint)=>joint.name);
    list.innerHTML=curves.map((curve,index)=>{
      const imageUrl=localMediaUrl(curve.image_url||curve.image?.url),points=Array.isArray(curve.points)?curve.points:[];
      const pointText=points.map((point)=>`${pdNumber(point.speed_rad_s*60/(2*Math.PI),6)}, ${pdNumber(point.torque_nm,6)}`).join("\n");
      return `<article class="motor-curve" data-motor-curve-index="${index}"><div class="motor-curve-heading"><h4>${esc(curve.label||`曲线 ${index+1}`)}</h4><span class="tag">${curve.confirmed&&curve.enabled!==false?"已确认 · 响应测试限矩":"图片参考 / 待确认"}</span><button class="button quiet small" type="button" data-remove-motor-curve="${index}">移除</button></div>${imageUrl?`<a class="motor-curve-image" href="${esc(imageUrl)}" target="_blank" rel="noopener"><img src="${esc(imageUrl)}" alt="${esc(curve.label||"电机扭矩转速曲线")}" loading="lazy"></a>`:'<p class="notice warning">曲线图片地址不可用，请重新上传。</p>'}<div class="form-grid two"><label class="field"><span>名称 / 工况说明</span><input data-motor-curve-field="label" value="${esc(curve.label||"")}" maxlength="120" aria-label="曲线 ${index+1} 名称"></label><label class="field"><span>母线电压 <em>V</em></span><input type="number" min="0.01" step="any" data-motor-curve-field="voltage_v" value="${esc(curve.voltage_v??"")}" placeholder="数值启用时必填" aria-label="曲线 ${index+1} 电压"></label><label class="field"><span>曲线描述的输出轴</span><select data-motor-curve-field="shaft" aria-label="曲线 ${index+1} 输出轴"><option value="joint" ${curve.shaft!=="module"?"selected":""}>最终关节输出轴</option><option value="module" ${curve.shaft==="module"?"selected":""}>电机 / 模组输出轴（厂家曲线所标轴）</option></select></label><label class="field"><span>关联关节</span><select multiple size="4" data-motor-curve-field="joint_names" aria-label="曲线 ${index+1} 关联关节">${joints.map((joint)=>`<option value="${esc(joint.name)}" ${(curve.joint_names||[]).includes(joint.name)?"selected":""}>${esc(joint.name)} · ${esc(joint.leg)} ${esc(joint.role)}</option>`).join("")}</select><small>按 Ctrl / ⌘ 多选；<button type="button" class="button quiet small" data-motor-curve-all="${index}">关联全部 12 关节</button></small></label>${curve.shaft==="module"?`<label class="field"><span>曲线所标输出轴 → 关节额外减速比</span><input type="number" min="0.000001" step="any" data-motor-curve-field="extra_reduction" value="${esc(curve.extra_reduction??1)}" aria-label="曲线 ${index+1} 额外减速比"><small>该轴直接带关节填 1；只计算其后的传动。厂家曲线若已含内部减速，不重复计入。这里的额外比与裸转子的总 N 分开填写，不自动联动。</small></label><label class="field"><span>额外传动效率 <em>0–1</em></span><input type="number" min="0.000001" max="1" step="any" data-motor-curve-field="efficiency" value="${esc(curve.efficiency??1)}" aria-label="曲线 ${index+1} 传动效率"></label>`:""}</div><details class="subsection" ${points.length?"open":""}><summary>录入曲线离散点 <span>${points.length} 点 · rpm / N·m</span></summary><p class="field-note">每行填“转速 rpm, 扭矩 N·m”，按转速递增。从 0 rpm 起，到扭矩 0 的最高转速止，扭矩不能随转速上升。请从指定电压、所选输出轴的同一条曲线取点，峰值和连续曲线不要混用。曲线零扭矩端点是包络边界，与上方“允许最高工作转速”分开填写；这里不模拟峰值持续时间。</p><label class="field"><span>离散点 <em>rpm, N·m</em></span><textarea class="motor-curve-points" data-motor-curve-points aria-label="曲线 ${index+1} 离散点" placeholder="0, 30&#10;100, 30&#10;250, 15&#10;350, 0">${esc(pointText)}</textarea></label><p class="field-note">测试使用速度绝对值上的对称、分段线性包络；它不包含回馈制动、温升或电池压降。上传图片本身不改变限矩。生成任务中仍使用配置的固定力矩限幅。</p></details><label class="confirm-box motor-curve-confirm"><input type="checkbox" data-motor-curve-confirm ${curve.confirmed&&curve.enabled!==false?"checked":""}><span>我已核对电压、输出轴、关联关节和曲线数值，将这些离散点用于 PD 响应测试的速度相关限矩。</span></label></article>`;
    }).join("")||'<p class="field-note motor-curves-empty">还没有上传曲线。没有曲线时，搜索使用逐关节配置的固定力矩限幅。</p>';
  }
  function curveIssue(curve,index) {
    if(!Number.isFinite(curve.voltage_v)||curve.voltage_v<=0)return "请填写曲线对应的正数电压";
    if(curve.shaft==="module"&&(!Number.isFinite(curve.extra_reduction)||curve.extra_reduction<=0||!Number.isFinite(curve.efficiency)||curve.efficiency<=0||curve.efficiency>1))return "请填写正数的额外传动比与 0–1 之间的传动效率";
    if(!curve.joint_names?.length)return "请至少关联一个关节";
    if((state.spec.motor_curves||[]).some((other,i)=>i!==index&&other.confirmed&&other.enabled!==false&&(other.joint_names||[]).some((name)=>curve.joint_names.includes(name))))return "同一关节只能启用一条 T–N 曲线，请先取消其他曲线的确认";
    const points=curve.points||[];
    if(points.length<2)return "请至少录入两个曲线点";
    if(points[0].speed_rad_s!==0||points[0].torque_nm<=0||points.at(-1).torque_nm!==0)return "曲线需要从 0 rpm 的正扭矩开始，并以扭矩 0 的最高转速结束";
    if(points.some((point,i)=>!Number.isFinite(point.speed_rad_s)||!Number.isFinite(point.torque_nm)||point.speed_rad_s<0||point.torque_nm<0||(i>0&&(point.speed_rad_s<=points[i-1].speed_rad_s||point.torque_nm>points[i-1].torque_nm))))return "转速需要严格递增，扭矩非负且不递增";
    return null;
  }
  async function uploadMotorCurve() {
    const file=$("#motor-curve-file").files?.[0];if(!file){toast("请先选择一张曲线图片","error");return;}
    const voltage=$("#motor-curve-voltage"),label=$("#motor-curve-label").value.trim();
    if(!voltage.value||!voltage.checkValidity()){voltage.reportValidity();toast("请填写这张曲线对应的电压","error");return;}
    await busy("正在保存电机曲线图片…",async()=>{
      const body=new FormData();body.append("file",file);if(voltage.value)body.append("voltage_v",voltage.value);if(label)body.append("label",label);
      const result=await api("/api/motor-curves",{method:"POST",body}),asset=result.asset||result;
      if(!asset.id||!localMediaUrl(asset.url))throw Error("上传未返回有效的本地图片");
      state.spec.motor_curves ||= [];state.spec.motor_curves.push({id:asset.id,image_id:asset.id,image_url:asset.url,image:clone(asset),label:label||file.name,voltage_v:voltage.value?Number(voltage.value):null,shaft:"joint",extra_reduction:1,efficiency:1,joint_names:[],points:[],confirmed:false,enabled:false});
      $("#motor-curve-file").value="";changed();renderMotorCurves();toast("曲线图片已保存；录入并确认离散点后才参与响应测试","success");
    });
  }
  function editMotorCurve(input) {
    const article=input.closest("[data-motor-curve-index]");if(!article)return;
    const index=Number(article.dataset.motorCurveIndex),curve=state.spec.motor_curves?.[index];if(!curve)return;
    const errorKey=`motor-curve:${curve.id}`;
    try {
      input.setCustomValidity("");const next=clone(curve);
      if(input.matches("[data-motor-curve-confirm]")) {
        if(input.checked){const error=curveIssue(next,index);if(error)throw Error(error);}
        next.confirmed=input.checked;next.enabled=input.checked;
      } else {
        next.confirmed=false;next.enabled=false;
        if(input.matches("[data-motor-curve-points]")) {
          next.points=input.value.trim()?input.value.trim().split(/\n+/).map((line,i)=>{const values=line.trim().split(/[,，;；\s]+/);if(values.length!==2||values.some((value)=>!value||!Number.isFinite(Number(value))||Number(value)<0))throw Error(`第 ${i+1} 行需要非负的转速 rpm 和扭矩 N·m`);return {speed_rad_s:Number(values[0])*2*Math.PI/60,torque_nm:Number(values[1])};}):[];
        } else {
          const key=input.dataset.motorCurveField;
          if(!key)return;
          if(key==="joint_names")next[key]=Array.from(input.selectedOptions,(option)=>option.value);
          else if(["voltage_v","extra_reduction","efficiency"].includes(key)){if(input.value&&!input.checkValidity())throw Error("请输入范围内的有效数值");next[key]=input.value?Number(input.value):null;}
          else next[key]=input.value;
          if(key==="shaft"&&next.shaft==="joint"){next.extra_reduction=1;next.efficiency=1;}
        }
      }
      state.spec.motor_curves[index]=next;state.invalidFields.delete(errorKey);changed();renderMotorCurves();renderPdInputs();renderParameterIndex();
    } catch(error) {if(input.matches("[data-motor-curve-confirm]"))input.checked=false;else state.invalidFields.add(errorKey);input.setCustomValidity(error.message);input.reportValidity();toast(error.message,"error");updateActions();}
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
    if(path==="terrain.border_width"){result.min=0;result.step="any";}
    if(path==="control.hip_reduction"){result.min=0.000000001;result.max=1;}
    if(key==="proportion"){result.min=0;result.step="any";}
    if(path==="terrain.border_height")result.min=0.000000001;
    if(path.startsWith("terrain.sub_terrains.")&&["step_width","platform_width","grid_width","horizontal_scale","vertical_scale","noise_step","downsampled_scale"].includes(key)){result.min=0.000000001;result.step="any";}
    return Object.entries(result).map(([name,value])=>`${name}="${value}"`).join(" ");
  }
  function terrainFieldError(path,value) {
    if(path==="terrain.difficulty_range"&&(!Array.isArray(value)||value.length!==2||!value.every(Number.isFinite)||value[0]>value[1]))return "生成难度区间需要两个端点：最小难度, 最大难度，例如 0, 1；不是 0,1,2… 的等级列表";
    if(path==="terrain.size"&&(!Array.isArray(value)||value.length!==2||!value.every((v)=>Number.isFinite(v)&&v>0)))return "单地块尺寸需要两个正数：长, 宽，单位 m，例如 8, 8";
    if(["terrain.num_rows","terrain.num_cols"].includes(path)&&(!Number.isInteger(value)||value<=0))return "地形行数与列数必须是正整数";
    if(path==="terrain.border_width"&&(!Number.isFinite(value)||value<0))return "整图外围平地宽度必须是非负数，单位 m";
    if(path==="terrain.curriculum"&&typeof value!=="boolean")return "按行生成难度课程必须选择开启或关闭";
    if(path==="terrain.border_height"&&(!Number.isFinite(value)||value<=0))return "整图外围平地厚度必须为正有限数值";
    if(path.startsWith("terrain.sub_terrains.")){
      const key=pathSegments(path).at(-1);
      if(["holes","merge_similar_heights","inverted","scale_with_difficulty"].includes(key))return typeof value==="boolean"?null:"此项必须选择开启或关闭";
      if(key.endsWith("_range")){
        if(!Array.isArray(value)||value.length!==2||!value.every(Number.isFinite)||value[0]>value[1])return "区间需要两个有序有限端点：最小值, 最大值";
        if(!["noise_range","slope_range"].includes(key)&&value[0]<0)return "高度区间的端点不可为负数";
        return null;
      }
      if(["max_merge_distance","num_waves"].includes(key))return Number.isInteger(value)&&value>0?null:"此项必须为正整数";
      if(key==="downsampled_scale"&&value===null)return null;
      if(!Number.isFinite(value)||value<0)return "此项必须为非负有限数值";
      if(["step_width","platform_width","grid_width","horizontal_scale","vertical_scale","noise_step","downsampled_scale"].includes(key)&&value===0)return "宽度、采样间距和步长必须大于零";
    }
    return null;
  }
  function terrainConfigurationIssues(parameters) {
    const errors=parameterLeaves(parameters.terrain||{},"terrain").map(({path,value})=>terrainFieldError(path,value)).filter(Boolean),t=parameters.terrain||{},types=t.sub_terrains;
    if(types){
      if(Object.values(types).every((item)=>Number.isFinite(item.proportion))&&Object.values(types).reduce((sum,item)=>sum+item.proportion,0)<=0)errors.push("子地形权重总和必须大于 0");
      if(Array.isArray(t.size)&&t.size.length===2&&t.size.every(Number.isFinite))for(const [name,item] of Object.entries(types))if(!item.holes&&2*(item.border_width||0)+(item.platform_width||0)>=Math.min(...t.size))errors.push(`${name} 的两侧边宽与中央平台必须容纳在单地块尺寸内`);
    }
    return [...new Set(errors)];
  }
  function renderField(path,value) {
    const key=pathSegments(path).at(-1);const metadata=fieldMetadata(path),terrainError=terrainFieldError(path,value);
    const readonly=path==="control.armature"||fixedKeys.has(key)||metadata.readonly||(path.startsWith("rewards.")&&pathSegments(path)[2]==="params");
    const label=metadata.label&&metadata.label!==key?metadata.label:fieldLabels[key]||key;
    const unit=metadata.unit||unitFor(path);const title=`${label}${unit?` / ${unit}`:""}`;
    const attr=`data-param="${esc(path)}" ${readonly?"disabled":""} aria-label="${esc(title)}"`;
    let input="",full=false;
    let options=metadata.options;
    if(!options && key==="activation")options=["elu","relu","selu","crelu","lrelu","tanh","sigmoid"];
    if(!options && key==="schedule")options=["adaptive","fixed"];
    if(!options&&path.startsWith("environment.mujoco."))options=({integrator:["euler","implicitfast"],cone:["pyramidal","elliptic"],jacobian:["dense","sparse","auto"],solver:["newton","cg"]})[key];
    if(path==="environment.mujoco.enableflags")input=`<select ${attr} data-type="mujoco-flags" multiple size="3">${["energy","invdiscrete","sleep"].map((flag)=>`<option value="${flag}" ${(value||[]).includes(flag)?"selected":""} ${flag==="sleep"&&effective().environment?.mujoco?.solver!=="newton"?"disabled":""}>${flag}</option>`).join("")}</select><small>支持 energy、invdiscrete、sleep；sleep 仅可与 newton 求解器一起启用。按 Ctrl / ⌘ 多选或取消。</small>`;
    else if(path==="environment.mujoco.disableflags")input=`<input type="text" ${attr} data-type="string-array" value="${esc((value||[]).join(", "))}" placeholder="以逗号分隔，留空表示不额外关闭">`;
    else if(options)input=`<select ${attr} data-type="${typeof value}">${options.map((option)=>{const v=typeof option==="object"?option.value:option;const l=typeof option==="object"?option.label:option;return `<option value="${esc(v)}" ${v===value?"selected":""}>${esc(l)}</option>`;}).join("")}</select>`;
    else if(typeof value==="boolean")input=`<select ${attr} data-type="boolean"><option value="true" ${value?"selected":""}>开启 · true</option><option value="false" ${!value?"selected":""}>关闭 · false</option></select>`;
    else if(typeof value==="number")input=`<input type="number" ${attr} data-type="number" ${numericConstraints(key,path,metadata)} value="${value}" required>`;
    else if(typeof value==="string")input=`<input type="text" ${attr} data-type="string" value="${esc(value)}">`;
    else if(Array.isArray(value)&&value.every((v)=>typeof v==="number"))input=`<input type="text" ${attr} data-type="number-array" value="${value.join(", ")}" placeholder="${path==="terrain.difficulty_range"?"最小难度, 最大难度，例如 0, 1（两个数）":path==="terrain.size"?"长, 宽，例如 8, 8（单位 m）":"以逗号分隔，如 0, 0, 0.3"}">`;
    else if(Array.isArray(value)&&value.every((v)=>typeof v==="string"))input=`<input type="text" ${attr} data-type="string-array" value="${esc(value.join(", "))}" placeholder="以逗号分隔">`;
    else {full=true;input=`<textarea ${attr} data-type="json" class="json-input">${esc(JSON.stringify(value,null,2))}</textarea>`;}
    return `<label class="field ${full?"full":""} ${terrainError?"invalid":""}"><span><span class="field-label-content">${esc(label)}${unit?`<span class="field-unit">${esc(unit)}</span>`:""}<span class="parameter-key">${esc(displayPath(path))}</span></span>${readonly?'<em class="source-badge template">固定契约</em>':sourceBadge(path)}</span>${input}${metadata.description?`<small>${esc(metadata.description)}</small>`:""}${terrainError?`<small class="validation-message">${esc(terrainError)}</small>`:""}</label>`;
  }
  function flattenFields(object,prefix="") {
    return Object.entries(object||{}).flatMap(([key,value])=>{
      const segment=encodeSegment(key);const path=prefix?`${prefix}.${segment}`:segment;
      if(jointPaths.has(path))return [];
      if(value && typeof value==="object"&&!Array.isArray(value)&&Object.keys(value).length)return flattenFields(value,path);
      return [{path,value}];
    });
  }
  function renderRewardWeights(rewards) {
    return `<p class="field-note">只修改奖励权重。函数参数保留当前模板值，足端、关节及碰撞对象由生成器按机器人绑定适配。步态周期、姿态容差等参数不会随权重自动重新标定，需要其他方案时应单独审阅。</p><div class="reward-weight-grid">${Object.entries(rewards||{}).map(([name,term])=>`<section class="reward-weight-row"><h4>${esc(name)}</h4>${renderField(`rewards.${encodeSegment(name)}.weight`,term.weight)}<details class="subsection"><summary>模板函数参数 <span>只读</span></summary><pre class="json-pre reward-template-params">${esc(JSON.stringify(term.params||{},null,2))}</pre></details></section>`).join("")}</div>`;
  }
  function parameterEntries() {
    function walk(object,prefix="") {
      return Object.entries(object||{}).flatMap(([key,value])=>{
        const path=prefix?`${prefix}.${encodeSegment(key)}`:encodeSegment(key);
        const parts=pathSegments(path),joint=jointPaths.has(path),reward=parts[0]==="rewards"&&parts[2]==="params";
        if(!joint&&!reward&&value&&typeof value==="object"&&!Array.isArray(value)&&Object.keys(value).length)return walk(value,path);
        const metadata=fieldMetadata(path),readonly=reward||fixedKeys.has(key)||metadata.readonly||path==="control.armature";
        return [{path,value,joint,readonly,label:fieldLabels[key]||metadata.label||key,group:parts[0],target:joint?(fieldStep(path)===3?"pose-control-card":fieldStep(path)===4?"motor-control-card":"joint-control-card"):`parameter-group-${parts[0]}-${fieldStep(path)}`,description:path==="control.armature"?"自动计算：电机转子惯量 × 关节总传动比²（只读）":reward?"模板函数参数（自动适配机器人绑定）":metadata.description||""}];
      });
    }
    const entries=walk(effective());
    entries.push({path:"model.mesh_face_limit",label:"单个网格面数上限",group:"model",target:"mesh-face-field",description:"导入时按最高面数初始化，可手动修改"});
    entries.push({path:"motor_curves",label:"电机 T–N 曲线 / 电压 / 输出轴",group:"motor",target:"motor-curves-card",description:"图片、离散点、关节关联与额外减速"});
    for(const key of ["rotor_inertia","gear_ratio","continuous_torque","velocity_limit"])entries.push({path:`${["rotor_inertia","gear_ratio"].includes(key)?"motor_inertia":"pd_analysis"}.${key}`,label:motorFieldHelp[key].label,group:"motor",target:["rotor_inertia","gear_ratio"].includes(key)?"motor-control-card":"pd-hardware-details",description:motorFieldHelp[key].note});
    return entries;
  }
  function renderParameterIndex() {
    const node=$("#parameter-index");if(!node)return;
    const entries=parameterEntries(),query=($("#parameter-search").value||"").trim().toLowerCase();
    const matches=entries.filter((entry)=>`${displayPath(entry.path)} ${entry.label} ${groupMeta[entry.group]?.label||entry.group} ${entry.description}`.toLowerCase().includes(query));
    const readonly=entries.filter((entry)=>entry.readonly).length,joint=entries.filter((entry)=>entry.joint).length;
    $("#parameter-count").textContent=`${entries.length} 项配置入口`;
    $("#parameter-coverage").textContent=`可编辑 ${entries.length-readonly} 项（含 ${joint} 类逐关节设置），只读 ${readonly} 项。高级设置已收在折叠区域；索引涵盖当前模板已接入的参数，固定观测结构与算法接口保持契约。`;
    $("#parameter-search-status").textContent=query?`找到 ${matches.length} 项；点击参数即可展开并定位。`:"奖励项只编辑权重，函数参数可在只读区域查看。";
    if(query)$("#parameter-index-details").open=true;
    node.innerHTML=matches.length?`<div class="table-scroll"><table class="data-table parameter-index-table"><thead><tr><th>参数与位置</th><th>状态 / 来源</th></tr></thead><tbody>${matches.map((entry)=>`<tr><td><button type="button" class="parameter-jump" data-parameter-jump="${esc(entry.path)}" data-parameter-target="${esc(entry.target)}"><strong>${esc(entry.label)}</strong><code>${esc(displayPath(entry.path))}</code><small>${esc(groupMeta[entry.group]?.label||({model:"模型资源",motor:"电机补充"})[entry.group]||entry.group)}${entry.description?` · ${esc(entry.description)}`:""}</small></button></td><td>${entry.readonly?'<em class="source-badge template">只读模板 / 契约</em>':entry.joint?'<em class="source-badge model">逐关节编辑</em>':entry.group==="motor"?'<em class="source-badge manual">可选人工补充</em>':sourceBadge(entry.path)}</td></tr>`).join("")}</tbody></table></div>`:'<p class="field-note">没有匹配项。可尝试参数原名；未接入的字段不会在索引中冒充可编辑配置。</p>';
  }
  function jumpToParameter(button) {
    const path=button.dataset.parameterJump,entry=parameterEntries().find((item)=>item.path===path);
    let node=$$("[data-param]").find((item)=>item.dataset.param===path)||$(`#${button.dataset.parameterTarget}`);
    if(entry?.group==="rewards"&&pathSegments(path)[2]==="params")node=$$(".reward-weight-row").find((item)=>$("h4",item)?.textContent===pathSegments(path)[1])||node;
    if(!node)return;
    const panel=node.closest("[data-panel]");if(panel)showStep(Number(panel.dataset.panel));
    if(path.startsWith("control.stiffness")||path.startsWith("control.damping")){state.pdUi.mode="manual";setPdPhase(3);$("#pd-manual-editor").open=true;}
    for(let parent=node;parent;parent=parent.parentElement)if(parent.tagName==="DETAILS")parent.open=true;

    node.scrollIntoView({behavior:"smooth",block:"center"});node.classList.add("parameter-highlight");setTimeout(()=>node.classList.remove("parameter-highlight"),1800);
    if(!node.disabled&&(node.matches("input,select,textarea")))node.focus({preventScroll:true});
  }
  function renderParameters() {
    const parameters=effective(),leaves=flattenFields(parameters),unknown=Object.keys(parameters).filter((key)=>!groupMeta[key]);
    for(let index=3;index<=10;index++){
      const groups=new Map();
      for(const field of leaves.filter((field)=>fieldStep(field.path)===index)){const group=pathSegments(field.path)[0];if(!groups.has(group))groups.set(group,[]);groups.get(group).push(field);}
      $(`#parameter-step-${index}`).innerHTML=Array.from(groups.entries()).map(([group,fields])=>{
        const meta={...(groupMeta[group]||{label:fieldLabels[group]||group,sub:group})};
        if(group==="control")meta.label=index===4?"动作公共设置":"控制时序与执行器延迟";
        if(group==="observations")meta.label=index===5?"观测延迟":"观测与噪声";
        if(group==="rewards")return `<div id="parameter-group-rewards-9" class="card parameter-card"><div class="card-top"><div><h3>奖励权重</h3><p>其余参数按模板保留。</p></div></div>${renderRewardWeights(parameters.rewards)}</div>`;
        const nested=new Map();for(const field of fields){const parts=pathSegments(field.path),sub=parts.length>2?parts[1]:"";if(!nested.has(sub))nested.set(sub,[]);nested.get(sub).push(field);}
        const contents=Array.from(nested.entries()).map(([sub,items])=>{
          if(group==="terrain"&&sub==="sub_terrains"){
            const types=new Map();for(const field of items){const name=pathSegments(field.path)[2];if(!types.has(name))types.set(name,[]);types.get(name).push(field);}
            return `<div class="terrain-type-list">${Array.from(types.entries()).map(([name,fields],i)=>{const metadata=fieldMetadata(fields[0].path),weight=fields.find((field)=>pathSegments(field.path).at(-1)==="proportion")?.value;const dimensions=fields.filter((field)=>/step_height_range|step_width|grid_width|grid_height_range/.test(field.path)).map((field)=>`${fieldMetadata(field.path).label||pathSegments(field.path).at(-1)} ${Array.isArray(field.value)?field.value.join("–"):field.value} m`).join(" · ");return `<details class="subsection terrain-type" ${i===0?"open":""}><summary><span><strong>${esc(metadata.terrain_label||name)}</strong><code>${esc(name)}</code></span><span>权重 ${esc(weight??"—")} · ${fields.length} 项</span></summary>${dimensions?`<p class="field-note">${esc(dimensions)}</p>`:""}<div class="form-grid two">${fields.map(({path,value})=>renderField(path,value)).join("")}</div></details>`;}).join("")}</div>`;
          }
          const content=`<div class="form-grid two">${items.map(({path,value})=>renderField(path,value)).join("")}</div>`;
          const label=sub==="sub_terrains"?"各类地形的比例与详细尺寸":({mujoco:"仿真器高级选项",policy:"策略网络 Policy",algorithm:"PPO 与 Estimator",runner:"训练预算 Runner",noise:"观测噪声",base_offset:"机体偏移",joint_offset:"关节偏移",ranges:"指令范围"})[sub]||sub;
          return sub?`<details class="subsection" ${sub==="runner"||group==="commands"||group==="terrain"?"open":""}><summary>${esc(label)} <span>${items.length} 项参数</span></summary>${content}</details>`:content;}).join("");
        return `<div id="parameter-group-${esc(group)}-${index}" class="card parameter-card"><div class="card-top"><div><h3>${esc(meta.label)}</h3><p>${esc(meta.sub||"")}</p></div></div>${group==="terrain"&&fields.some((field)=>field.path==="terrain.difficulty_range")?'<p class="field-note terrain-format-note">难度区间填两个端点（通常为 0, 1），不填等级列表。若要 0–5 共六个等级，将地形行数设为 6。按行课程开启时，每种子地形占一列，实际列数不采用 num_cols；地形尺寸是单地块大小，边宽在整图外围。平地任务未使用生成器时，这些网格参数不生效。</p>':""}${contents}</div>`;
      }).join("");
    }
    renderParameterIndex();updateMujocoFlagOptions();renderTerrainAnalysis();
  }
  function updateMujocoFlagOptions() {
    const flags=effective().environment?.mujoco?.enableflags||[],solver=effective().environment?.mujoco?.solver;
    const select=$$('[data-param]').find((input)=>input.dataset.param==="environment.mujoco.enableflags");
    if(select)for(const option of select.options)option.disabled=option.value==="sleep"&&solver!=="newton";
    const solverSelect=$$('[data-param]').find((input)=>input.dataset.param==="environment.mujoco.solver");
    if(solverSelect)for(const option of solverSelect.options)option.disabled=option.value==="cg"&&flags.includes("sleep");
  }
  function parseParameter(input) {
    const type=input.dataset.type;
    if(input.dataset.param==="environment.mujoco.solver"&&input.value!=="newton"&&(effective().environment?.mujoco?.enableflags||[]).includes("sleep"))throw Error("sleep 仅支持 newton；请先取消 sleep 再切换求解器");
    if(type==="mujoco-flags") {
      const flags=Array.from(input.selectedOptions,(option)=>option.value);
      if(flags.some((flag)=>!["energy","invdiscrete","sleep"].includes(flag)))throw Error("仅支持 energy、invdiscrete、sleep");
      if(flags.includes("sleep")&&effective().environment?.mujoco?.solver!=="newton")throw Error("sleep 仅支持 newton 求解器");
      return flags;
    }
    if(type==="number") {if(!input.value.trim()||!Number.isFinite(Number(input.value)))throw Error("请输入有效数值");if(!input.checkValidity())throw Error(input.validationMessage);const value=Number(input.value),terrainError=terrainFieldError(input.dataset.param,value);if(terrainError)throw Error(terrainError);return value;}
    if(type==="boolean")return input.value==="true";
    if(type==="number-array") {
      const values=input.value.trim().replace(/^\[|\]$/g,"").split(/[,，]/).map((v)=>v.trim());
      if(!values.length||values.some((v)=>!v||!Number.isFinite(Number(v))))throw Error("请用逗号分隔有效数值");
      const numbers=values.map(Number);const path=input.dataset.param;
      const terrainError=terrainFieldError(path,numbers);if(terrainError)throw Error(terrainError);
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
    const issues=localIssues(4);
    if(unconfirmedSteps().length)issues.push("请先确认所有配置步骤，再审阅生成文件。");if(issues.length){displayError(new Error(issues.join("\n")));return;}
    return busy("正在构建任务文件预览…",async()=>{
      const revision=state.revision;const result=await api("/api/plan",{method:"POST",body:{spec:generationSpec()}});
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
    const issues=localIssues(4);
    if(unconfirmedSteps().length)issues.push("请先确认所有配置步骤。");
    if(issues.length){displayError(new Error(issues.join("\n")));return;}
    if(!state.plan||state.planRevision!==state.revision||!state.spec.confirmed){displayError(new Error("请先审阅当前配置的文件预览，并勾选确认"));return;}
    if(hasBlockingIssues(state.plan.issues)){displayError(new Error("文件预览中仍有阻止生成的问题，请先修正"));return;}
    await busy("正在提交生成任务…",async()=>{
      const result=await api("/api/generate",{method:"POST",body:{spec:generationSpec()}});
      $("#generate-title").textContent="正在生成并检查任务";$("#generate-description").textContent="文件暂存、检查与写入过程记录在下方日志中。";
      $("#validation-card").hidden=false;
      if(result.job_id){state.job={id:result.job_id,status:result.status||"queued",mode:"generate"};state.generationRevision=state.revision;pollJob();}
      else {completeGeneration(result);}
    });
  }
  function completeGeneration(result) {
    state.generated=true;state.generatedRevision=state.generationRevision??state.revision;
    state.generatedTaskIds=result.task_ids||result.manifest?.task_ids||state.plan?.task_ids||taskIds();
    workflow.confirm(state.spec,"generate",stepSnapshots().generate,stepSummary(12));
    $("#generate-title").textContent="新任务已生成";$("#generate-description").textContent="独立任务文件已写入仓库，生成清单与验证记录可追溯。";
    $("#generate-button").textContent="任务已生成 ✓";
    const paths=result.paths||result.directories||result.output_dirs||result.output_roots||[];
    $("#generate-result").innerHTML=`<div class="notice success">已生成 ${state.generatedTaskIds.length} 个注册任务。${result.message?`\n${esc(result.message)}`:""}</div>${paths.length?`<ul class="result-paths">${paths.map((path)=>`<li>${esc(path)}</li>`).join("")}</ul>`:""}`;
    const cleanup=result.cleanup;
    if(cleanup){for(const entry of Object.values(state.spec.pd_analysis?.scopes||{})){const report=entry.response_report;if(!report)continue;report.videos=[];report.videos_removed_after_generation=true;for(const candidate of report.candidates||[])candidate.videos=[];}renderPd();if(state.draftId)saveDraft({quiet:true});}
    $("#generation-cleanup").hidden=!cleanup;
    if(cleanup)$("#generation-cleanup").textContent=`中间文件清理：${cleanup.status||"已完成"}${cleanup.removed_files!=null?` · ${cleanup.removed_files} 个文件`:""}${cleanup.removed_bytes!=null?` · ${formatBytes(cleanup.removed_bytes)}`:""}。保留草稿、输入模型、生成清单与曲线资料；本次 PD 视频和生成暂存已清理。${cleanup.message||cleanup.note||""}`;
    $("#validation-card").hidden=false;$("#commands-card").hidden=false;
    $("#validate-task").innerHTML=state.generatedTaskIds.map((id)=>`<option value="${esc(id)}">${esc(id)}</option>`).join("");
    renderCommands();updateSummary();updateActions();toast("任务已生成，CPU 检查结果见验证日志","success");
  }
  function renderCommands() {
    const ids=state.generatedTaskIds;
    $("#training-output-paths").innerHTML=trainingOutputs().map((item)=>`<div><strong>${esc(taskMeta[item.algorithm]?.label||item.algorithm)}</strong><code>${esc(item.path_template)}</code></div>`).join("");
    $("#task-commands").innerHTML=ids.map((id,index)=>{
      const train=`uv run --no-sync python -m scripts.train ${id} --env.scene.num-envs 1024 --agent.seed 42`;
      const play=`uv run --no-sync python -m scripts.play ${id} --agent zero --num-envs 1 --viewer viser`;
      const algorithm=Object.keys(state.spec.tasks).sort((a,b)=>(taskMeta[b]?.suffix.length||0)-(taskMeta[a]?.suffix.length||0)).find((key)=>id.startsWith(`${state.spec.robot.prefix}${taskMeta[key]?.suffix||""}-`))||"him";const output=trainingOutputs().find((item)=>item.algorithm===algorithm);
      return `<details ${index===0?"open":""} class="subsection"><summary>${esc(id)}</summary><div class="command-item"><div class="command-label"><span>训练</span><button class="button quiet small" data-copy-command="${esc(train)}">复制</button></div><pre class="command-code">${esc(train)}</pre><p class="field-note">模型与日志：<code>${esc(output?.path_template||"")}</code><br>默认不设置 run_name；添加 <code>--agent.run-name trial</code> 后目录以 <code>_trial</code> 结尾。</p></div><div class="command-item"><div class="command-label"><span>零动作回放 · 检查环境</span><button class="button quiet small" data-copy-command="${esc(play)}">复制</button></div><pre class="command-code">${esc(play)}</pre></div></details>`;
    }).join("");
  }
  async function runValidation(mode) {
    const task_id=$("#validate-task").value;if(!task_id)return;
    await busy("正在提交验证任务…",async()=>{
      const result=await api("/api/validate",{method:"POST",body:{task_id,mode}});
      state.job={id:result.job_id||result.id,status:result.status||"queued",mode};$("#job-log").textContent="验证已排队…";$("#validation-videos").replaceChildren();$("#validation-videos").hidden=true;pollJob();
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
      const videos=job.result?.videos||job.videos||[],videoHtml=videoMarkup(videos);
      $("#validation-videos").hidden=!videoHtml;
      if(videoHtml&&$("#validation-videos").dataset.job!==job.id){$("#validation-videos").innerHTML=`<h4>短仿真视频</h4><p class="field-note">固定预算仿真记录，仅用于检查模型、控制和任务运行。</p>${videoHtml}`;$("#validation-videos").dataset.job=job.id;}
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
  async function saveDraft({quiet=false}={}) {
    if(!state.inspection){toast("请先导入模型，再保存迁移草稿");return;}
    if(state.invalidFields.size){displayError(new Error("请先修正参数格式，再保存当前草稿"));return;}
    return busy("正在保存草稿与步骤确认…",async()=>{
      const result=await api("/api/drafts",{method:"POST",body:{spec:state.spec,name:state.spec.robot.prefix||"未命名机器人",...(state.draftId?{id:state.draftId}:{})}});
      state.draftId=result.id;if(!quiet)toast("草稿与步骤确认已保存在本项目中","success");return true;
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
      // Reimport refreshes model metadata; a saved global budget remains editable.
      if(spec.model.mesh_face_limit!==undefined)state.spec.model.mesh_face_limit=spec.model.mesh_face_limit;
      state.inspection=imported.inspection;state.importDefaults=clone(imported.spec||state.spec);state.draftId=id;state.modelXml=null;
      ensureMeshBudget();fillImportInput(imported);
      state.spec.version ||= 1;state.spec.parameters ||= {};state.spec.tasks ||= {};state.spec.confirmed=false;
      resetGeneration();state.previewOpen=false;state.invalidFields.clear();resetPreview();state.pd.sessions={};state.pdTest.sessions={};state.pendingEdits.clear();changed();renderAll();showStep(Math.min(11,state.spec.workflow?.last_step??0));toast("已恢复草稿和步骤确认，请重新审阅生成结果","success");
    });
  }
  function resetPreview() {$("#viewer-frame").hidden=true;$("#viewer-frame").src="about:blank";$(".viewer-empty").hidden=false;$("#start-preview").textContent="打开 3D 预览";}
  function resetGeneration() {
    state.generated=false;state.generatedRevision=-1;state.generatedTaskIds=[];state.generationRevision=undefined;
    clearTimeout(state.jobTimer);state.job=null;
    $("#generate-title").textContent="准备生成独立训练任务";$("#generate-description").textContent="目标目录已存在时将拒绝覆盖。文件在检查通过后写入仓库。";
    $("#generate-button").innerHTML='生成任务并检查 <span aria-hidden="true">→</span>';
    $("#generate-result").replaceChildren();$("#validation-card").hidden=true;$("#commands-card").hidden=true;
    $("#generation-cleanup").hidden=true;$("#validation-videos").hidden=true;delete $("#validation-videos").dataset.job;
  }
  function fillImportInput(result) {
    if(result.import_input){
      $("#model-root").value=result.import_input.root||"";
      $("#model-xml").value=result.import_input.xml||"";
    }
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
        const root=$("#model-root").value.trim();const xml=$("#model-xml").value.trim();
        if(!root&&!xml)throw Error("请填写模型目录或完整 XML / URDF 文件路径，两个输入框至少填写一项");
        body={root,xml};
      }
      const result=await api("/api/import",{method:"POST",body});
      if(!result.inspection||!result.spec)throw Error("模型解析未返回完整的配置与检查结果");
      state.pd.sessions={};state.pdTest.sessions={};state.pdUi={phase:0,mode:"search",inputConfirmed:{}};state.pendingEdits.clear();state.spec=result.spec;state.spec.model={...state.spec.model,...result.model};state.spec.parameters ||= {};state.spec.tasks ||= {him:defaultTask("him")};
      if(!Object.keys(state.spec.tasks).length)state.spec.tasks={him:defaultTask("him")};
      state.spec.robot ||= {slug:"",prefix:""};state.spec.bindings ||= {joints:[],feet:[]};
      state.inspection=result.inspection;ensureMeshBudget();fillImportInput(result);state.importDefaults=clone(state.spec);state.draftId=null;state.modelXml=null;state.previewOpen=false;state.invalidFields.clear();resetGeneration();resetPreview();changed();renderAll();
      $("#import-result").scrollIntoView({behavior:"smooth",block:"nearest"});
      toast(state.spec.model.source_format==="urdf"?"URDF 已转为 XML，请先检查转换报告":"模型已解析，请继续选择训练任务","success");
    });
  }
  async function copyText(text) {
    try {await navigator.clipboard.writeText(text);toast("已复制到剪贴板","success");}
    catch(_error){const area=document.createElement("textarea");area.value=text;area.style.position="fixed";area.style.opacity="0";document.body.append(area);area.select();const ok=document.execCommand("copy");area.remove();toast(ok?"已复制到剪贴板":"浏览器暂不允许复制，请手动选择文本",ok?"success":"");}
  }
  function renderAll() {ensureMotorInertia();renderImport();renderMeshBudget();renderTasks();renderBindings();renderConfigTabs();renderParameters();renderJointControls();renderPd();updateSummary();updateActions();}

  // Event delegation keeps dynamically generated forms tied to MigrationSpec.
  $("#step-nav").addEventListener("click",(event)=>{const button=event.target.closest("[data-step]");if(button&&!state.busy)showStep(Number(button.dataset.step));});
  $("#previous-step").addEventListener("click",()=>showStep(state.step-1));
  $("#next-step").addEventListener("click",confirmCurrentStep);
  document.addEventListener("click",(event)=>{const jump=event.target.closest("[data-workflow-jump]");if(jump&&!state.busy)showStep(Number(jump.dataset.workflowJump));});
  $("#pd-design").addEventListener("click",(event)=>{
    const phase=event.target.closest("[data-pd-phase]"),mode=event.target.closest("[data-pd-mode]");
    if(phase&&!phase.disabled)setPdPhase(Number(phase.dataset.pdPhase));
    if(mode){state.pdUi.mode=mode.dataset.pdMode;setPdPhase(mode.dataset.pdMode==="manual"?3:0);}
  });
  $("#pd-input-next").addEventListener("click",()=>{const issues=stepIssues(4).concat(stepIssues(5),stepIssues(3));if(issues.length){displayError(new Error([...new Set(issues)].join("\n")));return;}state.pdUi.inputConfirmed[state.scope]=pdInputFingerprint();setPdPhase(1);updatePdActions();});
  $("#pd-review-next").addEventListener("click",()=>setPdPhase(3));
  $("#import-form").addEventListener("submit",importModel);
  $("#import-result").addEventListener("click",(event)=>{
    if(event.target.closest("[data-show-model-xml]"))showModelXml();
    if(event.target.closest("[data-copy-model-xml]")&&state.modelXml)copyText(state.modelXml.xml);
  });
  function switchImport(mode) {state.importMode=mode;$("#path-fields").hidden=mode!=="path";$("#zip-fields").hidden=mode!=="zip";["path","zip"].forEach((key)=>{$(`#${key}-tab`).classList.toggle("active",key===mode);$(`#${key}-tab`).setAttribute("aria-selected",key===mode);});}
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
  $("#mesh-face-limit").addEventListener("input",(event)=>{
    const input=event.target;input.setCustomValidity("");const limit=Number(input.value);
    const valid=input.value.trim()!==""&&Number.isSafeInteger(limit)&&limit>0;
    state.spec.model.mesh_face_limit=valid?limit:null;
    if(valid)state.invalidFields.delete("model:mesh_face_limit");
    else {state.invalidFields.add("model:mesh_face_limit");input.setCustomValidity("单个网格面数上限需要填写正整数");}
    changed();renderMeshBudget({keepInput:true});
  });
  $("#config-tabs").addEventListener("click",(event)=>{const button=event.target.closest("[data-config-scope]");if(button&&button.dataset.configScope!==state.scope){state.scope=button.dataset.configScope;state.pdUi.phase=pdScopeEntry()?.response_report?2:0;renderConfigTabs();renderParameters();renderJointControls();renderPd();if(state.previewOpen)schedulePose();}});
  document.addEventListener("change",(event)=>{if(event.target.matches("[data-speed-unit]"))changeMotorSpeedUnit(event.target);});
  $("#batch-property").addEventListener("change",updateSpeedUnitControls);
  $("#hardware-batch-field").addEventListener("change",updateSpeedUnitControls);
  $("#hardware-apply-batch").addEventListener("click",()=>{const input=$("#hardware-batch-value");if(input.value.trim()===""||input.validity.badInput){toast("请先填写批量值","error");return;}applyHardwareBatch($("#hardware-batch-field").value,$("#hardware-batch-role").value,Number(input.value));});
  $("#main-content").addEventListener("focusout",(event)=>{const input=event.target;if(input.dataset.motorInertia==="gear_ratio"&&input.value.trim()==="")input.value="1";});
  $("#main-content").addEventListener("input",(event)=>{if(event.target.dataset.motorInertia)editMotorInertia(event.target);else if(event.target.dataset.pdOption||event.target.dataset.pdHardware)editPdInput(event.target);});
  $("#pd-design").addEventListener("click",(event)=>{if(event.target.closest("#pd-download"))downloadPd();if(event.target.closest("#pd-apply"))applyPd();if(event.target.closest("#pd-test-download"))downloadPdResponse();if(event.target.closest("#pd-test-apply"))applyPdResponse();if(event.target.closest("[data-pd-edit-controls]"))showStep(4);if(event.target.closest("[data-pd-edit-hardware]")){showStep(4);$("#pd-hardware-details").open=true;}});
  $("#pd-design").addEventListener("change",(event)=>{if(event.target.id==="pd-method"){state.pd.method=event.target.value;renderPdResults();}if(event.target.matches("[data-pd-grouping]")){const options=pdOptions();options.gain_grouping=event.target.value;pdScopeEntry(true).options=options;changed();}if(event.target.matches("[data-pd-test-candidate]")){state.pdTest.candidateId=event.target.value;renderPdResponse();}if(event.target.id==="pd-test-joint"){state.pdTest.jointName=event.target.value;renderPdResponse();}if(event.target.id==="pd-test-direction"){state.pdTest.direction=event.target.value;renderPdResponse();}});
  $("#pd-analyze").addEventListener("click",analyzePd);
  $("#pd-test").addEventListener("click",testPdResponse);
  $("#pd-test-stop").addEventListener("click",stopPdResponse);
  $("#motor-curve-upload").addEventListener("click",uploadMotorCurve);
  $("#motor-curves-list").addEventListener("change",(event)=>{if(event.target.matches("[data-motor-curve-field], [data-motor-curve-points], [data-motor-curve-confirm]"))editMotorCurve(event.target);});
  $("#motor-curves-list").addEventListener("input",()=>{invalidatePd("电机曲线正在编辑，请完成后重新搜索 PD。");updatePdActions();});
  $("#motor-curves-list").addEventListener("click",(event)=>{
    const removeButton=event.target.closest("[data-remove-motor-curve]"),allButton=event.target.closest("[data-motor-curve-all]");
    if(removeButton){const index=Number(removeButton.dataset.removeMotorCurve),curve=state.spec.motor_curves?.[index];if(!curve)return;state.spec.motor_curves.splice(index,1);state.invalidFields.delete(`motor-curve:${curve.id}`);changed();renderMotorCurves();}
    if(allButton){const curve=state.spec.motor_curves?.[Number(allButton.dataset.motorCurveAll)];if(!curve)return;curve.joint_names=state.spec.bindings.joints.filter((joint)=>joint.name).map((joint)=>joint.name);curve.confirmed=false;curve.enabled=false;changed();renderMotorCurves();}
  });
  $("#parameter-search").addEventListener("input",renderParameterIndex);
  $("#parameter-index").addEventListener("click",(event)=>{const button=event.target.closest("[data-parameter-jump]");if(button)jumpToParameter(button);});
  // An unfinished or invalid edit also invalidates results, before its change event.
  $("#main-content").addEventListener("input",(event)=>{const input=event.target;if(input.dataset.param){state.pendingEdits.add(`${state.scope}:${input.dataset.param}`);updatePdActions();}if(input.dataset.poseSlider!==undefined)applyJointInput(input,true);});
  $("#main-content").addEventListener("change",(event)=>{
    const input=event.target;if(!input.dataset.param)return;
    try{input.setCustomValidity("");const value=parseParameter(input);state.pendingEdits.delete(`${state.scope}:${input.dataset.param}`);updateFieldError(input,null);set(targetParameters(),input.dataset.param,value);changed({pose:input.dataset.param.startsWith("init_state")});const badge=input.closest(".field").querySelector(".source-badge");if(badge){badge.textContent="人工修改";badge.className="source-badge manual";}if(input.dataset.param.startsWith("control"))renderJointControls();if(input.dataset.param.startsWith("environment.mujoco."))updateMujocoFlagOptions();renderParameterIndex();}
    catch(error){updateFieldError(input,error.message);toast(error.message,"error");updateSummary();}
  });
  function applyJointInput(input,fromSlider=false) {
    const index=Number(fromSlider?input.dataset.poseSlider:input.dataset.joint);const key=fromSlider?"initial":input.dataset.jointControl;const joint=state.spec.bindings.joints[index];
    if(input.value===""&&["stiffness","damping"].includes(key)) {
      state.invalidFields.delete(`joint:${state.scope}:${index}:${key}`);setJointValue(key,joint,null);changed();renderPdInputs();renderParameterIndex();return;
    }
    if(input.value===""||!Number.isFinite(Number(input.value))||!input.checkValidity()){input.reportValidity();state.invalidFields.add(`joint:${state.scope}:${index}:${key}`);updateActions();return;}
    state.invalidFields.delete(`joint:${state.scope}:${index}:${key}`);setJointValue(key,joint,Number(input.value));
    if(key==="initial"){
      const peer=fromSlider?$(`[data-joint-control="initial"][data-joint="${index}"]`):$(`[data-pose-slider="${index}"]`);if(peer)peer.value=input.value;
    }
    changed({pose:key==="initial"});
    const badge=input.closest("td")?.querySelector("[data-control-source]");
    if(badge)badge.innerHTML=sourceBadge(`control.${key}`,joint);
    renderParameterIndex();if(key==="action_scale")renderActionScaling();
  }
  $("#main-content").addEventListener("change",(event)=>{if(event.target.dataset.jointControl)applyJointInput(event.target);});

  $("#apply-batch").addEventListener("click",()=>{
    const input=$("#batch-value");if(input.value===""||!Number.isFinite(Number(input.value))){toast("请填写批量设置的数值","error");return;}
    const key=$("#batch-property").value;const role=$("#batch-role").value;const value=Number(input.value);
    if(["continuous_torque","velocity_limit"].includes(key)){applyHardwareBatch(key,role,value);return;}
    if(key==="gear_ratio"&&value<=0||key==="gear_ratio"&&value>10000||key==="rotor_inertia"&&value>100||key!=="initial"&&value<0){toast("请填写范围内的非负数；传动比必须大于 0（直驱为 1）","error");return;}
    const matching=state.spec.bindings.joints.filter((joint)=>joint.role===role&&joint.name);
    if(!matching.length){toast("请先完成关节映射");return;}
    if(key==="initial"&&matching.some((joint)=>{const [min,max]=jointRange(joint);return value<min||value>max;})){toast("该角度超出至少一个同类关节的模型限位","error");return;}
    matching.forEach((joint)=>setJointValue(key,joint,value));changed({pose:key==="initial"});renderJointControls();toast(`已更新 ${matching.length} 个 ${role} 关节`,"success");
  });
  $("#reset-joints").addEventListener("click",()=>{
    resetMotorInertia();
    [...jointPaths].filter((path)=>fieldStep(path)===state.step).forEach((path)=>{remove(targetParameters(),path);const imported=state.scope==="common"?get(state.importDefaults?.parameters,path):undefined;if(imported!==undefined)set(targetParameters(),path,clone(imported));});
    state.invalidFields.forEach((key)=>{if(key.startsWith(`joint:${state.scope}:`))state.invalidFields.delete(key);});changed({pose:true});renderJointControls();renderParameters();toast("关节配置已恢复默认值");
  });
  $("#reset-config").addEventListener("click",()=>{
    resetMotorInertia();const target=targetParameters();
    for(const {path} of parameterLeaves(effective()).filter((field)=>fieldStep(field.path)===state.step)){
      remove(target,path);const imported=state.scope==="common"?get(state.importDefaults?.parameters,path):undefined;if(imported!==undefined)set(target,path,clone(imported));
    }
    for(const key of [...state.invalidFields,...state.pendingEdits])if(pendingFieldStep(key)===state.step){state.invalidFields.delete(key);state.pendingEdits.delete(key);}
    changed({pose:state.step===3});renderParameters();renderJointControls();renderPd();toast("仅恢复了本步配置，其余步骤保留。");
  });
  $("#main-content").addEventListener("input",(event)=>{if(event.target.id==="action-scale-base"){state.pendingEdits.add(`${state.scope}:control.action_scale`);updatePdActions();}});
  $("#main-content").addEventListener("change",(event)=>{const input=event.target;if(input.id!=="action-scale-base")return;const value=Number(input.value);if(!input.value.trim()||!Number.isFinite(value)||value<=0){state.invalidFields.add(`${state.scope}:control.action_scale`);input.setCustomValidity("基础动作缩放必须为正数");input.reportValidity();return;}input.setCustomValidity("");state.invalidFields.delete(`${state.scope}:control.action_scale`);state.pendingEdits.delete(`${state.scope}:control.action_scale`);set(targetParameters(),"control.action_scale",value);changed();renderActionScaling();renderParameterIndex();});
  $("#terrain-summary-button").addEventListener("click",()=>analyzeTerrain(false));$("#terrain-build-button").addEventListener("click",()=>analyzeTerrain(true));
  $("#pose-layout-toggle").addEventListener("click",()=>{state.poseFocus=!state.poseFocus;document.body.classList.toggle("pose-focus",state.step===3&&state.poseFocus);$("#pose-layout-toggle").textContent=state.poseFocus?"返回常规布局":"展开大画布";});
  $("#pose-fullscreen").addEventListener("click",async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch(error){toast(error.message,"error");}});
  $("#start-preview").addEventListener("click",openPreview);$("#refresh-plan").addEventListener("click",buildPlan);
  $("#file-tree").addEventListener("click",(event)=>{const node=event.target.closest("[data-file-index]");if(node)selectFile(Number(node.dataset.fileIndex));});
  $("#copy-file").addEventListener("click",()=>copyText(state.plan?.files[state.currentFile]?.content||""));
  $("#confirm-review").addEventListener("change",(event)=>{
    if(event.target.checked&&(!state.plan||state.planRevision!==state.revision||hasBlockingIssues(state.plan.issues))){event.target.checked=false;toast("请先生成无阻塞问题的文件预览","error");return;}
    state.spec.confirmed=event.target.checked;if(event.target.checked)workflow.confirm(state.spec,"review",stepSnapshots().review,stepSummary(11));else workflow.unconfirm(state.spec,"review");updateSummary();updateActions();
  });
  $("#generate-button").addEventListener("click",runGenerate);$("#validate-cpu").addEventListener("click",()=>runValidation("cpu"));$("#validate-simulation").addEventListener("click",()=>runValidation("simulation"));
  $("#stop-job").addEventListener("click",()=>busy("正在停止验证…",async()=>{await api(`/api/jobs/${encodeURIComponent(state.job.id)}/stop`,{method:"POST"});pollJob();}));
  $("#task-commands").addEventListener("click",(event)=>{const node=event.target.closest("[data-copy-command]");if(node)copyText(node.dataset.copyCommand);});
  $("#save-draft").addEventListener("click",saveDraft);$("#load-draft").addEventListener("click",listDrafts);$("#close-drafts").addEventListener("click",()=>$("#draft-dialog").close());
  $("#draft-list").addEventListener("click",(event)=>{const node=event.target.closest("[data-load-draft]");if(node)restoreDraft(node.dataset.loadDraft);});
  $("#draft-dialog").addEventListener("click",(event)=>{if(event.target===$("#draft-dialog")){const r=event.target.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)event.target.close();}});
  document.addEventListener("error",(event)=>{const video=event.target;if(video?.tagName!=="VIDEO")return;const figure=video.closest(".test-video");if(figure){const note=document.createElement("p");note.className="field-note";note.textContent="该视频已清理或不可用，响应曲线与测试报告仍可查看。";figure.replaceChildren(note);}},true);
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
