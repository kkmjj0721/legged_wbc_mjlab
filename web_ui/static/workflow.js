(function (global) {
  "use strict";

  const STEPS = Object.freeze([
    {id:"import",title:"导入机器人模型",hint:"选择 XML / URDF 与模型资源",group:"模型与任务"},
    {id:"tasks",title:"选择训练任务",hint:"确定算法与 Flat / Rough 组合",group:"模型与任务"},
    {id:"mapping",title:"确认机器人映射",hint:"核对关节顺序、足端与碰撞对象",group:"模型与任务"},
    {id:"pose",title:"确认初始姿态",hint:"检查机体高度、关节角与足端位置",group:"机器人与控制"},
    {id:"motors",title:"设置电机与动作",hint:"核对限矩、惯量、动作范围与 T–N 曲线",group:"机器人与控制"},
    {id:"timing",title:"设置时序与环境",hint:"确定步长、控制频率、延迟与环境数量",group:"机器人与控制"},
    {id:"pd",title:"确定 PD 增益",hint:"填写已知增益，或测试候选后选择应用",group:"机器人与控制"},
    {id:"commands",title:"设置指令与地形",hint:"确认速度范围、地形与课程",group:"任务与训练"},
    {id:"randomization",title:"设置随机化与观测",hint:"核对 Reset、动力学扰动与观测噪声",group:"任务与训练"},
    {id:"rewards",title:"确认奖励权重",hint:"调整权重并核对模板奖励参数",group:"任务与训练"},
    {id:"training",title:"设置训练参数",hint:"确认网络、算法与训练预算",group:"任务与训练"},
    {id:"review",title:"审阅生成文件",hint:"检查任务名称、文件内容与配置差异",group:"审阅与生成"},
    {id:"generate",title:"生成与验证",hint:"写入新任务并查看验证结果",group:"审阅与生成"},
  ].map((step,index) => Object.freeze({...step,index})));
  const ids = new Set(STEPS.map((step) => step.id));
  const isRecord = (value) => value !== null && typeof value === "object" && !Array.isArray(value);

  function requireId(id) {
    if (!ids.has(id)) throw new RangeError(`未知工作台步骤：${id}`);
  }

  // Object key order has no meaning in a draft; joint and task arrays do.
  function stable(value) {
    const ancestors = new Set();
    function ordered(current) {
      if (!current || typeof current !== "object") return current;
      if (ancestors.has(current)) throw new TypeError("步骤快照不能包含循环引用");
      ancestors.add(current);
      let result;
      if (Array.isArray(current)) {
        result = current.map(ordered);
      } else {
        result = Object.create(null);
        Object.keys(current).sort().forEach((key) => { result[key] = ordered(current[key]); });
      }
      ancestors.delete(current);
      return result;
    }
    return JSON.stringify(ordered(value)) ?? "null";
  }

  function ensure(spec) {
    if (!isRecord(spec)) throw new TypeError("MigrationSpec 必须是对象");
    if (!isRecord(spec.workflow)) spec.workflow = {};
    const workflow = spec.workflow;
    workflow.version = 1;
    if (!isRecord(workflow.confirmations)) workflow.confirmations = {};
    if (!Number.isInteger(workflow.last_step) || workflow.last_step < 0 || workflow.last_step >= STEPS.length) {
      workflow.last_step = 0;
    }
    return workflow;
  }

  function confirmation(spec,id) {
    const workflow = spec?.workflow;
    if (!isRecord(workflow) || workflow.version !== 1 || !isRecord(workflow.confirmations)) return null;
    const saved = workflow.confirmations[id];
    return isRecord(saved) && typeof saved.snapshot === "string" ? saved : null;
  }

  function status(spec,id,snapshot) {
    requireId(id);
    const saved = confirmation(spec,id);
    if (!saved) return "todo";
    return saved.snapshot === stable(snapshot) ? "confirmed" : "stale";
  }

  function confirm(spec,id,snapshot,summary = "") {
    requireId(id);
    const saved = {
      snapshot:stable(snapshot),
      summary:String(summary ?? ""),
      confirmed_at:new Date().toISOString(),
    };
    ensure(spec).confirmations[id] = saved;
    return saved;
  }

  function unconfirm(spec,id) {
    requireId(id);
    delete ensure(spec).confirmations[id];
  }

  function summarize(spec,snapshots = {}) {
    return STEPS.map((step) => {
      const saved = confirmation(spec,step.id);
      return {
        ...step,
        status:status(spec,step.id,snapshots[step.id]),
        summary:typeof saved?.summary === "string" ? saved.summary : "",
        confirmed_at:typeof saved?.confirmed_at === "string" ? saved.confirmed_at : null,
      };
    });
  }

  function changedSections(previous = {},current = {}) {
    return STEPS.filter((step) => stable(previous[step.id]) !== stable(current[step.id])).map((step) => step.id);
  }

  global.MigrationWorkflow = Object.freeze({STEPS,ensure,stable,status,confirm,unconfirm,summarize,changedSections});
})(typeof window === "undefined" ? globalThis : window);
