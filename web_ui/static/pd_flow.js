(function (global) {
  "use strict";

  function isReferenceOnly(report,candidate) {
    return !!candidate && (candidate.reference_only === true || (candidate.id === "current" && report?.pd_reference?.reference_only === true));
  }

  function hasGroundProtocol(report) {
    return report?.version === 3 && report.protocol === "joint_tracking_ground_v2";
  }

  function isEligible(report,candidate) {
    return hasGroundProtocol(report) && !!candidate && !isReferenceOnly(report,candidate)
      && candidate.summary?.eligible === true && candidate.summary.joint_eligible === true
      && candidate.summary.tracking_passed === true && candidate.tracking?.status === "passed"
      && candidate.tracking.passed === true && candidate.summary.budget_passed === true
      && candidate.torque_budget?.passed === true && candidate.summary.ground_passed === true
      && candidate.ground?.status === "passed" && candidate.ground.passed === true;
  }

  function organize({root,applicationPreview,applicationActions,report,selected,fresh,scopeLabel,applied} = {}) {
    if (!root) return null;
    const document = root.ownerDocument;
    const candidates = Array.isArray(report?.candidates) ? report.candidates : [];
    const joints = Array.isArray(selected?.joints) ? selected.joints : [];
    const summary = selected?.summary || {};
    const referenceOnly = isReferenceOnly(report,selected);
    const eligible = isEligible(report,selected);
    const format = (value,digits = 2) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "未提供";
    const text = (tag,content,className = "") => {
      const node = document.createElement(tag);
      node.textContent = content;
      if (className) node.className = className;
      return node;
    };
    let opened;
    try { opened = JSON.parse(root.dataset.pdDetails || "{}"); } catch (_error) { opened = {}; }
    if (!opened || typeof opened !== "object" || Array.isArray(opened)) opened = {};
    function detail(key,label,nodes) {
      let element = root.querySelector(`details[data-pd-detail="${key}"]`);
      if (element) return element;
      const contents = nodes.filter(Boolean);
      if (!contents.length) return null;
      element = document.createElement("details");
      element.className = "subsection pd-detail";
      element.dataset.pdDetail = key;
      element.append(text("summary",label));
      contents[0].before(element);
      contents.forEach((node) => element.append(node));
      element.open = opened[key] === true;
      element.addEventListener("toggle",() => {
        // Detached details can still dispatch a queued toggle after a re-render.
        if (!root.contains(element)) return;
        let current;
        try { current = JSON.parse(root.dataset.pdDetails || "{}"); } catch (_error) { current = {}; }
        if (!current || typeof current !== "object" || Array.isArray(current)) current = {};
        current[key] = element.open;
        root.dataset.pdDetails = JSON.stringify(current);
      });
      return element;
    }

    // Move the original controls so their IDs, handlers and radio state survive.
    const review = root.querySelector(".pd-response-review")?.closest("details") || applicationPreview?.querySelector(".pd-response-review")?.closest("details");
    const actions = root.querySelector(".pd-response-apply") || applicationActions?.querySelector(".pd-response-apply");
    if (applicationPreview) {
      applicationPreview.replaceChildren();
      if (review) applicationPreview.append(review);
    }
    if (applicationActions) {
      applicationActions.replaceChildren();
      if (actions) applicationActions.append(actions);
    }
    // The existing action updater retains its other guards; never enable here.
    const applyButton = actions?.querySelector("#pd-test-apply");
    if (applyButton && (!fresh || !eligible)) applyButton.disabled = true;

    const table = root.querySelector(".pd-response-table")?.closest(".table-scroll");
    const tableNote = table?.nextElementSibling?.classList.contains("field-note") ? table.nextElementSibling : null;
    const allCandidates = detail("candidates",`查看全部候选（${candidates.length} 组）`,[table,tableNote]);
    const chart = detail("responses","查看逐关节响应",[
      root.querySelector(".pd-chart-heading"),
      root.querySelector(".pd-response-chart"),
      root.querySelector(".pd-chart-legend"),
    ]);
    const heading = root.querySelector(".pd-section-heading");
    const oldExplanation = heading?.querySelector("p");
    if (oldExplanation) oldExplanation.textContent = "先核对单关节、真实动作幅值跟踪、整机落地及力矩预算。满足约束的候选按动作力矩需求等成本排序。";

    root.querySelectorAll("[data-pd-candidate-summary]").forEach((node) => node.remove());
    const card = document.createElement("section");
    card.className = "pd-candidate-summary";
    card.dataset.pdCandidateSummary = "true";
    card.setAttribute("aria-label","所选 PD 候选摘要");
    card.append(text("h4",selected ? `${referenceOnly ? "对照基准" : selected.id === report?.recommended_id && eligible ? "本次推荐候选" : "当前查看候选"} · ${selected.label || selected.id || "未命名候选"}` : "本次没有可供审阅的候选"));
    if (selected) {
      const total = Number.isFinite(summary.total_joints) ? summary.total_joints : joints.length || null;
      const measured = joints.filter((joint) => typeof joint?.metrics?.passed === "boolean");
      const passed = Number.isFinite(summary.passed_joints) ? summary.passed_joints : measured.length ? measured.filter((joint) => joint.metrics.passed).length : null;
      const groundStatus = {passed:"通过",failed:"未通过",skipped:"未运行"}[selected.ground?.status] || "缺少落地测试";
      const trackingStatus = selected.tracking?.passed === true ? "通过" : selected.tracking ? "未通过" : "缺少测试";
      const budgetStatus = selected.torque_budget?.passed === true ? "通过" : selected.torque_budget ? "未通过" : "缺少预算";
      card.append(text("strong",`单关节 ${passed ?? "—"} / ${total ?? "—"} · 动态目标：${trackingStatus} · 落地：${groundStatus} · 力矩预算：${budgetStatus}`));
      const settling = summary.worst_settling_time_s === null ? "未到位" : typeof summary.worst_settling_time_s === "number" && Number.isFinite(summary.worst_settling_time_s) ? `${format(summary.worst_settling_time_s,3)} s` : "未提供";
      const overshoot = typeof summary.max_overshoot_pct === "number" && Number.isFinite(summary.max_overshoot_pct) ? `${format(summary.max_overshoot_pct,1)}%` : "未提供";
      card.append(text("p",`最慢到位 ${settling} · 最大过冲 ${overshoot} · 作用范围：${scopeLabel || "当前配置"}`));
      const reasons = Array.isArray(selected.ground?.reasons) ? selected.ground.reasons : [];
      if (reasons.length) card.append(text("p",`落地诊断：${reasons.join("；")}`,"pd-ground-reason"));
      const warnings = Array.isArray(selected.ground?.warnings) ? selected.ground.warnings : [];
      if (warnings.length) card.append(text("p",`落地警告（参与排序）：${warnings.map((item) => typeof item === "string" ? item : item.message || JSON.stringify(item)).join("；")}`,"pd-ground-warning"));
      card.append(text("p",`动作预算最大限矩占比 ${typeof selected.torque_budget?.max_effort_ratio === "number" ? format(selected.torque_budget.max_effort_ratio * 100,1) + "%" : "未提供"} · 推荐在满足要求后综合比较动作力矩、跟踪误差及腿部触地成本。`));
    }
    let next;
    if (!fresh) next = "下一步：当前显示历史结果，请核对测试输入并重新测试后再审阅应用。";
    else if (!selected) next = "下一步：查看测试提示与日志，检查输入后重新测试。";
    else if (!hasGroundProtocol(report)) next = "旧报告缺少当前动态跟踪与力矩约束协议，不能应用；请使用当前设置重新搜索。";
    else if (referenceOnly) next = "下一步：展开全部候选，选择一个搜索得到的候选；当前基准仅作对照，不能应用。";
    else if (!eligible) next = "下一步：查看未通过的响应、动态跟踪、落地或力矩预算，或比较其他候选；全部约束通过后再审阅应用。";
    else if (applied === true) next = "当前候选已应用。请核对配置中的增益，然后确认本步。";
    else next = "下一步：前往「审阅并应用」，核对实际增益后点击应用；测试结果不会自动写入配置。";
    card.append(text("p",next));
    if (report?.pd_reference?.reference_only) card.append(text("small","20.0 / 0.5 是测试专用比较基准，不参与推荐，也未自动写入机器人配置。"));
    if (allCandidates) {
      const compare = text("button","比较其他候选","button secondary small");
      compare.type = "button";
      compare.addEventListener("click",() => {
        allCandidates.open = true;
        allCandidates.querySelector("input:checked")?.focus({preventScroll:true});
        allCandidates.scrollIntoView?.({block:"nearest",behavior:"smooth"});
      });
      card.append(compare);
    }
    if (heading) heading.after(card); else root.prepend(card);
    return {summary:card,candidates:allCandidates,responses:chart,review,actions};
  }

  global.MigrationPdFlow = Object.freeze({organize,isReferenceOnly,hasGroundProtocol,isEligible});
})(typeof window === "undefined" ? globalThis : window);
