(function (global) {
  "use strict";

  function organize({root,applicationPreview,applicationActions,report,selected,fresh,scopeLabel,applied} = {}) {
    if (!root) return null;
    const document = root.ownerDocument;
    const candidates = Array.isArray(report?.candidates) ? report.candidates : [];
    const joints = Array.isArray(selected?.joints) ? selected.joints : [];
    const summary = selected?.summary || {};
    const referenceOnly = !!selected && (selected.reference_only === true || (selected.id === "current" && report?.pd_reference?.reference_only === true));
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
    if (applyButton && (!fresh || referenceOnly || !selected || summary.eligible !== true)) applyButton.disabled = true;

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
    if (oldExplanation) oldExplanation.textContent = "先查看所选候选，完整比较与逐关节响应可按需展开。";

    root.querySelectorAll("[data-pd-candidate-summary]").forEach((node) => node.remove());
    const card = document.createElement("section");
    card.className = "pd-candidate-summary";
    card.dataset.pdCandidateSummary = "true";
    card.setAttribute("aria-label","所选 PD 候选摘要");
    card.append(text("h4",selected ? `${referenceOnly ? "对照基准" : selected.id === report?.recommended_id ? "本次推荐候选" : "当前查看候选"} · ${selected.label || selected.id || "未命名候选"}` : "本次没有可供审阅的候选"));
    if (selected) {
      const total = Number.isFinite(summary.total_joints) ? summary.total_joints : joints.length || null;
      const measured = joints.filter((joint) => typeof joint?.metrics?.passed === "boolean");
      const passed = Number.isFinite(summary.passed_joints) ? summary.passed_joints : measured.length ? measured.filter((joint) => joint.metrics.passed).length : null;
      card.append(text("strong",`通过关节 ${passed ?? "—"} / ${total ?? "—"}`));
      const settling = summary.worst_settling_time_s === null ? "未到位" : typeof summary.worst_settling_time_s === "number" && Number.isFinite(summary.worst_settling_time_s) ? `${format(summary.worst_settling_time_s,3)} s` : "未提供";
      const overshoot = typeof summary.max_overshoot_pct === "number" && Number.isFinite(summary.max_overshoot_pct) ? `${format(summary.max_overshoot_pct,1)}%` : "未提供";
      card.append(text("p",`最慢到位 ${settling} · 最大过冲 ${overshoot} · 作用范围：${scopeLabel || "当前配置"}`));
    }
    let next;
    if (!fresh) next = "下一步：当前显示历史结果，请核对测试输入并重新测试后再审阅应用。";
    else if (!selected) next = "下一步：查看测试提示与日志，检查输入后重新测试。";
    else if (referenceOnly) next = "下一步：展开全部候选，选择一个搜索得到的候选；当前基准仅作对照，不能应用。";
    else if (summary.eligible !== true) next = "下一步：展开逐关节响应查看未通过项，或比较其他候选；满足指标后再审阅应用。";
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

  global.MigrationPdFlow = Object.freeze({organize});
})(typeof window === "undefined" ? globalThis : window);
