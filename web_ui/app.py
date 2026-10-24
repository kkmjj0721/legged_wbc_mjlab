"""Local Flask application; importing this module starts no server or process."""

from __future__ import annotations

import json
import re
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


def create_app(repo_root=None, workspace=None):
    from flask import Flask, jsonify, render_template, request
    from werkzeug.exceptions import HTTPException

    from . import generator, models, templates_catalog
    from .preview import PosePreview
    from .validation import JobManager, check_conflicts, generate_transaction

    repository = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    work = Path(workspace or repository / ".artifacts" / "robot-migration").resolve()
    if not work.is_relative_to(repository / ".artifacts"):
        raise ValueError("Web 工具的工作目录必须位于项目 .artifacts 内")
    work.mkdir(parents=True, exist_ok=True)
    (work / "drafts").mkdir(exist_ok=True)
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=512 * 1024 * 1024, JSON_AS_ASCII=False)
    app.json.ensure_ascii = False
    preview = PosePreview()
    jobs = JobManager(repository, work)
    generation_lock = threading.Lock()
    app.extensions["migration"] = dict(preview=preview, jobs=jobs, workspace=work,
                                       close=lambda: (preview.stop(), jobs.close()))

    @app.before_request
    def require_local_request():
        host = urlsplit("//" + request.host).hostname
        if host not in ("127.0.0.1", "localhost", "::1"):
            return jsonify(error="local_only", message="此工具仅接受本机请求"), 403
        origin = request.headers.get("Origin")
        if origin and origin != request.host_url.rstrip("/"):
            return jsonify(error="origin_rejected", message="请求来源与本地工作台不一致"), 403

    @app.errorhandler(Exception)
    def handle_error(exc):
        if isinstance(exc, HTTPException):
            return jsonify(error=exc.name, message=exc.description), exc.code
        if isinstance(exc, KeyError):
            return jsonify(error="not_found", message=f"不存在的对象或缺少字段: {exc}"), 404
        if isinstance(exc, (ValueError, TypeError, FileNotFoundError)):
            return jsonify(error="invalid_input", message=str(exc)), 400
        app.logger.exception("迁移工作台请求失败")
        return jsonify(error="request_failed", message=str(exc)), 500

    def payload():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise ValueError("请求正文必须为 JSON 对象")
        return body

    def spec_from(body):
        spec = body.get("spec", body)
        if not isinstance(spec, dict):
            raise ValueError("spec 必须为配置对象")
        return spec

    def checked_spec(body, *, require_confirmed=True):
        spec = deepcopy(spec_from(body))
        if not require_confirmed:
            spec["confirmed"] = True
        issues = models.validate_spec(spec)
        errors = [item for item in issues if item.get("level", "error") == "error"]
        if errors:
            raise ValueError("；".join(item.get("message", str(item)) for item in errors))
        return spec, issues

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/templates")
    def templates():
        return jsonify(templates=templates_catalog.catalog(), examples=[
            dict(label="RI-4438", root=str(repository / "src/assets/robots/ri_4438/xmls"), xml="ri_4438.xml"),
            dict(label="Unitree Go2", root=str(repository / "src/assets/robots/go2/xmls"), xml="go2.xml"),
        ])

    @app.post("/api/import")
    def import_model():
        if "file" in request.files:
            result = models.import_zip(request.files["file"], workspace=work, xml=request.form.get("xml") or None)
        else:
            body = payload()
            result = models.import_model(body.get("root", ""), body.get("xml", ""), workspace=work)
        preview.stop()
        return jsonify(result)

    @app.post("/api/model-xml")
    def model_xml():
        body = payload()
        model = body.get("model", {})
        path = models.resolve_model(model)
        return jsonify(xml=path.read_text(encoding="utf-8"), path=model["xml"],
                       source_format=model.get("source_format", "mjcf"),
                       conversion=model.get("conversion"))

    def draft_path(draft_id):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", draft_id):
            raise ValueError("草稿 ID 格式无效")
        return work / "drafts" / f"{draft_id}.json"

    @app.get("/api/drafts")
    def drafts():
        entries = []
        for path in (work / "drafts").glob("*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                entries.append({key: value.get(key) for key in ("id", "name", "updated_at")})
            except (ValueError, OSError):
                continue
        return jsonify(drafts=sorted(entries, key=lambda item: item["updated_at"] or "", reverse=True))

    @app.post("/api/drafts")
    def save_draft():
        body = payload()
        draft_id = body.get("id") or uuid.uuid4().hex
        path = draft_path(draft_id)
        entry = dict(id=draft_id, name=str(body.get("name") or "未命名机器人"), spec=spec_from(body),
                     updated_at=datetime.now(timezone.utc).isoformat())
        temporary = path.with_suffix(f".{uuid.uuid4().hex}.pending")
        temporary.write_text(json.dumps(entry, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
        return jsonify(entry)

    @app.get("/api/drafts/<draft_id>")
    def get_draft(draft_id):
        entry = json.loads(draft_path(draft_id).read_text(encoding="utf-8"))
        preview.stop()
        return jsonify(entry)

    @app.post("/api/preview")
    def open_preview():
        spec = spec_from(payload())
        return jsonify(preview.start(models.resolve_model(spec), spec))

    @app.post("/api/preview/pose")
    def preview_pose():
        return jsonify(pose=preview.update(**payload()))

    @app.post("/api/preview/stop")
    def stop_preview():
        preview.stop()
        return jsonify(status="stopped")

    @app.post("/api/pd/analyze")
    def analyze_pd():
        from .pd_analysis import analyze

        body = payload()
        return jsonify(analyze(spec_from(body), scope=body.get("scope", "common"),
                               options=body.get("options")))

    @app.post("/api/plan")
    def plan():
        spec, issues = checked_spec(payload(), require_confirmed=False)
        try:
            jobs.check_spec(spec)
        except ValueError as exc:
            issues.append(dict(level="error", message=str(exc)))
        rendered = generator.render(spec)
        try:
            check_conflicts(repository, rendered["files"])
        except ValueError as exc:
            issues.append(dict(level="error", message=str(exc)))
        files = []
        for path, content in sorted(rendered["files"].items()):
            binary = isinstance(content, bytes)
            if binary:
                try:
                    shown = content.decode("utf-8")
                    kind = "text"
                except UnicodeDecodeError:
                    shown, kind = "", "binary"
            else:
                shown, kind = content, "text"
            size = len(content if binary else content.encode("utf-8"))
            files.append(dict(path=path, content=shown, size=size, kind=kind))
        return jsonify(files=files, task_ids=rendered["task_ids"], issues=issues, manifest=rendered["manifest"])

    @app.post("/api/generate")
    def generate():
        spec, _ = checked_spec(payload())
        rendered = generator.render(spec)
        check_conflicts(repository, rendered["files"])

        def transaction(job):
            with generation_lock:
                return generate_transaction(jobs, job, spec, rendered)

        return jsonify(jobs.submit("generate", rendered["task_ids"], operation=transaction)), 202

    @app.post("/api/validate")
    def validate():
        body = payload()
        mode = body.get("mode", "cpu")
        if mode not in ("cpu", "simulation"):
            raise ValueError("验证模式只能是 cpu 或 simulation")
        task_id = body.get("task_id")
        if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,150}", task_id):
            raise ValueError("请选择一个有效的 task_id")
        return jsonify(jobs.submit(mode, [task_id], spec=body.get("spec"))), 202

    @app.get("/api/jobs/<job_id>")
    def get_job(job_id):
        return jsonify(jobs.get(job_id))

    @app.post("/api/jobs/<job_id>/stop")
    def stop_job(job_id):
        return jsonify(jobs.stop(job_id))

    return app
