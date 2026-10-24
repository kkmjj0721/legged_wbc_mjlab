"""Serve the pinned URDF Studio canvas and only the active import's assets."""

from __future__ import annotations

from pathlib import Path

STUDIO_COMMIT = "23b96e2f3d5b1b68ad9629925d4479e9847d35a5"
STUDIO_DIST = Path(__file__).with_name("studio") / "dist"


def register_studio(app, preview):
    from flask import abort, jsonify, send_file, send_from_directory

    @app.get("/studio/")
    @app.get("/studio/<path:filename>")
    def studio_page(filename="index.html"):
        if not (STUDIO_DIST / "index.html").is_file():
            abort(503, "本地 URDF Studio 预览资源缺失，请按 web_ui/studio/README.md 重建")
        response = send_from_directory(STUDIO_DIST, filename)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; "
            "font-src 'self' data:; connect-src 'self' blob:; "
            "worker-src 'self' blob:; object-src 'none'; frame-ancestors 'self'"
        )
        return response

    @app.get("/api/preview/studio/<token>/model")
    def studio_model(token):
        return jsonify(preview.studio_model(token))

    @app.get("/api/preview/studio/<token>/state")
    def studio_state(token):
        response = jsonify(preview.studio_state(token))
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/preview/studio/<token>/asset/<path:filename>")
    def studio_asset(token, filename):
        path = preview.studio_asset(token, filename)
        response = send_file(path, conditional=True)
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response
