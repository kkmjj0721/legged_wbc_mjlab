"""Viser controls for LPACRL play; GUI threads only submit CPU requests."""

import math

from .play_control import PlaySelection


def geometry_label(parameters):
    names = {
        "step_height_m": ("height", "m"),
        "step_width": ("depth", "m"),
        "height_max_m": ("roughness", "m"),
        "gradient": ("gradient", ""),
    }
    return ", ".join(
        f"{label}={parameters[key]:.3g}{unit}"
        for key, (label, unit) in names.items() if key in parameters
    ) or "flat"


def create_play_gui(play, name, server, get_env_idx, request_action):
    c = play.controller
    terrain_map = c.terrain_map
    families = tuple(dict.fromkeys(family for family, _ in terrain_map.columns))

    def levels_for(family):
        return {
            f"{level}: {geometry_label(terrain_map.parameters[col])}": col
            for col, (f, level) in enumerate(terrain_map.columns) if f == family
        }

    modes = {"Manual": "manual", "Uniform tasks": "uniform"}
    if play.checkpoint_probabilities is not None:
        modes["Checkpoint teacher"] = "checkpoint"

    with server.gui.add_folder(f"{name.capitalize()} / Play"):
        server.gui.add_markdown(
            "Controls apply to the **selected environment**. Manual mode starts standing. "
            "Select a terrain and click **Apply & reset**; velocity sliders are live in Manual mode. "
            "Random modes sample the full catalog. **Replay** repeats the exact scene."
        )
        mode = server.gui.add_dropdown("Mode", options=tuple(modes), initial_value="Manual")
        family = server.gui.add_dropdown("Terrain", options=families, initial_value=families[0])
        initial_levels = levels_for(families[0])
        level = server.gui.add_dropdown("Level", options=tuple(initial_levels), initial_value=next(iter(initial_levels)))
        replica = server.gui.add_dropdown(
            "Replica", options=tuple(str(i) for i in range(terrain_map.replicas)), initial_value="0",
            visible=terrain_map.replicas > 1)
        yaw = server.gui.add_slider("Spawn yaw (deg)", min=-180, max=180, step=5, initial_value=0)
        resample = server.gui.add_checkbox("Resample on automatic reset", initial_value=False)
        sliders = [
            server.gui.add_slider(label, min=-max(limit, 0.01), max=max(limit, 0.01),
                                  step=0.01, initial_value=0.0, disabled=limit == 0)
            for label, limit in zip(("vx (m/s)", "vy (m/s)", "wz (rad/s)"), play.command_limits)
        ]
        hold = server.gui.add_checkbox("Hold heading when moving", initial_value=True)
        server.gui.add_markdown(
            "With heading hold enabled, nonzero wz turns manually; releasing wz locks the new heading. "
            "Zero velocity stops all three commands. Terrain and spawn yaw take effect on Apply & reset."
        )
        presets = server.gui.add_button_group("Command preset", options=("Stand", "Forward 0.5", "Backward 0.5"))
        apply = server.gui.add_button("Apply & reset", disabled=request_action is None)
        replay = server.gui.add_button("Replay", disabled=request_action is None)
        status = server.gui.add_markdown("")

    @family.on_update
    def _family_changed(_):
        choices = levels_for(family.value)
        level.options = tuple(choices)
        level.value = next(iter(choices))

    def update_visibility(_=None):
        manual = modes[mode.value] == "manual"
        for handle in (family, level, replica, yaw, hold):
            handle.disabled = not manual
        presets.visible = manual  # Viser button groups cannot be disabled.
        resample.disabled = manual
        for handle, limit in zip(sliders, play.command_limits):
            handle.disabled = not manual or limit == 0

    mode.on_update(update_visibility)
    update_visibility()

    def submit_command(_=None):
        if modes[mode.value] == "manual":
            play.request_command(get_env_idx(), tuple(float(s.value) for s in sliders), hold.value)

    for slider in sliders:
        slider.on_update(submit_command)
    hold.on_update(submit_command)

    @presets.on_click
    def _preset(event):
        vx = {"Stand": 0.0, "Forward 0.5": 0.5, "Backward 0.5": -0.5}[event.target.value]
        vx = max(-play.command_limits[0], min(play.command_limits[0], vx))
        with server.atomic():
            for slider, value in zip(sliders, (vx, 0.0, 0.0)):
                slider.value = value
        submit_command()

    @apply.on_click
    def _apply(_):
        selection = PlaySelection(
            column=levels_for(family.value)[level.value], replica=int(replica.value),
            yaw=math.radians(yaw.value), command=tuple(float(s.value) for s in sliders),
            heading_hold=hold.value, mode=modes[mode.value], resample_on_reset=resample.value,
        )
        play.request_reset(get_env_idx(), selection)
        request_action("CUSTOM", {"type": "gui_reset"})

    @replay.on_click
    def _replay(_):
        play.request_reset(get_env_idx())
        request_action("CUSTOM", {"type": "gui_reset"})

    def refresh_status():
        i = get_env_idx()
        scene = play.scenes[i]
        if scene is None:
            return
        selected = play.selections[i]
        f, lvl = terrain_map.columns[scene.column]
        vx, vy, wz = c.commands[i].detach().cpu().tolist()
        rotation = ""
        if (selected.mode != "manual" and c.catalog.heading_command) or (
                selected.mode == "manual" and selected.heading_hold):
            rotation = f" · heading target {math.degrees(float(c.heading_targets[i])):.1f}°"
        status.content = (
            f"**Current env {i} · {selected.mode}**\n\n"
            f"{f} / {lvl} · {geometry_label(terrain_map.parameters[scene.column])}\n\n"
            f"Spawn yaw {math.degrees(scene.yaw):.1f}° · replica {scene.replica}\n\n"
            f"Policy command: vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}{rotation}\n\n"
            + ("Manual commands" if selected.mode == "manual" else f"Task {scene.task}")
        )

    play.status_callback = refresh_status
    refresh_status()
