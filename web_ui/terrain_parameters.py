"""Shared terrain form descriptions and validation of generator overrides."""

import math


FIELD_METADATA = {
    "terrain.border_width": {
        "label": "整图外围平地宽度", "unit": "m",
        "description": "整个地形网格外围的平地边宽，不是每个地块或楼梯内部的边宽；可以为 0。",
    },
    "terrain.difficulty_range": {
        "label": "生成难度区间（两个端点）",
        "description": "恰好填写两个有限端点：下限, 上限，且下限不大于上限；通常为 0, 1。"
                       "这不是等级列表。若想要 0～5 共六个行等级，将 num_rows 设为 6，通常仍用区间 0, 1；"
                       "区间之外的难度是否适合，由具体子地形决定。",
    },
    "terrain.size": {
        "label": "单地块尺寸", "unit": "m",
        "description": "单个子地形地块的两个正尺寸 [X, Y]，不是整张地形地图的总尺寸。",
    },
    "terrain.num_rows": {
        "label": "地形行数 / 难度等级数",
        "description": "正整数；按行生成难度课程时，每行对应一个难度等级，"
                       "从区间下限到上限逐行插值。只有一行时使用下限。",
    },
    "terrain.num_cols": {
        "label": "随机模式地形列数",
        "description": "正整数；terrain.curriculum=false 时使用此列数。"
                       "terrain.curriculum=true 时实际每种子地形恰好占一列，忽略本数值。",
    },
    "terrain.curriculum": {
        "label": "按行生成难度课程",
        "description": "控制地形布局：true 时每种子地形占一列，难度沿行递增；"
                       "false 时各地块随机抽取类型和难度。"
                       "这与 curriculum.enabled 控制的训练运行时等级调整是不同设置。",
    },
    "terrain.border_height": {
        "label": "整图外围平地厚度", "unit": "m",
        "description": "外围边框向地面下方延伸的厚度；表面仍位于 Z=0。",
    },
}

SUB_FIELD_METADATA = {
    "proportion": {"label": "生成 / 出生权重", "description": "相对权重，后端按总和归一化。按行课程中每种类型仍各占一列，权重控制机器人出生比例；随机模式中控制地块抽样概率。0 不删除课程中的地形列。"},
    "step_height_range": {"label": "台阶高度区间", "unit": "m", "description": "[最小阶高, 最大阶高]，按当前地块难度线性插值。"},
    "step_width": {"label": "台阶踏面深度", "unit": "m", "description": "单级台阶沿行进方向的水平深度；修改后阶数由地块、平台和边宽共同计算。"},
    "platform_width": {"label": "中央平台宽度", "unit": "m", "description": "地块中央正方形平台的边长。"},
    "border_width": {"label": "地块内部平地边宽", "unit": "m", "description": "当前子地形外围的平地宽度，与整张地图的外围边宽不同；holes=true 时楼梯忽略此项。"},
    "holes": {"label": "保留孔洞", "description": "楼梯生成十字形踏步并保留四角空隙；随机网格保留间隙。"},
    "grid_width": {"label": "随机方块水平宽度", "unit": "m", "description": "每个随机高低方块的水平边长。越小通常生成越多 geom。"},
    "grid_height_range": {"label": "随机方块高度区间", "unit": "m", "description": "[最小高度, 最大高度]，当前难度决定随机高度幅度。"},
    "merge_similar_heights": {"label": "合并近似高度方块", "description": "合并高度相近的相邻方块，减少 MuJoCo geom 数量。"},
    "height_merge_threshold": {"label": "方块高度合并阈值", "unit": "m", "description": "高度差不超过此值的方块可合并。"},
    "max_merge_distance": {"label": "最大方块合并距离", "unit": "格", "description": "沿网格合并方块时允许跨越的最大格数。"},
    "horizontal_scale": {"label": "高度场水平采样间距", "unit": "m"},
    "vertical_scale": {"label": "高度场高度量化间距", "unit": "m"},
    "noise_range": {"label": "高度噪声区间", "unit": "m"},
    "noise_step": {"label": "高度噪声步长", "unit": "m"},
    "slope_range": {"label": "坡度比区间", "description": "高差 / 水平距离的两个端点，不是角度。"},
    "inverted": {"label": "反转坡面方向"},
    "scale_with_difficulty": {"label": "高度随难度缩放"},
    "downsampled_scale": {"label": "低分辨率采样间距", "unit": "m"},
    "amplitude_range": {"label": "波浪振幅区间", "unit": "m"},
    "num_waves": {"label": "波浪数量"},
    "base_thickness_ratio": {"label": "高度场基底厚度比", "description": "相对于地块高度范围的地下基底厚度比例。"},
}


def sub_terrain_label(name):
    if name == "flat":
        return "平地"
    if name == "box_random_grid":
        return "随机高低方块"
    if name.startswith("pyramid_stairs"):
        direction = "下楼梯" if name.startswith("pyramid_stairs_inv") else "上楼梯"
        suffix = name.rsplit("_", 1)[-1]
        return direction + (f" · 规格 {int(suffix) + 1}" if suffix.isdigit() else "")
    return name


def field_metadata(path):
    if path in FIELD_METADATA:
        return FIELD_METADATA[path]
    prefix = "terrain.sub_terrains."
    if path.startswith(prefix):
        name, _, field = path[len(prefix):].partition(".")
        item = dict(SUB_FIELD_METADATA.get(field, {}))
        item.update(terrain_name=name, terrain_label=sub_terrain_label(name))
        return item
    return {}


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate(terrain, label="terrain"):
    """Check only shared generator keys; preserve custom and nested schemas."""
    if not isinstance(terrain, dict):
        raise ValueError(f"{label} 必须为对象")
    if "difficulty_range" in terrain:
        value = terrain["difficulty_range"]
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or not all(_finite(item) for item in value) or value[0] > value[1]):
            raise ValueError(f"{label}.difficulty_range 必须恰好包含两个有限数值端点 [下限, 上限]，且下限 ≤ 上限；不是等级列表")
    if "size" in terrain:
        value = terrain["size"]
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or any(not _finite(item) or item <= 0 for item in value)):
            raise ValueError(f"{label}.size 必须恰好包含两个有限正数，表示单地块的 X / Y 尺寸（m）")
    for name in ("num_rows", "num_cols"):
        if name in terrain and (type(terrain[name]) is not int or terrain[name] <= 0):
            raise ValueError(f"{label}.{name} 必须为正整数")
    if "border_width" in terrain:
        value = terrain["border_width"]
        if not _finite(value) or value < 0:
            raise ValueError(f"{label}.border_width 必须为非负有限数值（m）")
    if "curriculum" in terrain and type(terrain["curriculum"]) is not bool:
        raise ValueError(f"{label}.curriculum 必须为布尔值 true / false")
    if "border_height" in terrain:
        value = terrain["border_height"]
        if not _finite(value) or value <= 0:
            raise ValueError(f"{label}.border_height 必须为正有限数值（m）")
    if "sub_terrains" not in terrain:
        return
    children = terrain["sub_terrains"]
    if not isinstance(children, dict) or not children:
        raise ValueError(f"{label}.sub_terrains 必须为非空对象")
    for name, values in children.items():
        prefix = f"{label}.sub_terrains.{name}"
        if not isinstance(values, dict):
            raise ValueError(f"{prefix} 必须为对象")
        for field, value in values.items():
            path = f"{prefix}.{field}"
            if field not in SUB_FIELD_METADATA:
                raise ValueError(f"{path} 不是可编辑的子地形参数")
            if field in ("holes", "merge_similar_heights", "inverted", "scale_with_difficulty"):
                if type(value) is not bool:
                    raise ValueError(f"{path} 必须为布尔值")
            elif field.endswith("_range"):
                if (not isinstance(value, (list, tuple)) or len(value) != 2
                        or not all(_finite(item) for item in value) or value[0] > value[1]
                        or field not in ("noise_range", "slope_range") and value[0] < 0):
                    raise ValueError(f"{path} 必须为两个有序有限端点 [下限, 上限]；高度区间不可为负")
            elif field in ("max_merge_distance", "num_waves"):
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{path} 必须为正整数")
            elif field == "downsampled_scale" and value is None:
                continue
            elif (not _finite(value) or value < 0
                  or field in ("step_width", "platform_width", "grid_width", "horizontal_scale", "vertical_scale", "noise_step", "downsampled_scale") and value == 0):
                raise ValueError(f"{path} 必须为有效的有限数值；宽度、采样间距和步长须大于 0，权重和边宽可为 0")
        if "size" in terrain and not values.get("holes", False):
            border = values.get("border_width", 0)
            platform = values.get("platform_width", 0)
            if 2 * border + platform >= min(terrain["size"]):
                raise ValueError(f"{prefix} 的两侧边宽与中央平台必须容纳在单地块尺寸内")
