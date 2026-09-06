import inspect
from copy import deepcopy


class BaseConfig:
    def __init__(self):
        self.init_member_classes(self)

    @classmethod
    def init_member_classes(cls, obj):
        for key in dir(obj):
            if key.startswith("__") or key == "__class__":
                continue
            value = getattr(obj, key)
            if inspect.isclass(value):
                instance = value()
                setattr(obj, key, instance)
                cls.init_member_classes(instance)
            elif isinstance(value, (dict, list, set)):
                setattr(obj, key, deepcopy(value))

    def to_dict(self):
        def _to_dict(obj):
            if not hasattr(obj, "__dict__"):
                return obj
            res = {}
            for k in dir(obj):
                if k.startswith("_"):
                    continue
                v = getattr(obj, k)
                if inspect.isclass(v) or (isinstance(v, object) and hasattr(v, "__dict__")):
                    res[k] = _to_dict(v)
                else:
                    res[k] = v
            return res
        return _to_dict(self)
