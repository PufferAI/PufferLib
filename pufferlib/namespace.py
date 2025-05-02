from pdb import set_trace as T
from types import SimpleNamespace
from collections.abc import Mapping

def __getitem__(self, key):
    return self.__dict__[key]

def __setitem__(self, key, value):
    self.__dict__[key] = value

def keys(self):
    return self.__dict__.keys()

def values(self):
    return self.__dict__.values()

def items(self):
    return self.__dict__.items()

def __iter__(self):
    return iter(self.__dict__)

def __len__(self):
    return len(self.__dict__)

class Namespace(SimpleNamespace, Mapping):
    __getitem__ = __getitem__
    __setitem__ = __setitem__
    __iter__ = __iter__
    __len__ = __len__
    keys = keys
    values = values
    items = items

def dataclass(cls):
    # Safely get annotations
    annotations = getattr(cls, '__annotations__', {})

    # Combine both annotated and non-annotated fields
    all_fields = {**{k: None for k in annotations.keys()}, **cls.__dict__}
    all_fields = {k: v for k, v in all_fields.items() if not callable(v) and not k.startswith('__')}

    def __init__(self, **kwargs):
        for field, default_value in all_fields.items():
            setattr(self, field, kwargs.get(field, default_value))

    cls.__init__ = __init__
    setattr(cls, "__getitem__", __getitem__)
    setattr(cls, "__setitem__", __setitem__)
    setattr(cls, "__iter__", __iter__)
    setattr(cls, "__len__", __len__)
    setattr(cls, "keys", keys)
    setattr(cls, "values", values)
    setattr(cls, "items", items)
    return cls

def namespace(self=None, **kwargs):
    if self is None:
        return Namespace(**kwargs)
    self.__dict__.update(kwargs)
