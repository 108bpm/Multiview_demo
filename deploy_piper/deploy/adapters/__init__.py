"""Adapter loading.

Built-in adapters keep their short names (``dummy``, ``lerobot``). User
adapters can be loaded by dotted class path, for example:

    --adapter=my_policy.deploy:MyPolicyAdapter
    --adapter=my_policy.deploy.MyPolicyAdapter

That keeps new policy deployment out of this registry: add the module to the
server's PYTHONPATH, point the config at the class, and launch.
"""
from __future__ import annotations

import importlib

from .base import PolicyAdapter as PolicyAdapter

_ADAPTERS = ("dummy", "lerobot")


def _load_dotted(name: str):
    if ":" in name:
        module_name, attr_name = name.split(":", 1)
    else:
        module_name, _, attr_name = name.rpartition(".")
    if not module_name or not attr_name:
        return None
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def make_adapter(name: str, **kwargs) -> PolicyAdapter:
    if name == "dummy":
        from .dummy import DummyAdapter

        return DummyAdapter(**kwargs)
    if name == "lerobot":
        from .lerobot import LerobotAdapter

        return LerobotAdapter(**kwargs)
    cls = _load_dotted(name)
    if cls is not None:
        adapter = cls(**kwargs)
        if not isinstance(adapter, PolicyAdapter):
            raise TypeError(
                f"{name} returned {type(adapter).__name__}, expected PolicyAdapter"
            )
        return adapter
    raise ValueError(
        f"Unknown adapter '{name}'. Use one of {', '.join(_ADAPTERS)} or a dotted "
        "class path like package.module:AdapterClass."
    )
