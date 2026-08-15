"""Plugin registry for evaluators (the "test modules").

Adding a new test module = drop a file in llmbench/evaluators/ that defines a
class subclassing Evaluator and decorates it with @register. It is discovered
automatically; no wiring elsewhere. This is the "microservice-style" seam —
each module is self-contained and independently registrable.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from llmbench.evaluators.base import Evaluator

_REGISTRY: dict[str, Type["Evaluator"]] = {}

# Whether discover() has run. This is deliberately not "is _REGISTRY empty?" —
# anything that imports one evaluator module directly registers that module and
# only that module, and an emptiness check would then read a registry of one as
# fully populated and never look for the rest.
_discovered = False


def register(cls: Type["Evaluator"]) -> Type["Evaluator"]:
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must set a class-level `name`")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"Evaluator name clash: {name!r} already registered")
    _REGISTRY[name] = cls
    return cls


def discover() -> None:
    """Import every module under llmbench.evaluators so decorators fire.

    Safe to call repeatedly: already-imported modules come from the import cache.
    """
    global _discovered
    import llmbench.evaluators as pkg

    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name in {"base"}:
            continue
        importlib.import_module(f"llmbench.evaluators.{mod.name}")
    _discovered = True


def get(name: str) -> Type["Evaluator"]:
    if not _discovered:
        discover()
    if name not in _REGISTRY:
        raise KeyError(f"No evaluator named {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    if not _discovered:
        discover()
    return sorted(_REGISTRY)
