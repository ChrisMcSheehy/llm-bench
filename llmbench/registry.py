"""Plugin registry for evaluators (the "test modules").

Adding a new test module = drop a file in llmbench/evaluators/ that defines a
class subclassing Evaluator and decorates it with @register. It is discovered
automatically; no wiring elsewhere. This is the "microservice-style" seam —
each module is self-contained and independently registrable.

A test module can also live in a *separate* installed package, which is what makes
this a tool other people can build on rather than one they have to fork (design E4).
"""
from __future__ import annotations

import importlib
import pkgutil
from importlib.metadata import entry_points
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


#: The entry-point group a separately-installed test module declares itself under. The
#: name is part of the public contract: changing it silently unregisters every plugin
#: anyone has published.
ENTRY_POINT_GROUP = "llmbench.evaluators"


def discover() -> None:
    """Import every test module, built in or installed, so decorators fire.

    Safe to call repeatedly: already-imported modules come from the import cache.
    """
    global _discovered
    import llmbench.evaluators as pkg

    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name in {"base"}:
            continue
        importlib.import_module(f"llmbench.evaluators.{mod.name}")
    _load_installed()
    _discovered = True


def _load_installed() -> None:
    """Import test modules published as separate distributions (design E4).

    A publisher declares one line in their own `pyproject.toml`::

        [project.entry-points."llmbench.evaluators"]
        mytest = "llmbench_mytest"

    Loading is all that is required, because `@register` is what registers - the same
    single rule the built-in scan relies on. That works whether the entry point names
    the module or a class inside it, since either import runs the decorator.

    **Built-ins are imported first, deliberately.** A plugin reusing a built-in name
    still raises the clash from `register`, and doing it in this order makes the message
    describe the plugin as the newcomer, which is the true account of what happened.

    A plugin that fails to import **stops discovery** rather than being skipped. A
    quietly absent test module is indistinguishable from one that was never installed,
    and a bench that silently runs fewer tests than you asked for is the failure this
    project treats most seriously. The error names the entry point so the culprit is
    the thing you read first, not something to be worked out.

    This executes third-party code, and deliberately: installing a package already
    grants that, and nothing here is discovered that the user did not choose to install.
    """
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            ep.load()
        except Exception as exc:
            raise RuntimeError(
                f"the installed test module {ep.name!r} (entry point {ep.value!r}) could "
                f"not be loaded: {exc!r}. Uninstall the package providing it, or fix it - "
                f"llmbench will not run a suite while a test module it should have is "
                f"missing."
            ) from exc


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
