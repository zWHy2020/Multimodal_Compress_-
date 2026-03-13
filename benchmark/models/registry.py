from __future__ import annotations

from typing import Callable, Dict

from benchmark.interfaces import JSCCMethodProtocol


class JSCCMethodRegistry:
    """Factory registry for plug-in single-modal JSCC methods."""

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[[], JSCCMethodProtocol]] = {}

    def register(self, name: str, factory: Callable[[], JSCCMethodProtocol]) -> None:
        key = name.strip().lower()
        if key in self._factories:
            raise ValueError(f"JSCC method already registered: {name}")
        self._factories[key] = factory

    def create(self, name: str) -> JSCCMethodProtocol:
        key = name.strip().lower()
        if key not in self._factories:
            raise KeyError(f"Unknown JSCC method: {name}")
        return self._factories[key]()

    def available(self) -> list[str]:
        return sorted(self._factories.keys())
