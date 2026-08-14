"""Explicit provider registry without global import side effects."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar


ProviderT = TypeVar("ProviderT")


class ProviderRegistry(Generic[ProviderT]):
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], ProviderT]] = {}

    def register(self, provider_id: str, factory: Callable[[], ProviderT]) -> None:
        if not provider_id or provider_id.strip() != provider_id:
            raise ValueError("provider_id must be a non-empty normalized string")
        if provider_id in self._factories:
            raise KeyError(f"provider already registered: {provider_id}")
        self._factories[provider_id] = factory

    def create(self, provider_id: str) -> ProviderT:
        try:
            factory = self._factories[provider_id]
        except KeyError as error:
            raise KeyError(f"unknown provider: {provider_id}; available={self.available()}") from error
        return factory()

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
