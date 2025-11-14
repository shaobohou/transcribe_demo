"""Shared async utilities for running coroutines."""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_async(coro_factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """
    Run async coroutine, handling event loop conflicts.

    Args:
        coro_factory: Function that creates the coroutine

    Returns:
        Result from coroutine

    Note:
        Handles RuntimeError when asyncio.run() fails (e.g., nested event loops).
        Creates new event loop as fallback.
    """
    try:
        return asyncio.run(coro_factory())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro_factory())
        finally:
            loop.close()
