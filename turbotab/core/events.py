"""The event bus behind SSE (docs/turbotab-next/BLUEPRINT.md §5).

``publish`` may be called from any thread (stage threads, the job runner's
dispatcher, request handlers). Each subscriber owns a bounded asyncio queue on
its own event loop; events are handed over with ``call_soon_threadsafe``, so
publishing never blocks and never touches a loop from the wrong thread.

A subscriber that falls behind does not stall anyone: when its queue is full,
everything still queued for it is dropped and a single ``("resync", {})`` is
queued in its place. Every pending event is superseded by the refetch that
``resync`` asks for, so dropping all of them (rather than just the oldest) loses
nothing and frees the queue at once.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, AsyncIterator

Event = tuple[str, dict[str, Any]]

RESYNC: Event = ("resync", {})
PING: Event = ("ping", {})


class _Subscriber:
    def __init__(self, loop: asyncio.AbstractEventLoop, maxsize: int):
        self.loop = loop
        self.queue: asyncio.Queue[Event] = asyncio.Queue(maxsize)

    def offer(self, event: Event) -> None:
        """Runs on the subscriber's loop."""
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            while not self.queue.empty():
                self.queue.get_nowait()
            self.queue.put_nowait(RESYNC)
            self.queue.put_nowait(event)


class EventBus:
    def __init__(self, maxsize: int = 256):
        if maxsize < 2:
            raise ValueError("a subscriber queue needs room for resync plus one event")
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._subscribers: dict[str, set[_Subscriber]] = {}

    def publish(self, pid: str, event_type: str, data: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers.get(pid, ()))
        event = (event_type, data)
        for sub in subscribers:
            try:
                sub.loop.call_soon_threadsafe(sub.offer, event)
            except RuntimeError:  # its loop is closed; the generator will not resume
                self._remove(pid, sub)

    def subscriber_count(self, pid: str) -> int:
        with self._lock:
            return len(self._subscribers.get(pid, ()))

    async def subscribe(
        self, pid: str, *, heartbeat: float | None = None, resync_first: bool = False
    ) -> AsyncIterator[Event]:
        """Yield ``(event_type, data)`` for ``pid`` until the consumer stops.

        The subscription starts on the first ``__anext__``. With
        ``resync_first`` the first event is ``("resync", {})``: an SSE route
        should pass it, because a browser's EventSource reconnects on its own
        and whatever was published while it was away is gone. With
        ``heartbeat`` set, ``("ping", {})`` is yielded after that many idle
        seconds (the SSE route may instead do its own).
        """
        sub = _Subscriber(asyncio.get_running_loop(), self._maxsize)
        with self._lock:
            self._subscribers.setdefault(pid, set()).add(sub)
        try:
            if resync_first:
                sub.offer(RESYNC)
            while True:
                if heartbeat is None:
                    event = await sub.queue.get()
                else:
                    try:
                        event = await asyncio.wait_for(sub.queue.get(), heartbeat)
                    except asyncio.TimeoutError:
                        event = PING
                yield event
        finally:
            self._remove(pid, sub)

    def _remove(self, pid: str, sub: _Subscriber) -> None:
        with self._lock:
            subs = self._subscribers.get(pid)
            if subs is not None:
                subs.discard(sub)
                if not subs:
                    del self._subscribers[pid]
