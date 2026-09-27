"""The event bus: thread -> asyncio delivery, overflow -> resync, cleanup."""
from __future__ import annotations

import asyncio
import threading

from turbotab.core.events import EventBus


async def _subscribed(bus: EventBus, pid: str, **kwargs):
    """Start a subscription and wait until it is registered."""
    gen = bus.subscribe(pid, **kwargs)
    first = asyncio.ensure_future(gen.__anext__())
    for _ in range(200):
        if bus.subscriber_count(pid):
            break
        await asyncio.sleep(0.005)
    return gen, first


def test_events_published_from_a_background_thread_reach_an_asyncio_subscriber():
    async def main():
        bus = EventBus()
        gen, first = await _subscribed(bus, "p1")

        def producer():
            for i in range(50):
                bus.publish("p1", "stage", {"i": i})
            bus.publish("other", "stage", {"i": -1})  # another project: not ours

        thread = threading.Thread(target=producer)
        thread.start()
        received = [await asyncio.wait_for(first, 5)]
        while len(received) < 50:
            received.append(await asyncio.wait_for(gen.__anext__(), 5))
        thread.join()
        await gen.aclose()
        return bus, received

    bus, received = asyncio.run(main())
    assert received == [("stage", {"i": i}) for i in range(50)]
    assert bus.subscriber_count("p1") == 0


def test_a_subscriber_that_falls_behind_gets_resync_then_the_newest_event():
    async def main():
        bus = EventBus(maxsize=4)
        gen, first = await _subscribed(bus, "p")
        thread = threading.Thread(
            target=lambda: [bus.publish("p", "job", {"n": n}) for n in range(10)]
        )
        thread.start()
        thread.join()
        await asyncio.sleep(0.05)  # let the loop run the handed-over callbacks
        out = [await asyncio.wait_for(first, 5)]
        while out[-1] != ("job", {"n": 9}):
            out.append(await asyncio.wait_for(gen.__anext__(), 5))
        await gen.aclose()
        return out

    out = asyncio.run(main())
    assert ("resync", {}) in out
    after = out[out.index(("resync", {})) + 1 :]
    assert after and after[-1] == ("job", {"n": 9})
    assert len(out) < 10  # the backlog was dropped, not delivered late


def test_resync_first_and_heartbeat():
    async def main():
        bus = EventBus()
        gen = bus.subscribe("p", heartbeat=0.05, resync_first=True)
        a = await asyncio.wait_for(gen.__anext__(), 1)
        b = await asyncio.wait_for(gen.__anext__(), 1)
        await gen.aclose()
        return a, b

    assert asyncio.run(main()) == (("resync", {}), ("ping", {}))


def test_publishing_with_no_subscribers_is_harmless():
    EventBus().publish("nobody", "stage", {})
