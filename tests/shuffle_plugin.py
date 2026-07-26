"""Shuffle collected test items without disturbing conftest discovery."""
import os, random

def pytest_collection_modifyitems(session, config, items):
    seed = int(os.environ.get("SHUFFLE_SEED", "0"))
    random.Random(seed).shuffle(items)
    print(f"\n[shuffle] {len(items)} items shuffled with seed {seed}")
