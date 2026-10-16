# turbotab/

TurboTab v2 lives here. The contract is `docs/turbotab-next/BLUEPRINT.md`, and §1 gives the layout.

| Path | What it is |
|---|---|
| `core/` | The engine host: workspace, datastore, decisions, the stage graph, jobs, methods and stages |
| `server/` | The FastAPI app, HTTP and SSE only |
| `frontend/` | The Vite + React app (read `frontend/CLAUDE.md` first) |
| `replay.py` | `python -m turbotab.replay <bundle.zip> --data <file>` rebuilds an export from its decision log |
| `*.py` beside them | The legacy app's domain modules (`engine`, `project`, `packs`, …). They stay because Classic or `core` imports them (BLUEPRINT §9.1). Their tests are the `test_*.py` files here. |
| `sample_data/` | The fixtures the tests and journeys read |
| `requirements.txt` | The legacy domain modules' minimal environment (pandas and numpy, no scikit-learn). Classic's `tests/test_the_guided_door_installs_without_the_app.py` checks it. It is not the app's environment. |

Run it from the repository root:

```bash
make turbotab-next
```

This builds the frontend when it is missing or stale, then serves the app and API on
<http://127.0.0.1:8787/>. `make turbotab-next-dev` prints how to run the Vite dev server beside the
API.

The test tiers are CI's: `.github/workflows/v2.yml` runs the fast tier (`-m "not slow"`, in shards)
on every push, and `v2-full.yml` runs the whole suite (50 to 58 minutes). Locally, run a targeted
selection with the flags the root `CLAUDE.md` gives, and `npm run check` in `turbotab/frontend`.

The legacy TurboTab app (the Guided door: `api.py`, `web/index.html`) was retired. Its README is
`docs/turbotab/archive/GUIDED_DOOR_README.md`, and its design record is `docs/turbotab/archive/`.
