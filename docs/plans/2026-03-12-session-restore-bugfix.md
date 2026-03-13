# Session Restore Bugfix Plan

## Goal
Fix the session restore/load bug where uploading a saved state file triggers:
`st.session_state.llm_backend cannot be modified after the widget with key llm_backend is instantiated.`

## User-observed acceptance target
- A previously downloaded session file can be uploaded/restored without raising the `llm_backend` widget-state mutation error.

## Scope
### In scope
- Session save/load / restore path
- Ordering of restored state relative to widget instantiation
- Handling of widget-owned UI keys during restore
- Minimal changes needed to make restore safe and repeatable

### Out of scope
- Redesign of session serialization format
- Broad changes to unrelated workflow state
- Export/methods specificity (handled in the next task)

## Likely files in scope
- `utils/session_manager.py`
- `utils/session_state.py`
- `utils/llm_ui.py`
- any page/helper where `llm_backend` widget is instantiated or restored

## Implementation guidance
1. Find where restored session state writes `llm_backend`.
2. Find where the widget with key `llm_backend` is instantiated.
3. Fix the ordering or filtering so restore does not write to a widget-owned key after widget creation.
4. Prefer a narrow, architecture-safe fix:
   - hydrate before widget creation, or
   - exclude/transcode transient widget keys during restore.
5. Keep save/load behavior compatible with existing session files if possible.

## Verification requirements
- `python3 -m py_compile` on all changed Python files
- targeted diff review
- if feasible, lightweight restore-path sanity check from code
- stop without commit

## Success criteria
- No `llm_backend` post-instantiation state mutation during restore
- Session restore remains compatible with current workflow architecture
- No broad UI-state regressions introduced
