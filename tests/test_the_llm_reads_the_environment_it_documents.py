"""The LLM reads the deployment variables the deployment files set.

docker-compose.yml has passed LLM_BACKEND, OLLAMA_BASE_URL, VLLM_BASE_URL,
OPENAI_API_KEY and ANTHROPIC_API_KEY into the app container since the
deployment layer was written, and UNIVERSITY_DEPLOYMENT.md documented four ways
to set them. Until Sep 2026 no Python file read any of them: the compose stack
pointed the app at an Ollama sidecar (`http://ollama:11434`) while
utils/llm_ui.py dialed a hard-coded `localhost:11434`, so from inside the
container the sidecar was unreachable and the feature failed with advice to run
`ollama serve`. The same shape had already been retired twice in this repository
(AUTH_MODE, COMPUTE_PROFILE), each time by deleting the variable. This time the
variables are made real, and this file keeps the two lists — what compose SETS
and what the module READS — from drifting apart again.
"""
from __future__ import annotations

import contextlib
import pathlib
import re
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ── the mirror: every variable compose sets, the app reads ─────────────────

def _compose_app_env_names():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    app_block = text.split("\n  app:", 1)[1].split("\n  ollama:", 1)[0]
    return set(re.findall(r"^\s*-\s*([A-Z_]+)=\$\{", app_block, re.MULTILINE))


def test_the_parser_sees_the_variables_it_is_about_to_check():
    """Positive control: an empty set would make the assertion below vacuous."""
    assert "OLLAMA_BASE_URL" in _compose_app_env_names()


def test_every_llm_variable_compose_sets_is_read_by_the_app():
    names = {n for n in _compose_app_env_names()
             if n.startswith(("LLM_", "OLLAMA_", "VLLM_", "OPENAI_", "ANTHROPIC_"))}
    assert names, "no LLM variables parsed out of docker-compose.yml's app service"
    source = (ROOT / "utils" / "llm_ui.py").read_text(encoding="utf-8")
    unread = sorted(n for n in names if f'"{n}"' not in source)
    assert not unread, (
        f"docker-compose.yml sets {unread} for the app and utils/llm_ui.py never "
        f"reads them — the AUTH_MODE / COMPUTE_PROFILE shape again: a setting that "
        f"looks like configuration and is not")


# ── the environment sets defaults ──────────────────────────────────────────

@pytest.fixture
def llm_ui():
    import utils.llm_ui as module
    return module


def test_the_backend_default_comes_from_the_environment(monkeypatch, llm_ui):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    assert llm_ui.env_llm_backend() == "ollama"
    monkeypatch.setenv("LLM_BACKEND", "Anthropic")
    assert llm_ui.env_llm_backend() == "anthropic"
    monkeypatch.setenv("LLM_BACKEND", "disabled")
    assert llm_ui.llm_disabled()
    # .env.example once listed "vllm" as a backend name. vLLM is the openai
    # backend pointed at VLLM_BASE_URL; an unknown name must not select a
    # backend nobody wrote.
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    assert llm_ui.env_llm_backend() == "ollama"


def test_the_ollama_address_comes_from_the_environment(monkeypatch, llm_ui):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert llm_ui.env_ollama_url() == "http://localhost:11434"
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434/")
    assert llm_ui.env_ollama_url() == "http://ollama:11434"


# ── the calls ──────────────────────────────────────────────────────────────

def _fake_anthropic(recorder, blocks):
    mod = types.ModuleType("anthropic")

    class _Messages:
        def create(self, **kwargs):
            recorder.update(kwargs)
            return types.SimpleNamespace(content=blocks)

    class Anthropic:
        def __init__(self, api_key=None):
            recorder["api_key"] = api_key
            self.messages = _Messages()

    mod.Anthropic = Anthropic
    return mod


def _fake_openai(recorder):
    mod = types.ModuleType("openai")

    class _Completions:
        def create(self, **kwargs):
            recorder.update(kwargs)
            message = types.SimpleNamespace(content=" fine ")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class OpenAI:
        def __init__(self, api_key=None, base_url=None):
            recorder["api_key"] = api_key
            recorder["base_url"] = base_url
            self.chat = types.SimpleNamespace(completions=_Completions())

    mod.OpenAI = OpenAI
    return mod


def test_the_anthropic_call_reads_the_text_block_not_the_first_block(monkeypatch, llm_ui):
    """Models that think adaptively put a `thinking` block before the text;
    `content[0].text` was the old code and it raises on those models."""
    recorder = {}
    blocks = [types.SimpleNamespace(type="thinking", thinking=""),
              types.SimpleNamespace(type="text", text="  The model is well calibrated.  ")]
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic(recorder, blocks))
    out = llm_ui._call_anthropic("ctx", "sys", "claude-sonnet-5", "sk-test")
    assert out == "The model is well calibrated."
    assert recorder["model"] == "claude-sonnet-5"
    assert recorder["api_key"] == "sk-test"
    assert "temperature" not in recorder, (
        "current Claude models reject sampling parameters with a 400")


def test_an_openai_compatible_endpoint_is_dialed_when_the_environment_names_one(monkeypatch, llm_ui):
    recorder = {}
    monkeypatch.setitem(sys.modules, "openai", _fake_openai(recorder))
    out = llm_ui._call_llm("ctx", "sys", backend="openai", model="", api_key="",
                           openai_base_url="http://vllm:8000/v1")
    assert out == "fine"
    assert recorder["base_url"] == "http://vllm:8000/v1"
    assert recorder["api_key"], "vLLM wants a non-empty key; the app supplies a placeholder"


# ── the runner: sidebar over environment, secrets stay server-side ─────────

class _FakeStreamlit:
    def __init__(self):
        self.session_state = {}

    def spinner(self, *_args, **_kwargs):
        return contextlib.nullcontext()


def test_a_server_side_key_is_used_when_the_sidebar_is_blank_and_never_stored(monkeypatch, llm_ui):
    st = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", st)
    monkeypatch.setenv("LLM_BACKEND", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-server")
    recorder = {}
    monkeypatch.setitem(sys.modules, "anthropic",
                        _fake_anthropic(recorder, [types.SimpleNamespace(type="text", text="ok")]))
    llm_ui._run_llm_call("ctx", "", "result_key")
    assert st.session_state["result_key"] == "ok"
    assert recorder["api_key"] == "sk-server"
    assert "anthropic_api_key" not in st.session_state, (
        "a server key copied into session state would reach a saved session")


def test_the_sidebar_key_wins_over_the_server_key(monkeypatch, llm_ui):
    st = _FakeStreamlit()
    st.session_state.update({"llm_backend": "anthropic", "anthropic_api_key": "sk-mine"})
    monkeypatch.setitem(sys.modules, "streamlit", st)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-server")
    recorder = {}
    monkeypatch.setitem(sys.modules, "anthropic",
                        _fake_anthropic(recorder, [types.SimpleNamespace(type="text", text="ok")]))
    llm_ui._run_llm_call("ctx", "", "result_key")
    assert recorder["api_key"] == "sk-mine"


def test_the_compose_sidecar_address_reaches_the_ollama_call(monkeypatch, llm_ui):
    st = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", st)
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    seen = {}

    def fake_ollama(context, system_prompt, model, url):
        seen["url"] = url
        return "hi"

    monkeypatch.setattr(llm_ui, "_call_ollama", fake_ollama)
    llm_ui._run_llm_call("ctx", "", "rk")
    assert seen["url"] == "http://ollama:11434", (
        "the compose stack sets OLLAMA_BASE_URL to the sidecar; a hard-coded "
        "localhost here is the app dialing its own container")
    assert st.session_state["rk"] == "hi"


def test_disabled_renders_nothing_and_calls_nothing(monkeypatch, llm_ui):
    st = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", st)
    monkeypatch.setenv("LLM_BACKEND", "disabled")
    called = []
    monkeypatch.setattr(llm_ui, "_call_llm", lambda *a, **k: called.append(1))
    llm_ui.render_interpretation_with_llm_button("ctx", key="k")
    llm_ui._run_llm_call("ctx", "", "rk")
    assert not called
    assert "rk" not in st.session_state
