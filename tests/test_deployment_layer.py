"""The deployment layer lives on main, and says only true things.

It used to live on a `university-docker` branch that drifted 96 commits and
three months behind main, kept alive by a bot that opened a sync PR on every
push and then skipped forever once one went unmerged. Two READMEs meant every
sync conflicted. The branch is gone; the eight files that mattered are here.

The sharper problem was what it claimed. `docker-compose.yml` offered
`AUTH_MODE=proxy` "if behind Shibboleth/CAS/KeyCloak", and `utils/auth.py`
existed to read the headers — but nothing in the app ever imported it. Setting
that variable looked like a lock and was not one, so an operator could put the
app on a university network believing it was gated when it was open to anyone
who could reach the port.

These tests keep the deployment story honest: the files are here, they parse,
they promise nothing the code does not do, and nothing points at a branch that
no longer exists.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

DEPLOYMENT_FILES = [
    "Dockerfile", "docker-compose.yml", ".dockerignore",
    ".env.example", "UNIVERSITY_DEPLOYMENT.md",
]
# Docs a researcher or an admin actually reads.
DOCS = ["README.md", "DEPLOYMENT.md", "UNIVERSITY_DEPLOYMENT.md", "QUICKSTART.md"]


def _read(name):
    p = ROOT / name
    return p.read_text() if p.exists() else ""


@pytest.mark.parametrize("name", DEPLOYMENT_FILES)
def test_the_deployment_layer_is_on_main(name):
    assert (ROOT / name).exists(), f"{name} is missing — it used to live on a branch"


def test_compose_parses():
    yaml = pytest.importorskip("yaml")
    spec = yaml.safe_load(_read("docker-compose.yml"))
    assert "app" in spec["services"]
    assert spec["services"]["app"]["build"] == "."


def test_compose_promises_no_authentication_the_app_does_not_perform():
    """The variable that looked like a lock and was not one."""
    yaml = pytest.importorskip("yaml")
    spec = yaml.safe_load(_read("docker-compose.yml"))
    env = spec["services"]["app"].get("environment") or []
    offered = [e.split("=")[0] for e in env]
    assert not [v for v in offered if v.startswith("AUTH_")], (
        f"compose offers {offered} — the app has no authentication, so an AUTH_ "
        f"variable here tells an operator they are protected when they are not")


def test_no_config_file_offers_an_auth_switch():
    for name in (".env.example", "docker-compose.yml"):
        body = _read(name)
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue          # explaining the removal is fine
            assert "AUTH_MODE" not in stripped, f"{name} still sets AUTH_MODE: {line}"
            assert "COMPUTE_PROFILE" not in stripped, f"{name} still sets COMPUTE_PROFILE"


def test_every_variable_compose_needs_has_a_default_or_is_documented():
    comp, env = _read("docker-compose.yml"), _read(".env.example")
    referenced = set(re.findall(r"\$\{([A-Z_]+)(?::-[^}]*)?\}", comp))
    defaulted = set(re.findall(r"\$\{([A-Z_]+):-", comp))
    documented = set(re.findall(r"^#?\s*([A-Z_]+)=", env, re.M))
    missing = sorted(referenced - defaulted - documented)
    assert not missing, f"compose needs {missing} with no default and no mention in .env.example"


@pytest.mark.parametrize("name", DOCS)
def test_no_document_points_at_the_deleted_branch(name):
    body = _read(name)
    assert "university-docker" not in body, (
        f"{name} still sends people to the university-docker branch")


@pytest.mark.parametrize("name", DOCS)
def test_no_document_claims_the_app_authenticates_anyone(name):
    """Describing a proxy in front is right; claiming the app logs people in is not.

    The distinction is the subject of the sentence, not the vocabulary. "OAuth2
    Proxy handles OIDC authentication" is exactly what should be documented;
    "Docker deployment with KeyCloak OIDC authentication", said of this app, is
    the claim that made an operator think the container was gated.
    """
    body = _read(name).lower()
    forbidden = [
        "keycloak oidc sso",                              # old README bullet
        "with keycloak oidc authentication",              # old README callout
        "docker + oidc auth",                             # old README badge
        "the app reads forwarded headers and displays",   # old guide sentence
        "auth_mode=proxy` is set",                        # old troubleshooting step
    ]
    for claim in forbidden:
        assert claim not in body, f"{name} claims the app performs auth: {claim!r}"


def test_the_sync_workflow_is_gone():
    assert not (ROOT / ".github/workflows/sync-university-docker.yml").exists(), (
        "the bot that opened a sync PR on every push is still installed")


def test_the_readme_tells_a_researcher_both_ways_to_run_it():
    body = _read("README.md")
    assert "docker compose up" in body, "the server route is not shown"
    assert "UNIVERSITY_DEPLOYMENT.md" in body, "the deployment guide is not linked"


def test_the_dockerfile_copies_a_tree_that_exists():
    """It was written against a tree 96 commits old."""
    assert "COPY requirements.txt" in _read("Dockerfile")
    for needed in ("requirements.txt", "app.py", "pages", "ml", "utils"):
        assert (ROOT / needed).exists(), f"Dockerfile builds on {needed}, which is missing"


def test_the_removed_modules_are_not_imported_anywhere():
    """auth.py and compute_config.py were dropped; nothing may reach for them."""
    for py in list(ROOT.glob("*.py")) + list(ROOT.glob("pages/*.py")) + list(ROOT.glob("utils/*.py")):
        body = py.read_text()
        assert "utils.auth" not in body and "utils.compute_config" not in body, (
            f"{py.name} imports a module that no longer exists")
