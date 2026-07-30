# Deployment attic

Three files rescued from the `university-docker` branch before it was deleted.

They are here because they existed on that branch and **nowhere else** — not on
`main`, not on `TurboTab`, not on the branch that folded the Docker layer into
`main`. Deleting the branch would have been the only copy gone. They are kept as
reference material, not as code: nothing in the app imports any of them, and
nothing here is on an import path.

Each is byte-identical to the blob it came from.

---

## `auth.py`

`utils/auth.py` on `university-docker`, 112 lines. Reads `X-Remote-User`,
`X-Remote-Email` and `X-Remote-Name` off `st.context.headers` and renders a
"signed in as…" sidebar badge. Written for Shibboleth / CAS / Keycloak / Azure AD
sitting in front of the app as a reverse proxy.

**It was never wired up.** A grep across every `.py` on that branch finds no
importer of `utils.auth`, `get_auth_mode`, `get_current_user`, or
`render_user_badge`. That mattered, because the same branch's `docker-compose.yml`
and `.env.example` advertised `AUTH_MODE=proxy` and four `AUTH_*` header
variables — so an administrator could set them, deploy on a campus network, and
believe the app was gated while the port was open to anyone who could reach it.
The fold-into-main removed those variables rather than leave them to mislead.

Keep this if you ever want the badge. It is a reasonable starting template — the
proxy is what does the authenticating either way; this only displays who the
proxy said you are. It is not, and never was, an access control.

## `compute_config.py`

`utils/compute_config.py` on `university-docker`, 117 lines. A `ComputeProfile`
dataclass and a `PROFILES` registry — `standard`, `high_performance`,
`enterprise` — carrying hardware-aware caps for SHAP background and evaluation
size, permutation repeats, PDP grid resolution, stability bootstrap, RFE CV
folds, Optuna trials, bootstrap resamples, CV folds, NN max epochs, sensitivity
seeds and dropout repeats.

Also never imported. `COMPUTE_PROFILE` was documented in three places and read by
none. It is a coherent piece of design rather than junk, which is why it is here:
if the app ever needs to scale its expensive analyses to the machine it is on,
this is the shape that work already took once.

## `deploy.yml`

`.github/workflows/deploy.yml` on `university-docker`, 27 lines. A self-hosted
runner deploy for a `clawserver` label.

**It never ran, and could not have.** It triggers on push to `main` but existed
only on `university-docker`, and GitHub resolves workflow files from the ref that
was pushed. Kept purely as the last record of the production host's operational
details, which exist in no other ref:

- server path — `/home/claw/.openclaw/workspace/glucose-mlp-interactive`
- systemd unit — `tabular-ml-lab`
- the TeX package set the manuscript PDF needs — `texlive-latex-base`,
  `texlive-latex-recommended`, `texlive-fonts-recommended`, `texlive-latex-extra`

That last line is the one worth keeping. `ci.yml` on `main` installs the same
four packages so CI compiles the manuscript the live server does; if the server
is ever rebuilt, this is the list.
