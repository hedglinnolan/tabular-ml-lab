# Security

## Supported version

The latest [release](https://github.com/hedglinnolan/tabular-ml-lab/releases/latest)
is supported. Fixes land on `main` and ship in the next release; older
releases are not patched.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private
vulnerability reporting, which is enabled for this repository:

https://github.com/hedglinnolan/tabular-ml-lab/security/advisories/new

Include the version, how you installed it (release zip, terminal, or Docker),
and enough to reproduce. You will get a reply from the maintainer; there is no
bounty program.

## What the app does with data, so you know the threat model

- **It runs where you install it.** There is no hosted service and no account
  system. On a laptop the app serves only `localhost`; on a server, the compose
  file publishes port 8501 on the host and the deployment guide expects your
  institution's single sign-on in front of it. The app authenticates nobody.
- **Nothing of yours is written to disk.** Uploads, fitted models and analysis
  state live in memory for the life of a browser session. Saved sessions are
  files the user downloads; API keys typed into the sidebar are never included
  in them. The one transient write is a scratch directory used to compile the
  manuscript to PDF, deleted immediately after.
- **The optional AI interpretation sends context off the machine only if a
  cloud backend is chosen.** What goes to the provider is column names, dataset
  shape, summary statistics and the result tables on screen, not raw rows. The
  local Ollama backend sends nothing anywhere. Server-side keys set in `.env`
  are read at call time and never shown or saved.
- **The double-click starters are not code-signed.** They download uv, a
  private Python and the packages in `requirements.txt` from their official
  sources; the release zip ships with a SHA-256 checksum for the archive.

Reports about any of those promises being false are exactly what this file is
for.
