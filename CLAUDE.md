# Project instructions

## Docs and comments

Apply `.claude/skills/terse-comments` to every comment and docstring, even when a request does not name the skill. Python and Markdown files must contain ASCII text. The `spec/` directory and vendored `.claude/` skills are exempt because they contain quoted or upstream text. Do not narrate code history. Git records history.

## Layout

The `foveate_tools` package contains synthetic calendar fixtures. The project uses a flat package because it has one distribution. The agent skill and its PEP 723 script live in `skills/foveate-images`. Add a runtime dependency only when the standard library cannot provide the required behavior. The ignored `runs/` directory contains generated calendars and evaluation output.

## Workflow

Run the self-check with `uv run python3 test_foveate_tools.py`. Run `uvx ruff@0.16.3 check .`, `uvx ruff@0.16.3 format .`, and `uvx pyright@1.1.411` before delivery. Commit freely during autonomous work. During interactive work, wait for the user before a commit unless the request includes publication.

## Foveation

Raw crops are the visual evidence. Annotated crops are navigation aids. Crop bounds use half-open, source-global coordinates after EXIF normalization. An exhaustive task must inspect every top-level anchor and deduplicate overlap by core ownership.
