# Contributing

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a pull request

Run the same checks used by continuous integration:

```bash
ruff check .
ruff format --check .
python -W error -m unittest discover -s tests -v
python -m compileall -q storepilot app.py tests
```

Keep pull requests focused. Forecasting and replenishment changes must include
tests that cover the business rule being changed. Do not include real customer,
sales or supplier data in issues, tests or screenshots.

## Commit messages

Use a short imperative subject, for example:

```text
Add shelf-life cap to replenishment target
Fix case-pack rounding under budget limit
Document stockout-censored sales handling
```

## Pull-request checklist

- [ ] Tests pass locally.
- [ ] New behaviour has test coverage.
- [ ] Data schema changes are documented.
- [ ] No credentials or real store data are included.
- [ ] Purchase orders remain drafts unless an explicit approval step is present.
