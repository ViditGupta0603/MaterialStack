# literature_advay

Research notes for MaterialStack: where our data comes from, how credible it is, which papers and
datasets could make the band-edge (VBM/CBM) and band-alignment predictions more accurate, and what
we have measured along the way.

| File | What's in it |
|---|---|
| [reading_guide.md](reading_guide.md) | What to read and which concepts to know before working on the project, in order |
| [datasets.md](datasets.md) | Every data source: what we ingest today (with a measured credibility check) and candidate sources to add, with status |
| [papers.md](papers.md) | Papers worth knowing, with the takeaway that matters for this project |
| [findings.md](findings.md) | Dated log of analyses, bugs and numbers found in our own pipeline |
| [references.bib](references.bib) | BibTeX for everything cited here (for the BTP report) |
| [scripts/](scripts/) | Analysis scripts behind the numbers in `findings.md` |

## Conventions

- **Add, don't overwrite.** New findings go at the top of `findings.md` with a date; if a later
  result supersedes an earlier one, mark the old entry *superseded* rather than deleting it.
- **Every number needs a source**: a script in `scripts/`, a paper, or a dataset version.
- **Credibility labels** used in `datasets.md`:
  - **High**: measured data or high-level theory, independently checked against experiment.
  - **Medium**: useful but with a known bias, coverage gap, or limited cross-checking.
  - **Low**: hypothetical structures, formula-derived values, or known to be broken in our copy.
- **Status labels** for candidate sources: `idea` → `evaluating` → `ingesting` → `in use` / `rejected (why)`.
- DOIs marked *(verify)* in `references.bib` have not been checked against the publisher yet.

## Running the scripts

From the repo root, after `build-db` and `train`:

```bash
PYTHONPATH=. .venv/bin/python literature_advay/scripts/source_credibility.py   # source cross-checks (F8, F9)
PYTHONPATH=. .venv/bin/python literature_advay/scripts/edge_analysis.py        # edge-model error breakdown (F2-F6)
PYTHONPATH=. .venv/bin/python literature_advay/scripts/surface_vacuum_check.py # surface vacuum alignment + interfaces (F1, F7)
```
