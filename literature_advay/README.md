# literature_advay

Research notes for MaterialStack: where our data comes from, how credible it is, which papers and
datasets could make the band-edge (VBM/CBM) and band-alignment predictions more accurate, and what
we have measured along the way.

| File | What's in it |
|---|---|
| [reading_guide.md](reading_guide.md) | What to read and which concepts to know before working on the project, in order |
| [datasets.md](datasets.md) | Every data source: what we ingest today (with a measured credibility check) and candidate sources to add, with status |
| [papers.md](papers.md) | Papers worth knowing, with the takeaway that matters for this project |
| [demo_stacks.md](demo_stacks.md) | Stacks to demo the tool, with the outputs it currently gives |
| [plan.md](plan.md) | Phased plan to reach an accurate, usable alignment screener, with benchmarks and done-criteria |
| [critique.md](critique.md) | Numbered list of everything wrong with the project, severity and fix status |
| [findings.md](findings.md) | Dated log of analyses, bugs and numbers found in our own pipeline |
| [references.bib](references.bib) | BibTeX for everything cited here (for the BTP report) |
| [scripts/](scripts/) | Helper scripts: open-access paper search for measured levels, headless UI check |

## Conventions

- **Add, don't overwrite.** New findings go at the top of `findings.md` with a date; if a later
  result supersedes an earlier one, mark the old entry *superseded* rather than deleting it.
- **Every number needs a source**: `validate.py` output, a script, a paper, or a dataset version.
- **Credibility labels** used in `datasets.md`:
  - **High**: measured data or high-level theory, independently checked against experiment.
  - **Medium**: useful but with a known bias, coverage gap, or limited cross-checking.
  - **Low**: hypothetical structures, formula-derived values, or known to be broken in our copy.
- **Status labels** for candidate sources: `idea` → `evaluating` → `ingesting` → `in use` / `rejected (why)`.
- DOIs marked *(verify)* in `references.bib` have not been checked against the publisher yet.

## Running the scripts

The analysis scripts behind findings F1–F21 worked on the old SQLite database and were removed in the
2026-10-06 simplification (CHANGELOG #26); their results stay in `findings.md`. Current numbers come from
`python validate.py` in the repo root (`results/metrics.csv`). Remaining scripts:

```bash
.venv/bin/python literature_advay/scripts/find_measured_levels.py "NiO" "NiOx ionization energy UPS"   # search open-access papers for measured IE/EA
.venv/bin/python literature_advay/scripts/ui_screenshots.py "TiO2\nMAPbI3"   # headless UI check (server on :8001)
```
