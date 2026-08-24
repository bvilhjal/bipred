# bipred web service

A small web front end for bipred: upload two GWAS summary-statistics files,
get back the joint two-trait estimates — genetic correlation, per-trait SNP
heritability, a model-implied MiXeR-style overlap summary, an unfiltered
fitted-panel LDSC-style moment diagnostic, and (optionally) per-trait
posterior-mean weight files — plus a harmonization report and full provenance.
The results page
breaks the harmonization report down per trait — rows in, and the count
removed by each QC step (non-finite values, duplicates, low per-variant N,
MAF/INFO floors) and by reference alignment (unmatched, palindromic,
allele-mismatch). It labels the block-jackknife SE on the moment-estimator
`r_g` separately from posterior SD across retained, autocorrelated joint-fit
sweeps; the latter is not a frequentist standard error or convergence test.
Critical fit warnings are visible and quarantine estimates and weight files.

This directory is **not** part of the installed `bipred` package. It is a
single-VM deployment: one web process, fit jobs as subprocesses, files on
local disk. There is no authentication; job URLs are unguessable tokens.
Put it behind TLS and access control if you expose it beyond a trusted
network.

## Quick start

From the repository checkout, with the `web` extra installed:

```bash
python -m pip install -e ".[web,fast]"
python -m webapp            # serves http://127.0.0.1:8000
```

Open the page and either upload two sumstats files or press the demo button.
The demo pair is synthetic (12k variants, known truth in
`webapp_data/caches/demo/meta.json`) and finishes in seconds; it exists to
exercise the flow, not to characterize the method — `benchmarks/` owns that.

Sumstats parsing is ldpred3's: TSV/TSV.GZ with common column aliases
(rsID/SNP, A1/A2, BETA/OR, SE, P, CHR/BP, EAF) detected automatically;
unusual headers can be mapped with `FIELD=COLUMN` overrides in the advanced
section. Each trait can use a detected per-variant N column, a constant
effective N, or a case/control split. The result records the actual basis and
the retained N range.

### Fetching from the GWAS Catalog

Instead of uploading a file, either trait can be a **GCST accession**
(e.g. `GCST90446168`, found on the
[GWAS Catalog](https://www.ebi.ac.uk/gwas/) study page). Typing one in
resolves it live — trait name, sample size, harmonised-file size — and
prefills the label and N when you have not entered them (case/control
counts give `4/(1/ncase+1/nctrl)`; your own entries always win). On submit
the runner downloads the harmonised file as a first `download` stage,
streams it while keeping only variants in the selected LD reference (the
raw files are ~90% off-reference variants), and normalizes the two catalog
layouts — the `hm_`-prefixed 2015-era schema and the current one, with
effects carried as beta, log(odds ratio), or z·se — into the plain TSV the
fit consumes. The schema handling is adapted from
`ldpred3/benchmarks/gwas_catalog_harvest.py`. Resolution caches the catalog's
harmonised-file index for a week and per-study metadata indefinitely under
`<data dir>/_meta/gwascat/`; the download itself records kept/seen counts
and the effect provenance on the results page. Requires network access on
the host, both at submit time and in the fit subprocess.

The `/catalog` page reads LDpred3's canonical, hashed benchmark registry
directly from the sibling checkout. It currently exposes 49 accessions that
completed an end-to-end fit and 37 documented rejected/failed deposits (36
preflight rejections plus one fit-stage failure), then merges later attempts
from this deployment. A completed LDpred3 phenotype satisfies the shared
input/QC/harmonization/LD contracts used by bipred; that is an input-
compatibility statement, not scientific endorsement of the phenotype or
estimate. The page reports the canonical table and registry hashes, row-level
runtime/peak RSS, and the current-profile CPU, OS, Python/numerical stack,
chains, sweeps, and worker/thread settings. It explicitly separates those 11
current-profile runs from the 38 legacy rows whose complete host snapshot was
not retained.

Local successes are recorded when a job completes. Deterministic failures —
no such study, no harmonised file, empty reference overlap, or structural
schema/QC failure — keep their reason; transient 5xx, timeout, and DNS errors
do not poison the registry. A later success upgrades an earlier failure. The
local registry lives in `<data dir>/_meta/gwascat/accessions.json`.

Two browser-side conveniences, both advisory only (the runner remains the
authority):

- Selecting a file shows a live header preview under the input — recognized
  fields mapped to columns, plus a warning for anything required that is
  missing — using the same alias table as the server. For `.gz` files this
  needs the browser's `DecompressionStream` (all current browsers); without
  it the preview degrades to a note and detection happens when the job runs.
- Validation errors re-render the form inline with your entries preserved
  (browsers clear file selections on reload, so files must be re-picked).

The job page updates live: a small poller hits `GET /jobs/<id>/status`
(JSON: `status`, `stage`, per-stage seconds, harmonization counts, `error`)
every 2 s and redirects to the results on completion. During the catalog
`download` stage the runner also reports compressed bytes read once a
second, which the page shows as MB read, percentage of the file size found
at resolve time, and MB/s. With JavaScript
disabled the page still renders the last server-side state and offers a
reload link.

## How it runs

Uploads are written under a `staging` job and transition atomically to
`queued` only after both inputs are durable. An in-process supervisor launches at
most `BIPRED_WEB_CONCURRENCY` fit subprocesses (`python -m webapp.runner`),
each pinned to one numerical thread (the same BLAS pins as the test suite, so
N jobs never oversubscribe the host and numerics stay deterministic).
`job.json` tracks validate → harmonize → ldsc → fit → (weights) with
per-stage seconds; a persisted `launching` claim prevents duplicate starts,
and startup reconciliation fails interrupted staging/running jobs rather than
leaving them stuck. The runner writes `result.json`, `munge.json`, and optional
weight files. Results record input/cache hashes, N basis and range, column
overrides, screen and sampling-error assumptions, source revision, CPU/OS/
Python/numerical backends and thread limits, wall/user/system CPU seconds,
peak RSS, and stage timings. Finished jobs are purged after
`BIPRED_WEB_TTL_DAYS` (default 7).

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `BIPRED_WEB_DATA` | `./webapp_data` | uploads, job dirs, demo cache |
| `BIPRED_WEB_CACHES` | — | real LD caches: `EUR=/path/eur.ld.npz;AFR=/path/afr.ld.npz` |
| `BIPRED_WEB_CONCURRENCY` | `2` | simultaneous fit subprocesses |
| `BIPRED_WEB_MAX_UPLOAD_MB` | `500` | per-file upload cap |
| `BIPRED_WEB_TTL_DAYS` | `7` | retention of finished jobs |
| `BIPRED_WEB_HOST` / `BIPRED_WEB_PORT` | `127.0.0.1` / `8000` | bind address |
| `BIPRED_WEB_LDPRED3_BENCHMARKS` | `../ldpred3/benchmarks` | canonical Catalog evidence directory |

## Real LD references

The default reference is the **UK Biobank European HapMap3** cache (the bigsnpr
Figshare LD reference converted to ldpred3 format, 1.05M variants from
362k samples), picked up automatically when present at its conventional
workspace location
`../ldpred3/benchmarks/.work/ldref-hm3/ldpred3_ldref_hm3.npz` — the same file
the real-data benchmarks use. The synthetic demo cache is accepted only by
`/demo`; normal uploads can never silently run against it.

To build further ancestries, use ldpred3 and register each through
`BIPRED_WEB_CACHES`:

```python
from ldpred3 import compute_ld_blocks, save_ld_blocks
blocks = compute_ld_blocks(dosages, chrom=chrom, block_size=500, quantize=True)
save_ld_blocks("eur.ld.npz", blocks, variant_ids, reference_af=af,
               n_ref=n_samples, counted_allele=a1, other_allele=a2,
               chrom=chrom, pos=pos)
```

Mount the file read-only on the host; uploads must match its ancestry and
genome build (matching is by rsID plus allele pair, so a build mismatch shows
up as a near-empty harmonization report, not a wrong fit).

## Limitations

- Single VM, no queue service: the supervisor is an asyncio loop in the web
  process (run **one** uvicorn worker). Swap in a real queue if concurrency
  needs outgrow one host.
- No accounts: the job URL is the capability. Anyone with the URL can see
  that job until it is purged.
- Multi-chain fits from the CLI are not exposed yet. The LD-consistency screen
  is an opt-in single-fit sensitivity analysis with fixed, recorded parameters;
  it is not a calibrated “conservatism” dial.
- `cross_corr=0` means no correlated sampling error. The form exposes this
  assumption, but does not substitute the fitted-panel moment intercept: that
  intercept can also contain confounding.
- Estimates come from the bipred/ldpred3 reimplementations; see
  `benchmarks/RESULTS.md` for how they compare against the original LDSC and
  MiXeR programs before quoting numbers publicly.
