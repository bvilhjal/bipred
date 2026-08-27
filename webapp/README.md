# bipred web service

A small web front end for bipred: upload two GWAS summary-statistics files,
get back the joint two-trait estimates — genetic correlation, per-trait SNP
heritability, a model-implied MiXeR-style overlap summary, an unfiltered
full-reference-score LDSC-style moment diagnostic, and (optionally) per-trait
posterior-mean weight files — plus a harmonization report and full provenance.
The results page breaks the harmonization report down per trait — rows in, and the count
removed by each QC step (non-finite values, duplicates, low per-variant N,
MAF/INFO floors) and by reference alignment (unmatched, palindromic,
allele-mismatch) and by the mandatory trait-local screen. It draws a schematic
Venn of each trait's post-screen variants against the shared fitted panel, plus
per-trait attrition bars with an explicit post-screen row. It
labels the block-jackknife SE on the moment-estimator
`r_g` separately from posterior SD across retained, autocorrelated joint-fit
sweeps; the latter is not a frequentist standard error or convergence test.
Critical fit warnings are visible and quarantine estimates and weight files;
the exact effect-cancellation, slab-scale, and retained-trace divergence
statistics and thresholds are reported separately.

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
the runner first obtains each harmonised file: it reuses a stored copy when
available, otherwise downloads and normalizes the two catalog layouts — the
`hm_`-prefixed 2015-era schema and the current one,
with effects carried as beta, log(odds ratio), or z·se — then filters locally
to the selected LD reference. The raw deposits are typically ~90%
off-reference variants. The schema handling is adapted from
`ldpred3/benchmarks/gwas_catalog_harvest.py`. Resolution caches the catalog's
harmonised-file index for a week and per-study metadata indefinitely under
`<data dir>/_meta/gwascat/`, but refreshes the file's size and validators on
every submission. Preparation records kept/seen counts, whether the job used
a stored copy or a network download, and the effect provenance on the results
page. Two Catalog traits are obtained concurrently, with at most two transfer
workers. Submission requires network access for accession resolution; the fit
subprocess needs it only when no compatible stored copy exists.

Each compatible deposit generation is fetched **once for the LD references
covered by its stored union**. The acquisition stage keeps one normalised
copy per accession under `<data dir>/catalog/`, filtered to the union of the
LD references registered at the time it was built, and every job filters
that copy locally into its own job directory. Re-running an analysis —
including against a different registered reference, which is exactly what a
per-reference file could not have served — then avoids transferring the
deposit body; the results page says `(stored copy)` rather than
`(network download)`, and the job page reports the local filter instead of a
byte count. Reuse is
keyed on the accession, harmonised-file URL, reported nonzero remote byte
count, and the ETag and/or Last-Modified validator when EBI supplies one,
plus the *content* hash of the LD cache. A changed validator therefore catches
an in-place, same-size re-deposit. A network GET is conditional on, and then
checked against, those same validators before its bytes can be published. If
EBI supplies neither validator, reuse
falls back to URL and size and cannot detect that unusual same-URL, same-size
case. Stored copies outlive the jobs that fetched them and are evicted
least-recently-used past `BIPRED_WEB_STORE_GB`, never within an hour of use.
On the first run after upgrading an older deployment, a completed job-local
Catalog file is promoted into this store without network access only when its
recorded URL, LD hash, producer version, compressed hash, schema, row counts,
variant membership, and sample-size metadata all validate exactly.

Every trait, whether uploaded or obtained from the Catalog, gets a second,
LD-reference-specific cache under
`<data dir>/prepared/`. It stores the sparse variants left only after per-trait
QC, allele harmonization, effect standardization, and the mandatory trait-local
LD-consistency screen, indexed in the full LD reference's order. The key covers
the normalized input content, exact LD-cache content hash, resolved sample-size
semantics, column overrides, QC and screen settings, preparation schema, and
bipred/ldpred3 versions, plus the NumPy version and BLAS/LAPACK implementation,
version, and integer API. It also binds the stored pre-screen univariate LDSC
QC record to the exact full-reference LD-score payload and original M. Labels,
the other trait, burn-in, iterations,
sampling-error correlation, thread counts, CPU dispatch, and weight output do
not alter that per-trait work.
A rerun or an A+B then A+C analysis can therefore reuse A's complete post-screen
artifact directly. Only intersection, allele-frequency checks, and LD subsetting
still rerun for each pair.
Prepared artifacts are checksummed, published atomically, rebuilt after
corruption, and
evicted least-recently-used past `BIPRED_WEB_PREPARED_GB`.

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
(JSON: `status`, `stage`, per-stage seconds, durable stage outcomes, joint-set
counts, `error`) every 2 s and redirects to the results on completion. The
visible stages follow the reusable-data boundaries:

| Stage | Reported |
|---|---|
| **Get Catalog data** (Catalog inputs only) | The two independent traits are fetched/reused concurrently. Each trait keeps its own live line with network MB, percent, and MB/s; stored-copy reuse and waits are explicit. The completed row retains downloaded/reused outcomes for both traits. |
| **Prepare each trait** | Two trait workers independently validate/hash their input, run QC and harmonization against one shared LD owner, and perform a quick free-intercept univariate LDSC check using precomputed full-reference scores and the original M. Gross attrition or implausible h²/intercept diagnostics are reported as warnings; structurally invalid inputs stop. |
| **Run LD-consistency screen** | Each trait proceeds directly from preparation into its mandatory DENTIST-inspired screen, without waiting for the other trait. Confirmed single-threaded reentrant BLAS permits both screens to overlap; otherwise the two screen calls serialize while preparation remains concurrent, avoiding known OpenMP-OpenBLAS corruption. The post-screen artifact is then stored. |
| **Combine the two traits** | Intersect the screened traits, check allele frequencies, and subset LD. This always reruns. |
| **Run LD-score diagnostic** | Reuse the selected reference's one precomputed LD-score vector, select the paired GWAS rows, regress with the original reference M, and initialize the sampler's two h² values. A missing or invalid reference artifact stops the job rather than silently changing the fitted method; a data-dependent regression failure is recorded and uses the deterministic default start. |
| **Fit bivariate model** | Sweep *k* of *burn-in + sampling*, labelled by phase. |
| **Write prediction weights** (optional) | The trait whose file is being written, or an explicit skip when critical warnings make weights unsafe. |

Long counters are throttled per trait/activity to once a second. Semantic
transitions and completed-stage summaries are persisted, so a fast cache hit
does not disappear between the browser's polls. With JavaScript disabled the
page still renders the last server-side state and offers a reload link.

Stage schema 4 uses `acquire`, overlapping `prepare`/`screen`, `pair`, `ldsc`,
`fit`, and `weights` in `job.json` and result provenance. `active_stages` keeps
both overlapping rows truthful. The page retains schema 3's serial trait-stage
semantics, schema 2's
combined optional-screen `pair` stage and schema 1's older
`download`/`validate`/`harmonize` mapping for completed jobs.

## How it runs

Uploads are written under a `staging` job and transition atomically to
`queued` only after both inputs are durable. An in-process supervisor launches at
most `BIPRED_WEB_CONCURRENCY` fit subprocesses (`python -m webapp.runner`),
each pinned to one numerical thread (the same BLAS pins as the test suite, so
N jobs never oversubscribe the host and numerics stay deterministic).
`job.json` tracks acquire → (prepare and screen per trait) → pair → ldsc → fit
→ (weights) with per-stage seconds; pairing starts only after both trait workers
join. A persisted `launching` claim prevents duplicate starts,
and startup reconciliation deletes unpublished staging uploads and fails
interrupted running jobs rather than leaving either stuck. The runner writes
`result.json`, `munge.json`, and optional weight files. Results record
input/cache hashes, N basis and range, column
overrides, screen and sampling-error assumptions, source revision, CPU/OS/
Python/numerical backends and thread limits, wall/user/system CPU seconds,
peak RSS, and stage timings. Finished jobs and abandoned staging uploads are
purged after `BIPRED_WEB_TTL_DAYS` (default 7); stored catalog downloads are kept
independently of jobs, under the `BIPRED_WEB_STORE_GB` budget, and every
trait's complete post-screen artifact—including derived arrays from an
upload—has its own `BIPRED_WEB_PREPARED_GB` budget and can outlive the job.
LD scores are likewise immutable reference data: every runner reads the small
`<cache>.ldscores.npz` sidecar once, uses it for each pre-screen univariate QC
regression and the post-pair diagnostic, and performs only trait-dependent
regressions.
It never squares a pair-specific LD subset. Because these scores initialize
the sampler, a registered reference is incomplete without a valid sidecar.
The cache fingerprint covers both NPZ metadata and all numerical mmap payloads.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `BIPRED_WEB_DATA` | `./webapp_data` | uploads, job dirs, demo cache |
| `BIPRED_WEB_CACHES` | — | real LD caches: `EUR=/path/eur.ld.npz;AFR=/path/afr.ld.npz` |
| `BIPRED_WEB_CONCURRENCY` | `1` | simultaneous fit subprocesses; raise only when the host can hold one private paired LD panel per job |
| `BIPRED_WEB_MAX_UPLOAD_MB` | `500` | per-file upload cap |
| `BIPRED_WEB_TTL_DAYS` | `7` | retention of finished jobs and stale staging uploads; must be finite and greater than zero |
| `BIPRED_WEB_STORE_GB` | `20` | byte budget for stored catalog downloads (`0` disables the cap) |
| `BIPRED_WEB_PREPARED_GB` | `20` | byte budget for QC'd, LD-aligned, screened traits, including uploads (`0` disables the cap) |
| `BIPRED_WEB_HOST` / `BIPRED_WEB_PORT` | `127.0.0.1` / `8000` | bind address |
| `BIPRED_WEB_LDPRED3_BENCHMARKS` | `../ldpred3/benchmarks` | canonical Catalog evidence directory |

## Real LD references

Real UK Biobank European caches are picked up automatically from the sibling
ldpred3 checkout when present:

- **HapMap3+** (LDpred2's default SNP set, 1.44M variants) at
  `../ldpred3/benchmarks/.work/ldref-hm3-plus/ldpred3_ldref_hm3_plus.npz`
- **HapMap3** (1.05M variants) at
  `../ldpred3/benchmarks/.work/ldref-hm3/ldpred3_ldref_hm3.npz`

HapMap3+ is the form default when both exist. Convert them with
`ldpred3/benchmarks/convert_bigsnpr_ldref.py` (`--panel hm3plus` for the
larger set). The synthetic demo cache is accepted only by `/demo`; normal
uploads can never silently run against it.

The bundled caches are ordinary compressed NPZ files. Loading one expands its
LD payload in the fit process. The mandatory screen evaluates only the selected
rows of each original source block, rather than first copying a near-complete
principal panel. Pairing then consumes the source block by block: the complete
source and complete pair-specific subset no longer coexist, and temporary
duplication is limited to the block currently being copied. The paired panel
itself is still private to each job, which is why concurrency defaults to one.
A reference saved by LDpred3 with `mmap=True` instead lets its read-only source
pages be shared by the operating system across jobs; each job's copied
pair-specific blocks remain private, and RSS tools may still count shared
mapped pages in every process.

Both source `map.csv` files already contain one LD score per reference
variant. On first use the webapp verifies exact row-for-row rsID order and
atomically imports that `ld` column into a sidecar bound to the complete
LD-cache generation; the workspace has both sidecars prebuilt. These scores
describe the original UKB source reference. They are deliberately labelled as
the source-map definition, rather than pretending to be a fresh contraction
of the spectrally floored/LR8 fitting cache. Jobs then gather
`scores[cache_indices]` but keep M at 1,444,196 or 1,054,330. The reported
participation-ratio effective rank is `M² / sum(scores)` for that recorded
score source; it is neither the exact algebraic rank nor, for the built-ins,
the rank of the transformed fitting cache.

To build further ancestries, use ldpred3 and register each through
`BIPRED_WEB_CACHES`:

```python
from ldpred3 import compute_ld_blocks, ld_scores, save_ld_blocks
from webapp.caches import write_ld_score_sidecar
blocks = compute_ld_blocks(dosages, chrom=chrom, block_size=500, quantize=True)
save_ld_blocks("eur.ld.npz", blocks, variant_ids, mmap=True, reference_af=af,
               n_ref=n_samples, counted_allele=a1, other_allele=a2,
               chrom=chrom, pos=pos)
write_ld_score_sidecar(
    "eur.ld.npz", ld_scores(blocks),
    source="locally computed full-reference LD blocks",
    algorithm="ldpred3.ld_scores-v1")
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
- Multi-chain fits from the CLI are not exposed yet. Every web fit runs the
  mandatory LD-consistency screen with fixed, recorded parameters. It is not a
  calibrated “conservatism” dial or the complete published DENTIST workflow.
- `cross_corr=0` means no correlated sampling error. The form exposes this
  assumption, but does not substitute the cross-trait LDSC-style intercept:
  that intercept can also contain confounding.
- Estimates come from the bipred/ldpred3 reimplementations; see
  `benchmarks/RESULTS.md` for how they compare against the original LDSC and
  MiXeR programs before quoting numbers publicly.
