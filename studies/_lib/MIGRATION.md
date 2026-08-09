# What has to move into `_lib/` before studies can have a `run.py`

The reports and results in `studies/` are real. The generation path is not yet
clean, and this records exactly why, so the next person does not have to
rediscover it.

## The problem

Producing today's estimates required monkeypatching six module globals of
`benchmarks/qc_factorial.py` from an external driver:

```python
Q.TRAITS, Q.PAIRS, Q.ARMS, Q.OUT, Q.validate_inputs, Q.read_aligned
```

That is not a criticism of the script -- it was written to answer one question
with a fixed design, and it does. It is the wrong shape for a study library,
and patching around it has already produced one real bug: dropping unused
traits from `TRAITS` raised `KeyError: 'LDL'`, because `main()` builds its
checksum set from hardcoded trait names.

## Four changes, in order

**1. Input validation must follow trait selection.** `main()` currently does:

```python
validate_inputs({"sumstats/jointGwasMc_LDL.txt.gz": TRAITS["LDL"][0]["path"], ...})
```

It should validate the files belonging to the traits a run actually uses.
This removes the coupling outright and makes trait subsets first-class.

**2. Trait configuration moves to the registry.** `TRAITS` becomes a lookup in
[`traits.toml`](traits.toml) rather than a literal in the script. Column
layouts become named vintages -- public releases use at least three
conventions, and which one a file uses is a property of the file, not of the
analysis.

**3. Dataset quirks become declared flags, not caller patches.** Today's run
needed three corrections applied by hand in a wrapper around `read_aligned`:

- `effect_is_odds_ratio` -- take `log()`, and note that `read_aligned` negates
  on allele flip, which is right for a beta and wrong for an OR. The correct
  transform is `sign(x) * log|x|`, since the flip of an odds ratio is its
  reciprocal. Getting this wrong produced `NaN` betas on 2,332 flipped rows.
- `frequency_is_maf` -- override with the reference AF; see `datasets.md`.
- `no_allele_frequency` -- substitute the reference AF, making the
  AF-concordance check vacuous rather than rejecting every variant.

Each belongs next to the dataset in the registry, applied once by `_lib`.

**4. Output path becomes an argument.** `qc_factorial.py` has no `--out`, so
any exploratory run either overwrites the committed artifact or requires
patching `OUT`. `real_ldl_cad.py` already has `--csv` and is the model.

## What not to change

`real_data_inputs.py` -- the manifest validation, clean-source gate and
provenance sidecars -- is the strongest part of the setup and should be reused
as-is. It is what allowed "the Windows LD reference is numerically equivalent
to the macOS one" to be a measured claim rather than a hope.

## Also worth doing at the same time

A **local manifest overlay**. `real_data_inputs.sha256` is currently both the
canonical evidence contract and the list of files a run is permitted to read.
Those are different things, and conflating them forces machine-local rebuilds
and exploratory downloads to edit a contract file. A gitignored
`benchmarks/local_inputs.sha256`, merged on top by `load_manifest`, would
remove that pressure entirely.
