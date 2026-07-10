# xscalenet

An efficient **Rust** reimplementation of the [ScaleNet](scalenet/) R package —
network reconstruction from high-dimensional data via spectral decomposition,
plus the **Spectral Consensus Strategy (SCS)** that fuses several embedded
reconstruction methods into one robust network.

This is the **core path** of the original package: spectral variable subsets →
ARACNE and/or Bayesian hill-climbing local reconstructions → gather → consensus,
with optional PAM discretization. It keeps everything in memory and reconstructs
subgraphs in parallel, avoiding the R version's per-subgraph disk round-trips.

## Algorithm (and where it maps to the R code)

| Stage | R source | Rust module |
|-------|----------|-------------|
| Load + factor-encode input | `setEnvironment.R` | `data.rs` |
| (optional) PAM discretization | `discretize.R` | `discretize.rs` |
| Affinity = mutual-information matrix (Miller–Madow, `mi.mm`) | `LaplacianRW.R` / `minet::build.mim` | `mi.rs` |
| Random-walk Laplacian `Lrw = I − D⁻¹W`, drop zero-degree nodes | `LaplacianRW.R` | `spectral.rs` |
| Symmetric eigendecomposition + two-line "elbow" for `k` | `computeEigenVectVal.R`, `computeEigenVectBestK.R` | `spectral.rs` |
| Spectral subsets: top-`m` / bottom-`m` per eigenvector | `createSubInputData.spectral.R` | `scalenet.rs::prepare` |
| ARACNE (DPI) local reconstruction | `rMethod.aracne.R` / `minet::aracne` | `aracne.rs` |
| Bayesian hill-climbing (BDeu score) | `rMethod.hc.R` / `bnlearn::hc` | `hc.rs` |
| Gather subgraphs → presence / orientation / frequency | `scaleNet.gatherSubGraphs.R` | `scalenet.rs::run_method` |
| Threshold + Spearman sign → edge list | `scaleNet.convertToScaleNetFormat.R` | `scalenet.rs`, `output.rs` |
| SCS rank-fusion consensus | `scs.R` | `scs.rs` |

## Build

```sh
cargo build --release
```

## Usage

```sh
xscalenet <input.tsv> <output_dir> [options]
```

`input.tsv`: tab-separated, **features in columns, observations in rows**, with a
header of feature names.

| Option | Default | Meaning |
|--------|---------|---------|
| `--methods` | `aracne,bayes_hc` | reconstruction methods to embed |
| `--var-perc` | `0.2` | fraction of variables per subgraph (`m`) |
| `--eigen-perc` | `auto` | fraction of eigenvectors (`auto` = elbow heuristic) |
| `--thresh` | `0.3,0.8` | presence-frequency thresholds |
| `--epsilon` | `0.001` | ARACNE DPI tolerance |
| `--restart` | `21` | hill-climbing random restarts |
| `--seed` | `6196` | PRNG seed |
| `--discretize` | off | PAM-discretize numeric columns first |

Example (matches the README `scs` call in the R package):

```sh
xscalenet scalenet/inst/extdata/pop2mat.txt out --discretize
```

### Outputs (same layout as the R package)

```
out/globalGraph/globalNet_presFreq<t>/edgesList.<method>.txt   # per method, per threshold
out/consensusGraph/globalNet_presFreq<t>/edgesList.txt          # fused consensus network
out/consensusGraph/globalNet_presFreq<t>/edgesList.consensus.rawAvg.txt
```

Edge-list columns: `x  y  epresenceScore  epresence  eorientScore  eorient  ecorr`.

## Validation

Run against the bundled R fixtures (`scalenet/tests/scalent_results/`, generated
by the original package on `pop2mat.txt`):

| Output | Result vs R fixture |
|--------|---------------------|
| **ARACNE** edges @ 0.3 / 0.8 | **exact match** — 205/205 and 168/168, zero differences, *and* `epresenceScore`/`ecorr` agree to 15 significant digits |
| Eigenvector count / subgraph structure | exact (`k=13`, `m=9`, 24 subgraphs) |
| Consensus format, ranking, `ecorr` | matches (top edges same nodes, same order, identical `ecorr`) |
| **bayes_hc** edges | ~80 % edge overlap (121/153 @ 0.3) |

The pipeline surrounding reconstruction (MI, Laplacian, eigen, gather, convert,
Spearman sign, consensus math) is **numerically exact** against R. The bayes_hc
differences are the expected consequence of *algorithmically-faithful* (not
bit-identical) structure learning: our hill-climber differs from `bnlearn` in
move tie-breaking, restart RNG, and search order, and BN structure search is
sensitive to all three. Consensus disagreements trace entirely to bayes_hc —
ARACNE contributes none.

## Performance

The whole `pop2mat` run (both methods, both thresholds, discretization) finishes
in well under a second. The two structurally expensive stages — the O(p²) MI
matrix and the 2·(k−1) subgraph reconstructions — are parallelized with `rayon`.
The dense symmetric eigendecomposition (via `nalgebra`, pure Rust, no system
LAPACK dependency) dominates for large feature counts.

## Not implemented (out of the chosen "core path" scope)

- Alternative subset strategies: `random`, `spectralKmeans`,
  `spectralFuzzyCmeans{Order,Sample}`, `spectralBipartition`.
- Plotting / PDF outputs (eigenvalue plots, best-k plot, `discretize.plot`).

These are comparison-only variants in the original package; the default
`spectral` path is the one exercised by the package's own README example.
