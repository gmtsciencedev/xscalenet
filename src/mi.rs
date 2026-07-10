//! Mutual information matrix with the Miller-Madow ("mi.mm") estimator, in nats.
//!
//! Reproduces `minet::build.mim(dataset, estimator = "mi.mm")`: pairwise MI,
//! negative values clamped to 0, and a zeroed diagonal.

use rayon::prelude::*;

/// Miller-Madow entropy of a discrete variable given its counts.
///
/// H_MM = H_MLE + (m - 1) / (2 N), where `m` is the number of non-empty cells.
fn entropy_mm(counts: &[u32], n: f64) -> f64 {
    let mut h = 0.0;
    let mut m = 0.0;
    for &c in counts {
        if c > 0 {
            let p = c as f64 / n;
            h -= p * p.ln();
            m += 1.0;
        }
    }
    h + (m - 1.0) / (2.0 * n)
}

/// Mutual information (nats, Miller-Madow) between two discrete columns.
///
/// `rx`/`ry` are the number of levels of `x`/`y`. Rows are assumed complete
/// (no missing values), as produced by the discrete encoding.
pub fn mi_mm(x: &[u32], y: &[u32], rx: u32, ry: u32) -> f64 {
    let n = x.len();
    let nf = n as f64;
    let rx = rx as usize;
    let ry = ry as usize;

    let mut joint = vec![0u32; rx * ry];
    let mut mx = vec![0u32; rx];
    let mut my = vec![0u32; ry];
    for i in 0..n {
        let xi = x[i] as usize;
        let yi = y[i] as usize;
        joint[xi * ry + yi] += 1;
        mx[xi] += 1;
        my[yi] += 1;
    }

    let hx = entropy_mm(&mx, nf);
    let hy = entropy_mm(&my, nf);
    let hxy = entropy_mm(&joint, nf);
    hx + hy - hxy
}

/// Build a full mutual-information matrix over the given columns.
///
/// `cols[k]` is the level-coded column and `levels[k]` its cardinality. The
/// returned matrix is symmetric, `>= 0`, with a zero diagonal.
pub fn build_mim(cols: &[&[u32]], levels: &[u32]) -> Vec<Vec<f64>> {
    let p = cols.len();
    let mut mim = vec![vec![0.0f64; p]; p];

    // Compute upper triangle in parallel, then mirror.
    let pairs: Vec<(usize, usize)> = (0..p)
        .flat_map(|i| (i + 1..p).map(move |j| (i, j)))
        .collect();
    let vals: Vec<f64> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let v = mi_mm(cols[i], cols[j], levels[i], levels[j]);
            if v < 0.0 {
                0.0
            } else {
                v
            }
        })
        .collect();
    for (k, &(i, j)) in pairs.iter().enumerate() {
        mim[i][j] = vals[k];
        mim[j][i] = vals[k];
    }
    mim
}
