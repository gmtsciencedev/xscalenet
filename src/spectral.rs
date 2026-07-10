//! Random-walk Laplacian and its spectral decomposition.
//!
//! Reproduces `LaplacianRW` + `computeEigenVectVal` + `computeEigenVectBestK`:
//! build the affinity (MIM), drop zero-degree nodes, form `Lrw = I - D^-1 W`,
//! take its symmetric eigendecomposition (R uses `eigen(symmetric = TRUE)`,
//! i.e. the lower triangle), sort eigenvalues ascending, and pick the number
//! of eigenvectors either from a fixed percentage or the two-line "elbow".

use nalgebra::DMatrix;

use crate::mi;

pub struct Spectral {
    /// Names of the nodes that survived the zero-degree filter, row order of
    /// `eigenvectors`.
    pub node_names: Vec<String>,
    /// Eigenvectors, truncated to `k` columns, sorted by ascending eigenvalue.
    /// `eigenvectors[i][c]` is component `i` of eigenvector `c`.
    pub eigenvectors: Vec<Vec<f64>>,
    /// Number of retained eigenvectors (`k`).
    pub k: usize,
    /// All eigenvalues (ascending), before truncation.
    pub eigenvalues: Vec<f64>,
}

/// Compute the spectral embedding.
///
/// * `cols` / `levels` — level-coded columns and cardinalities for every node.
/// * `names` — node names aligned with `cols`.
/// * `ncol_all` — original number of columns (used to turn `k%` into a count).
/// * `eigen_perc` — fraction of eigenvectors to keep, or `None` for the elbow.
pub fn compute(
    cols: &[&[u32]],
    levels: &[u32],
    names: &[String],
    ncol_all: usize,
    eigen_perc: Option<f64>,
    subset_m: usize,
) -> Result<Spectral, String> {
    // Affinity = mutual information matrix (>=0, zero diagonal).
    let w = mi::build_mim(cols, levels);
    let p = w.len();

    // Degrees = column/row sums.
    let degree: Vec<f64> = (0..p).map(|i| w[i].iter().sum::<f64>()).collect();

    // Identify and drop zero-degree nodes.
    let keep: Vec<usize> = (0..p).filter(|&i| degree[i] > 0.0).collect();
    if keep.is_empty() {
        return Err("all variables have zero degree".into());
    }
    if keep.len() < subset_m {
        return Err("number of connected variables is smaller than subgraph size".into());
    }
    let q = keep.len();
    let node_names: Vec<String> = keep.iter().map(|&i| names[i].clone()).collect();

    // Build the symmetric matrix R actually decomposes: Lrw = I - D^-1 W,
    // read from its lower triangle. Lower entry (i>j) is -W[i,j]/D[i]; the
    // diagonal is 1 (W has a zero diagonal). We mirror the lower triangle so
    // the matrix is exactly symmetric for nalgebra.
    let mut m = DMatrix::<f64>::zeros(q, q);
    for a in 0..q {
        let ia = keep[a];
        m[(a, a)] = 1.0;
        for b in 0..a {
            let ib = keep[b];
            let val = -w[ia][ib] / degree[ia]; // lower-triangle value (row a)
            m[(a, b)] = val;
            m[(b, a)] = val;
        }
    }

    // Symmetric eigendecomposition.
    let eig = m.symmetric_eigen();
    let vals = eig.eigenvalues;
    let vecs = eig.eigenvectors;

    // Sort ascending by eigenvalue.
    let mut order: Vec<usize> = (0..q).collect();
    order.sort_by(|&i, &j| vals[i].partial_cmp(&vals[j]).unwrap());
    let eigenvalues: Vec<f64> = order.iter().map(|&i| vals[i]).collect();

    // Determine k.
    let perc = match eigen_perc {
        Some(pc) => pc,
        None => best_k_perc(&eigenvalues),
    };
    let mut k = (perc * ncol_all as f64).ceil() as usize;
    if k < 2 {
        k = 2;
    }
    if k > q {
        k = q;
    }

    // Materialise the first k eigenvectors (column c = order[c]).
    let mut eigenvectors = vec![vec![0.0f64; k]; q];
    for c in 0..k {
        let col = order[c];
        for r in 0..q {
            eigenvectors[r][c] = vecs[(r, col)];
        }
    }

    Ok(Spectral {
        node_names,
        eigenvectors,
        k,
        eigenvalues,
    })
}

/// Simple OLS fit of `y ~ x`, returning (intercept, slope, r_squared).
fn lm(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    let slope = if sxx == 0.0 { 0.0 } else { sxy / sxx };
    let intercept = my - slope * mx;
    let r2 = if sxx == 0.0 || syy == 0.0 {
        0.0
    } else {
        (sxy * sxy) / (sxx * syy)
    };
    (intercept, slope, r2)
}

/// The two-regression elbow heuristic from `computeEigenVectBestK`.
///
/// Returns the retained fraction `best.k / p` (rounded to 4 digits like R).
fn best_k_perc(eigenvalues: &[f64]) -> f64 {
    let step = 10usize;
    let p = eigenvalues.len();
    let xs: Vec<f64> = (1..=p).map(|i| i as f64).collect();

    // First scan: symmetric windows around the middle, growing by `step`.
    let mut r_squared: Vec<f64> = Vec::new();
    let mut n = step;
    while n <= p {
        // R: n.start = ceil(p/2) - ceil(n/2); n.end = ceil(p/2) + ceil(n/2)
        let half_p = ((p as f64) / 2.0).ceil() as isize;
        let half_n = ((n as f64) / 2.0).ceil() as isize;
        let n_start = half_p - half_n;
        let n_end = half_p + half_n;
        if n_start <= 0 || n_end > p as isize {
            break;
        }
        let s = n_start as usize; // 1-based
        let e = n_end as usize;
        let (_, _, r2) = lm(&xs[s - 1..e], &eigenvalues[s - 1..e]);
        r_squared.push(r2);
        n += step;
    }
    if r_squared.is_empty() {
        // Not enough eigenvalues for the heuristic; fall back to a small default.
        return round4((2.0 / p as f64).max(0.0));
    }

    let max_r2 = r_squared.iter().cloned().fold(f64::MIN, f64::max);
    // n1 = max index (1-based, times step) where within 0.1 of the best R^2.
    let mut best_idx = 0usize;
    for (idx, &r2) in r_squared.iter().enumerate() {
        if round4(max_r2 - r2) < 0.1 {
            best_idx = idx + 1; // 1-based
        }
    }
    let n1 = best_idx * step;

    let half_p = ((p as f64) / 2.0).ceil() as isize;
    let half_n1 = ((n1 as f64) / 2.0).ceil() as isize;
    let n1_start = (half_p - half_n1).max(1) as usize;
    let n1_end = (half_p + half_n1).min(p as isize) as usize;

    let (int1, slope1, _) = lm(&xs[n1_start - 1..n1_end], &eigenvalues[n1_start - 1..n1_end]);
    let (int2, slope2, _) = lm(&xs[0..n1_start], &eigenvalues[0..n1_start]);

    let best_k = (int2 - int1) / (slope1 - slope2);
    round4(best_k / p as f64)
}

fn round4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}
