//! ARACNE reconstruction (`minet::aracne`) on a subset of variables.
//!
//! Builds the mutual-information matrix over the subset, then applies the Data
//! Processing Inequality: within every triangle (i, j, k) the weakest edge is
//! removed when its weight is below `min(w_ik, w_jk) - eps`. Surviving edges
//! are reported as present, unoriented, weighted by their MI.

use crate::mi;
use crate::recon::SubEdge;

/// Run ARACNE on the columns identified by `sub` (ascending global indices).
pub fn reconstruct(
    sub: &[usize],
    cols: &[Vec<u32>],
    levels: &[u32],
    eps: f64,
) -> Vec<SubEdge> {
    let m = sub.len();
    if m < 2 {
        return Vec::new();
    }
    let sub_cols: Vec<&[u32]> = sub.iter().map(|&g| cols[g].as_slice()).collect();
    let sub_levels: Vec<u32> = sub.iter().map(|&g| levels[g]).collect();
    let mut mim = mi::build_mim(&sub_cols, &sub_levels);

    // Data Processing Inequality. Decide removals reading the original weights,
    // then apply them (single pass, like minet).
    let mut remove = vec![vec![false; m]; m];
    for i in 0..m {
        for j in (i + 1)..m {
            for k in 0..m {
                if k == i || k == j {
                    continue;
                }
                let a = mim[i][j];
                let b = mim[i][k];
                let c = mim[j][k];
                // Only act when (i,j) is the weakest edge of the triangle.
                if a <= b && a <= c {
                    let other_min = b.min(c);
                    if a < other_min - eps {
                        remove[i][j] = true;
                        remove[j][i] = true;
                    }
                }
            }
        }
    }

    let mut edges = Vec::new();
    for i in 0..m {
        for j in (i + 1)..m {
            if remove[i][j] {
                mim[i][j] = 0.0;
            }
            if mim[i][j] > 0.0 {
                edges.push(SubEdge {
                    a: sub[i],
                    b: sub[j],
                    eorient: 1,
                    score: mim[i][j],
                });
            }
        }
    }
    edges
}
