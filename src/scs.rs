//! Spectral Consensus Strategy: rank-fuse the per-method edge lists into one
//! consensus network (`scs`).

use crate::data::Dataset;
use crate::scalenet::{EdgeList, Method};
use crate::util::npairs;

/// Per-method metadata used by the consensus.
pub struct MethodInfo {
    pub method: Method,
    /// Does the method provide edge orientation?
    pub ort: bool,
    /// Edge weight used for ranking: `true` = `ecorr`, `false` = `epresenceScore`.
    pub weight_ecorr: bool,
}

/// A consensus edge in the ranked "rawAvg" output.
pub struct RawEdge {
    pub idx: usize,
    pub avg_rank: f64,
    pub avg_ort: f64,
    pub ecorr: f64,
}

/// Build the consensus for one threshold from `(info, edgelist)` pairs, in the
/// method order given (later methods win ties for `ecorr`).
///
/// Returns `(consensus_edge_list, ranked_raw_edges)`.
pub fn consensus(
    data: &Dataset,
    methods: &[(MethodInfo, &EdgeList)],
) -> (EdgeList, Vec<RawEdge>) {
    let p = data.ncol();
    let n = npairs(p);

    // Per-method rank (1-based) of each inferred edge, and its edge count.
    let mut ranks: Vec<Vec<Option<usize>>> = Vec::with_capacity(methods.len());
    let mut counts: Vec<usize> = Vec::with_capacity(methods.len());

    for (info, el) in methods {
        let mut inferred: Vec<usize> = (0..n).filter(|&i| el.epresence[i] == 1).collect();
        // weight for ranking
        let weight = |idx: usize| -> f64 {
            let w = if info.weight_ecorr {
                el.ecorr[idx]
            } else {
                el.epresence_score[idx]
            };
            w.abs()
        };
        // Order by |weight| descending, NaN last (stable to keep determinism).
        inferred.sort_by(|&a, &b| {
            let wa = weight(a);
            let wb = weight(b);
            match (wa.is_nan(), wb.is_nan()) {
                (true, true) => a.cmp(&b),
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => wb.partial_cmp(&wa).unwrap().then(a.cmp(&b)),
            }
        });
        let mut rank = vec![None; n];
        for (r, &idx) in inferred.iter().enumerate() {
            rank[idx] = Some(r + 1);
        }
        counts.push(inferred.len());
        ranks.push(rank);
    }

    // Union of inferred edges.
    let union: Vec<usize> = (0..n)
        .filter(|&i| methods.iter().any(|(_, el)| el.epresence[i] == 1))
        .collect();

    let mut out = EdgeList {
        epresence: vec![0; n],
        eorient: vec![f64::NAN; n],
        epresence_score: vec![f64::NAN; n],
        eorient_score: vec![f64::NAN; n],
        ecorr: vec![f64::NAN; n],
    };

    let mut raw: Vec<RawEdge> = Vec::with_capacity(union.len());

    for &idx in &union {
        // Normalised score per method with data; lower is better.
        let mut norm_sum = 0.0;
        let mut norm_cnt = 0.0;
        let mut num = 0.0; // weighted orientation numerator
        let mut den = 0.0; // weighted orientation denominator
        let mut ecorr = f64::NAN;

        for (mi, (info, el)) in methods.iter().enumerate() {
            let nm = counts[mi];
            if nm == 0 {
                continue; // method inferred nothing -> skip (its column is NA)
            }
            let denom = (nm + 1) as f64;
            let norm = match ranks[mi][idx] {
                Some(r) => r as f64 / denom,
                None => 1.0, // (nm+1)/(nm+1)
            };
            norm_sum += norm;
            norm_cnt += 1.0;

            if info.ort {
                if let Some(_r) = ranks[mi][idx] {
                    // Orientation from this method's eorient: 1->0, then /2.
                    let mut o = el.eorient[idx];
                    if o == 1.0 {
                        o = 0.0;
                    }
                    let ortv = o / 2.0; // 2->1, -2->-1, 0->0
                    let w = 1.0 - norm;
                    num += w * ortv;
                    den += w;
                }
            }

            // ecorr: last method (in order) that inferred this edge wins.
            if el.epresence[idx] == 1 && !el.ecorr[idx].is_nan() {
                ecorr = el.ecorr[idx];
            }
        }

        let score_mean = if norm_cnt > 0.0 {
            norm_sum / norm_cnt
        } else {
            f64::NAN
        };
        let ort_wmean = if den != 0.0 { num / den } else { 0.0 };

        out.epresence[idx] = 1;
        out.epresence_score[idx] = score_mean;
        out.eorient_score[idx] = ort_wmean;
        out.eorient[idx] = if ort_wmean > 0.0 {
            1.0
        } else if ort_wmean < 0.0 {
            -1.0
        } else {
            0.0
        };
        out.ecorr[idx] = ecorr;

        raw.push(RawEdge {
            idx,
            avg_rank: score_mean,
            avg_ort: ort_wmean,
            ecorr,
        });
    }

    // Ranked raw output: ascending consensus score (best first).
    raw.sort_by(|a, b| {
        a.avg_rank
            .partial_cmp(&b.avg_rank)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    (out, raw)
}
