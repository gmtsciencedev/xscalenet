//! Per-method ScaleNet pipeline: spectral subsets -> local reconstruction ->
//! gather across subgraphs -> per-threshold edge lists.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::aracne;
use crate::data::Dataset;
use crate::hc;
use crate::recon::SubEdge;
use crate::spectral::{self, Spectral};
use crate::util::{self, npairs, pair_index};

/// Reconstruction method selection.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Method {
    Aracne,
    BayesHc,
}

impl Method {
    pub fn name(self) -> &'static str {
        match self {
            Method::Aracne => "aracne",
            Method::BayesHc => "bayes_hc",
        }
    }
}

/// Tunables carried through the pipeline.
#[derive(Clone)]
pub struct Params {
    pub var_perc: f64,
    pub eigen_perc: Option<f64>,
    pub pres_freq_thresh: Vec<f64>,
    // aracne
    pub epsilon: f64,
    // bayes_hc
    pub restart: usize,
    pub seed: u64,
}

/// A scaleNet-style edge table over all node pairs (combn order).
pub struct EdgeList {
    pub epresence: Vec<u8>,
    pub eorient: Vec<f64>,        // NaN = NA
    pub epresence_score: Vec<f64>, // NaN = NA
    pub eorient_score: Vec<f64>,   // NaN = NA (only used by consensus)
    pub ecorr: Vec<f64>,           // NaN = NA
}

impl EdgeList {
    fn empty(p: usize) -> EdgeList {
        let n = npairs(p);
        EdgeList {
            epresence: vec![0; n],
            eorient: vec![f64::NAN; n],
            epresence_score: vec![f64::NAN; n],
            eorient_score: vec![f64::NAN; n],
            ecorr: vec![f64::NAN; n],
        }
    }
}

/// Shared preprocessing (identical across reconstruction methods).
pub struct Prep {
    pub spectral: Spectral,
    pub subset_m: usize,
    /// Subgraphs: each carries the ascending global column indices it spans.
    pub subgraphs: Vec<Vec<usize>>,
}

/// Compute the spectral embedding and derive the spectral subgraphs.
pub fn prepare(data: &Dataset, params: &Params) -> Result<Prep, String> {
    let ncol = data.ncol();
    let subset_m = (params.var_perc * ncol as f64).ceil() as usize;

    let cols: Vec<&[u32]> = data.cols.iter().map(|c| c.as_slice()).collect();
    let sp = spectral::compute(
        &cols,
        &data.levels,
        &data.names,
        ncol,
        params.eigen_perc,
        subset_m,
    )?;

    // Map surviving node name -> global column index.
    let name_to_global: HashMap<&str, usize> = data
        .names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    let q = sp.node_names.len();
    let m = subset_m.min(q);
    let mut subgraphs = Vec::new();
    // Eigenvectors 2..k (skip the trivial smallest one at column 0).
    for c in 1..sp.k {
        // (surviving-row-index, value)
        let mut order: Vec<usize> = (0..q).collect();
        order.sort_by(|&i, &j| {
            sp.eigenvectors[j][c]
                .partial_cmp(&sp.eigenvectors[i][c])
                .unwrap()
        }); // descending by value

        let top: Vec<usize> = order[..m]
            .iter()
            .map(|&r| name_to_global[sp.node_names[r].as_str()])
            .collect();
        let bottom: Vec<usize> = order[q - m..]
            .iter()
            .map(|&r| name_to_global[sp.node_names[r].as_str()])
            .collect();

        let mut pos = top;
        pos.sort_unstable(); // preserve original column order
        let mut neg = bottom;
        neg.sort_unstable();
        subgraphs.push(pos);
        subgraphs.push(neg);
    }

    Ok(Prep {
        spectral: sp,
        subset_m,
        subgraphs,
    })
}

/// Run one reconstruction method over all subgraphs and gather the results.
///
/// Returns one `EdgeList` per presence-frequency threshold (same order as
/// `params.pres_freq_thresh`).
pub fn run_method(
    data: &Dataset,
    prep: &Prep,
    method: Method,
    params: &Params,
) -> Vec<EdgeList> {
    let p = data.ncol();

    // Reconstruct every subgraph (embarrassingly parallel).
    let results: Vec<Vec<SubEdge>> = prep
        .subgraphs
        .par_iter()
        .enumerate()
        .map(|(idx, sub)| match method {
            Method::Aracne => aracne::reconstruct(sub, &data.cols, &data.levels, params.epsilon),
            Method::BayesHc => hc::reconstruct(
                sub,
                &data.cols,
                &data.levels,
                params.restart,
                params.seed.wrapping_add(idx as u64),
            ),
        })
        .collect();

    // Gather: presence / forward / backward / possible counts per pair.
    let n = npairs(p);
    let mut presence = vec![0u32; n];
    let mut forward = vec![0u32; n];
    let mut backward = vec![0u32; n];
    let mut possible = vec![0u32; n];
    let mut scores: HashMap<usize, Vec<f64>> = HashMap::new();

    for (sub, edges) in prep.subgraphs.iter().zip(results.iter()) {
        // Every pair within the subset is "possible" (visited).
        for i in 0..sub.len() {
            for j in (i + 1)..sub.len() {
                possible[pair_index(sub[i], sub[j], p)] += 1;
            }
        }
        for e in edges {
            let idx = pair_index(e.a, e.b, p);
            presence[idx] += 1;
            match e.eorient {
                2 => forward[idx] += 1,
                -2 => backward[idx] += 1,
                _ => {}
            }
            if !e.score.is_nan() {
                scores.entry(idx).or_default().push(e.score);
            }
        }
    }

    // presence.freq and presence.ort per pair.
    let mut freq = vec![f64::NAN; n];
    let mut pres_ort = vec![f64::NAN; n];
    for idx in 0..n {
        if possible[idx] > 0 {
            freq[idx] = presence[idx] as f64 / possible[idx] as f64;
        }
        if freq[idx] > 0.0 {
            pres_ort[idx] = if forward[idx] > backward[idx] {
                2.0
            } else if backward[idx] > forward[idx] {
                -2.0
            } else {
                1.0
            };
        }
    }

    // Averaged presence score (mean over unique observed values), like convert.
    let mut avg_score = vec![f64::NAN; n];
    for (&idx, vals) in &scores {
        let mut uniq: Vec<f64> = vals.clone();
        // unique() then mean (R uses unique on the concatenated strings)
        uniq.sort_by(|a, b| a.partial_cmp(b).unwrap());
        uniq.dedup();
        avg_score[idx] = uniq.iter().sum::<f64>() / uniq.len() as f64;
    }

    // Build one edge list per threshold.
    params
        .pres_freq_thresh
        .iter()
        .map(|&thr| {
            let mut el = EdgeList::empty(p);
            for idx in 0..n {
                if freq[idx].is_nan() || freq[idx] < thr {
                    continue;
                }
                el.epresence[idx] = 1;
                el.eorient[idx] = pres_ort[idx]; // 1 / 2 / -2
                el.epresence_score[idx] = avg_score[idx];
            }
            el
        })
        .collect()
}

/// Fill in the Spearman-based sign (`ecorr`) for every present edge.
pub fn fill_ecorr(data: &Dataset, el: &mut EdgeList) {
    let p = data.ncol();
    let pairs = util::all_pairs(p);
    for (idx, &(a, b)) in pairs.iter().enumerate() {
        if el.epresence[idx] == 1 {
            el.ecorr[idx] = data.spearman(a, b);
        }
    }
}
