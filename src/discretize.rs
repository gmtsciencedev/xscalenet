//! Optional PAM-based discretization of numeric columns, mirroring the R
//! `discretize` (default `pamk`) with Tukey outlier handling and small-cluster
//! merging. Non-numeric columns are passed through unchanged.

/// Discretize a column-major raw string matrix. Numeric columns are replaced by
/// integer cluster labels (as strings, 1-based, ordered by medoid value).
pub fn discretize_raw(raw: &[Vec<String>], max_clusters: usize) -> Vec<Vec<String>> {
    raw.iter()
        .map(|col| match parse_numeric(col) {
            Some(vals) => discretize_column(&vals, max_clusters),
            None => col.clone(),
        })
        .collect()
}

/// Parse a column to `f64`, with `NaN` for empty/`NA`. Returns `None` if any
/// non-empty cell is not numeric (i.e. the column is categorical).
fn parse_numeric(col: &[String]) -> Option<Vec<f64>> {
    let mut out = Vec::with_capacity(col.len());
    for c in col {
        let t = c.trim();
        if t.is_empty() || t == "NA" || t == "NaN" {
            out.push(f64::NAN);
        } else {
            match t.parse::<f64>() {
                Ok(v) => out.push(v),
                Err(_) => return None,
            }
        }
    }
    Some(out)
}

fn discretize_column(values: &[f64], max_clusters: usize) -> Vec<String> {
    let n = values.len();
    let not_na: Vec<usize> = (0..n).filter(|&i| !values[i].is_nan()).collect();

    // Output labels; NA stays as "NA".
    let mut labels: Vec<String> = values
        .iter()
        .map(|v| if v.is_nan() { "NA".to_string() } else { "1".to_string() })
        .collect();

    if not_na.is_empty() {
        return labels;
    }
    // Too few non-NA, or constant column -> single class.
    let uniq_vals: Vec<f64> = {
        let mut u: Vec<f64> = not_na.iter().map(|&i| values[i]).collect();
        u.sort_by(|a, b| a.partial_cmp(b).unwrap());
        u.dedup();
        u
    };
    if not_na.len() < 3 || uniq_vals.len() == 1 {
        return labels; // all 1 already
    }

    // Tukey outliers using the type-7 quartiles.
    let (lowerq, upperq) = (quantile(values, 0.25), quantile(values, 0.75));
    let iqr = upperq - lowerq;
    let k_out = 1.5;
    let up_thr = upperq + iqr * k_out;
    let lo_thr = lowerq - iqr * k_out;
    let pos_out: Vec<usize> = (0..n)
        .filter(|&i| !values[i].is_nan() && values[i] > up_thr)
        .collect();
    let neg_out: Vec<usize> = (0..n)
        .filter(|&i| !values[i].is_nan() && values[i] < lo_thr)
        .collect();

    let is_out: Vec<bool> = {
        let mut o = vec![false; n];
        for &i in pos_out.iter().chain(neg_out.iter()) {
            o[i] = true;
        }
        o
    };
    // Non-NA, non-outlier indices and values.
    let core_idx: Vec<usize> = not_na.iter().cloned().filter(|&i| !is_out[i]).collect();
    let core_vals: Vec<f64> = core_idx.iter().map(|&i| values[i]).collect();

    // Cluster the core values.
    let (mut clustering, mut medoids): (Vec<usize>, Vec<f64>) = {
        let core_uniq: Vec<f64> = {
            let mut u = core_vals.clone();
            u.sort_by(|a, b| a.partial_cmp(b).unwrap());
            u.dedup();
            u
        };
        if core_uniq.len() <= 1 {
            (vec![1; core_vals.len()], vec![mean(&core_vals)])
        } else {
            let mut kmax = max_clusters;
            if core_vals.len() <= kmax {
                kmax = core_vals.len().saturating_sub(1);
            }
            let kmax = kmax.max(2);
            let (cl, med) = pamk(&core_vals, 2, kmax);
            (cl, med)
        }
    };
    let core_medoid_count = medoids.len();

    // Reassemble a per-row cluster vector (1-based), inserting outliers.
    let mut clust = vec![0usize; n]; // 0 = unassigned/NA
    for (pos, &i) in core_idx.iter().enumerate() {
        clust[i] = clustering[pos];
    }
    // Positive outliers -> largest-medoid cluster (or a new one if single cluster).
    if !pos_out.is_empty() {
        if core_medoid_count > 1 {
            let maxm = argmax(&medoids) + 1;
            for &i in &pos_out {
                clust[i] = maxm;
            }
        } else {
            for &i in &pos_out {
                clust[i] = 2;
            }
            medoids.push(mean(&pos_out.iter().map(|&i| values[i]).collect::<Vec<_>>()));
        }
    }
    // Negative outliers -> smallest-medoid cluster (or a new one).
    if !neg_out.is_empty() {
        if core_medoid_count > 1 {
            let minm = argmin(&medoids) + 1;
            for &i in &neg_out {
                clust[i] = minm;
            }
        } else {
            for &i in &neg_out {
                clust[i] = 3;
            }
            medoids.push(mean(&neg_out.iter().map(|&i| values[i]).collect::<Vec<_>>()));
        }
    }

    // Present cluster labels (non-zero), and merge small clusters.
    let assigned: Vec<usize> = (0..n).filter(|&i| clust[i] != 0).collect();
    let mut cl_vec: Vec<usize> = assigned.iter().map(|&i| clust[i]).collect();
    merge_small(&mut cl_vec, &mut medoids, core_medoid_count);
    for (pos, &i) in assigned.iter().enumerate() {
        clust[i] = cl_vec[pos];
    }

    // Order cluster labels by medoid value (ascending) -> final 1-based labels.
    let mut order: Vec<usize> = (0..medoids.len()).collect();
    order.sort_by(|&a, &b| medoids[a].partial_cmp(&medoids[b]).unwrap());
    let mut remap = vec![0usize; medoids.len() + 1];
    for (new, &old) in order.iter().enumerate() {
        remap[old + 1] = new + 1; // old cluster label (1-based) -> new
    }
    for &i in &not_na {
        if clust[i] != 0 && clust[i] <= medoids.len() {
            labels[i] = remap[clust[i]].to_string();
        }
    }
    // guard against drift in clustering length
    let _ = &mut clustering;
    labels
}

/// Merge clusters smaller than a size threshold into their nearest (by medoid)
/// neighbour, until none are too small or only two remain.
fn merge_small(cl: &mut [usize], medoids: &mut Vec<f64>, core_medoid_count: usize) {
    if cl.is_empty() {
        return;
    }
    let total = cl.len();
    loop {
        let kmax = *cl.iter().max().unwrap();
        if kmax <= 2 || medoids.len() < kmax {
            break;
        }
        let small_size = if core_medoid_count == 1 {
            (0.75 * total as f64 / 3.0).ceil() as usize
        } else {
            (0.75 * total as f64 / kmax as f64).ceil() as usize
        };
        let mut sizes = vec![0usize; kmax + 1];
        for &c in cl.iter() {
            sizes[c] += 1;
        }
        let small: Vec<usize> = (1..=kmax).filter(|&c| sizes[c] < small_size).collect();
        if small.is_empty() {
            break;
        }

        // Best pair (smallest medoid difference) involving a small cluster.
        let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
        for a in 1..=kmax {
            for b in (a + 1)..=kmax {
                let d = (medoids[a - 1] - medoids[b - 1]).abs();
                pairs.push((a, b, d));
            }
        }
        pairs.sort_by(|x, y| x.2.partial_cmp(&y.2).unwrap());
        let merge = pairs
            .iter()
            .find(|(a, b, _)| small.contains(a) || small.contains(b));
        let (ca, cb) = match merge {
            Some(&(a, b, _)) => (a, b),
            None => break,
        };

        // Renumber: merged pair becomes label kmax-1; the rest fill 1..kmax-2.
        let mut conv = vec![0usize; kmax + 1];
        conv[ca] = kmax - 1;
        conv[cb] = kmax - 1;
        let mut next = 1usize;
        for c in 1..=kmax {
            if conv[c] == 0 {
                conv[c] = next;
                next += 1;
            }
        }
        // New medoids: merged = mean of the two; others carried over.
        let mut new_med = vec![0.0f64; kmax - 1];
        new_med[kmax - 2] = (medoids[ca - 1] + medoids[cb - 1]) / 2.0;
        for c in 1..=kmax {
            let nc = conv[c];
            if nc != kmax - 1 {
                new_med[nc - 1] = medoids[c - 1];
            }
        }
        for c in cl.iter_mut() {
            *c = conv[*c];
        }
        *medoids = new_med;
    }
}

// ---- PAM (partitioning around medoids), 1-D, manhattan distance ----

fn dist(a: f64, b: f64) -> f64 {
    (a - b).abs()
}

/// Run PAM for every `k` in `[kmin, kmax]`, keep the one with the best average
/// silhouette width. Returns (1-based clustering, medoid values).
fn pamk(x: &[f64], kmin: usize, kmax: usize) -> (Vec<usize>, Vec<f64>) {
    let mut best_asw = f64::MIN;
    let mut best: (Vec<usize>, Vec<f64>) = (vec![1; x.len()], vec![mean(x)]);
    for k in kmin..=kmax {
        if k >= x.len() {
            break;
        }
        let (assign, med_idx) = pam(x, k);
        let asw = avg_silhouette(x, &assign, k);
        if asw > best_asw {
            best_asw = asw;
            // Compact to only the clusters that actually received points, so
            // labels are contiguous 1..K' and align with the medoid vector.
            let mut present: Vec<usize> = assign.clone();
            present.sort_unstable();
            present.dedup();
            let mut remap = vec![0usize; k];
            for (new, &old) in present.iter().enumerate() {
                remap[old] = new + 1;
            }
            let clustering: Vec<usize> = assign.iter().map(|&a| remap[a]).collect();
            let medoids: Vec<f64> = present.iter().map(|&c| x[med_idx[c]]).collect();
            best = (clustering, medoids);
        }
    }
    best
}

/// Classic PAM: greedy BUILD then SWAP. Returns (assignment 0-based cluster per
/// point, medoid point indices).
fn pam(x: &[f64], k: usize) -> (Vec<usize>, Vec<usize>) {
    let n = x.len();
    // BUILD
    let mut medoids: Vec<usize> = Vec::with_capacity(k);
    // first medoid: minimizes total distance
    let first = (0..n)
        .min_by(|&a, &b| {
            let sa: f64 = x.iter().map(|&v| dist(x[a], v)).sum();
            let sb: f64 = x.iter().map(|&v| dist(x[b], v)).sum();
            sa.partial_cmp(&sb).unwrap()
        })
        .unwrap();
    medoids.push(first);
    while medoids.len() < k {
        // nearest medoid distance for each point
        let nd: Vec<f64> = (0..n)
            .map(|i| medoids.iter().map(|&m| dist(x[i], x[m])).fold(f64::MAX, f64::min))
            .collect();
        let cand = (0..n)
            .filter(|i| !medoids.contains(i))
            .max_by(|&a, &b| {
                let ga: f64 = (0..n).map(|i| (nd[i] - dist(x[i], x[a])).max(0.0)).sum();
                let gb: f64 = (0..n).map(|i| (nd[i] - dist(x[i], x[b])).max(0.0)).sum();
                ga.partial_cmp(&gb).unwrap()
            })
            .unwrap();
        medoids.push(cand);
    }

    // SWAP
    loop {
        let mut best_delta = -1e-12;
        let mut best_swap: Option<(usize, usize)> = None; // (medoid position, point)
        let cur_cost = total_cost(x, &medoids);
        for mi in 0..medoids.len() {
            for h in 0..n {
                if medoids.contains(&h) {
                    continue;
                }
                let mut trial = medoids.clone();
                trial[mi] = h;
                let c = total_cost(x, &trial);
                let delta = c - cur_cost;
                if delta < best_delta {
                    best_delta = delta;
                    best_swap = Some((mi, h));
                }
            }
        }
        match best_swap {
            Some((mi, h)) => medoids[mi] = h,
            None => break,
        }
    }

    let assign: Vec<usize> = (0..n)
        .map(|i| {
            medoids
                .iter()
                .enumerate()
                .min_by(|(_, &a), (_, &b)| dist(x[i], x[a]).partial_cmp(&dist(x[i], x[b])).unwrap())
                .map(|(pos, _)| pos)
                .unwrap()
        })
        .collect();
    (assign, medoids)
}

fn total_cost(x: &[f64], medoids: &[usize]) -> f64 {
    (0..x.len())
        .map(|i| medoids.iter().map(|&m| dist(x[i], x[m])).fold(f64::MAX, f64::min))
        .sum()
}

fn avg_silhouette(x: &[f64], assign: &[usize], k: usize) -> f64 {
    let n = x.len();
    if k < 2 {
        return -1.0;
    }
    let mut s_sum = 0.0;
    for i in 0..n {
        let ci = assign[i];
        let mut same_sum = 0.0;
        let mut same_cnt = 0usize;
        let mut other = vec![(0.0f64, 0usize); k];
        for j in 0..n {
            if j == i {
                continue;
            }
            let d = dist(x[i], x[j]);
            if assign[j] == ci {
                same_sum += d;
                same_cnt += 1;
            } else {
                other[assign[j]].0 += d;
                other[assign[j]].1 += 1;
            }
        }
        let a = if same_cnt > 0 {
            same_sum / same_cnt as f64
        } else {
            0.0
        };
        let b = other
            .iter()
            .enumerate()
            .filter(|(c, (_, cnt))| *c != ci && *cnt > 0)
            .map(|(_, (sum, cnt))| sum / *cnt as f64)
            .fold(f64::MAX, f64::min);
        let s = if same_cnt == 0 || b == f64::MAX {
            0.0
        } else {
            (b - a) / a.max(b)
        };
        s_sum += s;
    }
    s_sum / n as f64
}

// ---- small numeric helpers ----

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn argmax(v: &[f64]) -> usize {
    (0..v.len())
        .max_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap())
        .unwrap()
}

fn argmin(v: &[f64]) -> usize {
    (0..v.len())
        .min_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap())
        .unwrap()
}

/// Type-7 (R default) quantile over non-NA values.
fn quantile(values: &[f64], p: f64) -> f64 {
    let mut v: Vec<f64> = values.iter().cloned().filter(|x| !x.is_nan()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return v[0];
    }
    let h = (n as f64 - 1.0) * p;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    v[lo] + (h - lo as f64) * (v[hi] - v[lo])
}
