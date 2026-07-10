//! Bayesian-network structure learning by hill climbing with the BDeu score
//! (`bnlearn::hc(data, score = "bde")`).
//!
//! Empty-graph start, greedy add/delete/reverse moves that preserve acyclicity,
//! plus random restarts with perturbation. The BDeu score is decomposable, so
//! per-node local scores are cached. Orientation of each learned arc is mapped
//! back onto the ordered global pair.

use std::collections::HashMap;

use crate::recon::SubEdge;

/// Imaginary sample size for BDeu (bnlearn's `iss` default is 1).
const ISS: f64 = 1.0;

struct Scorer<'a> {
    cols: Vec<&'a [u32]>,
    levels: Vec<u32>,
    nrow: usize,
    cache: HashMap<(usize, Vec<usize>), f64>,
}

impl<'a> Scorer<'a> {
    fn new(sub: &[usize], cols: &'a [Vec<u32>], levels: &[u32]) -> Scorer<'a> {
        let sub_cols: Vec<&[u32]> = sub.iter().map(|&g| cols[g].as_slice()).collect();
        let sub_levels: Vec<u32> = sub.iter().map(|&g| levels[g]).collect();
        let nrow = sub_cols.first().map(|c| c.len()).unwrap_or(0);
        Scorer {
            cols: sub_cols,
            levels: sub_levels,
            nrow,
            cache: HashMap::new(),
        }
    }

    /// BDeu local score of `child` given `parents` (parents need not be sorted).
    fn local(&mut self, child: usize, parents: &[usize]) -> f64 {
        let mut key_parents = parents.to_vec();
        key_parents.sort_unstable();
        if let Some(&v) = self.cache.get(&(child, key_parents.clone())) {
            return v;
        }
        let v = self.compute_local(child, &key_parents);
        self.cache.insert((child, key_parents), v);
        v
    }

    fn compute_local(&self, child: usize, parents: &[usize]) -> f64 {
        let r_c = self.levels[child] as usize;
        // Number of parent configurations (mixed radix over parent cardinalities).
        let mut q: usize = 1;
        for &p in parents {
            q = q.saturating_mul(self.levels[p] as usize);
        }

        let alpha_ij = ISS / q as f64;
        let alpha_ijk = ISS / (q as f64 * r_c as f64);
        let lg_alpha_ij = libm::lgamma(alpha_ij);
        let lg_alpha_ijk = libm::lgamma(alpha_ijk);

        // Count N_ijk indexed by (parent config, child level). Only configs
        // that actually occur contribute a non-zero term.
        let mut counts: HashMap<usize, Vec<u32>> = HashMap::new();
        let child_col = self.cols[child];
        for row in 0..self.nrow {
            let mut cfg = 0usize;
            for &p in parents {
                cfg = cfg * self.levels[p] as usize + self.cols[p][row] as usize;
            }
            let entry = counts.entry(cfg).or_insert_with(|| vec![0u32; r_c]);
            entry[child_col[row] as usize] += 1;
        }

        let mut score = 0.0;
        for cell in counts.values() {
            let n_ij: u32 = cell.iter().sum();
            score += lg_alpha_ij - libm::lgamma(alpha_ij + n_ij as f64);
            for &n_ijk in cell {
                if n_ijk > 0 {
                    score += libm::lgamma(alpha_ijk + n_ijk as f64) - lg_alpha_ijk;
                }
            }
        }
        score
    }
}

/// DAG over `m` local nodes, stored as an adjacency matrix (`edge[u][v]` = u->v).
struct Dag {
    m: usize,
    edge: Vec<Vec<bool>>,
}

impl Dag {
    fn empty(m: usize) -> Dag {
        Dag {
            m,
            edge: vec![vec![false; m]; m],
        }
    }

    fn parents_of(&self, v: usize) -> Vec<usize> {
        (0..self.m).filter(|&u| self.edge[u][v]).collect()
    }

    /// Is `target` reachable from `start` following arc directions?
    fn reachable(&self, start: usize, target: usize) -> bool {
        let mut stack = vec![start];
        let mut seen = vec![false; self.m];
        seen[start] = true;
        while let Some(u) = stack.pop() {
            if u == target {
                return true;
            }
            for v in 0..self.m {
                if self.edge[u][v] && !seen[v] {
                    seen[v] = true;
                    stack.push(v);
                }
            }
        }
        false
    }

    fn total_score(&self, sc: &mut Scorer) -> f64 {
        (0..self.m)
            .map(|v| sc.local(v, &self.parents_of(v)))
            .sum()
    }
}

/// Tiny deterministic PRNG (xorshift64*), so restarts are reproducible.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Greedy hill climbing from the current DAG state until no move improves the
/// score. Mutates `dag`; returns the resulting total score.
fn climb(dag: &mut Dag, sc: &mut Scorer, max_iter: usize) -> f64 {
    let m = dag.m;
    let mut score = dag.total_score(sc);
    for _ in 0..max_iter {
        let mut best_delta = 1e-10;
        // op: (kind, from, to). kind 0=add, 1=delete, 2=reverse.
        let mut best_op: Option<(u8, usize, usize)> = None;

        for from in 0..m {
            for to in 0..m {
                if from == to {
                    continue;
                }
                if dag.edge[from][to] {
                    // DELETE from->to
                    let mut p_to = dag.parents_of(to);
                    let base_to = sc.local(to, &p_to);
                    p_to.retain(|&x| x != from);
                    let del = sc.local(to, &p_to) - base_to;
                    if del > best_delta {
                        best_delta = del;
                        best_op = Some((1, from, to));
                    }

                    // REVERSE from->to  =>  to->from
                    // Valid iff removing from->to and adding to->from stays acyclic:
                    // i.e. `from` must not reach `to` through other arcs.
                    dag.edge[from][to] = false;
                    let acyclic = !dag.reachable(from, to);
                    dag.edge[from][to] = true;
                    if acyclic {
                        // delta = change on `to` (lose parent from) + change on
                        // `from` (gain parent to)
                        let del_to = del; // score(to | p_to\{from}) - score(to | p_to)
                        let mut p_from = dag.parents_of(from);
                        let base_from = sc.local(from, &p_from);
                        p_from.push(to);
                        let del_from = sc.local(from, &p_from) - base_from;
                        let rev = del_to + del_from;
                        if rev > best_delta {
                            best_delta = rev;
                            best_op = Some((2, from, to));
                        }
                    }
                } else if !dag.edge[to][from] {
                    // No arc between the two in either direction: consider ADD.
                    // Adding from->to is acyclic iff `to` does not reach `from`.
                    if !dag.reachable(to, from) {
                        let mut p_to = dag.parents_of(to);
                        let base_to = sc.local(to, &p_to);
                        p_to.push(from);
                        let add = sc.local(to, &p_to) - base_to;
                        if add > best_delta {
                            best_delta = add;
                            best_op = Some((0, from, to));
                        }
                    }
                }
            }
        }

        match best_op {
            None => break,
            Some((0, from, to)) => {
                dag.edge[from][to] = true;
                score += best_delta;
            }
            Some((1, from, to)) => {
                dag.edge[from][to] = false;
                score += best_delta;
            }
            Some((2, from, to)) => {
                dag.edge[from][to] = false;
                dag.edge[to][from] = true;
                score += best_delta;
            }
            _ => break,
        }
    }
    score
}

/// Apply `n` random legal single-arc perturbations to `dag`.
fn perturb(dag: &mut Dag, rng: &mut Rng, n: usize) {
    let m = dag.m;
    if m < 2 {
        return;
    }
    for _ in 0..n {
        // Try a handful of times to find a legal random move.
        for _ in 0..20 {
            let from = rng.below(m);
            let mut to = rng.below(m);
            if from == to {
                to = (to + 1) % m;
            }
            if dag.edge[from][to] {
                dag.edge[from][to] = false; // delete
                break;
            } else if dag.edge[to][from] {
                // reverse if acyclic
                dag.edge[to][from] = false;
                if !dag.reachable(from, to) {
                    dag.edge[from][to] = true;
                    break;
                }
                dag.edge[to][from] = true; // revert
            } else if !dag.reachable(to, from) {
                dag.edge[from][to] = true; // add
                break;
            }
        }
    }
}

/// Learn a Bayesian network by hill climbing and return the inferred arcs as
/// oriented edges on the global pairs.
pub fn reconstruct(
    sub: &[usize],
    cols: &[Vec<u32>],
    levels: &[u32],
    restarts: usize,
    seed: u64,
) -> Vec<SubEdge> {
    let m = sub.len();
    if m < 2 {
        return Vec::new();
    }
    // hc requires each variable to have >= 2 levels; drop single-level ones.
    let active: Vec<usize> = (0..m).filter(|&i| levels[sub[i]] >= 2).collect();
    if active.len() < 2 {
        return Vec::new();
    }
    let sub_active: Vec<usize> = active.iter().map(|&i| sub[i]).collect();

    let mut sc = Scorer::new(&sub_active, cols, levels);
    let ma = sub_active.len();
    let max_iter = 10_000;

    let mut best = Dag::empty(ma);
    let mut best_score = climb(&mut best, &mut sc, max_iter);

    let mut rng = Rng(seed | 1);
    for _ in 0..restarts {
        // Restart from a perturbation of the current best.
        let mut cand = Dag {
            m: ma,
            edge: best.edge.clone(),
        };
        perturb(&mut cand, &mut rng, 1);
        let s = climb(&mut cand, &mut sc, max_iter);
        if s > best_score + 1e-10 {
            best_score = s;
            best = cand;
        }
    }

    // Emit arcs. `sub_active` is ascending in global index, so local u<v means
    // global sub_active[u] < sub_active[v].
    let mut edges = Vec::new();
    for u in 0..ma {
        for v in 0..ma {
            if best.edge[u][v] {
                let (a, b, eorient) = if sub_active[u] < sub_active[v] {
                    (sub_active[u], sub_active[v], 2) // forward a->b
                } else {
                    (sub_active[v], sub_active[u], -2) // backward b->a
                };
                edges.push(SubEdge {
                    a,
                    b,
                    eorient,
                    score: f64::NAN,
                });
            }
        }
    }
    edges
}
