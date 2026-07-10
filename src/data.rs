//! Input data handling: load a tab-separated matrix (features in columns,
//! observations in rows) and encode each column into discrete integer levels.
//!
//! Mirrors the R behaviour in `setEnvironment`: every column is turned into a
//! factor and then into integer codes. We keep both the integer level codes
//! (for mutual information / Bayesian scoring) and, since the codes double as
//! the numeric representation R uses, they are reused for the Spearman sign.

use std::fs;
use std::path::Path;

/// A discrete data matrix stored column-major.
#[derive(Clone)]
pub struct Dataset {
    /// Feature (node) names, one per column.
    pub names: Vec<String>,
    /// Column-major level codes: `cols[j][i]` is the level of observation `i`
    /// for feature `j`, in `0..levels[j]`.
    pub cols: Vec<Vec<u32>>,
    /// Number of distinct levels for each column.
    pub levels: Vec<u32>,
    /// Number of observations (rows).
    pub nrow: usize,
}

impl Dataset {
    pub fn ncol(&self) -> usize {
        self.names.len()
    }

    /// Load from a TSV file with a header row of feature names.
    pub fn from_tsv<P: AsRef<Path>>(path: P) -> Result<Dataset, String> {
        let text = fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.as_ref().display()))?;
        Dataset::from_tsv_str(&text)
    }

    pub fn from_tsv_str(text: &str) -> Result<Dataset, String> {
        let (names, raw, nrow) = Dataset::parse_tsv_raw(text)?;
        Ok(Dataset::from_raw_columns(names, raw, nrow))
    }

    /// Parse a TSV into (names, column-major raw string cells, nrow) without
    /// encoding. Used when discretization must run before factor-encoding.
    pub fn parse_tsv_raw(
        text: &str,
    ) -> Result<(Vec<String>, Vec<Vec<String>>, usize), String> {
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header = lines.next().ok_or("empty input")?;
        let names: Vec<String> = header.split('\t').map(|s| s.trim().to_string()).collect();
        let ncol = names.len();

        let mut raw: Vec<Vec<String>> = vec![Vec::new(); ncol];
        for (r, line) in lines.enumerate() {
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() != ncol {
                return Err(format!(
                    "row {} has {} fields, expected {ncol}",
                    r + 1,
                    fields.len()
                ));
            }
            for (j, f) in fields.iter().enumerate() {
                raw[j].push(f.trim().to_string());
            }
        }
        let nrow = raw.first().map(|c| c.len()).unwrap_or(0);
        Ok((names, raw, nrow))
    }

    /// Build a dataset from column-major raw string cells by factor-encoding
    /// each column. Levels are ordered by first appearance of the sorted unique
    /// values so the encoding is deterministic.
    pub fn from_raw_columns(names: Vec<String>, raw: Vec<Vec<String>>, nrow: usize) -> Dataset {
        let mut cols = Vec::with_capacity(names.len());
        let mut levels = Vec::with_capacity(names.len());
        for col in &raw {
            let mut uniq: Vec<&String> = col.iter().collect();
            uniq.sort();
            uniq.dedup();
            // Map value -> code in sorted order (matches R factor level ordering).
            let coded: Vec<u32> = col
                .iter()
                .map(|v| uniq.binary_search(&v).unwrap() as u32)
                .collect();
            levels.push(uniq.len() as u32);
            cols.push(coded);
        }
        Dataset {
            names,
            cols,
            levels,
            nrow,
        }
    }

    /// Spearman correlation between two columns (average-rank Pearson),
    /// matching R's `cor(x, y, method = "spearman")` on the numeric codes.
    pub fn spearman(&self, a: usize, b: usize) -> f64 {
        let x = rank_avg(&self.cols[a]);
        let y = rank_avg(&self.cols[b]);
        pearson(&x, &y)
    }
}

/// Average ranks (ties get the mean of their positions), 1-based like R.
fn rank_avg(v: &[u32]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| v[i].cmp(&v[j]));
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && v[idx[j]] == v[idx[i]] {
            j += 1;
        }
        // positions i..j share the same value; average rank = mean of (i+1..=j)
        let avg = ((i + 1 + j) as f64) / 2.0; // mean of consecutive integers i+1..j
        for &k in &idx[i..j] {
            ranks[k] = avg;
        }
        i = j;
    }
    ranks
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n == 0.0 {
        return f64::NAN;
    }
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
    if sxx == 0.0 || syy == 0.0 {
        return f64::NAN;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}
