//! Small helpers shared across the pipeline.

/// Number of unordered pairs among `p` items.
pub fn npairs(p: usize) -> usize {
    p * (p.saturating_sub(1)) / 2
}

/// Index of the ordered pair `(a, b)` with `a < b` in `combn` order over `p`
/// items (row-major over the upper triangle), matching R's `combn(names, 2)`.
pub fn pair_index(a: usize, b: usize, p: usize) -> usize {
    debug_assert!(a < b && b < p);
    let base = a * (p - 1) - a * (a.wrapping_sub(1)) / 2;
    base + (b - a - 1)
}

/// Enumerate all pairs in `combn` order.
pub fn all_pairs(p: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::with_capacity(npairs(p));
    for a in 0..p {
        for b in (a + 1)..p {
            v.push((a, b));
        }
    }
    v
}

/// Format a value the way the R outputs do: `NA` for missing, otherwise a
/// full-precision round-trippable number (integers without a decimal point).
pub fn fmt(x: f64) -> String {
    if x.is_nan() {
        "NA".to_string()
    } else if x == x.trunc() && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{}", x)
    }
}
