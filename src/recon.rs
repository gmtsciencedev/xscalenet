//! Shared types for local network reconstruction.

/// An inferred edge from a subgraph reconstruction.
///
/// `a` and `b` are global column indices with `a < b`. `eorient` follows the
/// ScaleNet convention on the ordered pair `(a, b)`:
///   * `2`  forward  (a -> b)
///   * `-2` backward (b -> a)
///   * `1`  present but unoriented
/// `score` is the edge weight (ARACNE MI weight; `NaN` for hill-climbing).
#[derive(Clone, Copy, Debug)]
pub struct SubEdge {
    pub a: usize,
    pub b: usize,
    pub eorient: i32,
    pub score: f64,
}
