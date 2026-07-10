//! Writers for the scaleNet / consensus edge-list files (same layout as the R
//! package so outputs can be compared directly).

use std::fs;
use std::io::Write;
use std::path::Path;

use crate::data::Dataset;
use crate::scalenet::EdgeList;
use crate::scs::RawEdge;
use crate::util::{all_pairs, fmt};

/// Write a full scaleNet edge list (all node pairs, combn order).
pub fn write_edge_list(path: &Path, data: &Dataset, el: &EdgeList) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let mut f = fs::File::create(path)?;
    writeln!(f, "x\ty\tepresenceScore\tepresence\teorientScore\teorient\tecorr")?;
    for (idx, (a, b)) in all_pairs(data.ncol()).into_iter().enumerate() {
        let xa = &data.names[a];
        let yb = &data.names[b];
        writeln!(
            f,
            "{xa}_<<_{yb}\t{xa}\t{yb}\t{}\t{}\t{}\t{}\t{}",
            fmt(el.epresence_score[idx]),
            el.epresence[idx],
            fmt(el.eorient_score[idx]),
            fmt(el.eorient[idx]),
            fmt(el.ecorr[idx]),
        )?;
    }
    Ok(())
}

/// Write the ranked consensus "rawAvg" output.
pub fn write_raw_consensus(
    path: &Path,
    data: &Dataset,
    raw: &[RawEdge],
) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let pairs = all_pairs(data.ncol());
    let mut f = fs::File::create(path)?;
    writeln!(f, "x\ty\tavg.rank\tavg.ort\tecorr")?;
    for e in raw {
        let (a, b) = pairs[e.idx];
        let xa = &data.names[a];
        let yb = &data.names[b];
        writeln!(
            f,
            "{xa}_<<_{yb}\t{xa}\t{yb}\t{}\t{}\t{}",
            fmt(e.avg_rank),
            fmt(e.avg_ort),
            fmt(e.ecorr),
        )?;
    }
    Ok(())
}
