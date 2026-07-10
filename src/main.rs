//! Command-line entry point for xscalenet.
//!
//! Usage:
//!   xscalenet <input.tsv> <output_dir> [options]
//!
//! Options (defaults match the README `scs` example):
//!   --methods aracne,bayes_hc     reconstruction methods to embed
//!   --var-perc 0.2                fraction of variables per subgraph
//!   --eigen-perc auto             fraction of eigenvectors (auto = elbow)
//!   --thresh 0.3,0.8              presence-frequency thresholds
//!   --epsilon 0.001               ARACNE DPI tolerance
//!   --restart 21                  hill-climbing random restarts
//!   --seed 6196                   PRNG seed
//!   --discretize                  PAM-discretize numeric columns first

use std::path::PathBuf;
use std::process::exit;

use xscalenet::data::Dataset;
use xscalenet::discretize;
use xscalenet::output;
use xscalenet::scalenet::{self, Method, Params};
use xscalenet::scs::{self, MethodInfo};

struct Args {
    input: String,
    outdir: String,
    methods: Vec<Method>,
    params: Params,
    discretize: bool,
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() < 3 {
        return Err(format!(
            "usage: {} <input.tsv> <output_dir> [--methods aracne,bayes_hc] \
             [--var-perc 0.2] [--eigen-perc auto] [--thresh 0.3,0.8] \
             [--epsilon 0.001] [--restart 21] [--seed 6196] [--discretize]",
            argv.first().map(|s| s.as_str()).unwrap_or("xscalenet")
        ));
    }
    let input = argv[1].clone();
    let outdir = argv[2].clone();

    let mut methods = vec![Method::Aracne, Method::BayesHc];
    let mut discretize = false;
    let mut params = Params {
        var_perc: 0.2,
        eigen_perc: None,
        pres_freq_thresh: vec![0.3, 0.8],
        epsilon: 0.001,
        restart: 21,
        seed: 6196,
    };

    let mut i = 3;
    while i < argv.len() {
        let a = argv[i].clone();
        // Fetch the value for options that take one.
        let mut value = || -> Result<String, String> {
            i += 1;
            argv.get(i)
                .cloned()
                .ok_or_else(|| format!("missing value for {a}"))
        };
        match a.as_str() {
            "--discretize" => discretize = true,
            "--methods" => {
                methods = value()?
                    .split(',')
                    .map(|m| match m.trim() {
                        "aracne" => Ok(Method::Aracne),
                        "bayes_hc" => Ok(Method::BayesHc),
                        other => Err(format!("unknown method: {other}")),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--var-perc" => params.var_perc = value()?.parse().map_err(|e| format!("{e}"))?,
            "--eigen-perc" => {
                let v = value()?;
                params.eigen_perc = if v == "auto" || v == "-1" {
                    None
                } else {
                    Some(v.parse().map_err(|e| format!("{e}"))?)
                };
            }
            "--thresh" => {
                params.pres_freq_thresh = value()?
                    .split(',')
                    .map(|t| t.trim().parse::<f64>().map_err(|e| format!("{e}")))
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--epsilon" => params.epsilon = value()?.parse().map_err(|e| format!("{e}"))?,
            "--restart" => params.restart = value()?.parse().map_err(|e| format!("{e}"))?,
            "--seed" => params.seed = value()?.parse().map_err(|e| format!("{e}"))?,
            other => return Err(format!("unknown option: {other}")),
        }
        i += 1;
    }

    Ok(Args {
        input,
        outdir,
        methods,
        params,
        discretize,
    })
}

fn info_for(method: Method) -> MethodInfo {
    match method {
        // aracne: no orientation, ranked by presence score
        Method::Aracne => MethodInfo {
            method,
            ort: false,
            weight_ecorr: false,
        },
        // bayes_hc: oriented, ranked by correlation magnitude
        Method::BayesHc => MethodInfo {
            method,
            ort: true,
            weight_ecorr: true,
        },
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    // Load and (optionally) discretize.
    let text = std::fs::read_to_string(&args.input)
        .map_err(|e| format!("cannot read {}: {e}", args.input))?;
    let (names, raw, nrow) = Dataset::parse_tsv_raw(&text)?;
    let raw = if args.discretize {
        eprintln!("# Discretizing numeric columns (pamk)...");
        discretize::discretize_raw(&raw, 5)
    } else {
        raw
    };
    let data = Dataset::from_raw_columns(names, raw, nrow);
    eprintln!(
        "# Loaded {} variables x {} observations",
        data.ncol(),
        data.nrow
    );

    // Shared preprocessing (spectral embedding + subgraphs).
    let prep = scalenet::prepare(&data, &args.params)?;
    eprintln!(
        "# Kept k={} eigenvectors, {} subgraphs, m={} vars/subgraph",
        prep.spectral.k,
        prep.subgraphs.len(),
        prep.subset_m
    );

    // Run each method, keeping per-threshold edge lists in memory.
    // methods_lists[method_i][thresh_i]
    let mut methods_lists = Vec::new();
    for &method in &args.methods {
        eprintln!("# Reconstructing with {}...", method.name());
        let mut lists = scalenet::run_method(&data, &prep, method, &args.params);
        for el in &mut lists {
            scalenet::fill_ecorr(&data, el);
        }
        // Write per-method global edge lists.
        for (ti, thr) in args.params.pres_freq_thresh.iter().enumerate() {
            let path: PathBuf = [
                &args.outdir,
                "globalGraph",
                &format!("globalNet_presFreq{thr}"),
                &format!("edgesList.{}.txt", method.name()),
            ]
            .iter()
            .collect();
            output::write_edge_list(&path, &data, &lists[ti])
                .map_err(|e| format!("write {}: {e}", path.display()))?;
        }
        methods_lists.push(lists);
    }

    // Consensus per threshold.
    for (ti, thr) in args.params.pres_freq_thresh.iter().enumerate() {
        let pairs: Vec<(MethodInfo, &scalenet::EdgeList)> = args
            .methods
            .iter()
            .enumerate()
            .map(|(mi, &m)| (info_for(m), &methods_lists[mi][ti]))
            .collect();
        let (cons, raw) = scs::consensus(&data, &pairs);

        let base: PathBuf = [
            &args.outdir,
            "consensusGraph",
            &format!("globalNet_presFreq{thr}"),
        ]
        .iter()
        .collect();
        output::write_edge_list(&base.join("edgesList.txt"), &data, &cons)
            .map_err(|e| format!("write consensus: {e}"))?;
        output::write_raw_consensus(&base.join("edgesList.consensus.rawAvg.txt"), &data, &raw)
            .map_err(|e| format!("write consensus raw: {e}"))?;

        let n_edges = cons.epresence.iter().filter(|&&p| p == 1).count();
        eprintln!("# Consensus @ presFreq {thr}: {n_edges} edges");
    }

    eprintln!("# Done. Outputs under {}", args.outdir);
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        exit(1);
    }
}
