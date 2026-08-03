//! Standalone offline compaction tool: reads N input SST files, compacts
//! them (drops obsolete versions and tombstones, bottommost semantics —
//! see compactor::compaction module doc comment for scope), writes one
//! output SST. Used for the RocksDB-vs-compactor logical-equivalence
//! oracle test: point this at the same pre-compaction SSTs RocksDB's own
//! `ldb compact` was run against, then diff decoded contents.

use compactor::compaction::{CompactionOptions, compact};
use compactor::format::sst_builder::SstBuilderOptions;
use std::env;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();
    let rt = tokio::runtime::Runtime::new().expect("failed to build tokio runtime");
    rt.block_on(run(args));
}

async fn run(mut args: Vec<String>) {
    let for_ingest = args
        .iter()
        .position(|a| a == "--for-ingest")
        .map(|i| {
            args.remove(i);
            true
        })
        .unwrap_or(false);

    if args.len() < 3 {
        eprintln!(
            "usage: compact_files [--for-ingest] <output.sst> <input1.sst> [input2.sst ...]\n   or: compact_files [--for-ingest] <output.sst> --list <path-list-file>"
        );
        std::process::exit(1);
    }
    let output_path = &args[1];
    let input_paths: Vec<String> = if args[2] == "--list" {
        let list_path = args.get(3).unwrap_or_else(|| {
            eprintln!("--list requires a path-list file argument");
            std::process::exit(1);
        });
        std::fs::read_to_string(list_path)
            .unwrap_or_else(|e| {
                eprintln!("failed to read {}: {}", list_path, e);
                std::process::exit(1);
            })
            .lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    } else {
        args[2..].to_vec()
    };
    let input_paths = &input_paths[..];

    let mut sources = Vec::with_capacity(input_paths.len());
    for path in input_paths {
        let reader = match compactor::io::open(Path::new(path)).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("failed to open {}: {}", path, e);
                std::process::exit(1);
            }
        };
        match compactor::compaction::load_arena_source(reader).await {
            Ok(s) => sources.push(s),
            Err(e) => {
                eprintln!("failed to load {}: {}", path, e);
                std::process::exit(1);
            }
        }
    }

    let opts = CompactionOptions {
        sst_builder: SstBuilderOptions {
            base_context_checksum: 0x4242_4242,
            external_sst_global_seqno: if for_ingest { Some(0) } else { None },
            ..Default::default()
        },
    };

    let output_bytes = match compact(sources, &opts) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("compaction failed: {}", e);
            std::process::exit(1);
        }
    };

    if let Err(e) = std::fs::write(output_path, &output_bytes) {
        eprintln!("failed to write {}: {}", output_path, e);
        std::process::exit(1);
    }
    println!(
        "wrote {} bytes to {} from {} input SSTs",
        output_bytes.len(),
        output_path,
        input_paths.len()
    );
}
