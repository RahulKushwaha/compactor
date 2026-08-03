//! End-to-end oracle test against a REAL RocksDB instance (not the
//! self-contained `oracle` crate's naive reference — this shells out to
//! `rocksdb_ldb` / `rocksdb_sst_dump`, exercising the actual thing
//! compactor is meant to replace).
//!
//! Protocol (see conversation history for why each step is shaped this
//! way — earlier attempts hit two real gotchas worth remembering):
//! 1. Run a real workload against a live RocksDB DB with
//!    `--auto_compaction=false`, then force one extra flush-triggering
//!    write+delete pair at the end — `ldb`'s single-shot delete process
//!    does not reliably flush its own tombstone to an SST before exiting
//!    when it's the LAST operation in a sequence (observed directly: the
//!    tombstone sat in the WAL and was invisible to a raw SST scan until a
//!    subsequent write forced a memtable flush). Always verify a delete's
//!    tombstone is present in an on-disk SST before trusting a snapshot.
//! 2. Capture ground truth via `ldb scan` on the untouched original DB.
//! 3. Copy the raw pre-compaction SSTs, compact them with compactor's own
//!    `compact_files --for-ingest` binary.
//! 4. `ldb ingest_extern_sst` rejects ANY raw internal SST — even ones
//!    RocksDB itself wrote — unless it carries the SstFileWriter-specific
//!    `rocksdb.external_sst_file.version`/`.global_seqno` properties
//!    (confirmed directly against a genuine RocksDB-internal file). Hence
//!    `--for-ingest`, which stamps those and forces every key's seqno to 0
//!    per SstFileWriter's own contract.
//! 5. Ingest into a FRESH empty DB (not the original) and diff `ldb scan`
//!    output against the ground truth from step 2.
//!
//! Requires `rocksdb_ldb` and `rocksdb_sst_dump` on PATH (Homebrew rocksdb
//! formula). Skips (passes trivially with a message) if unavailable, since
//! this is a real-binary integration test, not a pure-Rust unit test.

use std::path::{Path, PathBuf};
use std::process::Command;

/// `compact_files` is a binary owned by the `compactor` crate, a sibling
/// package (not a dependency of `fuzzer`'s own Cargo.toml in the binary
/// sense), so `CARGO_BIN_EXE_*` isn't set for it here. Locate it relative
/// to this test binary's own path instead: integration test binaries and
/// workspace-built binaries land in the same `target/<profile>/` directory.
fn find_compact_files_binary() -> PathBuf {
    let mut dir = std::env::current_exe().expect("failed to get current exe path");
    loop {
        dir.pop();
        let candidate = dir.join("compact_files");
        if candidate.is_file() {
            return candidate;
        }
        if dir.file_name().map(|n| n == "target").unwrap_or(false) {
            panic!(
                "could not locate compact_files binary under {}",
                dir.display()
            );
        }
        if dir.parent().is_none() {
            panic!("could not locate compact_files binary (walked up to filesystem root)");
        }
    }
}

fn have_binary(name: &str) -> bool {
    Command::new(name)
        .arg("--help")
        .output()
        .map(|o| o.status.success() || !o.stdout.is_empty() || !o.stderr.is_empty())
        .unwrap_or(false)
}

fn run_ldb(db: &Path, extra_args: &[&str]) -> (String, String, bool) {
    let mut cmd = Command::new("rocksdb_ldb");
    cmd.arg(format!("--db={}", db.display()));
    cmd.args(extra_args);
    let out = cmd.output().expect("failed to spawn rocksdb_ldb");
    (
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
        out.status.success(),
    )
}

fn sst_files_in(dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .expect("failed to read db dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "sst").unwrap_or(false))
        .collect()
}

fn sst_dump_scan(path: &Path) -> String {
    let out = Command::new("rocksdb_sst_dump")
        .args([&format!("--file={}", path.display()), "--command=scan"])
        .output()
        .expect("failed to spawn rocksdb_sst_dump");
    String::from_utf8_lossy(&out.stdout).to_string()
}

fn normalize_scan_line(line: &str) -> Option<String> {
    // ldb scan format: "key ==> value"
    // sst_dump scan format: "'key' seq:N, type:T => value"
    if let Some((k, v)) = line.split_once(" ==> ") {
        return Some(format!("{} ==> {}", k.trim(), v.trim()));
    }
    if let Some((k, v)) = line.split_once(" => ") {
        let key = k.trim().trim_matches('\'');
        return Some(format!("{} ==> {}", key, v.trim()));
    }
    None
}

#[test]
fn compactor_output_ingests_into_rocksdb_and_reads_back_identically() {
    if !have_binary("rocksdb_ldb") || !have_binary("rocksdb_sst_dump") {
        eprintln!(
            "SKIP: rocksdb_ldb / rocksdb_sst_dump not found on PATH (install via `brew install rocksdb`)"
        );
        return;
    }

    let tmp = std::env::temp_dir().join(format!("compactor_rocksdb_oracle_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let original_db = tmp.join("original_db");

    // Step 1: real workload, 3 generations of overwrites + 3 deletes.
    for generation in 1..=3 {
        for i in 1..=50 {
            let (_, stderr, ok) = run_ldb(
                &original_db,
                &[
                    "--auto_compaction=false",
                    "put",
                    &format!("key{:05}", i),
                    &format!("gen{}_value_{:05}", generation, i),
                    "--create_if_missing",
                ],
            );
            assert!(ok, "put failed: {}", stderr);
        }
    }
    for i in [5, 15, 25] {
        let (_, stderr, ok) = run_ldb(
            &original_db,
            &["--auto_compaction=false", "delete", &format!("key{:05}", i)],
        );
        assert!(ok, "delete failed: {}", stderr);
    }
    // Force-flush trailing WAL state (see module doc comment point 1):
    // one more put+delete pair whose own tombstone we don't care about,
    // just needed to push the *previous* delete's memtable entry to disk.
    run_ldb(
        &original_db,
        &["--auto_compaction=false", "put", "zzz_flush_trigger", "x"],
    );
    run_ldb(
        &original_db,
        &["--auto_compaction=false", "delete", "zzz_flush_trigger"],
    );
    run_ldb(
        &original_db,
        &["--auto_compaction=false", "put", "zzz_flush_trigger2", "x"],
    );

    // Verify every delete's tombstone actually landed in an on-disk SST
    // before trusting anything downstream.
    for key in ["key00005", "key00015", "key00025", "zzz_flush_trigger"] {
        let mut found = false;
        for sst in sst_files_in(&original_db) {
            if sst_dump_scan(&sst).contains(&format!("'{}'", key))
                && sst_dump_scan(&sst).contains(", type:0")
            {
                // cheap containment check is enough here: type:0 anywhere
                // alongside the key string on the same scan is sufficient
                // signal this file holds *a* record for that key; exact
                // per-line matching isn't needed for this presence check.
                if sst_dump_scan(&sst)
                    .lines()
                    .any(|l| l.contains(&format!("'{}'", key)) && l.contains("type:0"))
                {
                    found = true;
                }
            }
        }
        assert!(
            found,
            "delete tombstone for {} not flushed to any SST before snapshot — WAL flush timing bug in test setup",
            key
        );
    }

    // Step 2: ground truth.
    let (scan_out, stderr, ok) = run_ldb(&original_db, &["--auto_compaction=false", "scan"]);
    assert!(ok, "ground truth scan failed: {}", stderr);
    let mut ground_truth: Vec<String> = scan_out
        .lines()
        .filter_map(normalize_scan_line)
        .filter(|l| !l.contains("zzz_flush_trigger"))
        .collect();
    ground_truth.sort();
    assert_eq!(
        ground_truth.len(),
        47,
        "expected 47 live keys (150 puts - 3 deletes), got: {:?}",
        ground_truth
    );

    // Step 3: snapshot pre-compaction SSTs, run compactor.
    let pre_compaction_dir = tmp.join("pre_compaction_ssts");
    std::fs::create_dir_all(&pre_compaction_dir).unwrap();
    let sst_paths = sst_files_in(&original_db);
    for sst in &sst_paths {
        std::fs::copy(sst, pre_compaction_dir.join(sst.file_name().unwrap())).unwrap();
    }

    let filelist_path = tmp.join("filelist.txt");
    let filelist_contents: String = sst_files_in(&pre_compaction_dir)
        .iter()
        .map(|p| p.to_str().unwrap().to_string())
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&filelist_path, filelist_contents).unwrap();

    let output_sst = tmp.join("compactor_output.sst");
    let compact_bin = find_compact_files_binary();
    let out = Command::new(&compact_bin)
        .args([
            "--for-ingest",
            output_sst.to_str().unwrap(),
            "--list",
            filelist_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run compact_files binary");
    assert!(
        out.status.success(),
        "compact_files failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Step 4: verify validity, then ingest into a FRESH db.
    let check = Command::new("rocksdb_sst_dump")
        .args([
            &format!("--file={}", output_sst.display()),
            "--command=check",
        ])
        .output()
        .unwrap();
    assert!(
        check.status.success(),
        "compactor output failed sst_dump check: {}",
        String::from_utf8_lossy(&check.stderr)
    );

    let fresh_db = tmp.join("fresh_db");
    let (_, stderr, ok) = run_ldb(
        &fresh_db,
        &[
            "--create_if_missing",
            "ingest_extern_sst",
            output_sst.to_str().unwrap(),
        ],
    );
    assert!(ok, "ingest failed: {}", stderr);

    // Step 5: read back and diff against ground truth.
    let (scan_out, stderr, ok) = run_ldb(&fresh_db, &["scan"]);
    assert!(ok, "fresh db scan failed: {}", stderr);
    let mut fresh_scan: Vec<String> = scan_out.lines().filter_map(normalize_scan_line).collect();
    fresh_scan.sort();

    assert_eq!(
        fresh_scan, ground_truth,
        "compactor's output, ingested into a fresh RocksDB instance, does not match the original DB's ground-truth scan"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// A different, more direct form of "usable by RocksDB": swap a real DB's
/// own on-disk SSTs for compactor's compacted output (not ingest into a
/// fresh DB), regenerate the MANIFEST via `ldb repair` (RocksDB's own
/// file-driven recovery path — rescans SSTs on disk and rebuilds metadata,
/// exactly the mechanism a "restart RocksDB with our files" scenario needs
/// since compactor has no manifest-writing code of its own), then reopen
/// and scan.
///
/// Confirmed by direct testing that this requires TWO properties beyond
/// what `--for-ingest` mode needs:
/// - `column_family_id`/`column_family_name` must be present and match the
///   target CF ("default"/0) — `ldb repair` rejected a file missing this
///   with "Table #N: inconsistent column family name" and recovered 0
///   files. Note this is a DIFFERENT requirement from `--for-ingest`'s
///   seqno-zeroing; this test does NOT use `--for-ingest` (repair opens the
///   file as a normal internal SST, not an external ingest file), so
///   `column_family_id`/`name` alone (now always stamped, see
///   properties.rs) are what make this path work.
/// - Stale WAL/MANIFEST files from the original DB must be removed before
///   repair, or repair may try to replay logs referencing keys already
///   folded into the compacted file.
#[test]
fn compactor_output_replaces_real_dbs_files_and_survives_repair_and_restart() {
    if !have_binary("rocksdb_ldb") || !have_binary("rocksdb_sst_dump") {
        eprintln!(
            "SKIP: rocksdb_ldb / rocksdb_sst_dump not found on PATH (install via `brew install rocksdb`)"
        );
        return;
    }

    let tmp = std::env::temp_dir().join(format!("compactor_restart_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let db = tmp.join("db");

    for generation in 1..=3 {
        for i in 1..=50 {
            let (_, stderr, ok) = run_ldb(
                &db,
                &[
                    "--auto_compaction=false",
                    "put",
                    &format!("key{:05}", i),
                    &format!("gen{}_value_{:05}", generation, i),
                    "--create_if_missing",
                ],
            );
            assert!(ok, "put failed: {}", stderr);
        }
    }
    for i in [5, 15, 25] {
        let (_, stderr, ok) = run_ldb(
            &db,
            &["--auto_compaction=false", "delete", &format!("key{:05}", i)],
        );
        assert!(ok, "delete failed: {}", stderr);
    }
    // Force-flush the last delete's tombstone (see the other test in this
    // file for why this is needed).
    run_ldb(&db, &["--auto_compaction=false", "put", "zzz_trigger", "x"]);

    let (scan_out, stderr, ok) = run_ldb(&db, &["--auto_compaction=false", "scan"]);
    assert!(ok, "ground truth scan failed: {}", stderr);
    let mut ground_truth: Vec<String> = scan_out
        .lines()
        .filter_map(normalize_scan_line)
        .filter(|l| !l.contains("zzz_trigger"))
        .collect();
    ground_truth.sort();
    assert_eq!(
        ground_truth.len(),
        47,
        "unexpected ground truth: {:?}",
        ground_truth
    );

    // Snapshot pre-compaction SSTs (before touching the live db further).
    let pre_compaction_dir = tmp.join("pre_compaction_ssts");
    std::fs::create_dir_all(&pre_compaction_dir).unwrap();
    for sst in sst_files_in(&db) {
        std::fs::copy(&sst, pre_compaction_dir.join(sst.file_name().unwrap())).unwrap();
    }
    let filelist_path = tmp.join("filelist.txt");
    let filelist_contents: String = sst_files_in(&pre_compaction_dir)
        .iter()
        .map(|p| p.to_str().unwrap().to_string())
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&filelist_path, filelist_contents).unwrap();

    // Compact with compactor (plain internal-SST mode, NOT --for-ingest —
    // this file will be opened as a normal SST during repair/recovery, not
    // ingested).
    let output_sst = tmp.join("compacted.sst");
    let compact_bin = find_compact_files_binary();
    let out = Command::new(&compact_bin)
        .args([
            output_sst.to_str().unwrap(),
            "--list",
            filelist_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run compact_files binary");
    assert!(
        out.status.success(),
        "compact_files failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let check = Command::new("rocksdb_sst_dump")
        .args([
            &format!("--file={}", output_sst.display()),
            "--command=check",
        ])
        .output()
        .unwrap();
    assert!(
        check.status.success(),
        "compactor output failed sst_dump check: {}",
        String::from_utf8_lossy(&check.stderr)
    );

    // "Stop RocksDB" (nothing to stop here — no live process — just
    // manipulate its files directly): remove ALL existing SSTs, WALs, and
    // manifests, drop in only the compacted file, then let `ldb repair`
    // (RocksDB's own file-driven recovery) rebuild the manifest from
    // what's on disk.
    for entry in std::fs::read_dir(&db).unwrap().filter_map(|e| e.ok()) {
        let path = entry.path();
        let name = path.file_name().unwrap().to_string_lossy();
        if name.ends_with(".sst")
            || name.ends_with(".log")
            || name.starts_with("MANIFEST-")
            || name.starts_with("LOG.old.")
        {
            std::fs::remove_file(&path).unwrap();
        }
    }
    std::fs::copy(&output_sst, db.join("999999.sst")).unwrap();

    let (_, stderr, ok) = run_ldb(&db, &["repair"]);
    assert!(ok, "ldb repair failed: {}", stderr);
    assert!(
        stderr.contains("recovered 1 files") || stderr.contains("recovered 1 file"),
        "repair did not recover our file as expected, stderr: {}",
        stderr
    );

    // "Restart" (reopen) and scan.
    let (scan_out, stderr, ok) = run_ldb(&db, &["scan"]);
    assert!(ok, "post-repair scan failed: {}", stderr);
    let mut post_repair: Vec<String> = scan_out.lines().filter_map(normalize_scan_line).collect();
    post_repair.sort();

    assert_eq!(
        post_repair, ground_truth,
        "RocksDB, restarted with only compactor's compacted file in place, does not read back the same data"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
