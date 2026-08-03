use fuzzer::{generate_sources, run_compactor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Differential test: compactor's real compaction path vs oracle's
/// independent naive reference, across many random input shapes. This is
/// the replacement for the ad-hoc randomized checks that used to live
/// inline in compactor's own test files — now a dedicated fuzzer crate that
/// diffs against a deliberately independent implementation.
#[test]
fn compactor_matches_oracle_across_random_inputs() {
    let mut rng = StdRng::seed_from_u64(42);

    for trial in 0..100 {
        let num_sources = rng.gen_range(1..=12);
        let num_keys = rng.gen_range(0..=200);
        let per_source = generate_sources(&mut rng, num_sources, num_keys);

        let compactor_out = run_compactor(&per_source);
        let oracle_out = oracle::naive_compact(&per_source);

        assert_eq!(
            compactor_out, oracle_out,
            "trial {}: compactor diverged from oracle (num_sources={}, num_keys={})",
            trial, num_sources, num_keys
        );
    }
}
