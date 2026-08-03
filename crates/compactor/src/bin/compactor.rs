use compactor::format::internal_key;
use compactor::sst::SstReader;
use std::env;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();
    let rt = tokio::runtime::Runtime::new().expect("failed to build tokio runtime");
    rt.block_on(run(args));
}

async fn run(args: Vec<String>) {
    let Some(path) = args.get(1) else {
        eprintln!("usage: compactor <path-to-sst-file>");
        std::process::exit(1);
    };

    let reader = match compactor::io::open(Path::new(path)).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to open {}: {}", path, e);
            std::process::exit(1);
        }
    };

    let sst = match SstReader::open(reader).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to open SST: {}", e);
            std::process::exit(1);
        }
    };

    let dump_all = args.get(2).map(|s| s.as_str()) == Some("--dump-all");

    if !dump_all {
        println!(
            "magic=0x{:016x} format_version={} checksum_type={:?}",
            sst.footer.table_magic_number, sst.footer.format_version, sst.footer.checksum_type,
        );
    }

    let mut count = 0usize;
    let mut printed = 0usize;
    let result = sst
        .for_each_entry(|key, value| {
            count += 1;
            match internal_key::split(key) {
                Ok(ik) => {
                    if dump_all {
                        println!(
                            "{} ==> {}",
                            String::from_utf8_lossy(ik.user_key),
                            String::from_utf8_lossy(value)
                        );
                    } else if printed < 20 {
                        println!(
                            "  user_key={:?} seq={} type={:?} value_len={}",
                            String::from_utf8_lossy(ik.user_key),
                            ik.sequence,
                            ik.value_type,
                            value.len()
                        );
                        printed += 1;
                    }
                }
                Err(err) => println!("  <bad internal key: {}>", err),
            }
        })
        .await;

    if let Err(e) = result {
        eprintln!("failed to iterate entries: {}", e);
        std::process::exit(1);
    }

    if !dump_all {
        println!("total entries: {}", count);
        if count > printed {
            println!("  ... ({} more)", count - printed);
        }
    }
}
