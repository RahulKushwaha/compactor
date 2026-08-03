//! Async file-read abstraction, one backend: `BlockingFileReader` (std
//! positional read wrapped in `spawn_blocking`).
//!
//! A `tokio-uring`-backed variant was attempted for Linux (true async disk
//! I/O, no threadpool hop) but hit a real, structural incompatibility, not
//! just a porting bug: confirmed by actually compiling it on the Linux dev
//! desktop (this had only run on macOS before, where it's cfg'd out).
//! `tokio_uring::fs::File` holds an `Rc` internally and is thread-affine by
//! design (one io_uring instance per OS thread, via `tokio_uring::start`'s
//! single-threaded reactor) — it cannot satisfy `Send + Sync`, and even
//! forcing that would be a lie: this crate's compaction path spawns work
//! across OS threads (subcompaction sharding via `spawn_blocking`, and
//! `compact-rocksdb`'s worker thread pool calling in from outside any
//! `tokio_uring::start` reactor at all), which io_uring's thread affinity
//! fundamentally conflicts with. A correct fix would need a per-thread
//! io_uring instance pool with reads routed to the thread that owns the
//! fd — a real redesign, not attempted here. The blocking backend below is
//! what actually fixed this crate's I/O-bound slowdown earlier (batching
//! each file into one `read_at` instead of one per block); io_uring was
//! always the secondary, unverified path, not the source of that fix.
//!
//! Every read returns a freshly-allocated `Vec<u8>`.

use std::io;
use std::sync::Arc;

/// Async positional file reader. Implementors must support concurrent calls
/// to `read_at` from multiple tasks against the same open file.
pub trait FileReader: Send + Sync + 'static {
    fn read_at(
        &self,
        offset: u64,
        len: usize,
    ) -> impl std::future::Future<Output = io::Result<Vec<u8>>> + Send;

    fn file_size(&self) -> u64;
}

impl<T: FileReader> FileReader for Arc<T> {
    async fn read_at(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        (**self).read_at(offset, len).await
    }

    fn file_size(&self) -> u64 {
        (**self).file_size()
    }
}

pub use blocking::BlockingFileReader as PlatformFileReader;

/// Open `path` with the platform-appropriate backend, wrapped in an `Arc` so
/// many concurrent block reads (subcompaction shards, etc.) can share one
/// open file handle.
pub async fn open(path: &std::path::Path) -> io::Result<Arc<PlatformFileReader>> {
    PlatformFileReader::open(path).await.map(Arc::new)
}

mod blocking {
    use super::FileReader;
    use std::fs::File;
    use std::io;
    #[cfg(unix)]
    use std::os::unix::fs::FileExt;
    use std::path::Path;

    pub struct BlockingFileReader {
        file: File,
        file_size: u64,
    }

    impl BlockingFileReader {
        pub async fn open(path: &Path) -> io::Result<Self> {
            let path = path.to_owned();
            tokio::task::spawn_blocking(move || {
                let file = File::open(&path)?;
                let file_size = file.metadata()?.len();
                Ok(BlockingFileReader { file, file_size })
            })
            .await
            .expect("spawn_blocking panicked")
        }
    }

    impl FileReader for BlockingFileReader {
        async fn read_at(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
            // SAFETY-free approach: std::fs::File is Send+Sync-safe to share
            // via &self for positional reads (each read_at call is independent,
            // no shared cursor). We duplicate the fd view via try_clone to keep
            // the closure 'static-friendly for spawn_blocking without unsafe.
            let file = self.file.try_clone()?;
            tokio::task::spawn_blocking(move || {
                let mut buf = vec![0u8; len];
                file.read_exact_at(&mut buf, offset)?;
                Ok(buf)
            })
            .await
            .expect("spawn_blocking panicked")
        }

        fn file_size(&self) -> u64 {
            self.file_size
        }
    }
}
