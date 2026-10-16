//! Entry points that exist only in test builds (`--features test-hooks`).

use pyo3::prelude::*;

/// Panic inside the extension, so a test can see the panic reach Python as
/// `PanicException` rather than abort the interpreter.
#[pyfunction]
pub(crate) fn _panic_for_test() {
    #[expect(clippy::panic, reason = "the probe exists to panic")]
    {
        panic!("pg-phenotype test hook: deliberate panic");
    }
}

/// How many times a crossing has released free memory in this process.
#[pyfunction]
pub(crate) fn _release_count() -> usize {
    crate::crossing::RELEASES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Each Relationship Category's `(code, up, down)` meioses, as the
/// Relative-Pair SE's MRCA walk reads them.
#[pyfunction]
pub(crate) fn _meioses() -> Vec<(&'static str, u8, u8)> {
    pg_phenotype_core::relative_pairs::meioses().collect()
}
