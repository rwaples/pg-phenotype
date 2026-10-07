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
