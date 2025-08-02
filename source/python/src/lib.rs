use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "autonomy_in_rust_for_python")]
fn autonomy_in_rust_for_python(_module: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
