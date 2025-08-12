use pyo3::prelude::*;

mod errors;
mod integrations;
mod logs;
mod nodes;

#[pymodule]
#[pyo3(name = "autonomy_in_rust_for_python")]
fn autonomy_in_rust_for_python(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<nodes::PyNode>()?;
    module.add_class::<nodes::PyMailbox>()?;

    module.add_class::<integrations::mcp::PyMcpClient>()?;
    module.add_class::<integrations::mcp::PyMcpServer>()?;

    module.add_function(wrap_pyfunction!(logs::info, module)?)?;
    module.add_function(wrap_pyfunction!(logs::error, module)?)?;
    module.add_function(wrap_pyfunction!(logs::warn, module)?)?;
    module.add_function(wrap_pyfunction!(logs::debug, module)?)?;

    Ok(())
}
