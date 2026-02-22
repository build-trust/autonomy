//! Sandbox module for agent process isolation using Landlock and seccomp.
//!
//! This module provides OS-level sandboxing for agent command execution:
//! - Landlock: Filesystem access control (Linux 5.13+)
//! - Seccomp: System call filtering (Linux)
//!
//! On non-Linux systems, sandboxing gracefully degrades to a no-op with warnings.

mod landlock;
mod seccomp;

use pyo3::prelude::*;

#[cfg(target_os = "linux")]
use std::path::PathBuf;

/// Check if the current platform supports sandboxing.
#[pyfunction]
pub fn is_sandbox_supported() -> bool {
    cfg!(target_os = "linux")
}

/// Check if Landlock is supported on this kernel.
#[pyfunction]
pub fn is_landlock_supported() -> bool {
    #[cfg(target_os = "linux")]
    {
        landlock::is_supported()
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// Check if seccomp is supported on this system.
#[pyfunction]
pub fn is_seccomp_supported() -> bool {
    #[cfg(target_os = "linux")]
    {
        seccomp::is_supported()
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// Sandbox configuration for agent execution.
#[pyclass]
#[derive(Clone, Debug)]
pub struct SandboxConfig {
    /// Workspace directory (read/write access)
    #[pyo3(get, set)]
    pub workspace_path: String,

    /// Additional read-only paths
    #[pyo3(get, set)]
    pub readonly_paths: Vec<String>,

    /// Whether to block network access via seccomp
    #[pyo3(get, set)]
    pub block_network: bool,

    /// Whether to enable Landlock filesystem isolation
    #[pyo3(get, set)]
    pub enable_landlock: bool,

    /// Whether to enable seccomp syscall filtering
    #[pyo3(get, set)]
    pub enable_seccomp: bool,
}

#[pymethods]
impl SandboxConfig {
    #[new]
    #[pyo3(signature = (workspace_path, readonly_paths=None, block_network=true, enable_landlock=true, enable_seccomp=true))]
    pub fn new(
        workspace_path: String,
        readonly_paths: Option<Vec<String>>,
        block_network: bool,
        enable_landlock: bool,
        enable_seccomp: bool,
    ) -> Self {
        let default_readonly = vec![
            "/usr".to_string(),
            "/lib".to_string(),
            "/lib64".to_string(),
            "/bin".to_string(),
            "/sbin".to_string(),
            "/etc".to_string(),
            "/proc".to_string(),
            "/dev/null".to_string(),
            "/dev/zero".to_string(),
            "/dev/urandom".to_string(),
            "/tmp".to_string(),
        ];

        Self {
            workspace_path,
            readonly_paths: readonly_paths.unwrap_or(default_readonly),
            block_network,
            enable_landlock,
            enable_seccomp,
        }
    }

    /// Create a config for read-only mode (no write access to workspace)
    #[staticmethod]
    pub fn read_only(workspace_path: String) -> Self {
        Self {
            workspace_path,
            readonly_paths: vec![
                "/usr".to_string(),
                "/lib".to_string(),
                "/lib64".to_string(),
                "/bin".to_string(),
                "/sbin".to_string(),
                "/etc".to_string(),
                "/proc".to_string(),
                "/dev/null".to_string(),
                "/dev/zero".to_string(),
                "/dev/urandom".to_string(),
                "/tmp".to_string(),
            ],
            block_network: true,
            enable_landlock: true,
            enable_seccomp: true,
        }
    }
}

/// Result of applying sandbox restrictions.
#[pyclass]
#[derive(Clone, Debug)]
pub struct SandboxResult {
    #[pyo3(get)]
    pub success: bool,

    #[pyo3(get)]
    pub landlock_applied: bool,

    #[pyo3(get)]
    pub seccomp_applied: bool,

    #[pyo3(get)]
    pub message: String,
}

#[pymethods]
impl SandboxResult {
    fn __repr__(&self) -> String {
        format!(
            "SandboxResult(success={}, landlock={}, seccomp={}, message='{}')",
            self.success, self.landlock_applied, self.seccomp_applied, self.message
        )
    }
}

/// Apply sandbox restrictions to the current process.
///
/// WARNING: This is irreversible! Once applied, sandbox restrictions cannot be removed.
/// This should only be called in a forked child process before executing a command.
///
/// Args:
///   config: SandboxConfig specifying the sandbox parameters
///   read_only_workspace: If true, workspace is read-only (no write access)
///
/// Returns:
///   SandboxResult indicating what was applied
#[pyfunction]
#[pyo3(signature = (config, read_only_workspace=false))]
#[allow(unused_variables)]
pub fn apply_sandbox(config: &SandboxConfig, read_only_workspace: bool) -> PyResult<SandboxResult> {
    #[allow(unused_mut)]
    let mut landlock_applied = false;
    #[allow(unused_mut)]
    let mut seccomp_applied = false;
    let mut messages = Vec::new();

    #[cfg(target_os = "linux")]
    {
        // Apply Landlock first (filesystem restrictions)
        if config.enable_landlock {
            match apply_landlock(config, read_only_workspace) {
                Ok(()) => {
                    landlock_applied = true;
                    messages.push("Landlock applied".to_string());
                }
                Err(e) => {
                    messages.push(format!("Landlock failed: {}", e));
                }
            }
        }

        // Apply seccomp second (syscall filtering)
        if config.enable_seccomp {
            match apply_seccomp(config) {
                Ok(()) => {
                    seccomp_applied = true;
                    messages.push("Seccomp applied".to_string());
                }
                Err(e) => {
                    messages.push(format!("Seccomp failed: {}", e));
                }
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        messages.push("Sandbox not supported on this platform".to_string());
    }

    let success = landlock_applied || seccomp_applied || !cfg!(target_os = "linux");

    Ok(SandboxResult {
        success,
        landlock_applied,
        seccomp_applied,
        message: messages.join("; "),
    })
}

#[cfg(target_os = "linux")]
fn apply_landlock(config: &SandboxConfig, read_only_workspace: bool) -> Result<(), String> {
    use landlock::LandlockSandbox;

    let workspace = PathBuf::from(&config.workspace_path);
    let readonly: Vec<PathBuf> = config.readonly_paths.iter().map(PathBuf::from).collect();

    let sandbox = LandlockSandbox::new(workspace, readonly, read_only_workspace)
        .map_err(|e| format!("Failed to create Landlock sandbox: {}", e))?;

    sandbox
        .apply()
        .map_err(|e| format!("Failed to apply Landlock: {}", e))
}

#[cfg(target_os = "linux")]
fn apply_seccomp(config: &SandboxConfig) -> Result<(), String> {
    use seccomp::SeccompFilter;

    let filter = SeccompFilter::new(config.block_network);

    filter
        .apply()
        .map_err(|e| format!("Failed to apply seccomp: {}", e))
}

/// Register sandbox module with Python.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let sandbox = PyModule::new(parent.py(), "sandbox")?;

    sandbox.add_function(wrap_pyfunction!(is_sandbox_supported, &sandbox)?)?;
    sandbox.add_function(wrap_pyfunction!(is_landlock_supported, &sandbox)?)?;
    sandbox.add_function(wrap_pyfunction!(is_seccomp_supported, &sandbox)?)?;
    sandbox.add_function(wrap_pyfunction!(apply_sandbox, &sandbox)?)?;

    sandbox.add_class::<SandboxConfig>()?;
    sandbox.add_class::<SandboxResult>()?;

    parent.add_submodule(&sandbox)?;

    Ok(())
}
