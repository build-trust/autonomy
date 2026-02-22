//! Landlock filesystem sandbox implementation.
//!
//! Landlock is a Linux security module that enables unprivileged processes
//! to restrict their own filesystem access. This is perfect for sandboxing
//! agent command execution.
//!
//! Requirements:
//! - Linux kernel 5.13+ for basic support
//! - Linux kernel 5.19+ for network restrictions (not used here)
//!
//! Reference: https://docs.kernel.org/userspace-api/landlock.html

#![allow(dead_code)]

use std::path::PathBuf;

#[cfg(target_os = "linux")]
use std::os::unix::io::AsRawFd;

/// Check if Landlock is supported on this kernel.
pub fn is_supported() -> bool {
    #[cfg(target_os = "linux")]
    {
        // Try to create a Landlock ruleset to check support
        use std::io::Error;

        // syscall numbers for landlock
        const SYS_LANDLOCK_CREATE_RULESET: i64 = 444;
        const LANDLOCK_CREATE_RULESET_VERSION: u32 = 1 << 0;

        let result = unsafe {
            libc::syscall(
                SYS_LANDLOCK_CREATE_RULESET,
                std::ptr::null::<libc::c_void>(),
                0usize,
                LANDLOCK_CREATE_RULESET_VERSION,
            )
        };

        if result >= 0 {
            // Got a valid ABI version, Landlock is supported
            true
        } else {
            let err = Error::last_os_error();
            // ENOSYS means syscall not available
            // EOPNOTSUPP means Landlock is disabled
            err.raw_os_error() != Some(libc::ENOSYS) && err.raw_os_error() != Some(libc::EOPNOTSUPP)
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// Landlock sandbox for filesystem isolation.
#[derive(Debug)]
pub struct LandlockSandbox {
    /// Workspace directory with read/write (or read-only) access
    workspace: PathBuf,

    /// Paths with read-only access
    readonly_paths: Vec<PathBuf>,

    /// Whether workspace should be read-only
    workspace_readonly: bool,
}

impl LandlockSandbox {
    /// Create a new Landlock sandbox configuration.
    ///
    /// # Arguments
    /// * `workspace` - Directory for read/write access (unless workspace_readonly is true)
    /// * `readonly_paths` - Additional paths with read-only access
    /// * `workspace_readonly` - If true, workspace is also read-only
    pub fn new(
        workspace: PathBuf,
        readonly_paths: Vec<PathBuf>,
        workspace_readonly: bool,
    ) -> Result<Self, String> {
        // Ensure workspace directory exists
        if !workspace.exists() {
            std::fs::create_dir_all(&workspace)
                .map_err(|e| format!("Failed to create workspace directory: {}", e))?;
        }

        Ok(Self {
            workspace,
            readonly_paths,
            workspace_readonly,
        })
    }

    /// Apply Landlock restrictions to the current process.
    ///
    /// WARNING: This is irreversible! Once applied, restrictions cannot be removed.
    pub fn apply(&self) -> Result<(), String> {
        #[cfg(target_os = "linux")]
        {
            self.apply_linux()
        }

        #[cfg(not(target_os = "linux"))]
        {
            Err("Landlock is only supported on Linux".to_string())
        }
    }

    #[cfg(target_os = "linux")]
    fn apply_linux(&self) -> Result<(), String> {
        use std::io::Error;

        // Landlock syscall numbers
        const SYS_LANDLOCK_CREATE_RULESET: i64 = 444;
        const SYS_LANDLOCK_ADD_RULE: i64 = 445;
        const SYS_LANDLOCK_RESTRICT_SELF: i64 = 446;

        // Landlock constants
        const LANDLOCK_RULE_PATH_BENEATH: u32 = 1;

        // Access rights for ABI v1 (Linux 5.13+)
        const LANDLOCK_ACCESS_FS_EXECUTE: u64 = 1 << 0;
        const LANDLOCK_ACCESS_FS_WRITE_FILE: u64 = 1 << 1;
        const LANDLOCK_ACCESS_FS_READ_FILE: u64 = 1 << 2;
        const LANDLOCK_ACCESS_FS_READ_DIR: u64 = 1 << 3;
        const LANDLOCK_ACCESS_FS_REMOVE_DIR: u64 = 1 << 4;
        const LANDLOCK_ACCESS_FS_REMOVE_FILE: u64 = 1 << 5;
        const LANDLOCK_ACCESS_FS_MAKE_CHAR: u64 = 1 << 6;
        const LANDLOCK_ACCESS_FS_MAKE_DIR: u64 = 1 << 7;
        const LANDLOCK_ACCESS_FS_MAKE_REG: u64 = 1 << 8;
        const LANDLOCK_ACCESS_FS_MAKE_SOCK: u64 = 1 << 9;
        const LANDLOCK_ACCESS_FS_MAKE_FIFO: u64 = 1 << 10;
        const LANDLOCK_ACCESS_FS_MAKE_BLOCK: u64 = 1 << 11;
        const LANDLOCK_ACCESS_FS_MAKE_SYM: u64 = 1 << 12;

        // All read access
        const READ_ACCESS: u64 =
            LANDLOCK_ACCESS_FS_EXECUTE | LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR;

        // All write access
        const WRITE_ACCESS: u64 = LANDLOCK_ACCESS_FS_WRITE_FILE
            | LANDLOCK_ACCESS_FS_REMOVE_DIR
            | LANDLOCK_ACCESS_FS_REMOVE_FILE
            | LANDLOCK_ACCESS_FS_MAKE_CHAR
            | LANDLOCK_ACCESS_FS_MAKE_DIR
            | LANDLOCK_ACCESS_FS_MAKE_REG
            | LANDLOCK_ACCESS_FS_MAKE_SOCK
            | LANDLOCK_ACCESS_FS_MAKE_FIFO
            | LANDLOCK_ACCESS_FS_MAKE_BLOCK
            | LANDLOCK_ACCESS_FS_MAKE_SYM;

        // All access rights
        const ALL_ACCESS: u64 = READ_ACCESS | WRITE_ACCESS;

        // Ruleset attribute structure
        #[repr(C)]
        struct LandlockRulesetAttr {
            handled_access_fs: u64,
        }

        // Path beneath structure
        #[repr(C)]
        struct LandlockPathBeneathAttr {
            allowed_access: u64,
            parent_fd: i32,
        }

        // Create ruleset
        let ruleset_attr = LandlockRulesetAttr {
            handled_access_fs: ALL_ACCESS,
        };

        let ruleset_fd = unsafe {
            libc::syscall(
                SYS_LANDLOCK_CREATE_RULESET,
                &ruleset_attr as *const LandlockRulesetAttr,
                std::mem::size_of::<LandlockRulesetAttr>(),
                0u32,
            )
        };

        if ruleset_fd < 0 {
            let err = Error::last_os_error();
            return Err(format!("landlock_create_ruleset failed: {}", err));
        }

        let ruleset_fd = ruleset_fd as i32;

        // Helper to add a path rule
        let add_path_rule = |path: &PathBuf, access: u64| -> Result<(), String> {
            if !path.exists() {
                // Skip non-existent paths silently
                return Ok(());
            }

            let fd = match std::fs::File::open(path) {
                Ok(f) => f,
                Err(e) => {
                    // Skip paths we can't open (permission issues, etc.)
                    tracing::debug!("Skipping path {:?}: {}", path, e);
                    return Ok(());
                }
            };

            let path_beneath = LandlockPathBeneathAttr {
                allowed_access: access,
                parent_fd: fd.as_raw_fd(),
            };

            let result = unsafe {
                libc::syscall(
                    SYS_LANDLOCK_ADD_RULE,
                    ruleset_fd,
                    LANDLOCK_RULE_PATH_BENEATH,
                    &path_beneath as *const LandlockPathBeneathAttr,
                    0u32,
                )
            };

            if result < 0 {
                let err = Error::last_os_error();
                tracing::warn!("landlock_add_rule failed for {:?}: {}", path, err);
                // Continue anyway - some paths may not be rulable
            }

            Ok(())
        };

        // Add workspace with appropriate access
        let workspace_access = if self.workspace_readonly {
            READ_ACCESS
        } else {
            ALL_ACCESS
        };
        add_path_rule(&self.workspace, workspace_access)?;

        // Add read-only paths
        for path in &self.readonly_paths {
            add_path_rule(path, READ_ACCESS)?;
        }

        // Before restricting, drop privileges if we're root
        // (Landlock works better without CAP_SYS_ADMIN)
        unsafe {
            // Set no_new_privs to prevent privilege escalation
            if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0 {
                let err = Error::last_os_error();
                // Close ruleset fd before returning error
                libc::close(ruleset_fd);
                return Err(format!("prctl(PR_SET_NO_NEW_PRIVS) failed: {}", err));
            }
        }

        // Apply the ruleset
        let result = unsafe { libc::syscall(SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0u32) };

        // Close the ruleset fd regardless of result
        unsafe {
            libc::close(ruleset_fd);
        }

        if result < 0 {
            let err = Error::last_os_error();
            return Err(format!("landlock_restrict_self failed: {}", err));
        }

        tracing::debug!(
            "Landlock applied: workspace={:?}, readonly_paths={}, workspace_readonly={}",
            self.workspace,
            self.readonly_paths.len(),
            self.workspace_readonly
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_supported() {
        // Just check it doesn't panic
        let _ = is_supported();
    }

    #[test]
    fn test_sandbox_creation() {
        let temp_dir = std::env::temp_dir().join("landlock_test");
        let sandbox = LandlockSandbox::new(temp_dir, vec![], false);
        assert!(sandbox.is_ok());
    }
}
