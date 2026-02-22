//! Seccomp syscall filtering implementation.
//!
//! Seccomp (secure computing mode) is a Linux kernel feature that restricts
//! the system calls a process can make. This provides defense-in-depth
//! by blocking dangerous syscalls that could be used to escape the sandbox.
//!
//! Reference: https://www.kernel.org/doc/html/latest/userspace-api/seccomp_filter.html

#![allow(dead_code)]

/// Check if seccomp is supported on this system.
pub fn is_supported() -> bool {
    #[cfg(target_os = "linux")]
    {
        use std::io::Error;

        // Try prctl with PR_GET_SECCOMP to check if seccomp is available
        let result = unsafe { libc::prctl(libc::PR_GET_SECCOMP, 0, 0, 0, 0) };

        if result < 0 {
            let err = Error::last_os_error();
            // EINVAL means seccomp is not built into kernel
            err.raw_os_error() != Some(libc::EINVAL)
        } else {
            true
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// Seccomp filter for syscall restrictions.
#[derive(Debug)]
pub struct SeccompFilter {
    /// Whether to block network-related syscalls
    block_network: bool,
}

impl SeccompFilter {
    /// Create a new seccomp filter configuration.
    ///
    /// # Arguments
    /// * `block_network` - If true, block socket creation syscalls
    pub fn new(block_network: bool) -> Self {
        Self { block_network }
    }

    /// Apply seccomp filter to the current process.
    ///
    /// WARNING: This is irreversible! Once applied, restrictions cannot be removed.
    pub fn apply(&self) -> Result<(), String> {
        #[cfg(target_os = "linux")]
        {
            self.apply_linux()
        }

        #[cfg(not(target_os = "linux"))]
        {
            Err("Seccomp is only supported on Linux".to_string())
        }
    }

    #[cfg(target_os = "linux")]
    fn apply_linux(&self) -> Result<(), String> {
        use std::io::Error;

        // BPF constants
        const BPF_LD: u16 = 0x00;
        const BPF_W: u16 = 0x00;
        const BPF_ABS: u16 = 0x20;
        const BPF_JMP: u16 = 0x05;
        const BPF_JEQ: u16 = 0x10;
        const BPF_K: u16 = 0x00;
        const BPF_RET: u16 = 0x06;

        // Seccomp return values
        const SECCOMP_RET_ALLOW: u32 = 0x7fff_0000;
        const SECCOMP_RET_ERRNO: u32 = 0x0005_0000;
        const SECCOMP_RET_KILL_PROCESS: u32 = 0x8000_0000;

        // Seccomp operation
        const SECCOMP_SET_MODE_FILTER: u32 = 1;

        // Offset of syscall number in seccomp_data
        const SYSCALL_NR_OFFSET: u32 = 0;

        // EPERM error code
        const EPERM: u32 = 1;

        // BPF instruction
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct SockFilter {
            code: u16,
            jt: u8,
            jf: u8,
            k: u32,
        }

        #[repr(C)]
        struct SockFprog {
            len: u16,
            filter: *const SockFilter,
        }

        // Build the filter
        let mut filter: Vec<SockFilter> = Vec::new();

        // Load syscall number
        filter.push(SockFilter {
            code: BPF_LD | BPF_W | BPF_ABS,
            jt: 0,
            jf: 0,
            k: SYSCALL_NR_OFFSET,
        });

        // Syscalls to block (with EPERM)
        // These are dangerous syscalls that could be used to escape the sandbox
        let blocked_syscalls: Vec<(i64, &str)> = vec![
            (libc::SYS_ptrace, "ptrace"), // Process debugging/inspection
            (libc::SYS_process_vm_readv, "process_vm_readv"), // Cross-process memory read
            (libc::SYS_process_vm_writev, "process_vm_writev"), // Cross-process memory write
            (libc::SYS_mount, "mount"),   // Filesystem mounting
            (libc::SYS_umount2, "umount2"), // Filesystem unmounting
            (libc::SYS_pivot_root, "pivot_root"), // Change root filesystem
            (libc::SYS_chroot, "chroot"), // Change root directory
            (libc::SYS_reboot, "reboot"), // System reboot
            (libc::SYS_kexec_load, "kexec_load"), // Load new kernel
            (libc::SYS_init_module, "init_module"), // Load kernel module
            (libc::SYS_finit_module, "finit_module"), // Load kernel module from fd
            (libc::SYS_delete_module, "delete_module"), // Unload kernel module
            (libc::SYS_acct, "acct"),     // Process accounting
            (libc::SYS_swapon, "swapon"), // Enable swap
            (libc::SYS_swapoff, "swapoff"), // Disable swap
            (libc::SYS_setns, "setns"),   // Join namespace
            (libc::SYS_unshare, "unshare"), // Create new namespace
            (libc::SYS_perf_event_open, "perf_event_open"), // Performance monitoring
            (libc::SYS_bpf, "bpf"),       // BPF operations
            (libc::SYS_userfaultfd, "userfaultfd"), // User-space page fault handling
        ];

        // Add network blocking if requested
        let network_syscalls: Vec<(i64, &str)> = if self.block_network {
            vec![
                (libc::SYS_socket, "socket"),         // Create socket
                (libc::SYS_socketpair, "socketpair"), // Create socket pair
                (libc::SYS_connect, "connect"),       // Connect socket
                (libc::SYS_accept, "accept"),         // Accept connection
                (libc::SYS_accept4, "accept4"),       // Accept connection
                (libc::SYS_bind, "bind"),             // Bind socket
                (libc::SYS_listen, "listen"),         // Listen on socket
                (libc::SYS_sendto, "sendto"),         // Send data
                (libc::SYS_recvfrom, "recvfrom"),     // Receive data
                (libc::SYS_sendmsg, "sendmsg"),       // Send message
                (libc::SYS_recvmsg, "recvmsg"),       // Receive message
            ]
        } else {
            vec![]
        };

        // Calculate total blocked syscalls for jump offsets
        let total_blocked = blocked_syscalls.len() + network_syscalls.len();

        // For each blocked syscall, add a check
        // Each check is: if (syscall == X) return ERRNO(EPERM)
        // Jump offset calculation: if match (jt), jump to return EPERM instruction
        // if no match (jf), continue to next check

        for (i, (syscall_nr, _name)) in blocked_syscalls
            .iter()
            .chain(network_syscalls.iter())
            .enumerate()
        {
            // Calculate jump offset to EPERM return
            // After all checks, we have: EPERM return, then ALLOW return
            // Jump to position (total_blocked - i - 1) to reach EPERM return
            let jump_to_eperm = (total_blocked - i - 1) as u8;

            filter.push(SockFilter {
                code: BPF_JMP | BPF_JEQ | BPF_K,
                jt: jump_to_eperm,
                jf: 0,
                k: *syscall_nr as u32,
            });
        }

        // Default: Allow syscall
        filter.push(SockFilter {
            code: BPF_RET | BPF_K,
            jt: 0,
            jf: 0,
            k: SECCOMP_RET_ALLOW,
        });

        // Return EPERM for blocked syscalls
        filter.push(SockFilter {
            code: BPF_RET | BPF_K,
            jt: 0,
            jf: 0,
            k: SECCOMP_RET_ERRNO | EPERM,
        });

        // Actually the jump logic above is wrong. Let me fix it.
        // The issue is that jt jumps forward by jt instructions, not to absolute position.
        // Let me rebuild the filter correctly.

        filter.clear();

        // Load syscall number
        filter.push(SockFilter {
            code: BPF_LD | BPF_W | BPF_ABS,
            jt: 0,
            jf: 0,
            k: SYSCALL_NR_OFFSET,
        });

        // Combine all blocked syscalls
        let all_blocked: Vec<i64> = blocked_syscalls
            .iter()
            .chain(network_syscalls.iter())
            .map(|(nr, _)| *nr)
            .collect();

        // For each blocked syscall, check and jump to deny
        // Structure: check1, check2, ..., checkN, ALLOW, DENY
        // Each check: if match, jump to DENY (which is at end)
        for syscall_nr in &all_blocked {
            // After this instruction, there are (remaining_checks + 2) instructions
            // remaining_checks = all_blocked.len() - current_index - 1
            // Then ALLOW, then DENY
            // So to jump to DENY, we need to skip: remaining_checks + 1 instructions
            // But we're iterating, so let's calculate differently

            filter.push(SockFilter {
                code: BPF_JMP | BPF_JEQ | BPF_K,
                jt: 0, // Will fix in post-processing
                jf: 0,
                k: *syscall_nr as u32,
            });
        }

        // ALLOW instruction (index = 1 + all_blocked.len())
        let allow_idx = 1 + all_blocked.len();
        filter.push(SockFilter {
            code: BPF_RET | BPF_K,
            jt: 0,
            jf: 0,
            k: SECCOMP_RET_ALLOW,
        });

        // DENY instruction (index = allow_idx + 1)
        let deny_idx = allow_idx + 1;
        filter.push(SockFilter {
            code: BPF_RET | BPF_K,
            jt: 0,
            jf: 0,
            k: SECCOMP_RET_ERRNO | EPERM,
        });

        // Fix jump offsets: each check instruction at index i needs to jump to deny_idx
        // Jump offset = deny_idx - i - 1 (relative jump)
        for i in 0..all_blocked.len() {
            let check_idx = 1 + i; // Account for the LOAD instruction at index 0
            let jump_offset = deny_idx - check_idx - 1;
            filter[check_idx].jt = jump_offset as u8;
        }

        let prog = SockFprog {
            len: filter.len() as u16,
            filter: filter.as_ptr(),
        };

        // Set no_new_privs first (required for unprivileged seccomp)
        let result = unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) };
        if result < 0 {
            let err = Error::last_os_error();
            return Err(format!("prctl(PR_SET_NO_NEW_PRIVS) failed: {}", err));
        }

        // Apply the seccomp filter
        let result = unsafe {
            libc::syscall(
                libc::SYS_seccomp,
                SECCOMP_SET_MODE_FILTER,
                0u32,
                &prog as *const SockFprog,
            )
        };

        if result < 0 {
            let err = Error::last_os_error();
            return Err(format!("seccomp(SET_MODE_FILTER) failed: {}", err));
        }

        tracing::debug!(
            "Seccomp applied: blocked {} syscalls, network_blocked={}",
            all_blocked.len(),
            self.block_network
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
    fn test_filter_creation() {
        let filter = SeccompFilter::new(true);
        assert!(filter.block_network);
    }
}
