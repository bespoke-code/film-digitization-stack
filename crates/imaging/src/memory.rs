//! Memory budget derived from the physical memory of the machine.

/// Upper limit for the memory the application plans to use.
pub const MAX_MEMORY_BUDGET: u64 = 4 << 30;
/// Budget used when the physical memory cannot be read.
const FALLBACK_MEMORY_BUDGET: u64 = 2 << 30;

/// Half of the physical memory, at most [`MAX_MEMORY_BUDGET`].
///
/// Half leaves room for the operating system, the GPU preview and other
/// applications; on an 8 GB machine this gives the full 4 GB.
pub fn memory_budget() -> u64 {
    budget_for(physical_memory())
}

fn budget_for(physical: Option<u64>) -> u64 {
    physical
        .filter(|bytes| *bytes > 0)
        .map(|bytes| (bytes / 2).min(MAX_MEMORY_BUDGET))
        .unwrap_or(FALLBACK_MEMORY_BUDGET)
}

#[cfg(target_os = "macos")]
fn physical_memory() -> Option<u64> {
    let output = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    String::from_utf8(output.stdout).ok()?.trim().parse().ok()
}

#[cfg(target_os = "linux")]
fn physical_memory() -> Option<u64> {
    let total = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| parse_meminfo(&text));
    // A container limit is smaller than the host memory and wins.
    let limit = std::fs::read_to_string("/sys/fs/cgroup/memory.max")
        .ok()
        .and_then(|text| text.trim().parse::<u64>().ok());
    match (total, limit) {
        (Some(total), Some(limit)) => Some(total.min(limit)),
        (total, limit) => total.or(limit),
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn physical_memory() -> Option<u64> {
    None
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_meminfo(text: &str) -> Option<u64> {
    let line = text.lines().find(|line| line.starts_with("MemTotal:"))?;
    let kibibytes: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kibibytes * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_is_half_the_memory_up_to_four_gigabytes() {
        assert_eq!(budget_for(Some(8 << 30)), 4 << 30);
        assert_eq!(budget_for(Some(32 << 30)), 4 << 30);
        assert_eq!(budget_for(Some(4 << 30)), 2 << 30);
        assert_eq!(budget_for(None), FALLBACK_MEMORY_BUDGET);
        assert_eq!(budget_for(Some(0)), FALLBACK_MEMORY_BUDGET);
    }

    #[test]
    fn meminfo_total_is_read_in_bytes() {
        let text = "MemTotal:        8048576 kB\nMemFree:          123456 kB\n";
        assert_eq!(parse_meminfo(text), Some(8_048_576 * 1024));
        assert_eq!(parse_meminfo("MemFree: 1 kB"), None);
    }

    #[test]
    fn this_machine_reports_a_budget_within_the_limit() {
        let budget = memory_budget();
        assert!(budget > 0 && budget <= MAX_MEMORY_BUDGET);
    }
}
