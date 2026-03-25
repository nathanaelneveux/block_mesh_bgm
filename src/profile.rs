#![cfg(feature = "internal-profiler")]

use std::cell::Cell;
use std::time::Duration;

/// AO-specific timing and event counters for one profiled mesh call.
#[derive(Clone, Debug, Default)]
pub struct AoProfile {
    /// Time spent building AO keys from opaque occupancy columns.
    pub key_build: Duration,
    /// Total time spent in the AO carry merge loop.
    pub carry_total: Duration,
    /// Time spent deciding which bits continue into the next row.
    pub continue_mask: Duration,
    /// Time spent emitting rows with no incoming carry.
    pub emit_single: Duration,
    /// Time spent emitting rows that terminate active carry.
    pub emit_terminal: Duration,
    /// Time spent emitting rows that both terminate and continue carry.
    pub emit_mixed: Duration,
    /// Time spent emitting unit-only AO slices.
    pub unit_emit: Duration,
    /// Number of AO face slices visited.
    pub slices: u32,
    /// Number of AO face slices that hit the unit-only path.
    pub unit_slices: u32,
    /// Number of AO face slices that hit the carry path.
    pub carry_slices: u32,
    /// Number of non-empty AO carry rows processed.
    pub carry_rows: u32,
    /// Number of rows emitted by the single-row fast path.
    pub single_rows: u32,
    /// Number of rows emitted by the terminal-row fast path.
    pub terminal_rows: u32,
    /// Number of rows emitted by the mixed-row path.
    pub mixed_rows: u32,
    /// Total visible bits seen by the AO carry path.
    pub visible_bits: u64,
    /// Total overlapping bits considered for continuation.
    pub overlapping_bits: u64,
    /// Overlapping bits whose AO keys matched across adjacent rows.
    pub ao_compatible_overlap_bits: u64,
    /// Total overlapping bits that actually continued after AO checks.
    pub continued_bits: u64,
    /// Visible opaque bits that needed real AO signature generation.
    pub opaque_key_bits: u64,
    /// Visible non-opaque bits that bypassed AO signature generation.
    pub passthrough_key_bits: u64,
    /// Rows whose opaque visible bits all shared one AO key.
    pub uniform_opaque_rows: u32,
    /// Rows whose visible bits were all non-opaque and therefore AO-neutral.
    pub passthrough_rows: u32,
    /// Quads emitted by the single-row fast path.
    pub single_quads: u64,
    /// Quads emitted by the terminal-row fast path.
    pub terminal_quads: u64,
    /// Quads emitted by the mixed-row path.
    pub mixed_quads: u64,
    /// Unit quads emitted by the unit-only path.
    pub unit_quads: u64,
    /// Rows whose overlapping bits had no AO-compatible candidates at all.
    pub ao_rejected_rows: u32,
}

thread_local! {
    static ACTIVE_PROFILE: Cell<*mut AoProfile> = const { Cell::new(std::ptr::null_mut()) };
}

/// Runs a closure with AO profiling enabled for the current thread.
pub fn with_ao_profile<R>(profile: &mut AoProfile, f: impl FnOnce() -> R) -> R {
    ACTIVE_PROFILE.with(|slot| {
        let previous = slot.replace(profile as *mut AoProfile);
        let result = f();
        slot.set(previous);
        result
    })
}

fn with_active_profile(f: impl FnOnce(&mut AoProfile)) {
    ACTIVE_PROFILE.with(|slot| {
        let ptr = slot.get();
        if !ptr.is_null() {
            // SAFETY: `with_ao_profile` installs a valid mutable pointer for the
            // current thread and restores the previous pointer before returning.
            unsafe { f(&mut *ptr) };
        }
    });
}

pub(crate) fn record_key_build(duration: Duration) {
    with_active_profile(|profile| profile.key_build += duration);
}

pub(crate) fn record_carry_total(duration: Duration) {
    with_active_profile(|profile| profile.carry_total += duration);
}

pub(crate) fn record_continue_mask(duration: Duration) {
    with_active_profile(|profile| profile.continue_mask += duration);
}

pub(crate) fn record_emit_single(duration: Duration) {
    with_active_profile(|profile| profile.emit_single += duration);
}

pub(crate) fn record_emit_terminal(duration: Duration) {
    with_active_profile(|profile| profile.emit_terminal += duration);
}

pub(crate) fn record_emit_mixed(duration: Duration) {
    with_active_profile(|profile| profile.emit_mixed += duration);
}

pub(crate) fn record_unit_emit(duration: Duration) {
    with_active_profile(|profile| profile.unit_emit += duration);
}

pub(crate) fn record_slice(unit_only: bool) {
    with_active_profile(|profile| {
        profile.slices += 1;
        if unit_only {
            profile.unit_slices += 1;
        } else {
            profile.carry_slices += 1;
        }
    });
}

pub(crate) fn record_carry_row(row_bits: u64, overlapping_bits: u64) {
    with_active_profile(|profile| {
        profile.carry_rows += 1;
        profile.visible_bits += row_bits.count_ones() as u64;
        profile.overlapping_bits += overlapping_bits.count_ones() as u64;
    });
}

pub(crate) fn record_continued_bits(continue_mask: u64) {
    with_active_profile(|profile| {
        profile.continued_bits += continue_mask.count_ones() as u64;
    });
}

pub(crate) fn record_ao_overlap_candidates(matching_bits: u64) {
    with_active_profile(|profile| {
        profile.ao_compatible_overlap_bits += matching_bits.count_ones() as u64;
        if matching_bits == 0 {
            profile.ao_rejected_rows += 1;
        }
    });
}

pub(crate) fn record_single_row() {
    with_active_profile(|profile| profile.single_rows += 1);
}

pub(crate) fn record_terminal_row() {
    with_active_profile(|profile| profile.terminal_rows += 1);
}

pub(crate) fn record_mixed_row() {
    with_active_profile(|profile| profile.mixed_rows += 1);
}

pub(crate) fn record_key_row(opaque_bits: u32, passthrough_bits: u32, uniform_opaque: bool) {
    with_active_profile(|profile| {
        profile.opaque_key_bits += opaque_bits as u64;
        profile.passthrough_key_bits += passthrough_bits as u64;
        if uniform_opaque {
            profile.uniform_opaque_rows += 1;
        }
        if opaque_bits == 0 && passthrough_bits != 0 {
            profile.passthrough_rows += 1;
        }
    });
}

pub(crate) fn record_single_quads(quads: usize) {
    with_active_profile(|profile| profile.single_quads += quads as u64);
}

pub(crate) fn record_terminal_quads(quads: usize) {
    with_active_profile(|profile| profile.terminal_quads += quads as u64);
}

pub(crate) fn record_mixed_quads(quads: usize) {
    with_active_profile(|profile| profile.mixed_quads += quads as u64);
}

pub(crate) fn record_unit_quads(quads: usize) {
    with_active_profile(|profile| profile.unit_quads += quads as u64);
}
