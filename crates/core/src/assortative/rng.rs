//! The draw streams: SplitMix64 keyed by (seed, draw), so a draw is the same
//! under any thread count or batch schedule (pedsum
//! `assortative_kernels.py:891-944`, `:990-1008`, `:1356-1381`).

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX_1: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_2: u64 = 0x94D0_49BB_1331_11EB;
const BOOTSTRAP_DOMAIN: u64 = 0xD1B5_4A32_D192_ED03;
const TWO_POW_MINUS_53: f64 = 1.0 / (1u64 << 53) as f64;

/// SplitMix64's output function.
#[inline]
fn mix64(z: u64) -> u64 {
    let z = (z ^ (z >> 30)).wrapping_mul(MIX_1);
    let z = (z ^ (z >> 27)).wrapping_mul(MIX_2);
    z ^ (z >> 31)
}

/// One draw's stream.
pub(crate) struct Stream(u64);

impl Stream {
    /// Permutation draw `draw` of `seed`.
    pub fn permutation(seed: i64, draw: u64) -> Stream {
        Stream(mix64(mix64(seed as u64).wrapping_add(draw)))
    }

    /// Bootstrap draw `draw` of `seed`, a domain of its own.
    pub fn bootstrap(seed: i64, draw: u64) -> Stream {
        Stream(mix64(
            mix64(seed as u64 ^ BOOTSTRAP_DOMAIN).wrapping_add(draw),
        ))
    }

    /// A uniform integer in `[0, n)`: the top 53 bits as a double, times `n`.
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_add(GOLDEN);
        // Below 2^53, so the signed conversion is exact and one instruction.
        let u = ((mix64(self.0) >> 11) as i64) as f64 * TWO_POW_MINUS_53;
        // A signed conversion, numba's int(): u * n lies in [0, n).
        ((u * n as f64) as i64).min(n as i64 - 1) as usize
    }
}

/// Fill `order` (row indices listed block by block) with permutation draw
/// `draw` of `seed`: Fisher-Yates within each block, a block of one taking
/// no number.
pub(crate) fn shuffle_index(order: &mut [u32], block_start: &[usize], seed: i64, draw: u64) {
    let mut stream = Stream::permutation(seed, draw);
    for b in 0..block_start.len() - 1 {
        let (lo, hi) = (block_start[b], block_start[b + 1]);
        for (t, o) in order[lo..hi].iter_mut().enumerate() {
            *o = (lo + t) as u32;
        }
        for t in (1..hi - lo).rev() {
            let j = stream.below(t + 1);
            order.swap(lo + t, lo + j);
        }
    }
}

/// Fill `w` with bootstrap draw `draw` of `seed`: `G = mult.len()` Mate
/// Networks drawn with replacement, each pair weighted by its network's
/// multiplicity.
pub(crate) fn network_weights(
    w: &mut [f64],
    mult: &mut [u32],
    labels: &[usize],
    seed: i64,
    draw: u64,
) {
    network_multiplicities(mult, seed, draw);
    for (wi, &l) in w.iter_mut().zip(labels) {
        *wi = mult[l] as f64;
    }
}

/// Fill `mult` with how often draw `draw` of `seed` takes each of its
/// `mult.len()` networks.
pub(crate) fn network_multiplicities(mult: &mut [u32], seed: i64, draw: u64) {
    let g = mult.len();
    mult.fill(0);
    let mut stream = Stream::bootstrap(seed, draw);
    for _ in 0..g {
        mult[stream.below(g)] += 1;
    }
}
