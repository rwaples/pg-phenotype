//! The Relative-Pair SE's heap, measured by a counting allocator: one test,
//! as every test in the binary shares the counter.

use pg_phenotype_core::assortative::Strata;
use pg_phenotype_core::relative_pairs::{correlation, Form, Se};
use pg_phenotype_core::{PedigreeArg, PedigreeInput, Trait, TraitKind};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

struct Counting;

static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);

fn grow(bytes: usize) {
    let live = LIVE.fetch_add(bytes as u64, Relaxed) + bytes as u64;
    PEAK.fetch_max(live, Relaxed);
}

// SAFETY: every call goes to `System` unchanged; only the counters are added.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            grow(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            grow(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) };
        LIVE.fetch_sub(layout.size() as u64, Relaxed);
    }

    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, layout, size) };
        if !q.is_null() {
            if size > layout.size() {
                grow(size - layout.size());
            } else {
                LIVE.fetch_sub((layout.size() - size) as u64, Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// `f`'s result and its peak heap above what was live before it.
fn peak<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let out = f();
    (out, PEAK.load(Relaxed) - base)
}

/// Founders mating at random over `generations`, two to four children a
/// couple, a fifth of mothers having a second partner.
struct Columns {
    ids: Vec<i64>,
    mother: Vec<i64>,
    father: Vec<i64>,
    generation: Vec<i64>,
}

fn generated(founders: usize, generations: i64, seed: u64) -> Columns {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move |bound: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % bound as u64) as usize
    };
    let mut cols = Columns {
        ids: Vec::new(),
        mother: Vec::new(),
        father: Vec::new(),
        generation: Vec::new(),
    };
    let push = |cols: &mut Columns, mother: i64, father: i64, generation: i64| {
        cols.ids.push(cols.ids.len() as i64);
        cols.mother.push(mother);
        cols.father.push(father);
        cols.generation.push(generation);
    };
    let mut parents: Vec<i64> = (0..founders as i64).collect();
    for _ in 0..founders {
        push(&mut cols, -1, -1, 0);
    }
    for generation in 1..=generations {
        let (women, men) = parents.split_at(parents.len() / 2);
        let mut children = Vec::new();
        for &mother in women {
            let partners = if next(5) == 0 { 2 } else { 1 };
            for _ in 0..partners {
                let father = men[next(men.len())];
                for _ in 0..2 + next(3) {
                    children.push(cols.ids.len() as i64);
                    push(&mut cols, mother, father, generation);
                }
            }
        }
        let keep = founders.min(children.len());
        parents = children[children.len() - keep..].to_vec();
    }
    cols
}

/// With the SE, and with the covariance, a call that passes the budget's
/// checks holds no more heap than the budget and its inputs (the call's
/// heap without any category: the validated pedigree, the members' labels
/// and values), at the smallest budget it passes and at twice that, at one
/// thread and four, with generation strata and with twelve.
#[test]
fn a_call_with_the_se_stays_within_its_budget() {
    let cols = generated(200, 4, 7);
    let n = cols.ids.len();
    let input = PedigreeInput {
        ids: &cols.ids,
        mother: &cols.mother,
        father: &cols.father,
        twin: None,
        sex: None,
    };
    let known: Vec<bool> = (0..n).map(|i| i % 11 != 0).collect();
    let twelve: Vec<i64> = (0..n as i64).map(|i| i % 12).collect();
    let noise = |i: usize, t: usize| ((i * 7919 + 13 + 31 * t) % 101) as f64 / 101.0;
    for (kind, n_traits, labels) in [
        (TraitKind::Continuous, 1, None),
        (TraitKind::Binary, 2, Some(&cols.generation)),
        (TraitKind::Continuous, 2, Some(&cols.generation)),
        (TraitKind::Binary, 2, Some(&twelve)),
    ] {
        let values: Vec<Vec<f64>> = (0..n_traits)
            .map(|t| {
                (0..n)
                    .map(|i| match kind {
                        _ if (i + t) % 9 == 0 => f64::NAN,
                        TraitKind::Binary => f64::from(u8::from(noise(i, t) < 0.4)),
                        _ => noise(i, t) + 0.5 * noise(cols.mother[i].max(0) as usize, t),
                    })
                    .collect()
            })
            .collect();
        let traits: Vec<Trait<'_>> = values
            .iter()
            .map(|values| Trait {
                values,
                kind,
                n_levels: None,
            })
            .collect();
        let strata = labels.map(|labels| Strata {
            labels,
            known: &known,
        });
        for (threads, se) in [1, 4]
            .into_iter()
            .flat_map(|t| [(t, Se::Estimates), (t, Se::Covariance)])
        {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let run = |budget: u64, categories: Option<&[_]>| {
                pool.install(|| {
                    correlation(
                        PedigreeArg::Columns(input),
                        &traits,
                        strata,
                        categories,
                        budget as i64,
                        se,
                    )
                })
            };
            let (mut lo, mut hi) = (1u64, 1u64 << 31);
            while hi - lo > hi / 100 {
                let mid = lo + (hi - lo) / 2;
                if run(mid, None).is_ok() {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let (_, inputs) = peak(|| run(1 << 31, Some(&[])).unwrap());
            for budget in [hi, 2 * hi] {
                let (res, held) = peak(|| run(budget, None).unwrap());
                assert!(res
                    .cells
                    .iter()
                    .any(|c| c.estimates[0].se.is_some_and(|se| se.is_ok())));
                let covariance = res.covariance.as_ref().map(|c| c.of(Form::Pooled).unwrap());
                assert_eq!(
                    covariance.is_some_and(|m| m.iter().any(|v| v.is_finite() && *v != 0.0)),
                    se == Se::Covariance
                );
                let what =
                    format!("{kind:?} x{n_traits}, {se:?}, {threads} threads, budget {budget}");
                assert!(
                    held <= budget + inputs,
                    "{what}: {held} > {budget} + {inputs}"
                );
            }
        }
    }
}
