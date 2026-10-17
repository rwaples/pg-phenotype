//! Each Relative-Pair estimate's influence (ADR 0013): one entry's
//! contribution to `r̂ − r`, already divided by the fit's entries, so that
//! the cluster sandwich SE is `√meat`.
//!
//! Every estimator's influence is a [`Quadratic`] in the entry's two
//! values, `x` (trait `view.first` of the first member) and `y` (trait
//! `view.second` of the second), one per block cell (first code, first
//! label, second code, second label) of the [`super::Blocks`] the fit read.
//! A binary influence is a constant per cell.  [`Weights`] pulls those
//! back to the engine cells of each block an estimate reads, so a key's
//! influence is a linear function of its rows' exact stride sums.

use super::clusters::Sum;
use super::{observed, Layout, Request, SeReason, View};
use crate::error::Error;
use crate::input::TraitKind;
use crate::polychoric::{cell_influence, CellInfluence, Point, Reason as FitReason, Tables};
use pedigree_graph_core::relationships::{MomentsTable, Side, Statistic};

/// `one + x·x + y·y + xx·x² + yy·y² + xy·x·y`: one entry's influence.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Quadratic {
    one: f64,
    x: f64,
    y: f64,
    xx: f64,
    yy: f64,
    xy: f64,
}

impl Quadratic {
    fn constant(one: f64) -> Quadratic {
        Quadratic {
            one,
            ..Quadratic::default()
        }
    }

    /// `(c[0]·uv + c[1]·u + c[2]·v + c[3]·u² + c[4]·v² + c[5]) / n` with
    /// `u = (x − mx)/sx` and `v = (y − my)/sy`, expanded in `x` and `y`.
    /// Applied to raw sums its terms cancel by about `(mx/sx)²`, which the
    /// midrange centring of [`super::Members`]' values keeps small.
    fn standardised(mx: f64, sx: f64, my: f64, sy: f64, c: [f64; 6], n: f64) -> Quadratic {
        // u = a x + p, v = b y + q.
        let (a, b) = (1.0 / sx, 1.0 / sy);
        let (p, q) = (-mx * a, -my * b);
        Quadratic {
            one: (c[0] * p * q + c[1] * p + c[2] * q + c[3] * p * p + c[4] * q * q + c[5]) / n,
            x: (c[0] * a * q + c[1] * a + 2.0 * c[3] * a * p) / n,
            y: (c[0] * b * p + c[2] * b + 2.0 * c[4] * b * q) / n,
            xx: c[3] * a * a / n,
            yy: c[4] * b * b / n,
            xy: c[0] * a * b / n,
        }
    }

    /// The same quadratic in units `x / sx`, `y / sy`.
    fn scaled(self, sx: f64, sy: f64) -> Quadratic {
        Quadratic {
            one: self.one,
            x: self.x * sx,
            y: self.y * sy,
            xx: self.xx * sx * sx,
            yy: self.yy * sy * sy,
            xy: self.xy * sx * sy,
        }
    }
}

/// An influence's quadratics on one block, per (first label, second label)
/// of the view's observed labels; a continuous view reads only `[0][0]`.
type Labels = [[Quadratic; 2]; 2];

/// The blocks, by (first code, second code), an estimate reads.
enum Reads {
    /// The same quadratics on every block of `rows` x `cols`.
    Uniform {
        rows: Box<[usize]>,
        cols: Box<[usize]>,
        at: Box<Labels>,
    },
    /// Each listed block `(s, u)` with its own quadratics.
    PerBlock(Box<[(usize, usize, Labels)]>),
}

/// One estimate's influence in its Blocks' coordinates: only the blocks it
/// reads, so its size does not grow with the square of the strata.
pub(super) struct Influence {
    view: View,
    reads: Reads,
}

impl Influence {
    /// The same quadratics on every block of `rows` x `cols`.
    fn uniform(view: View, rows: &[usize], cols: &[usize], at: Labels) -> Influence {
        Influence {
            view,
            reads: Reads::Uniform {
                rows: rows.into(),
                cols: cols.into(),
                at: Box::new(at),
            },
        }
    }

    /// Each listed block `(s, u)` with its own quadratics.
    fn per_block(view: View, at: impl IntoIterator<Item = (usize, usize, Labels)>) -> Influence {
        Influence {
            view,
            reads: Reads::PerBlock(at.into_iter().collect()),
        }
    }

    /// Each block `(s, u)` the estimate reads, and its quadratics there.
    fn blocks(&self) -> impl Iterator<Item = (usize, usize, &Labels)> {
        let (uniform, per_block) = match &self.reads {
            Reads::Uniform { rows, cols, at } => (Some((rows, cols, &**at)), None),
            Reads::PerBlock(blocks) => (None, Some(blocks)),
        };
        let uniform = uniform.into_iter().flat_map(|(rows, cols, at)| {
            rows.iter()
                .flat_map(move |&s| cols.iter().map(move |&u| (s, u, at)))
        });
        let per_block = per_block
            .into_iter()
            .flat_map(|blocks| blocks.iter().map(|(s, u, at)| (*s, *u, at)));
        uniform.chain(per_block)
    }

    /// How many blocks it reads.
    fn n_blocks(&self) -> usize {
        match &self.reads {
            Reads::Uniform { rows, cols, .. } => rows.len() * cols.len(),
            Reads::PerBlock(blocks) => blocks.len(),
        }
    }

    /// The bytes it holds, its slot in a request's list included.
    pub(super) fn bytes(influence: &Result<Influence, SeReason>) -> u64 {
        let heap = match influence {
            Ok(Influence {
                reads: Reads::Uniform { rows, cols, .. },
                ..
            }) => (rows.len() + cols.len()) * size_of::<usize>() + size_of::<Labels>(),
            Ok(Influence {
                reads: Reads::PerBlock(blocks),
                ..
            }) => blocks.len() * size_of::<(usize, usize, Labels)>(),
            Err(_) => 0,
        };
        (size_of::<Result<Influence, SeReason>>() + heap) as u64
    }

    /// Pearson's `r` over the blocks `rows` x `cols`, from their summed
    /// `[label, label]` `table`: `(uv − r/2 (u² + v²)) / N`, `u` and `v`
    /// standardised by the table's means and `1/N` SDs.
    pub(super) fn pearson(
        view: View,
        (rows, cols): (&[usize], &[usize]),
        table: &MomentsTable<'_>,
        r: f64,
        name: &str,
    ) -> Result<Influence, Error> {
        let n = table.counts()[0] as f64;
        let side = |side: Side, column: usize| -> Result<(f64, f64), Error> {
            let mean = table.derive(Statistic::Mean(side), column, name)?[0];
            let m2 = table.derive(Statistic::M2(side), column, name)?[0];
            Ok((mean, (m2 / n).sqrt()))
        };
        let ((mx, sx), (my, sy)) = (
            side(Side::First, view.first)?,
            side(Side::Second, view.second)?,
        );
        let q =
            Quadratic::standardised(mx, sx, my, sy, [1.0, 0.0, 0.0, -r / 2.0, -r / 2.0, 0.0], n);
        Ok(Influence::uniform(
            view,
            rows,
            cols,
            [[q, Quadratic::default()], [Quadratic::default(); 2]],
        ))
    }

    /// Phi, the Pearson of the 0/1 values, over the blocks `rows` x `cols`
    /// of the 2 x 2 `counts`: a constant per cell.
    pub(super) fn phi(
        view: View,
        (rows, cols): (&[usize], &[usize]),
        counts: [[u64; 2]; 2],
        phi: f64,
    ) -> Influence {
        let [[a, b], [c, d]] = counts.map(|row| row.map(|v| v as f64));
        let n = a + b + c + d;
        let (pa, pb) = ((c + d) / n, (b + d) / n);
        let (sa, sb) = ((pa * (1.0 - pa)).sqrt(), (pb * (1.0 - pb)).sqrt());
        let at = [0.0, 1.0].map(|i| {
            [0.0, 1.0].map(|j| {
                let (u, v) = ((i - pa) / sa, (j - pb) / sb);
                Quadratic::constant((u * v - phi / 2.0 * (u * u + v * v)) / n)
            })
        });
        Influence::uniform(view, rows, cols, at)
    }

    /// The two-step tetrachoric over the blocks `rows` x `cols`, fitted as
    /// one 2 x 2 table: `influence.at(0, 0, i, j)` on each.
    pub(super) fn tetrachoric(
        view: View,
        (rows, cols): (&[usize], &[usize]),
        influence: &CellInfluence,
    ) -> Influence {
        let at = [0, 1].map(|i| [0, 1].map(|j| Quadratic::constant(influence.at(0, 0, i, j))));
        Influence::uniform(view, rows, cols, at)
    }

    /// The stratified tetrachoric: `influence.at(s, u, i, j)` on each kept
    /// block of the `k x k` known strata.
    pub(super) fn stratified_tetrachoric(
        view: View,
        k: usize,
        kept: &[bool],
        influence: &CellInfluence,
    ) -> Influence {
        let at = (0..k * k).filter(|&b| kept[b]).map(|b| {
            let (s, u) = (b / k, b % k);
            let at = [0, 1].map(|i| [0, 1].map(|j| Quadratic::constant(influence.at(s, u, i, j))));
            (s, u, at)
        });
        Influence::per_block(view, at)
    }

    /// The stratified Pearson `ρ = Σ z₁ z₂ / N`, each side standardised
    /// within its stratum, per kept block `(s, u)`:
    ///
    /// ```text
    /// (z₁ z₂ − ρ − A_s z₁ / N_s − B_s (z₁² − 1) / (2 N_s)
    ///            − A'_u z₂ / N'_u − B'_u (z₂² − 1) / (2 N'_u)) / N
    /// ```
    ///
    /// with `N_s`, `A_s = Σ z₂` and `B_s = Σ z₁ z₂` over the kept entries
    /// whose first member is in `s`, and the primed sums over those whose
    /// second is in `u`: the influence of assortative's
    /// `kernels::pearson_influence`, one block at a time.
    pub(super) fn stratified_pearson(
        view: View,
        scales: [&[(f64, f64)]; 2],
        blocks: &[StratifiedBlock],
        rho: f64,
    ) -> Influence {
        let [rows, cols] = scales;
        let (mut first, mut second) = (vec![[0.0; 3]; rows.len()], vec![[0.0; 3]; cols.len()]);
        let mut n = 0.0;
        for b in blocks {
            let (f, s) = (&mut first[b.s], &mut second[b.u]);
            *f = [f[0] + b.n, f[1] + b.zb, f[2] + b.zab];
            *s = [s[0] + b.n, s[1] + b.za, s[2] + b.zab];
            n += b.n;
        }
        let at = blocks.iter().map(|b| {
            let ([ns, a, bb], [nu, a2, b2]) = (first[b.s], second[b.u]);
            let ((mx, sx), (my, sy)) = (rows[b.s], cols[b.u]);
            let c = [
                1.0,
                -a / ns,
                -a2 / nu,
                -bb / (2.0 * ns),
                -b2 / (2.0 * nu),
                -rho + bb / (2.0 * ns) + b2 / (2.0 * nu),
            ];
            let q = Quadratic::standardised(mx, sx, my, sy, c, n);
            (
                b.s,
                b.u,
                [[q, Quadratic::default()], [Quadratic::default(); 2]],
            )
        });
        Influence::per_block(view, at)
    }
}

/// One populated kept block of a stratified Pearson fit: its strata, its
/// entries, and over them `Σ z₁`, `Σ z₂` and `Σ z₁ z₂`.
pub(super) struct StratifiedBlock {
    pub(super) s: usize,
    pub(super) u: usize,
    pub(super) n: f64,
    pub(super) za: f64,
    pub(super) zb: f64,
    pub(super) zab: f64,
}

/// A two-step tetrachoric fit's influence, built by `build` from its cell
/// influences, or why it has none.
pub(super) fn two_step(
    fit: &Result<(Tables, Point), FitReason>,
    build: impl FnOnce(&CellInfluence) -> Influence,
) -> Result<Influence, SeReason> {
    let (tables, point) = fit.as_ref().map_err(|_| SeReason::NoEstimate)?;
    if point.boundary == Some(true) {
        return Err(SeReason::Boundary);
    }
    let influence = cell_influence(tables, point.value).map_err(|_| SeReason::SandwichUndefined)?;
    Ok(build(&influence))
}

/// One estimate's weight on an engine cell: the stride slots of `x`, `y`,
/// `x²`, `y²` and `x·y` its view reads, and its quadratic in quantized
/// units.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Weight {
    pub(super) estimate: u32,
    slots: [u16; 5],
    w: Quadratic,
}

impl Weight {
    /// The stride slots `view` reads with `k` value columns.
    fn slots(view: View, k: usize) -> [u16; 5] {
        if k == 0 {
            return [0; 5];
        }
        let product = if k == 1 { 0 } else { view.product() };
        [
            1 + view.first,
            1 + k + view.second,
            1 + 2 * k + view.first,
            1 + 3 * k + view.second,
            1 + 4 * k + product,
        ]
        .map(|slot| slot as u16)
    }

    /// The influence of the entries whose stride sums are `m`.
    #[inline]
    pub(super) fn apply(&self, m: &[f64]) -> f64 {
        let [x, y, xx, yy, xy] = self.slots.map(usize::from);
        let w = &self.w;
        w.one * m[0] + w.x * m[x] + w.y * m[y] + w.xx * m[xx] + w.yy * m[yy] + w.xy * m[xy]
    }
}

/// Per trait, each joint label that shows it with the trait's label there:
/// the engine labels one block cell spreads over.
struct Shown {
    n_joint: usize,
    /// Engine labels: codes times joint labels.
    n_labels: usize,
    traits: Vec<Vec<(usize, usize)>>,
}

impl Shown {
    fn of(layout: Layout) -> Shown {
        let n_joint = layout.n_joint();
        let traits = (0..layout.n_traits)
            .map(|t| {
                (0..n_joint)
                    .map(|joint| (joint, layout.digit(joint, t)))
                    .filter(|&(_, label)| label < observed(layout.kind))
                    .collect()
            })
            .collect();
        Shown {
            n_joint,
            n_labels: layout.n_codes * n_joint,
            traits,
        }
    }

    /// The engine cells of one category.
    fn n_cells(&self) -> usize {
        self.n_labels * self.n_labels
    }

    /// How many engine cells `influence` reads: per block, those whose
    /// labels show both traits its view reads.
    fn n_read(&self, influence: &Influence) -> u64 {
        let view = influence.view;
        let per_block = self.traits[view.first].len() * self.traits[view.second].len();
        (influence.n_blocks() * per_block) as u64
    }

    /// `f(cell, i, j, at)` on each engine cell `influence` reads, `(i, j)`
    /// the block cell's labels and `at` the block's quadratics.
    fn each(&self, influence: &Influence, mut f: impl FnMut(usize, usize, usize, &Labels)) {
        let view = influence.view;
        let (firsts, seconds) = (&self.traits[view.first], &self.traits[view.second]);
        for (s, u, at) in influence.blocks() {
            for &(a, i) in firsts {
                let row = (s * self.n_joint + a) * self.n_labels + u * self.n_joint;
                for &(b, j) in seconds {
                    f(row + b, i, j, at);
                }
            }
        }
    }
}

/// One request's influences pulled back to engine cells.  A request's
/// categories (a group's members) share one table by (first label, second
/// label), as their tables are summed before any fit.
pub(super) struct Weights {
    /// Per engine cell of one category, `first label * labels + second
    /// label`: its range of `items`, estimates ascending.
    offsets: Vec<u32>,
    items: Vec<Weight>,
    pub(super) n_estimates: usize,
}

impl Weights {
    /// `influences` in estimate order, each read on every engine cell of
    /// the blocks it reads, scaled from the trait's units to its
    /// quantization steps (`exponents`).  One pass counts each cell's
    /// weights and one places them, influence by influence, so the work is
    /// the weights made.
    pub(super) fn build(
        layout: Layout,
        exponents: &[i64],
        influences: &[Result<Influence, SeReason>],
    ) -> Weights {
        let k = exponents.len();
        let step = |t: usize| exponents.get(t).map_or(1.0, |&e| super::step_of(e));
        let shown = Shown::of(layout);
        let n_cells = shown.n_cells();
        let mut offsets = vec![0u32; n_cells + 1];
        for influence in influences.iter().flatten() {
            shown.each(influence, |cell, _, _, _| offsets[cell + 1] += 1);
        }
        for cell in 0..n_cells {
            offsets[cell + 1] += offsets[cell];
        }
        let mut next = offsets[..n_cells].to_vec();
        let mut items = vec![Weight::default(); offsets[n_cells] as usize];
        for (e, influence) in influences.iter().enumerate() {
            let Ok(influence) = influence else { continue };
            let view = influence.view;
            let slots = Weight::slots(view, k);
            let (sx, sy) = (step(view.first), step(view.second));
            shown.each(influence, |cell, i, j, at| {
                let place = &mut next[cell];
                items[*place as usize] = Weight {
                    estimate: e as u32,
                    slots,
                    w: at[i][j].scaled(sx, sy),
                };
                *place += 1;
            });
        }
        Weights {
            offsets,
            items,
            n_estimates: influences.len(),
        }
    }

    /// The bytes [`Weights::build`] holds at its peak for `influences`,
    /// counted without building them.
    pub(super) fn bytes(layout: Layout, influences: &[Result<Influence, SeReason>]) -> u64 {
        let shown = Shown::of(layout);
        let items: u64 = influences
            .iter()
            .flatten()
            .map(|influence| shown.n_read(influence))
            .sum();
        items
            .saturating_mul(size_of::<Weight>() as u64)
            .saturating_add(offset_bytes(shown.n_cells() as u64))
    }

    /// The engine cells of one category.
    pub(super) fn cells(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The weights on engine cell `cell` of one category.
    #[inline]
    pub(super) fn of(&self, cell: usize) -> &[Weight] {
        &self.items[self.offsets[cell] as usize..self.offsets[cell + 1] as usize]
    }
}

/// [`Weights::build`]'s offsets and their placing cursor over `n_cells`.
fn offset_bytes(n_cells: u64) -> u64 {
    n_cells
        .saturating_mul(2)
        .saturating_add(1)
        .saturating_mul(4)
}

/// The bytes the SE keeps back from the pass, and how many estimates a
/// sweep takes, from the request shapes alone: enough to sweep one request
/// at a time after the pass ([`sweeps`]).
///
/// A request with `v` views and `k` known strata fits per view `e(1 + k)`
/// estimates (`e` = 2 binary, 1 continuous), each over at most `k + 1`
/// first and second codes, plus a stratified one over at most `k²`
/// blocks.  An engine cell whose labels show the view's traits is read by
/// at most `2e + 1` of a view's estimates: the pooled, one stratum cell's,
/// the stratified fit.  Kept back: `ancestry`, every request's influences,
/// which live until the last sweep, and one sweep's: its request's
/// [`Weights`] and `n_rows · batch` person totals, a [`Sum`] each.
/// `batch` is every estimate of the largest request unless that passes a
/// quarter of the budget, then as many as fit it (at least one).
pub(super) fn reserve(
    layout: Layout,
    requests: &[Request],
    strata: Option<usize>,
    n_rows: usize,
    budget: u64,
    ancestry: u64,
) -> (u64, usize) {
    let e = if layout.kind == TraitKind::Binary {
        2
    } else {
        1
    };
    let (k, stratified) = strata.map_or((0, 0), |k| (k, 1));
    let shown = Shown::of(layout);
    let n_shown = shown.traits.iter().map(Vec::len).max().unwrap_or(0) as u64;
    let n_codes = layout.n_codes as u64;
    let slot = size_of::<Result<Influence, SeReason>>() as u64;
    let uniform = slot + 2 * n_codes * size_of::<usize>() as u64 + size_of::<Labels>() as u64;
    let per_block = ((k * k) as u64)
        .saturating_mul(size_of::<(usize, usize, Labels)>() as u64)
        .saturating_add(slot);
    let read = (n_codes * n_shown).saturating_mul(n_codes * n_shown);
    let (mut influences, mut weights, mut estimates) = (0u64, 0u64, 1usize);
    for request in requests {
        let views = View::of(layout.n_traits, request.symmetric).len() as u64;
        let fits = e * (1 + k) as u64;
        estimates = estimates.max(views as usize * (fits as usize + stratified));
        influences = influences.saturating_add(views.saturating_mul(
            (fits * uniform).saturating_add(per_block.saturating_mul(stratified as u64)),
        ));
        let items = read.saturating_mul(views * (2 * e + stratified as u64));
        weights = weights.max(
            items
                .saturating_mul(size_of::<Weight>() as u64)
                .saturating_add(offset_bytes(shown.n_cells() as u64)),
        );
    }
    let per_estimate = person_bytes(n_rows);
    let batch = if per_estimate.saturating_mul(estimates as u64) <= budget / 4 {
        estimates
    } else {
        ((budget / 4 / per_estimate) as usize).max(1)
    };
    let reserve = ancestry
        .saturating_add(influences)
        .saturating_add(weights)
        .saturating_add(per_estimate.saturating_mul(batch as u64));
    (reserve, batch)
}

/// One estimate's person totals in a sweep.
fn person_bytes(n_rows: usize) -> u64 {
    (n_rows as u64)
        .saturating_mul(size_of::<Sum>() as u64)
        .max(1)
}

/// The bytes of the smallest plan [`sweeps`] makes: one request at a time,
/// the largest [`Weights`] and `batch` estimates' person totals.
pub(super) fn one_sweep(requests: &[(u64, usize)], n_rows: usize, batch: usize) -> u64 {
    let weights = requests.iter().map(|&(w, _)| w).max().unwrap_or(0);
    weights.saturating_add(person_bytes(n_rows).saturating_mul(batch as u64))
}

/// How many requests sweep at once and how many estimates a sweep takes,
/// in the `free` bytes the pass and the fits leave: each request's
/// estimates in one sweep, as many requests at once as `free` holds, at
/// most `threads`; failing that, [`reserve`]'s plan of one request at a
/// time, `batch` estimates a sweep.  `requests` gives each request's
/// [`Weights::bytes`] and estimates.
pub(super) fn sweeps(
    free: u64,
    threads: usize,
    requests: &[(u64, usize)],
    n_rows: usize,
    batch: usize,
) -> (usize, usize) {
    let per_estimate = person_bytes(n_rows);
    let need = requests
        .iter()
        .map(|&(weights, n)| weights.saturating_add(per_estimate.saturating_mul(n as u64)))
        .max()
        .unwrap_or(0)
        .max(1);
    let most = requests.iter().map(|&(_, n)| n).max().unwrap_or(1);
    match (free / need) as usize {
        0 => (1, batch),
        fit => (fit.min(threads).min(requests.len()).max(1), most.max(1)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn labels(seed: usize) -> Labels {
        let q = |i: usize| {
            let v = (seed * 4 + i) as f64;
            Quadratic {
                one: v + 0.5,
                x: v - 1.25,
                y: 2.0 * v,
                xx: -v,
                yy: v / 3.0,
                xy: 1.0 / (v + 1.0),
            }
        };
        [[q(0), q(1)], [q(2), q(3)]]
    }

    fn layout(kind: TraitKind, n_traits: usize, n_codes: usize) -> Layout {
        Layout {
            kind,
            n_traits,
            n_labels: if kind == TraitKind::Binary { 3 } else { 2 },
            n_codes,
        }
    }

    /// A request's influences shaped as a stratified cell's fits give
    /// them: per view the pooled and each stratum cell's `e`, an undefined
    /// one, and a stratified fit over two blocks in three.
    fn shaped(layout: Layout, symmetric: bool) -> Vec<Result<Influence, SeReason>> {
        let (n, k) = (layout.n_codes, layout.n_codes - 1);
        let e = if layout.kind == TraitKind::Binary {
            2
        } else {
            1
        };
        let all: Vec<usize> = (0..n).collect();
        let mut out = Vec::new();
        for view in View::of(layout.n_traits, symmetric) {
            for _ in 0..e {
                out.push(Ok(Influence::uniform(view, &all, &all, labels(out.len()))));
            }
            for s in 0..k {
                let cols = if symmetric { &all[s..=s] } else { &all[..] };
                for _ in 0..e {
                    out.push(Ok(Influence::uniform(view, &[s], cols, labels(out.len()))));
                }
            }
            out.push(Err(SeReason::NoEstimate));
            let blocks = (0..k * k)
                .filter(|b| b % 3 != 1)
                .map(|b| (b / k, b % k, labels(b)));
            out.push(Ok(Influence::per_block(view, blocks)));
        }
        out
    }

    /// The weights on an engine cell, every influence tried on it: the
    /// build before it read only the blocks each influence reads.
    fn on_cell<'a>(
        layout: Layout,
        exponents: &'a [i64],
        influences: &'a [Result<Influence, SeReason>],
    ) -> impl Fn(usize) -> Vec<Weight> + 'a {
        let n_joint = layout.n_joint();
        let n_labels = layout.n_codes * n_joint;
        let step = |t: usize| exponents.get(t).map_or(1.0, |&e| super::super::step_of(e));
        let blocks: Vec<Option<HashMap<(usize, usize), &Labels>>> = influences
            .iter()
            .map(|influence| {
                let influence = influence.as_ref().ok()?;
                Some(influence.blocks().map(|(s, u, at)| ((s, u), at)).collect())
            })
            .collect();
        move |cell| {
            let (first, second) = (cell / n_labels, cell % n_labels);
            influences
                .iter()
                .zip(&blocks)
                .enumerate()
                .filter_map(|(e, (influence, blocks))| {
                    let (influence, blocks) = (influence.as_ref().ok()?, blocks.as_ref()?);
                    let view = influence.view;
                    let (i, j) = (
                        layout.digit(first % n_joint, view.first),
                        layout.digit(second % n_joint, view.second),
                    );
                    if i >= observed(layout.kind) || j >= observed(layout.kind) {
                        return None;
                    }
                    let at = blocks.get(&(first / n_joint, second / n_joint))?;
                    Some(Weight {
                        estimate: e as u32,
                        slots: Weight::slots(view, exponents.len()),
                        w: at[i][j].scaled(step(view.first), step(view.second)),
                    })
                })
                .collect()
        }
    }

    #[test]
    fn weights_are_every_influence_tried_on_every_engine_cell() {
        for (layout, exponents) in [
            (layout(TraitKind::Binary, 2, 5), vec![]),
            (layout(TraitKind::Binary, 1, 1), vec![]),
            (layout(TraitKind::Continuous, 2, 4), vec![-3, 5]),
            (layout(TraitKind::Continuous, 1, 3), vec![7]),
        ] {
            for symmetric in [true, false] {
                let influences = shaped(layout, symmetric);
                let weights = Weights::build(layout, &exponents, &influences);
                let expect = on_cell(layout, &exponents, &influences);
                let mut items = 0;
                for cell in 0..weights.cells() {
                    let expected = expect(cell);
                    assert_eq!(weights.of(cell), expected, "{layout:?} cell {cell}");
                    items += expected.len();
                }
                assert!(items > 0);
                assert_eq!(
                    Weights::bytes(layout, &influences),
                    (items * size_of::<Weight>()) as u64 + offset_bytes(weights.cells() as u64)
                );
            }
        }
    }

    /// Eighty strata of two binary traits: 531,441 engine cells and 492
    /// influences a request, which tried on every cell took seconds a
    /// request.  The weights read only each influence's blocks, and the
    /// reserve covers them and the influences.
    #[test]
    fn eighty_strata_weights_are_built_from_the_blocks_read() {
        let layout = layout(TraitKind::Binary, 2, 81);
        for symmetric in [true, false] {
            let influences = shaped(layout, symmetric);
            let weights = Weights::build(layout, &[], &influences);
            assert_eq!(weights.cells(), 729 * 729);
            let expect = on_cell(layout, &[], &influences);
            for cell in (0..weights.cells()).step_by(997) {
                assert_eq!(weights.of(cell), expect(cell), "cell {cell}");
            }
            let request = Request {
                name: String::new(),
                members: Vec::new(),
                symmetric,
                roles: None,
            };
            let (reserve, batch) = super::reserve(layout, &[request], Some(80), 0, u64::MAX, 0);
            let held: u64 = influences.iter().map(Influence::bytes).sum();
            assert!(reserve >= held + Weights::bytes(layout, &influences) + batch as u64);
        }
    }
}
