//! A validated pedigree that several methods, or several calls of one,
//! share (ADR 0006).

use crate::assortative::{mate_networks, mating_pairs, MatingPairs};
use crate::error::{Error, Position};
use crate::input::PedigreeInput;
use pedigree_graph_core::graph::{self, Columns, IdIndex, Limits, SexEncoding};
use pedigree_graph_core::relationships;
use std::borrow::Cow;
use std::ops::Deref;
use std::sync::{Arc, OnceLock};

/// A pedigree pedigree-graph-core's validator accepted, in its input's row
/// order, with the Mating Pairs it finds once used, the Mate Networks once a
/// call with inference needs them, and an id index once ids are looked up.
///
/// It keeps only what no method parameter changes.  The caches are filled
/// serially: a Rayon worker waiting on one must not steal a job that waits
/// on the same cache.
#[derive(Debug)]
pub struct Pedigree {
    /// Shared with every Prep built from it.
    ids: Arc<Vec<i64>>,
    mother_ids: Vec<i64>,
    father_ids: Vec<i64>,
    mother_rows: Vec<i32>,
    father_rows: Vec<i32>,
    twin_rows: Vec<i32>,
    pairs: OnceLock<MatingPairs>,
    /// The Mate Network of each of `pairs`.
    networks: OnceLock<Vec<usize>>,
    index: OnceLock<IdIndex>,
}

impl Pedigree {
    /// Validate `input` by pedigree-graph-core's own rules (ADR 0001).
    ///
    /// # Errors
    ///
    /// Any pedigree-graph-core validation error, with its code.
    pub fn new(input: PedigreeInput<'_>) -> Result<Pedigree, Error> {
        let built = graph::build(
            Columns {
                ids: input.ids,
                mother: input.mother,
                father: input.father,
                twin: input.twin,
                sex: input.sex,
                generation: None,
                birth_year: None,
            },
            SexEncoding::Simace,
            Limits::default(),
        )?;
        Ok(Pedigree {
            ids: Arc::new(built.ids),
            mother_ids: built.mother_ids,
            father_ids: built.father_ids,
            mother_rows: built.mother_rows,
            father_rows: built.father_rows,
            twin_rows: built.twin_rows,
            pairs: OnceLock::new(),
            networks: OnceLock::new(),
            index: OnceLock::new(),
        })
    }

    /// The number of rows.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether there are no rows.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Row ids, in input order.
    pub fn ids(&self) -> &[i64] {
        &self.ids
    }

    /// The row of each of `ids`.
    ///
    /// # Errors
    ///
    /// At the first id that is not a row's, [`Error::UnknownId`], or that
    /// repeats an earlier one, [`Error::DuplicateLookupId`].
    pub fn rows(&self, ids: &[i64]) -> Result<Vec<usize>, Error> {
        let index = self.index.get_or_init(|| IdIndex::build(&self.ids));
        checked_rows(
            index,
            self.len(),
            ids,
            |id, position| Error::UnknownId {
                id,
                position: position.into(),
            },
            |id, positions| Error::DuplicateLookupId {
                id,
                positions: positions.map(Position::from),
            },
        )
    }

    /// `values`, one per entry of `ids`, moved to the rows those ids name;
    /// a row no id names gets `missing`.
    ///
    /// # Errors
    ///
    /// [`Error::PairLength`] when `values` is not `ids`'s length, then the
    /// errors of [`Pedigree::rows`].
    pub fn align(&self, ids: &[i64], values: &[f64], missing: f64) -> Result<Vec<f64>, Error> {
        if values.len() != ids.len() {
            return Err(Error::PairLength {
                columns: ["ids", "values"],
                expected_length: ids.len(),
                actual_length: values.len(),
            });
        }
        let mut out = vec![missing; self.len()];
        for (row, &value) in self.rows(ids)?.into_iter().zip(values) {
            out[row] = value;
        }
        Ok(out)
    }

    /// The id index a lookup built and kept, or one for the caller alone,
    /// which never enters the Pedigree.
    pub(crate) fn index_or_build(&self) -> Cow<'_, IdIndex> {
        match self.index.get() {
            Some(index) => Cow::Borrowed(index),
            None => Cow::Owned(IdIndex::build(&self.ids)),
        }
    }

    #[cfg(test)]
    pub(crate) fn cached_index(&self) -> Option<&IdIndex> {
        self.index.get()
    }

    pub(crate) fn shared_ids(&self) -> Arc<Vec<i64>> {
        Arc::clone(&self.ids)
    }

    /// Mother row per row, `-1` when missing or external.
    pub(crate) fn mother_rows(&self) -> &[i32] {
        &self.mother_rows
    }

    /// Father row per row, `-1` when missing or external.
    pub(crate) fn father_rows(&self) -> &[i32] {
        &self.father_rows
    }

    /// MZ co-twin row per row, `-1` when missing or external.
    pub(crate) fn twin_rows(&self) -> &[i32] {
        &self.twin_rows
    }

    /// The pedigree as pedigree-graph-core's relationship engine reads it.
    ///
    /// # Errors
    ///
    /// None for a validated pedigree; pedigree-graph-core's column checks
    /// pass through.
    pub(crate) fn relationships(&self) -> Result<relationships::Pedigree<'_>, Error> {
        Ok(relationships::Pedigree::try_new(
            &self.mother_rows,
            &self.father_rows,
            &self.twin_rows,
            &self.mother_ids,
            &self.father_ids,
        )?)
    }

    /// Every Mating Pair, built on first use.
    pub(crate) fn mating_pairs(&self) -> &MatingPairs {
        self.pairs
            .get_or_init(|| mating_pairs(&self.ids, &self.mother_rows, &self.father_rows))
    }

    /// The Mate Network of each of [`Pedigree::mating_pairs`], built on
    /// first use.
    pub(crate) fn networks(&self) -> &[usize] {
        self.networks.get_or_init(|| {
            let pairs = self.mating_pairs();
            mate_networks(&pairs.mothers, &pairs.fathers)
        })
    }
}

/// The row of each of `ids` in `index` over `n` rows, with the caller's
/// errors for an id that is not a row's and one listed twice.
pub(crate) fn checked_rows(
    index: &IdIndex,
    n: usize,
    ids: &[i64],
    unknown: impl Fn(i64, usize) -> Error,
    duplicate: impl Fn(i64, [usize; 2]) -> Error,
) -> Result<Vec<usize>, Error> {
    // Each stored position names a distinct row, so it is below n.
    let mut seen = vec![u32::MAX; n];
    let mut rows = Vec::with_capacity(ids.len());
    for (position, &id) in ids.iter().enumerate() {
        let row = index.row(id);
        if row < 0 {
            return Err(unknown(id, position));
        }
        let row = row as usize;
        if seen[row] != u32::MAX {
            return Err(duplicate(id, [seen[row] as usize, position]));
        }
        seen[row] = position as u32;
        rows.push(row);
    }
    Ok(rows)
}

/// What a method takes: a built [`Pedigree`], or columns it validates for
/// that call alone, after its own parameters.
#[derive(Clone, Copy, Debug)]
pub enum PedigreeArg<'a> {
    Columns(PedigreeInput<'a>),
    Built(&'a Pedigree),
}

impl<'a> PedigreeArg<'a> {
    /// The Pedigree, validating columns.
    pub(crate) fn resolve(self) -> Result<Resolved<'a>, Error> {
        Ok(match self {
            PedigreeArg::Columns(input) => Resolved::Owned(Box::new(Pedigree::new(input)?)),
            PedigreeArg::Built(pedigree) => Resolved::Built(pedigree),
        })
    }
}

/// A Pedigree a call borrowed or built for itself.
pub(crate) enum Resolved<'a> {
    Owned(Box<Pedigree>),
    Built(&'a Pedigree),
}

impl Deref for Resolved<'_> {
    type Target = Pedigree;

    fn deref(&self) -> &Pedigree {
        match self {
            Resolved::Owned(pedigree) => pedigree,
            Resolved::Built(pedigree) => pedigree,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows out of id order: 30 and 31 are founders, 10 and 11 their children.
    fn columns() -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        (
            vec![10, 30, 11, 31],
            vec![30, -1, 30, -1],
            vec![31, -1, 31, -1],
        )
    }

    fn built(ids: &[i64], mother: &[i64], father: &[i64]) -> Result<Pedigree, Error> {
        Pedigree::new(PedigreeInput {
            ids,
            mother,
            father,
            twin: None,
            sex: None,
        })
    }

    #[test]
    fn pedigree_errors_pass_through_with_their_code() {
        let err = built(&[1, 1], &[-1, -1], &[-1, -1]).unwrap_err();
        assert_eq!(err.code(), "duplicate_id");
        let err = built(&[1, 2], &[2, 1], &[-1, -1]).unwrap_err();
        assert_eq!(err.code(), "cycle");
    }

    #[test]
    fn len_and_ids_follow_the_input_rows() {
        let (ids, mother, father) = columns();
        let ped = built(&ids, &mother, &father).unwrap();
        assert_eq!(ped.len(), 4);
        assert!(!ped.is_empty());
        assert_eq!(ped.ids(), &ids);
        assert_eq!(ped.mother_rows(), &[1, -1, 1, -1]);
        assert_eq!(ped.father_ids, father);
    }

    #[test]
    fn caches_are_built_once() {
        let (ids, mother, father) = columns();
        let ped = built(&ids, &mother, &father).unwrap();
        let pairs = ped.mating_pairs();
        assert_eq!(
            (pairs.mothers.as_slice(), pairs.fathers.as_slice()),
            (&[1][..], &[3][..])
        );
        assert!(std::ptr::eq(pairs, ped.mating_pairs()));
        let networks = ped.networks();
        assert_eq!(networks, &[0]);
        assert!(std::ptr::eq(networks, ped.networks()));
    }

    #[test]
    fn rows_and_align_follow_the_ids() {
        let (ids, mother, father) = columns();
        let ped = built(&ids, &mother, &father).unwrap();
        assert_eq!(ped.rows(&[31, 10]).unwrap(), vec![3, 0]);
        assert_eq!(ped.rows(&[]).unwrap(), Vec::<usize>::new());
        let index: *const IdIndex = ped.index.get().unwrap();
        let aligned = ped.align(&[11, 30], &[1.5, f64::NAN], -9.0).unwrap();
        let bits: Vec<u64> = aligned.iter().map(|v| v.to_bits()).collect();
        let want: Vec<u64> = [-9.0, f64::NAN, 1.5, -9.0]
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(bits, want);
        assert!(std::ptr::eq(index, ped.index.get().unwrap()));
    }

    #[test]
    fn rows_and_align_reject_unknown_and_repeated_ids() {
        let (ids, mother, father) = columns();
        let ped = built(&ids, &mother, &father).unwrap();
        assert_eq!(
            ped.rows(&[10, 99]).unwrap_err(),
            Error::UnknownId {
                id: 99,
                position: 1.into()
            }
        );
        assert_eq!(
            ped.rows(&[-1]).unwrap_err(),
            Error::UnknownId {
                id: -1,
                position: 0.into()
            }
        );
        assert_eq!(
            ped.rows(&[31, 10, 31]).unwrap_err(),
            Error::DuplicateLookupId {
                id: 31,
                positions: [0.into(), 2.into()]
            }
        );
        let err = ped.align(&[10, 11], &[1.0], f64::NAN).unwrap_err();
        assert_eq!(err.code(), "pair_length_mismatch");
        let err = ped.align(&[10, 10], &[1.0, 2.0], f64::NAN).unwrap_err();
        assert_eq!(err.code(), "duplicate_lookup_id");
    }

    #[test]
    fn concurrent_first_use_sees_one_cache() {
        let (ids, mother, father) = columns();
        let ped = built(&ids, &mother, &father).unwrap();
        let seen: Vec<(usize, usize)> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    s.spawn(|| {
                        let pairs: *const MatingPairs = ped.mating_pairs();
                        let networks: *const [usize] = ped.networks();
                        (pairs as usize, networks as *const usize as usize)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert!(seen.windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn a_built_arg_borrows_and_columns_build() {
        let (ids, mother, father) = columns();
        let ped = built(&ids, &mother, &father).unwrap();
        let resolved = PedigreeArg::Built(&ped).resolve().unwrap();
        assert!(std::ptr::eq(&*resolved, &ped));
        let input = PedigreeInput {
            ids: &ids,
            mother: &mother,
            father: &father,
            twin: None,
            sex: None,
        };
        let owned = PedigreeArg::Columns(input).resolve().unwrap();
        assert_eq!(owned.ids(), ped.ids());
    }
}
