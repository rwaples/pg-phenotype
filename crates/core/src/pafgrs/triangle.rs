//! Lossless dictionary coding of a chunk's relative-relative kinship.
//!
//! Pedigree kinship takes few distinct values (82 over the 205M entries of
//! simACE `cure_rA50_200k` at degree 3), so each chunk stores its distinct
//! values once and every entry as the narrowest index that fits.  Decoding
//! returns the stored bits exactly.

use std::collections::HashMap;

/// A chunk's triangle entries over a dictionary of their distinct values.
#[derive(Debug)]
pub(crate) struct Triangle {
    dict: Vec<f32>,
    codes: Codes,
}

#[derive(Debug)]
enum Codes {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F32(Vec<f32>),
}

impl Triangle {
    /// Dictionary in first-occurrence order, so it is a function of the
    /// values alone.
    pub(crate) fn encode(values: &[f32]) -> Triangle {
        let mut index: HashMap<u32, u32> = HashMap::new();
        let mut dict: Vec<f32> = Vec::new();
        let mut codes: Vec<u32> = Vec::with_capacity(values.len());
        for v in values {
            let code = *index.entry(v.to_bits()).or_insert_with(|| {
                dict.push(*v);
                dict.len() as u32 - 1
            });
            codes.push(code);
        }
        let codes = if dict.len() <= 1 << 8 {
            Codes::U8(codes.iter().map(|&c| c as u8).collect())
        } else if dict.len() <= 1 << 16 {
            Codes::U16(codes.iter().map(|&c| c as u16).collect())
        } else {
            return Triangle {
                dict: Vec::new(),
                codes: Codes::F32(values.to_vec()),
            };
        };
        Triangle { dict, codes }
    }

    #[inline]
    pub(crate) fn get(&self, at: usize) -> f32 {
        match &self.codes {
            Codes::U8(c) => self.dict[c[at] as usize],
            Codes::U16(c) => self.dict[c[at] as usize],
            Codes::F32(v) => v[at],
        }
    }

    pub(crate) fn bytes(&self) -> usize {
        self.dict.len() * 4
            + match &self.codes {
                Codes::U8(c) => c.len(),
                Codes::U16(c) => c.len() * 2,
                Codes::F32(v) => v.len() * 4,
            }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_every_width() {
        for n_distinct in [1usize, 200, 300, 70_000] {
            let values: Vec<f32> = (0..n_distinct * 2)
                .map(|i| (i % n_distinct) as f32 / 1024.0)
                .collect();
            let tri = Triangle::encode(&values);
            for (at, v) in values.iter().enumerate() {
                assert_eq!(tri.get(at).to_bits(), v.to_bits());
            }
        }
    }

    #[test]
    fn negative_zero_keeps_its_bits() {
        let tri = Triangle::encode(&[0.0, -0.0, 0.25]);
        assert_eq!(tri.get(1).to_bits(), (-0.0f32).to_bits());
    }
}
