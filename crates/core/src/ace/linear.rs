//! The ACE Fit's dense linear algebra: the pivoted QR least squares, and
//! for generalized least squares the Cholesky factor and its inverse.
//! Matrices are small (a row per cell), so every routine is the plain
//! textbook loop.

/// The least-squares `b` of `X b = y` for each right-hand side `y` of
/// `rhs`, `X` given by columns, or `None` when `X` has rank below its
/// columns.  The first right-hand side takes exactly the operations it took
/// alone: each is reflected and back-substituted on its own.
///
/// Householder QR with column pivoting (each step takes the remaining
/// column of largest norm), so `|R_kk|` falls along the diagonal and its
/// first entry is the largest, over rows sorted by decreasing largest entry.
/// Without the sort, a cell of n = 3 beside one of n = 1e9 loses about
/// 5e-12 of a Falconer A; with it the error stays near 1e-15 of the
/// correlations' scale.  Full rank needs every `|R_kk|` above
/// `m * eps * |R_11|` (`m >= p` rows): Householder QR is backward stable, the
/// computed `R` being the exact factor of `X + dX` with `|dX|` of order
/// `m * eps * |X|`, so a diagonal below that bound is
/// indistinguishable from an exact zero (the bound numpy's `matrix_rank`
/// puts on singular values).  The design's entries are `sqrt(n)` times `2^-d`
/// (`d <= 5`), 0 or 1, so an identified design sits orders of magnitude
/// above the bound unless its weights differ by a factor near `1e20`.
pub(super) fn least_squares(columns: Vec<Vec<f64>>, rhs: Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let (m, p) = (rhs.first().map_or(0, Vec::len), columns.len());
    if m < p {
        return None;
    }
    let largest = |i: usize| columns.iter().map(|c| c[i].abs()).fold(0.0, f64::max);
    let mut by_size: Vec<usize> = (0..m).collect();
    by_size.sort_by(|&i, &j| largest(j).total_cmp(&largest(i)));
    let sorted = |c: &Vec<f64>| -> Vec<f64> { by_size.iter().map(|&i| c[i]).collect() };
    let mut columns: Vec<Vec<f64>> = columns.iter().map(sorted).collect();
    let mut rhs: Vec<Vec<f64>> = rhs.iter().map(sorted).collect();
    let mut order: Vec<usize> = (0..p).collect();
    let mut diagonal = Vec::with_capacity(p);
    for k in 0..p {
        let norms: Vec<f64> = columns[k..]
            .iter()
            .map(|c| c[k..].iter().map(|x| x * x).sum())
            .collect();
        let pivot = (0..norms.len()).max_by(|&i, &j| norms[i].total_cmp(&norms[j]))? + k;
        columns.swap(k, pivot);
        order.swap(k, pivot);
        let alpha = norms[pivot - k].sqrt();
        if alpha == 0.0 {
            return None;
        }
        let r_kk = if columns[k][k] >= 0.0 { -alpha } else { alpha };
        let mut v = columns[k][k..].to_vec();
        v[0] -= r_kk;
        let vv: f64 = v.iter().map(|x| x * x).sum();
        let reflect = |target: &mut [f64]| {
            let f = 2.0 * v.iter().zip(&*target).map(|(a, b)| a * b).sum::<f64>() / vv;
            target.iter_mut().zip(&v).for_each(|(t, a)| *t -= f * a);
        };
        for column in &mut columns[k + 1..] {
            reflect(&mut column[k..]);
        }
        for y in &mut rhs {
            reflect(&mut y[k..]);
        }
        columns[k][k] = r_kk;
        diagonal.push(r_kk);
    }
    let tolerance = m as f64 * f64::EPSILON * diagonal.first().map_or(0.0, |d| d.abs());
    if diagonal.iter().any(|d| d.abs() <= tolerance) {
        return None;
    }
    let solve = |y: &Vec<f64>| {
        let mut b = vec![0.0; p];
        for k in (0..p).rev() {
            let tail: f64 = (k + 1..p).map(|j| columns[j][k] * b[j]).sum();
            b[k] = (y[k] - tail) / columns[k][k];
        }
        let mut solution = vec![0.0; p];
        for (k, &column) in order.iter().enumerate() {
            solution[column] = b[k];
        }
        solution
    };
    Some(rhs.iter().map(solve).collect())
}

/// The lower Cholesky factor `L` (`a = L Lᵀ`) of the `m x m` symmetric
/// `a`, row-major, read from its lower triangle; `None` when a pivot falls
/// to `m * eps * max(diag)` or below (or is NaN), so `a` is not numerically
/// positive definite.
pub(super) fn cholesky(a: &[f64], m: usize) -> Option<Vec<f64>> {
    let largest = (0..m).map(|i| a[i * m + i]).fold(0.0, f64::max);
    let tolerance = m as f64 * f64::EPSILON * largest;
    let mut l = vec![0.0; m * m];
    for j in 0..m {
        let pivot = a[j * m + j] - (0..j).map(|k| l[j * m + k] * l[j * m + k]).sum::<f64>();
        if pivot.is_nan() || pivot <= tolerance {
            return None;
        }
        let l_jj = pivot.sqrt();
        l[j * m + j] = l_jj;
        for i in j + 1..m {
            let dot: f64 = (0..j).map(|k| l[i * m + k] * l[j * m + k]).sum();
            l[i * m + j] = (a[i * m + j] - dot) / l_jj;
        }
    }
    Some(l)
}

/// `L⁻¹` of the `m x m` lower-triangular `L` with a nonzero diagonal, by
/// forward substitution, row-major and lower-triangular.
pub(super) fn lower_inverse(l: &[f64], m: usize) -> Vec<f64> {
    let mut inverse = vec![0.0; m * m];
    for j in 0..m {
        inverse[j * m + j] = 1.0 / l[j * m + j];
        for i in j + 1..m {
            let dot: f64 = (j..i).map(|k| l[i * m + k] * inverse[k * m + j]).sum();
            inverse[i * m + j] = -dot / l[i * m + i];
        }
    }
    inverse
}

#[cfg(test)]
mod tests {
    use super::*;

    fn product(a: &[f64], b: &[f64], m: usize) -> Vec<f64> {
        let mut out = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..m {
                out[i * m + j] = (0..m).map(|k| a[i * m + k] * b[k * m + j]).sum();
            }
        }
        out
    }

    fn transpose(a: &[f64], m: usize) -> Vec<f64> {
        (0..m * m).map(|x| a[(x % m) * m + x / m]).collect()
    }

    fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() <= tolerance, "{actual:?} vs {expected:?}");
        }
    }

    #[test]
    fn cholesky_factors_and_inverts_a_positive_definite_matrix() {
        let a = [4.0, 2.0, 0.6, 2.0, 5.0, 1.0, 0.6, 1.0, 3.0];
        let l = cholesky(&a, 3).unwrap();
        assert_eq!([l[1], l[2], l[5]], [0.0; 3]);
        assert_close(&product(&l, &transpose(&l, 3), 3), &a, 1e-14);
        let inverse = lower_inverse(&l, 3);
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_close(&product(&inverse, &l, 3), &identity, 1e-15);
        assert_eq!(cholesky(&[], 0), Some(vec![]));
    }

    #[test]
    fn cholesky_refuses_a_matrix_that_is_not_positive_definite() {
        assert_eq!(cholesky(&[1.0, 2.0, 2.0, 1.0], 2), None);
        assert_eq!(cholesky(&[1.0, 1.0, 1.0, 1.0], 2), None);
        assert_eq!(cholesky(&[f64::NAN, 0.0, 0.0, 1.0], 2), None);
        assert_eq!(cholesky(&[-1.0], 1), None);
    }

    #[test]
    fn each_right_hand_side_is_its_own_least_squares_solution() {
        let columns = vec![vec![1.0, 1.0, 1.0, 1.0], vec![0.0, 1.0, 2.0, 3.5]];
        let rhs = vec![vec![0.1, 0.9, 2.2, 3.1], vec![1.0, 0.0, 0.0, 0.0]];
        let both = least_squares(columns.clone(), rhs.clone()).unwrap();
        let first = least_squares(columns.clone(), vec![rhs[0].clone()]).unwrap();
        let second = least_squares(columns, vec![rhs[1].clone()]).unwrap();
        assert_eq!(both, [first[0].clone(), second[0].clone()]);
        assert_eq!(least_squares(vec![vec![1.0, 2.0, 3.0, 4.0]; 2], rhs), None);
    }
}
