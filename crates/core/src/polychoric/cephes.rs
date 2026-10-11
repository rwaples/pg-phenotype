//! The SciPy special functions pedsum's NumPy layer calls: `ndtr`, and
//! `owens_t` with the cephes `erf`, `erfc` and `expm1` it uses.
//!
//! Ported line by line from xsf at f7b85f505a94fd398024478d2e747b01dd2d7a6d
//! (the submodule SciPy 1.18.1 vendors): `cephes/ndtr.h`, `cephes/unity.h`,
//! `cephes/polevl.h` and `cephes/owens_t.h` (Patefield & Tandy 2000, J Stat
//! Softw 5(5), Boost-licensed translation by Benjamin Sobotta).  The numba
//! kernels' `0.5 * erfc(-x / sqrt 2)` is [`kernel_ndtr`], on the platform
//! `erfc` numba calls.

use std::f64::consts::{FRAC_1_SQRT_2, PI};

const MAXLOG: f64 = 7.097_827_128_933_839_730_962_063_185_871E2;

fn polevl(x: f64, coef: &[f64]) -> f64 {
    coef[1..].iter().fold(coef[0], |ans, &c| ans * x + c)
}

fn p1evl(x: f64, coef: &[f64]) -> f64 {
    coef[1..].iter().fold(x + coef[0], |ans, &c| ans * x + c)
}

#[allow(clippy::excessive_precision)]
const NDTR_P: [f64; 9] = [
    2.46196981473530512524E-10,
    5.64189564831068821977E-1,
    7.46321056442269912687E0,
    4.86371970985681366614E1,
    1.96520832956077098242E2,
    5.26445194995477358631E2,
    9.34528527171957607540E2,
    1.02755188689515710272E3,
    5.57535335369399327526E2,
];
#[allow(clippy::excessive_precision)]
const NDTR_Q: [f64; 8] = [
    1.32281951154744992508E1,
    8.67072140885989742329E1,
    3.54937778887819891062E2,
    9.75708501743205489753E2,
    1.82390916687909736289E3,
    2.24633760818710981792E3,
    1.65666309194161350182E3,
    5.57535340817727675546E2,
];
#[allow(clippy::excessive_precision)]
const NDTR_R: [f64; 6] = [
    5.64189583547755073984E-1,
    1.27536670759978104416E0,
    5.01905042251180477414E0,
    6.16021097993053585195E0,
    7.40974269950448939160E0,
    2.97886665372100240670E0,
];
#[allow(clippy::excessive_precision)]
const NDTR_S: [f64; 6] = [
    2.26052863220117276590E0,
    9.39603524938001434673E0,
    1.20489539808096656605E1,
    1.70814450747565897222E1,
    9.60896809063285878198E0,
    3.36907645100081516050E0,
];
#[allow(clippy::excessive_precision)]
const NDTR_T: [f64; 5] = [
    9.60497373987051638749E0,
    9.00260197203842689217E1,
    2.23200534594684319226E3,
    7.00332514112805075473E3,
    5.55923013010394962768E4,
];
#[allow(clippy::excessive_precision)]
const NDTR_U: [f64; 5] = [
    3.35617141647503099647E1,
    5.21357949780152679795E2,
    4.59432382970980127987E3,
    2.26290000613890934246E4,
    4.92673942608635921086E4,
];

/// cephes `erfc`.
pub(crate) fn erfc(a: f64) -> f64 {
    if a.is_nan() {
        return f64::NAN;
    }
    let x = a.abs();
    if x < 1.0 {
        return 1.0 - erf(a);
    }
    let z = -a * a;
    if z >= -MAXLOG {
        let z = z.exp();
        let (p, q) = if x < 8.0 {
            (polevl(x, &NDTR_P), p1evl(x, &NDTR_Q))
        } else {
            (polevl(x, &NDTR_R), p1evl(x, &NDTR_S))
        };
        let mut y = (z * p) / q;
        if a < 0.0 {
            y = 2.0 - y;
        }
        if y != 0.0 {
            return y;
        }
    }
    if a < 0.0 {
        2.0
    } else {
        0.0
    }
}

/// cephes `erf`.
pub(crate) fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return -erf(-x);
    }
    if x.abs() > 1.0 {
        return 1.0 - erfc(x);
    }
    let z = x * x;
    x * polevl(z, &NDTR_T) / p1evl(z, &NDTR_U)
}

/// SciPy `ndtr`: P(Z <= a).
pub(crate) fn ndtr(a: f64) -> f64 {
    if a.is_nan() {
        return f64::NAN;
    }
    let x = a * FRAC_1_SQRT_2;
    let z = x.abs();
    if z < 1.0 {
        0.5 + 0.5 * erf(x)
    } else {
        let y = 0.5 * erfc(z);
        if x > 0.0 {
            1.0 - y
        } else {
            y
        }
    }
}

/// pedsum's numba `ndtr`: `0.5 * erfc(-x / sqrt 2)` on the C library's `erfc`.
#[inline]
pub(crate) fn kernel_ndtr(x: f64) -> f64 {
    0.5 * c_erfc(-x / std::f64::consts::SQRT_2)
}

#[allow(clippy::excessive_precision)]
const ERX: f64 = 8.45062911510467529297e-01;
#[allow(clippy::excessive_precision)]
const PP: [f64; 5] = [
    1.28379167095512558561e-01,
    -3.25042107247001499370e-01,
    -2.84817495755985104766e-02,
    -5.77027029648944159157e-03,
    -2.37630166566501626084e-05,
];
#[allow(clippy::excessive_precision)]
const QQ: [f64; 6] = [
    0.0,
    3.97917223959155352819e-01,
    6.50222499887672944485e-02,
    5.08130628187576562776e-03,
    1.32494738004321644526e-04,
    -3.96022827877536812320e-06,
];
#[allow(clippy::excessive_precision)]
const PA: [f64; 7] = [
    -2.36211856075265944077e-03,
    4.14856118683748331666e-01,
    -3.72207876035701323847e-01,
    3.18346619901161753674e-01,
    -1.10894694282396677476e-01,
    3.54783043256182359371e-02,
    -2.16637559486879084300e-03,
];
#[allow(clippy::excessive_precision)]
const QA: [f64; 7] = [
    0.0,
    1.06420880400844228286e-01,
    5.40397917702171048937e-01,
    7.18286544141962662868e-02,
    1.26171219808761642112e-01,
    1.36370839120290507362e-02,
    1.19844998467991074170e-02,
];
#[allow(clippy::excessive_precision)]
const RA: [f64; 8] = [
    -9.86494403484714822705e-03,
    -6.93858572707181764372e-01,
    -1.05586262253232909814e+01,
    -6.23753324503260060396e+01,
    -1.62396669462573470355e+02,
    -1.84605092906711035994e+02,
    -8.12874355063065934246e+01,
    -9.81432934416914548592e+00,
];
#[allow(clippy::excessive_precision)]
const SA: [f64; 9] = [
    0.0,
    1.96512716674392571292e+01,
    1.37657754143519042600e+02,
    4.34565877475229228821e+02,
    6.45387271733267880336e+02,
    4.29008140027567833386e+02,
    1.08635005541779435134e+02,
    6.57024977031928170135e+00,
    -6.04244152148580987438e-02,
];
#[allow(clippy::excessive_precision)]
const RB: [f64; 7] = [
    -9.86494292470009928597e-03,
    -7.99283237680523006574e-01,
    -1.77579549177547519889e+01,
    -1.60636384855821916062e+02,
    -6.37566443368389627722e+02,
    -1.02509513161107724954e+03,
    -4.83519191608651397019e+02,
];
#[allow(clippy::excessive_precision)]
const SB: [f64; 8] = [
    0.0,
    3.03380607434824582924e+01,
    3.25792512996573918826e+02,
    1.53672958608443695994e+03,
    3.19985821950859553908e+03,
    2.55305040643316442583e+03,
    4.74528541206955367215e+02,
    -2.24409524465858183362e+01,
];

/// The C library's `erfc`: glibc 2.39 `sysdeps/ieee754/dbl-64/s_erf.c`
/// (Sun fdlibm, evaluated in glibc's split form), on the C library's `exp`.
pub(crate) fn c_erfc(x: f64) -> f64 {
    let hx = (x.to_bits() >> 32) as u32 as i32;
    let ix = hx & 0x7fff_ffff;
    if ix >= 0x7ff0_0000 {
        return f64::from(((hx as u32) >> 31) << 1) + 1.0 / x;
    }
    if ix < 0x3feb_0000 {
        if ix < 0x3c70_0000 {
            return 1.0 - x;
        }
        let z = x * x;
        let r1 = PP[0] + z * PP[1];
        let z2 = z * z;
        let r2 = PP[2] + z * PP[3];
        let z4 = z2 * z2;
        let s1 = 1.0 + z * QQ[1];
        let s2 = QQ[2] + z * QQ[3];
        let s3 = QQ[4] + z * QQ[5];
        let r = r1 + z2 * r2 + z4 * PP[4];
        let s = s1 + z2 * s2 + z4 * s3;
        let y = r / s;
        if hx < 0x3fd0_0000 {
            return 1.0 - (x + x * y);
        }
        let r = x * y + (x - 0.5);
        return 0.5 - r;
    }
    if ix < 0x3ff4_0000 {
        let s = x.abs() - 1.0;
        let p1 = PA[0] + s * PA[1];
        let s2 = s * s;
        let q1 = 1.0 + s * QA[1];
        let s4 = s2 * s2;
        let p2 = PA[2] + s * PA[3];
        let s6 = s4 * s2;
        let q2 = QA[2] + s * QA[3];
        let p3 = PA[4] + s * PA[5];
        let q3 = QA[4] + s * QA[5];
        let p = p1 + s2 * p2 + s4 * p3 + s6 * PA[6];
        let q = q1 + s2 * q2 + s4 * q3 + s6 * QA[6];
        return if hx >= 0 {
            (1.0 - ERX) - p / q
        } else {
            1.0 + (ERX + p / q)
        };
    }
    if ix >= 0x403c_0000 {
        return if hx > 0 { 0.0 } else { 2.0 };
    }
    let xa = x.abs();
    let s = 1.0 / (xa * xa);
    let (r, big_s);
    if ix < 0x4006_DB6D {
        let r1 = RA[0] + s * RA[1];
        let s2 = s * s;
        let s1 = 1.0 + s * SA[1];
        let s4 = s2 * s2;
        let r2 = RA[2] + s * RA[3];
        let s6 = s4 * s2;
        let ss2 = SA[2] + s * SA[3];
        let s8 = s4 * s4;
        let r3 = RA[4] + s * RA[5];
        let ss3 = SA[4] + s * SA[5];
        let r4 = RA[6] + s * RA[7];
        let ss4 = SA[6] + s * SA[7];
        r = r1 + s2 * r2 + s4 * r3 + s6 * r4;
        big_s = s1 + s2 * ss2 + s4 * ss3 + s6 * ss4 + s8 * SA[8];
    } else {
        if hx < 0 && ix >= 0x4018_0000 {
            return 2.0;
        }
        let r1 = RB[0] + s * RB[1];
        let s2 = s * s;
        let s1 = 1.0 + s * SB[1];
        let s4 = s2 * s2;
        let r2 = RB[2] + s * RB[3];
        let s6 = s4 * s2;
        let ss2 = SB[2] + s * SB[3];
        let r3 = RB[4] + s * RB[5];
        let ss3 = SB[4] + s * SB[5];
        let ss4 = SB[6] + s * SB[7];
        r = r1 + s2 * r2 + s4 * r3 + s6 * RB[6];
        big_s = s1 + s2 * ss2 + s4 * ss3 + s6 * ss4;
    }
    let z = f64::from_bits(xa.to_bits() & 0xffff_ffff_0000_0000);
    let e = (-z * z - 0.5625).exp() * ((z - xa) * (z + xa) + r / big_s).exp();
    if hx > 0 {
        e / xa
    } else {
        2.0 - e / xa
    }
}

#[allow(clippy::excessive_precision)]
const UNITY_EP: [f64; 3] = [
    1.2617719307481059087798E-4,
    3.0299440770744196129956E-2,
    9.9999999999999999991025E-1,
];
#[allow(clippy::excessive_precision)]
const UNITY_EQ: [f64; 4] = [
    3.0019850513866445504159E-6,
    2.5244834034968410419224E-3,
    2.2726554820815502876593E-1,
    2.0000000000000000000897E0,
];

/// cephes `expm1`.
fn expm1(x: f64) -> f64 {
    if !x.is_finite() {
        if x.is_nan() || x > 0.0 {
            return x;
        }
        return -1.0;
    }
    if !(-0.5..=0.5).contains(&x) {
        return x.exp() - 1.0;
    }
    let xx = x * x;
    let r = x * polevl(xx, &UNITY_EP);
    let r = r / (polevl(xx, &UNITY_EQ) - r);
    r + r
}

const SELECT_METHOD: [usize; 120] = [
    0, 0, 1, 12, 12, 12, 12, 12, 12, 12, 12, 15, 15, 15, 8, 0, 1, 1, 2, 2, 4, 4, 13, 13, 14, 14,
    15, 15, 15, 8, 1, 1, 2, 2, 2, 4, 4, 14, 14, 14, 14, 15, 15, 15, 9, 1, 1, 2, 4, 4, 4, 4, 6, 6,
    15, 15, 15, 15, 15, 9, 1, 2, 2, 4, 4, 5, 5, 7, 7, 16, 16, 16, 11, 11, 10, 1, 2, 4, 4, 4, 5, 5,
    7, 7, 16, 16, 16, 11, 11, 11, 1, 2, 3, 3, 5, 5, 7, 7, 16, 16, 16, 16, 16, 11, 11, 1, 2, 3, 3,
    5, 5, 17, 17, 17, 17, 16, 16, 16, 11, 11,
];
const HRANGE: [f64; 14] = [
    0.02, 0.06, 0.09, 0.125, 0.26, 0.4, 0.6, 1.6, 1.7, 2.33, 2.4, 3.36, 3.4, 4.8,
];
const ARANGE: [f64; 7] = [0.025, 0.09, 0.15, 0.36, 0.5, 0.9, 0.99999];
const ORD: [f64; 18] = [
    2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 12.0, 18.0, 10.0, 20.0, 30.0, 0.0, 4.0, 7.0, 8.0, 20.0, 0.0, 0.0,
];
const METHODS: [u8; 18] = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6];
#[allow(clippy::excessive_precision)]
const C: [f64; 31] = [
    1.0,
    -1.0,
    1.0,
    -0.9999999999999998,
    0.9999999999999839,
    -0.9999999999993063,
    0.9999999999797337,
    -0.9999999995749584,
    0.9999999933226235,
    -0.9999999188923242,
    0.9999992195143483,
    -0.9999939351372067,
    0.9999613559769055,
    -0.9997955636651394,
    0.9990927896296171,
    -0.9965938374119182,
    0.9891001713838613,
    -0.9700785580406933,
    0.9291143868326319,
    -0.8542058695956156,
    0.737965260330301,
    -0.585234698828374,
    0.4159977761456763,
    -0.25882108752419436,
    0.13755358251638927,
    -0.060795276632595575,
    0.021633768329987153,
    -0.005934056934551867,
    0.0011743414818332946,
    -0.0001489155613350369,
    9.072354320794358e-06,
];
#[allow(clippy::excessive_precision)]
const PTS: [f64; 13] = [
    0.35082039676451715489E-02,
    0.31279042338030753740E-01,
    0.85266826283219451090E-01,
    0.16245071730812277011E+00,
    0.25851196049125434828E+00,
    0.36807553840697533536E+00,
    0.48501092905604697475E+00,
    0.60277514152618576821E+00,
    0.71477884217753226516E+00,
    0.81475510988760098605E+00,
    0.89711029755948965867E+00,
    0.95723808085944261843E+00,
    0.99178832974629703586E+00,
];
#[allow(clippy::excessive_precision)]
const WTS: [f64; 13] = [
    0.18831438115323502887E-01,
    0.18567086243977649478E-01,
    0.18042093461223385584E-01,
    0.17263829606398753364E-01,
    0.16243219975989856730E-01,
    0.14994592034116704829E-01,
    0.13535474469662088392E-01,
    0.11886351605820165233E-01,
    0.10070377242777431897E-01,
    0.81130545742299586629E-02,
    0.60419009528470238773E-02,
    0.38862217010742057883E-02,
    0.16793031084546090448E-02,
];

fn get_method(h: f64, a: f64) -> usize {
    let ihint = HRANGE.iter().position(|&r| h <= r).unwrap_or(14);
    let iaint = ARANGE.iter().position(|&r| a <= r).unwrap_or(7);
    SELECT_METHOD[iaint * 15 + ihint]
}

fn norm1(x: f64) -> f64 {
    erf(x / 2.0_f64.sqrt()) / 2.0
}

fn norm2(x: f64) -> f64 {
    erfc(x / 2.0_f64.sqrt()) / 2.0
}

fn t1(h: f64, a: f64, m: f64) -> f64 {
    let mut j = 1.0;
    let mut jj = 1.0;
    let hs = -0.5 * h * h;
    let dhs = hs.exp();
    let as_ = a * a;
    let mut aj = a / (2.0 * PI);
    let mut dj = expm1(hs);
    let mut gj = hs * dhs;
    let mut val = a.atan() / (2.0 * PI);
    loop {
        val += dj * aj / jj;
        if m <= j {
            break;
        }
        j += 1.0;
        jj += 2.0;
        aj *= as_;
        dj = gj - dj;
        gj *= hs / j;
    }
    val
}

fn t2(h: f64, a: f64, ah: f64, m: f64) -> f64 {
    let mut i = 1.0;
    // C++ `int maxi = 2 * m + 1` truncates; every ORD entry is whole.
    let maxi = 2.0 * m + 1.0;
    let hs = h * h;
    let as_ = -a * a;
    let y = 1.0 / hs;
    let mut val = 0.0;
    let mut vi = a * (-0.5 * ah * ah).exp() / (2.0 * PI).sqrt();
    let mut z = (ndtr(ah) - 0.5) / h;
    loop {
        val += z;
        if maxi <= i {
            break;
        }
        z = y * (vi - i * z);
        vi *= as_;
        i += 2.0;
    }
    // C++ `val *= exp(..) / sqrt(2 pi)` divides first.
    val * ((-0.5 * hs).exp() / (2.0 * PI).sqrt())
}

fn t3(h: f64, a: f64, ah: f64) -> f64 {
    let aa = a * a;
    let hh = h * h;
    let y = 1.0 / hh;
    let mut vi = a * (-ah * ah / 2.0).exp() / (2.0 * PI).sqrt();
    let mut zi = norm1(ah) / h;
    let mut result = 0.0;
    for (i, &c) in C.iter().enumerate() {
        result += zi * c;
        zi = y * ((2 * i + 1) as f64 * zi - vi);
        vi *= aa;
    }
    result * ((-hh / 2.0).exp() / (2.0 * PI).sqrt())
}

fn t4(h: f64, a: f64, m: f64) -> f64 {
    let maxi = 2.0 * m + 1.0;
    let hh = h * h;
    let naa = -a * a;
    let mut i = 1.0;
    let mut ai = a * (-hh * (1.0 - naa) / 2.0).exp() / (2.0 * PI);
    let mut yi = 1.0;
    let mut result = 0.0;
    loop {
        result += ai * yi;
        if maxi <= i {
            break;
        }
        i += 2.0;
        yi = (1.0 - hh * yi) / i;
        ai *= naa;
    }
    result
}

fn t5(h: f64, a: f64) -> f64 {
    let aa = a * a;
    let nhh = -0.5 * h * h;
    let mut result = 0.0;
    for (&pt, &wt) in PTS.iter().zip(&WTS) {
        let r = 1.0 + aa * pt;
        result += wt * (nhh * r).exp() / r;
    }
    result * a
}

fn t6(h: f64, a: f64) -> f64 {
    let normh = norm2(h);
    let y = 1.0 - a;
    let r = y.atan2(1.0 + a);
    let mut result = normh * (1.0 - normh) / 2.0;
    if r != 0.0 {
        result -= r * (-y * h * h / (2.0 * r)).exp() / (2.0 * PI);
    }
    result
}

fn dispatch(h: f64, a: f64, ah: f64) -> f64 {
    if h == 0.0 {
        return a.atan() / (2.0 * PI);
    }
    if a == 0.0 {
        return 0.0;
    }
    if a == 1.0 {
        return norm2(-h) * norm2(h) / 2.0;
    }
    let index = get_method(h, a);
    let m = ORD[index];
    match METHODS[index] {
        1 => t1(h, a, m),
        2 => t2(h, a, ah, m),
        3 => t3(h, a, ah),
        4 => t4(h, a, m),
        5 => t5(h, a),
        _ => t6(h, a),
    }
}

/// Owen's T function `T(h, a)` (SciPy `owens_t`).
pub(crate) fn owens_t(h: f64, a: f64) -> f64 {
    if h.is_nan() || a.is_nan() {
        return f64::NAN;
    }
    let h = h.abs();
    let fabs_a = a.abs();
    let fabs_ah = fabs_a * h;
    let result = if fabs_a == f64::INFINITY {
        0.5 * norm2(h)
    } else if h == f64::INFINITY {
        0.0
    } else if fabs_a <= 1.0 {
        dispatch(h, fabs_a, fabs_ah)
    } else if fabs_ah <= 0.67 {
        let normh = norm1(h);
        let normah = norm1(fabs_ah);
        0.25 - normh * normah - dispatch(fabs_ah, 1.0 / fabs_a, h)
    } else {
        let normh = norm2(h);
        let normah = norm2(fabs_ah);
        (normh + normah) / 2.0 - normh * normah - dispatch(fabs_ah, 1.0 / fabs_a, h)
    };
    if a < 0.0 {
        -result
    } else {
        result
    }
}
