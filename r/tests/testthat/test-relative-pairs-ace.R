# relative_pairs_ace(): the same Rust core as pg_phenotype.relative_pairs.ace,
# so values agree bit for bit.

ped <- read_fixture("pedigree_small.csv")
# Python: a familial pair on these rows, rounded to 0.1.
x <- c(-0.4, 0.9, -1.4, 1.1, -0.2, -1.9, 1.3, 1.1, -1.9, 0.8, -0.1, -1.1, 0.0, 0.0, -0.1, 0.7, -0.4, -0.6, -0.6, 1.5, -0.7, 0.0, -0.5, -0.6, -0.7, -0.1, 1.6, 2.4, -1.8, 0.7, 1.7, 0.5, 1.1, -0.9, 0.0, 1.4, 1.4, -1.8, -0.7, 1.1, 0.4, 1.7, -1.1, -0.4, 0.6, -1.4, 0.4, 1.6, -1.6, -1.2, -0.6, 0.4, -1.1, 0.4, 1.4, 0.8, -1.2, 0.7, 0.3, 2.6, -0.2, 0.0, -0.4, 1.2, -0.5, 2.1, -1.4, 0.5, -0.7)
y <- c(-1.0, 0.6, 0.7, 1.2, -0.4, 0.0, -0.4, 0.3, -0.5, 2.2, 2.4, -0.7, 0.0, -1.8, -0.9, 0.9, 1.5, -2.5, 0.7, 1.6, 0.9, 0.5, -1.0, 0.4, 0.9, 1.6, 1.3, 1.6, -0.2, 0.4, 1.4, 1.2, 0.0, -0.6, 1.0, -1.9, 1.7, -3.8, 0.7, 2.2, 1.0, 2.5, -1.8, -2.5, 1.8, -1.1, -1.4, 1.4, -2.3, -1.6, -0.4, -0.8, -1.6, 2.7, 2.5, -0.5, -2.0, 2.7, -1.4, 2.8, -0.7, -1.2, -1.0, 1.2, 1.6, 1.5, -1.8, 1.4, -0.1)
cats <- c("MZ", "MO", "FO", "FS", "MHS", "PHS", "GP", "Av", "1C")
shared <- c("MZ", "FS", "MHS")
two <- relative_pairs_correlation(ped, list(trait(x), trait(y)), categories = cats)

test_that("every share and cross value is Python's bit for bit", {
  # Python: ace(correlation(ped, [Trait(x), Trait(y)], categories=cats), shared_environment=shared).
  fit <- relative_pairs_ace(two, shared_environment = shared)
  expect_s3_class(fit, "pgphenotype_ace")
  shares <- function(t) c(t$A, t$C, t$E)
  expect_identical(shares(fit$traits[[1L]]), as.numeric(c(
    "0x1.c6aedb2e90cb0p-3", "0x1.3c7efa0654752p-4", "0x1.66c469f3913eap-1"
  )))
  expect_identical(shares(fit$traits[[2L]]), as.numeric(c(
    "0x1.64ec128c9152cp-2", "0x1.a6d0bf56be4ddp-3", "0x1.c7ab8dc80f866p-2"
  )))
  cross <- vapply(c("sA", "sC", "sE", "rA", "rC", "rE"), function(k) fit$cross[[k]], 0, USE.NAMES = FALSE)
  expect_identical(cross, as.numeric(c(
    "0x1.40173e982888ap-3", "0x1.905739ada3788p-4", "0x1.5397c952d6fdfp-2",
    "0x1.1faa3d7026b4fp-1", "0x1.8c35b542125ebp-1", "0x1.30136a239f555p-1"
  )))
  expect_false(fit$cross$rA_out_of_range)
  expect_identical(vapply(fit$traits, `[[`, 0L, "trait"), 1:2)
  expect_length(fit$rows, 32L)
  last <- fit$rows[[32L]]
  expect_identical(c(last$entry, last$first_trait, last$second_trait), c("within_person", "1", "2"))
  expect_true(last$within_person)
  expect_null(last$kinship)
  mo <- fit$rows[[6L]]
  expect_identical(c(mo$first_trait, mo$second_trait), c(2L, 1L))
  expect_identical(mo$value, two$cells[[6L]]$crude[[1L]]$value)
  expect_identical(fit$settings$estimate, "pooled")
  expect_identical(fit$metadata$pedigree_graph_core_rev, pg_core_rev())
})

test_that("the recipes are their closed forms", {
  one <- relative_pairs_correlation(ped, trait(x), categories = cats)
  r <- function(code) Filter(function(cell) cell$category == code, one$cells)[[1L]]$crude[[1L]]$value
  sibs <- relative_pairs_ace(one, categories = "FS", components = "A")$traits[[1L]]
  expect_equal(sibs$A, 2 * r("FS"), tolerance = 1e-15)
  expect_null(sibs$C)
  expect_identical(sibs$C_reason, "not_requested")
  falconer <- relative_pairs_ace(one, categories = c("MZ", "FS"), shared_environment = c("MZ", "FS"))$traits[[1L]]
  expect_equal(falconer$A, 2 * (r("MZ") - r("FS")), tolerance = 1e-14)
  expect_equal(falconer$C, 2 * r("FS") - r("MZ"), tolerance = 1e-14)
  expect_null(relative_pairs_ace(one, components = "A")$cross)
})

test_that("the estimate follows the result unless given", {
  s <- relative_pairs_correlation(ped, list(trait(x), trait(y)), categories = c("FS", "MO"),
                                  stratum = rep(c(0, 1, 2), length.out = nrow(ped)))
  fit <- relative_pairs_ace(s, components = "A")
  expect_identical(fit$settings$estimate, "stratified")
  expect_identical(fit$rows[[1L]]$value, s$cells[[1L]]$stratified$estimate$value)
  expect_identical(fit$rows[[1L]]$weight, s$cells[[1L]]$stratified$n)
  # FS's cross cell weighs its complete orientations, MO's its pairs.
  expect_identical(c(s$cells[[3L]]$first_trait, s$cells[[3L]]$second_trait), 1:2)
  expect_identical(fit$rows[[3L]]$weight, s$cells[[3L]]$stratified$n_entries)
  expect_identical(fit$rows[[5L]]$weight, s$cells[[5L]]$stratified$n)
  pooled <- relative_pairs_ace(s, components = "A", estimate = "pooled")
  expect_identical(pooled$rows[[1L]]$value, s$cells[[1L]]$crude[[1L]]$value)
  expect_identical(pooled$rows[[3L]]$weight, s$cells[[3L]]$n_entries)
  expect_true(s$cells[[3L]]$n_entries > s$cells[[3L]]$n)
  expect_identical(pooled$rows[[5L]]$weight, s$cells[[5L]]$n)
  expect_pgp_error(relative_pairs_ace(two, components = "A", estimate = "stratified"), "usage", "no_stratified_form")
})

test_that("phi reads each binary cell's phi, pooled only", {
  b <- relative_pairs_correlation(ped, list(trait(as.numeric(x > 0), kind = "binary"),
                                            trait(as.numeric(y > 0), kind = "binary")), categories = cats)
  fit <- relative_pairs_ace(b, components = "A", estimator = "phi", estimate = "pooled")
  phi <- vapply(c(b$cells, list(b$within_person)), function(cell) {
    Filter(function(e) e$estimator == "phi", cell$crude)[[1L]]$value
  }, 0)
  expect_identical(vapply(fit$rows, `[[`, 0, "value"), phi)
  b$cells <- lapply(b$cells, function(cell) {
    cell$stratified <- list(n = 1L, estimate = list(value = 0.1, reason = NULL))
    cell
  })
  expect_pgp_error(relative_pairs_ace(b, components = "A", estimator = "phi"), "usage", "no_stratified_form")
})

test_that("arguments are checked", {
  usage <- function(expr) expect_pgp_error(expr, "usage")
  usage(relative_pairs_ace(two$cells, components = "A"))
  usage(relative_pairs_ace(two, components = character()))
  usage(relative_pairs_ace(two, components = "E"))
  usage(relative_pairs_ace(two, components = c("A", "A")))
  usage(relative_pairs_ace(two, components = 1))
  usage(relative_pairs_ace(two, components = "A", estimator = "tetrachoric"))
  usage(relative_pairs_ace(two, components = "A", estimate = "both"))
  usage(relative_pairs_ace(two, components = "A", estimate = c("pooled", "pooled")))
  usage(relative_pairs_ace(two, components = "A", categories = 1))
  usage(relative_pairs_ace(two, shared_environment = NA_character_))
  expect_pgp_error(relative_pairs_ace(two), "usage", "missing_shared_environment")
  err <- expect_pgp_error(relative_pairs_ace(two, shared_environment = c("MZ", "XX")), "validation", "unknown_category")
  expect_identical(err$fields$position, 2)
  err <- expect_pgp_error(relative_pairs_ace(two, categories = c("FS", "sibs"), components = "A"),
                          "validation", "unknown_entry")
  expect_identical(err$fields$position, 2)
  expect_pgp_error(relative_pairs_ace(two, components = "A", estimator = "phi"), "usage", "missing_estimator")
  g <- relative_pairs_correlation(ped, trait(x), categories = list(HS = c("MHS", "PHS")))
  expect_pgp_error(relative_pairs_ace(g, shared_environment = "MHS"), "validation", "mixed_shared_environment")
})

test_that("the fit prints", {
  expect_output(print(relative_pairs_ace(two, shared_environment = shared)), "pgphenotype_ace")
  expect_output(print(relative_pairs_ace(two, components = "A")), "C undefined: not_requested")
})

two_cov <- relative_pairs_correlation(ped, list(trait(x), trait(y)), categories = cats, se = TRUE, covariance = TRUE)
ses <- function(fit) {
  c(unlist(lapply(fit$traits, function(t) c(t$A_se, t$C_se, t$E_se))),
    vapply(c("sA", "sC", "sE", "rA", "rC", "rE"), function(k) fit$cross[[paste0(k, "_se")]] %||% NA_real_, 0,
           USE.NAMES = FALSE))
}

test_that("with the covariance every SE is Python's bit for bit", {
  # Python: ace(correlation(ped, [Trait(x), Trait(y)], categories=cats, se=True, covariance=True),
  # shared_environment=shared, weights=weights).
  pairs <- relative_pairs_ace(two_cov, shared_environment = shared, weights = "pairs")
  expect_identical(pairs$traits[[1L]]$A, relative_pairs_ace(two, shared_environment = shared)$traits[[1L]]$A)
  expect_identical(ses(pairs), as.numeric(c(
    "0x1.682439d197635p-4", "0x1.8ff95c18a5dc0p-3", "0x1.b538b3cddd351p-4",
    "0x1.05c97700cf294p-3", "0x1.1739e4341d1cdp-3", "0x1.3e1d06e735df6p-3",
    "0x1.0658022c5ff58p-3", "0x1.1b3574ec7f618p-3", "0x1.e83d347345b65p-4",
    "0x1.2f86dde2747c8p-2", "0x1.5a7ee3225b130p-2", "0x1.f7753fff1df0cp-5"
  )))
  iv <- relative_pairs_ace(two_cov, shared_environment = shared, weights = "inverse_variance")
  expect_identical(iv$traits[[1L]]$A, as.numeric("0x1.47337de2865f4p-2"))
  expect_identical(ses(iv), as.numeric(c(
    "0x1.fadc5360f4d6fp-4", "0x1.03f86f1101f34p-3", "0x1.9f13a0829f609p-5",
    "0x1.f5ce968d12507p-4", "0x1.f0bf62e8ee47ap-4", "0x1.6a9ffcb6b4c06p-5",
    NA, "0x1.abfcf11013fa9p-4", "0x1.b2ec5e4c50c41p-4",
    "0x1.386f1efc14a16p-3", "0x1.e332db905c729p-3", "0x1.a7be3bac5e599p-1"
  )))
  expect_identical(iv$settings$weights, "inverse_variance")
  expect_identical(iv$cross$sA_se_reason, "negative_variance")
  # The fixture's covariance is not positive definite over the cells each solve reads.
  gls <- relative_pairs_ace(two_cov, shared_environment = shared, weights = "gls")
  expect_null(gls$traits[[1L]]$A)
  expect_identical(c(gls$traits[[1L]]$A_reason, gls$traits[[1L]]$A_se_reason, gls$cross$rA_se_reason),
                   rep("covariance_not_positive_definite", 3L))
  expect_identical(c(gls$settings$weights, gls$traits[[1L]]$weights), c("gls", "gls"))
  expect_identical(pairs$rows[[4L]]$variance, two_cov$covariance$pooled[4L, 4L])
  expect_identical(pairs$rows[[32L]]$variance, two_cov$covariance$pooled[32L, 32L])
  expect_output(print(pairs), "SE")
})

test_that("Falconer's SEs are the closed forms of the covariance", {
  one <- relative_pairs_correlation(ped, trait(x), categories = cats, se = TRUE, covariance = TRUE)
  s <- one$covariance$pooled
  mm <- s["MZ:1:1", "MZ:1:1"]
  mf <- s["MZ:1:1", "FS:1:1"]
  ff <- s["FS:1:1", "FS:1:1"]
  for (weights in c("pairs", "inverse_variance")) {
    t <- relative_pairs_ace(one, categories = c("FS", "MZ"), shared_environment = c("MZ", "FS"),
                            weights = weights)$traits[[1L]]
    expect_equal(t$A_se, sqrt(4 * mm - 8 * mf + 4 * ff), tolerance = 1e-12)
    expect_equal(t$C_se, sqrt(mm - 4 * mf + 4 * ff), tolerance = 1e-12)
    expect_equal(t$E_se, sqrt(mm), tolerance = 1e-12)
  }
})

# Each solve's weighting: the traits', then the cross solve's.
weighting <- function(fit) c(vapply(fit$traits, `[[`, "", "weights"), fit$cross$weights)
solved <- function(fit) fit[c("traits", "cross", "rows")]

test_that("the default weights are gls per fit with the covariance read, else pairs", {
  b <- function(v) trait(as.numeric(v > 0.2), kind = "binary")
  binary <- relative_pairs_correlation(ped, list(b(x), b(y)), categories = cats, se = TRUE, covariance = TRUE)
  expect_false(is.null(binary$covariance$phi))
  fs <- relative_pairs_ace(binary, categories = "FS", components = "A")
  expect_identical(c(fs$settings$weights, weighting(fs)), c("default", rep("gls", 3L)))
  expect_identical(solved(fs), solved(relative_pairs_ace(binary, categories = "FS", components = "A", weights = "gls")))
  phi <- relative_pairs_ace(binary, components = "A", estimator = "phi")
  expect_identical(weighting(phi), rep("pairs", 3L))
  expect_identical(solved(phi), solved(relative_pairs_ace(binary, components = "A", estimator = "phi", weights = "pairs")))
  # Without the covariance the default is pairs: the fit before the covariance existed.
  plain <- relative_pairs_ace(two, shared_environment = shared)
  expect_identical(weighting(plain), rep("pairs", 3L))
  expect_identical(solved(plain), solved(relative_pairs_ace(two, shared_environment = shared, weights = "pairs")))
})

test_that("the default falls back to pairs for each fit whose covariance has no Cholesky factor", {
  by_default <- relative_pairs_ace(two_cov, shared_environment = shared)
  expect_identical(weighting(by_default), rep("pairs", 3L))
  expect_identical(solved(by_default), solved(relative_pairs_ace(two_cov, shared_environment = shared, weights = "pairs")))
  expect_false(is.null(by_default$traits[[1L]]$A))
  # FS and MO: each trait's block has a factor, the cross block has none.
  fit <- relative_pairs_ace(two_cov, categories = c("FS", "MO"), components = "A")
  expect_identical(weighting(fit), c("gls", "gls", "pairs"))
  gls <- relative_pairs_ace(two_cov, categories = c("FS", "MO"), components = "A", weights = "gls")
  pairs <- relative_pairs_ace(two_cov, categories = c("FS", "MO"), components = "A", weights = "pairs")
  expect_identical(fit$traits, gls$traits)
  expect_identical(fit$cross[c("sA", "sA_se", "sE", "sE_se", "rows")], pairs$cross[c("sA", "sA_se", "sE", "sE_se", "rows")])
  expect_identical(gls$cross$sA_reason, "covariance_not_positive_definite")
  expect_output(print(fit), "pairs weights")
})

test_that("SEs need the covariance", {
  fit <- relative_pairs_ace(two, shared_environment = shared)
  expect_null(fit$traits[[1L]]$A_se)
  expect_identical(c(fit$traits[[1L]]$A_se_reason, fit$cross$rA_se_reason), c("no_covariance", "no_covariance"))
  expect_null(fit$rows[[1L]]$variance)
  expect_identical(fit$settings$weights, "default")
  expect_pgp_error(relative_pairs_ace(two, shared_environment = shared, weights = "gls"), "usage", "missing_covariance")
  expect_pgp_error(relative_pairs_ace(two, components = "A", weights = "iv"), "usage")
  short <- two_cov
  short$cells <- short$cells[-1L]
  err <- expect_pgp_error(relative_pairs_ace(short, components = "A"), "validation", "covariance_shape_mismatch")
  expect_identical(err$fields$cells, 31)
})
