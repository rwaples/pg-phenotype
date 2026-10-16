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
    "0x1.fc8100937c6d9p-3", "0x1.0dade082b5139p-4", "0x1.5f2a03caca423p-1"
  )))
  expect_identical(shares(fit$traits[[2L]]), as.numeric(c(
    "0x1.64b480fb3ddecp-2", "0x1.a70115a24ec98p-3", "0x1.c7caf4339abc8p-2"
  )))
  cross <- vapply(c("sA", "sC", "sE", "rA", "rC", "rE"), function(k) fit$cross[[k]], 0, USE.NAMES = FALSE)
  expect_identical(cross, as.numeric(c(
    "0x1.58f9827525a36p-3", "0x1.7ab1f7a202ebdp-4", "0x1.4c8ff7e74093dp-2",
    "0x1.254066770b240p-1", "0x1.95ed7bd103127p-1", "0x1.2cf24d4a858e1p-1"
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
