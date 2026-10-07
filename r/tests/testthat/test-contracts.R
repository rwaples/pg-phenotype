test_that("output is one row per proband in pedigree input-row order", {
  ped <- small_pedigree()[c(5, 2, 7, 1, 4, 6, 3), ]
  s <- score_small(pafgrs_prepare(ped))
  expect_s3_class(s, "data.frame")
  expect_identical(names(s), c("id", "est", "var", "n_relatives"))
  expect_identical(s$id, ped$id)
  b <- score_small_pair(pafgrs_prepare(ped), h2 = c(0.4, 0.3), rg = 0.2)
  expect_identical(names(b), c("id", "est1", "est2", "var1", "var2", "cov12", "n_relatives", "n_obs1", "n_obs2"))
  expect_identical(b$id, ped$id)
})

test_that("scores follow ids, not row positions", {
  ped <- small_pedigree()
  case <- small_case()
  base <- score_small(pafgrs_prepare(ped), case)
  perm <- c(4L, 7L, 1L, 6L, 2L, 5L, 3L)
  shuffled <- list(trait = trait(case$trait$values[perm]), age = case$age[perm])
  got <- score_small(pafgrs_prepare(ped[perm, ]), shuffled)
  expect_identical(got$id, base$id[perm])
  expect_identical(got$est, base$est[perm])
  expect_identical(got$var, base$var[perm])
})

test_that("a probands subset scores those rows, in input-row order, as the full run does", {
  ped <- small_pedigree()
  full <- score_small(pafgrs_prepare(ped, ndegree = 3))
  sub <- score_small(pafgrs_prepare(ped, ndegree = 3, probands = c(16, 12, 10)))
  keep <- full$id %in% c(10L, 12L, 16L)
  expect_identical(sub$id, c(10L, 12L, 16L))
  expect_identical(sub$est, full$est[keep])
  expect_identical(attr(sub, "metadata")$n_probands, 3L)
})

test_that("NA and -1 parents are the same, and ids keep their storage", {
  ped <- small_pedigree()
  minus <- ped
  minus$mother[is.na(minus$mother)] <- -1L
  minus$father[is.na(minus$father)] <- -1L
  a <- score_small(pafgrs_prepare(ped))
  expect_identical(score_small(pafgrs_prepare(minus)), a)
  dbl <- ped
  dbl$id <- as.double(dbl$id)
  expect_identical(score_small(pafgrs_prepare(dbl))$id, as.double(ped$id))
  i64 <- ped
  i64$id <- as_integer64(ped$id)
  got <- score_small(pafgrs_prepare(i64))
  expect_identical(class(got$id), "integer64")
  expect_identical(unclass(got$id), unclass(i64$id))
  expect_identical(got$est, a$est)
})

test_that("the missingness rules hold", {
  ped <- data.frame(id = 1:4, mother = c(NA, NA, 1L, 1L), father = c(NA, NA, 2L, 2L))
  table <- pafgrs_cip(c(0, 80), c(0, 0.1))
  age <- c(30, NA, 40, NA)
  s <- pafgrs_score_univariate(pafgrs_prepare(ped), trait(c(NA, 0, 0, 1)), age, table, h2 = 0.5)
  expect_identical(attr(s, "metadata")$controls_without_age, 1L)
  # Row 4 (a case without age) and row 3 (a control with age) inform the founders.
  expect_identical(s$n_relatives[1:2], c(2L, 2L))
  logical_status <- trait(c(NA, FALSE, FALSE, TRUE))
  expect_identical(pafgrs_score_univariate(pafgrs_prepare(ped), logical_status, age, table, h2 = 0.5), s)
})

test_that("a proband with no informative relative gets the prior", {
  ped <- data.frame(id = 1:3, mother = NA, father = NA)
  case <- list(trait = trait(c(1, 0, 1)), age = c(30, 40, 50))
  s <- score_small(pafgrs_prepare(ped), case, h2 = 0.3)
  expect_identical(s$est, c(0, 0, 0))
  expect_identical(s$var, c(0.3, 0.3, 0.3))
  expect_identical(s$n_relatives, c(0L, 0L, 0L))
  b <- score_small_pair(pafgrs_prepare(ped), case, h2 = c(0.3, 0.5), rg = 0.4)
  expect_identical(b$cov12, rep(0.4 * sqrt(0.3 * 0.5), 3))
})

test_that("metadata carries the parameters and the versions", {
  prep <- pafgrs_prepare(small_pedigree(), ndegree = 3)
  meta <- attr(score_small(prep), "metadata")
  expect_identical(meta$h2, 0.4)
  expect_identical(meta$prevalence, 0.1)
  expect_identical(meta$threshold, small_cip()$threshold)
  expect_identical(meta$ndegree, 3L)
  expect_identical(meta$n_probands, 7L)
  expect_identical(meta$pg_phenotype_version, pgphenotype_version())
  expect_identical(meta$pedigree_graph_core_rev, pg_core_rev())
  expect_identical(meta$controls_without_age, 2L)
  b <- attr(score_small_pair(prep, h2 = c(0.4, 0.2), rg = 0.5), "metadata")
  expect_identical(b$rho_within, 0.5 * sqrt(0.4 * 0.2))
  expect_identical(b$h2, c(0.4, 0.2))
  expect_identical(b$controls_without_age, c(2L, 2L))
  given <- attr(score_small_pair(prep, h2 = c(0.4, 0.2), rg = 0.5, rho_within = 0.3), "metadata")
  expect_identical(given$rho_within, 0.3)
})

test_that("pafgrs_cip validates and derives the threshold", {
  table <- pafgrs_cip(c(0, 30, 80), c(0, 0.01, 0.07))
  expect_identical(table$prevalence, 0.07)
  expect_equal(table$threshold, stats::qnorm(0.07, lower.tail = FALSE), tolerance = 1e-15)
  bad <- list(
    list(c(0, 1), c(0, 1), 2), list(c(1, 0), c(0, 0.1), 2),
    list(c(0, 1), c(0, 0), 2), list(numeric(), numeric(), 1), list(c(0, NA), c(0, 0.1), 2)
  )
  for (case in bad) {
    err <- expect_pgp_error(pafgrs_cip(case[[1]], case[[2]]), "parameter", "invalid_cip")
    expect_identical(err$fields$position, case[[3]])
  }
  expect_pgp_error(pafgrs_cip("0", 0.1), "usage")
})

test_that("prep, cip and trait print a summary", {
  prep <- pafgrs_prepare(small_pedigree(), probands = c(12, 13))
  expect_output(print(prep), "<pgphenotype_pafgrs_prep> 7 rows, 2 probands, ndegree 2", fixed = TRUE)
  expect_output(print(small_cip()), "prevalence 0.1")
  expect_output(print(small_case()$trait), "<pgphenotype_trait> binary, 7 values, 1 missing", fixed = TRUE)
  expect_output(print(trait(c("b", "a"))), "categorical, 2 values, 0 missing; levels a, b", fixed = TRUE)
})
