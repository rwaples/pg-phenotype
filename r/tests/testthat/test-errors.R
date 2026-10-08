test_that("errors are classed conditions with a code and 1-based fields", {
  ped <- small_pedigree()
  err <- expect_pgp_error(pafgrs_prepare(ped, probands = c(10, 99)), "validation", "unknown_proband")
  expect_s3_class(err, "pgphenotype_error")
  expect_identical(err$fields, list(id = 99, position = 2))
  expect_match(conditionMessage(err), "probands[2] = 99", fixed = TRUE)
  err <- expect_pgp_error(pafgrs_prepare(ped, probands = c(12, 13, 12)), "validation", "duplicate_proband")
  expect_identical(err$fields, list(id = 12, positions = c(1, 3)))
  err <- expect_pgp_error(pafgrs_prepare(ped, probands = c(12, NA)), "validation", "invalid_integer_value")
  expect_identical(err$fields$position, 2)
})

test_that("the pedigree is validated", {
  ped <- small_pedigree()
  err <- expect_pgp_error(pafgrs_prepare(ped[c("id", "mother")]), "validation", "missing_field")
  expect_identical(err$fields, list(field = "father"))
  frac <- ped
  frac$mother[3] <- 10.5
  err <- expect_pgp_error(pafgrs_prepare(frac), "validation", "invalid_integer_value")
  expect_identical(err$fields[c("field", "position", "value")], list(field = "mother", position = 3, value = 10.5))
  dup <- ped
  dup$id[2] <- 10L
  expect_pgp_error(pafgrs_prepare(dup), "validation", "duplicate_id")
  cyc <- ped
  cyc$mother[1] <- 15L
  expect_pgp_error(pafgrs_prepare(cyc), "validation")
  expect_pgp_error(pafgrs_prepare(ped, ndegree = 0), "validation", "degree_out_of_range")
  err <- expect_pgp_error(pafgrs_prepare(ped, ndegree = 6), "validation", "degree_out_of_range")
  expect_identical(err$fields, list(value = 6, minimum = 1, maximum = 5))
  expect_pgp_error(pafgrs_prepare(ped, ndegree = 1.5), "usage")
  expect_pgp_error(pafgrs_prepare(ped, ndegree = "2"), "usage")
  expect_pgp_error(pafgrs_prepare(1:3), "usage")
})

test_that("score parameters are checked before the trait", {
  prep <- pafgrs_prepare(small_pedigree())
  bad <- list(trait = trait(0.5), age = 1)
  for (h2 in c(0, 1.2, NA, NaN)) {
    err <- expect_pgp_error(score_small(prep, bad, h2 = h2), "parameter", "parameter_out_of_range")
    expect_identical(err$fields$name, "h2")
    expect_identical(err$fields$domain, "(0, 1]")
  }
  expect_pgp_error(score_small_pair(prep, bad, h2 = c(0.3, 1.5), rg = 0.2), "parameter",
                   "parameter_out_of_range")
  expect_pgp_error(score_small_pair(prep, bad, h2 = c(0.3, 0.5), rg = 1.2), "parameter",
                   "parameter_out_of_range")
  expect_pgp_error(score_small_pair(prep, bad, h2 = c(0.3, 0.5), rg = 0.2, rho_within = -1.5),
                   "parameter", "parameter_out_of_range")
  expect_pgp_error(score_small_pair(prep, bad, h2 = c(0.9, 0.9), rg = 0, rho_within = 0.5),
                   "parameter", "inconsistent_parameters")
})

test_that("score arguments of the wrong type are usage errors", {
  prep <- pafgrs_prepare(small_pedigree())
  case <- small_case()
  table <- small_cip()
  expect_pgp_error(pafgrs_score_univariate(prep, case$trait$values, case$age, table, 0.3), "usage")
  expect_pgp_error(pafgrs_score_univariate(prep, case$trait, case$age, list(), 0.3), "usage")
  expect_pgp_error(pafgrs_score_univariate(prep, case$trait, "50", table, 0.3), "usage")
  pair <- list(case$trait, case$trait)
  ages <- list(case$age, case$age)
  expect_pgp_error(pafgrs_score_bivariate(prep, case$trait, ages, list(table, table), c(0.3, 0.5), 0.2), "usage")
  expect_pgp_error(pafgrs_score_bivariate(prep, pair, case$age, list(table, table), c(0.3, 0.5), 0.2), "usage")
  expect_pgp_error(pafgrs_score_bivariate(prep, pair, ages, list(table), c(0.3, 0.5), 0.2), "usage")
  expect_pgp_error(pafgrs_score_bivariate(prep, pair, ages, list(table, table), 0.3, 0.2), "usage")
})

test_that("PA-FGRS takes binary traits only", {
  prep <- pafgrs_prepare(data.frame(id = 1:2, mother = NA, father = NA))
  table <- pafgrs_cip(c(0, 1), c(0, 0.1))
  err <- expect_pgp_error(pafgrs_score_univariate(prep, trait(c(1.5, 2)), c(10, 20), table, 0.5),
                          "validation", "trait_kind_mismatch")
  expect_identical(err$fields, list(field = "trait", expected = "binary", actual = "continuous"))
  err <- expect_pgp_error(
    pafgrs_score_bivariate(prep, list(trait(c(1, 0)), trait(c("a", "b"))), list(c(10, 20), c(10, 20)),
                           list(table, table), h2 = c(0.5, 0.5), rg = 0),
    "validation", "trait_kind_mismatch"
  )
  expect_identical(err$fields, list(field = "trait2", expected = "binary", actual = "categorical"))
})

test_that("trait values and ages are validated against the pedigree", {
  ped <- data.frame(id = 1L, mother = NA, father = NA)
  table <- pafgrs_cip(c(0, 1), c(0, 0.1))
  cases <- list(
    list(trait(2, kind = "binary"), 10, "invalid_trait_value", "trait", 2),
    list(trait(1), -1, "invalid_age", "age", -1),
    list(trait(1), Inf, "invalid_age", "age", Inf)
  )
  for (case in cases) {
    err <- expect_pgp_error(pafgrs_score_univariate(pafgrs_prepare(ped), case[[1]], case[[2]], table, h2 = 0.5),
                            "validation", case[[3]])
    want <- if (case[[4]] == "trait") {
      list(field = case[[4]], kind = "binary", position = 1, value = case[[5]])
    } else {
      list(field = case[[4]], position = 1, value = case[[5]])
    }
    expect_identical(err$fields, want)
  }
  err <- expect_pgp_error(
    pafgrs_score_bivariate(pafgrs_prepare(ped), list(trait(1), trait(3, kind = "binary")), list(10, 10),
                           list(table, table), h2 = c(0.5, 0.5), rg = 0),
    "validation", "invalid_trait_value"
  )
  expect_identical(err$fields$field, "trait2")
  err <- expect_pgp_error(
    pafgrs_score_bivariate(pafgrs_prepare(ped), list(trait(1), trait(1)), list(-2, 10),
                           list(table, table), h2 = c(0.5, 0.5), rg = 0),
    "validation", "invalid_age"
  )
  expect_identical(err$fields$field, "age1")
  two <- pafgrs_prepare(data.frame(id = 1:2, mother = NA, father = NA))
  err <- expect_pgp_error(pafgrs_score_univariate(two, trait(1), c(10, 20), table, h2 = 0.5),
                          "validation", "trait_length_mismatch")
  expect_identical(err$fields, list(field = "trait", expected_length = 2, actual_length = 1))
  err <- expect_pgp_error(pafgrs_score_univariate(two, trait(c(1, 0)), 10, table, h2 = 0.5),
                          "validation", "trait_length_mismatch")
  expect_identical(err$fields$field, "age")
})

test_that("a prep that did not survive serialization is a usage error", {
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path))
  saveRDS(pafgrs_prepare(small_pedigree()), path)
  restored <- readRDS(path)
  err <- expect_pgp_error(score_small(restored), "usage")
  expect_match(conditionMessage(err), "saveRDS", fixed = TRUE)
  expect_pgp_error(score_small(list()), "usage")
})
