# pedigree(): a validated pedigree that methods share (ADR 0006).

test_that("ids keep their storage type through pedigree() and columns", {
  ped <- small_pedigree()
  storages <- list(
    integer = ped,
    double = transform(ped, id = as.double(id), mother = as.double(mother), father = as.double(father)),
    integer64 = transform(ped, id = as_integer64(id))
  )
  for (name in names(storages)) {
    x <- storages[[name]]
    built <- pedigree(x)
    expect_identical(built$ids, x$id, info = name)
    expect_identical(length(built), nrow(x), info = name)
    from_columns <- score_small(pafgrs_prepare(x))
    from_built <- score_small(pafgrs_prepare(built))
    expect_identical(from_built, from_columns, info = name)
    expect_identical(from_built$id, x$id, info = name)
  }
})

test_that("a pedigree gives what its columns give, in any call order", {
  ped <- small_pedigree()
  built <- pedigree(ped)
  y <- trait(c(0.3, -1.2, 0.5, 0.1, 0.9, -0.4, 0.2))
  stratum <- c(1, 1, 2, 2, NA, 2, 2)
  calls <- list(
    function(p) assortative_mate_correlation(p, y, stratum = stratum, permutations = 9, min_stratum_networks = 1),
    function(p) assortative_mate_correlation(p, y, permutations = 9),
    function(p) assortative_mate_correlation(p, list(y, y), bootstrap = 9)
  )
  for (call in calls) expect_identical(call(built), call(ped))
  expect_output(print(built), "<pgphenotype_pedigree> 7 rows", fixed = TRUE)
})

test_that("with columns, parameters fail before the pedigree", {
  bad <- data.frame(id = c(1L, 1L, 2L), mother = NA, father = NA)
  expect_pgp_error(pafgrs_prepare(bad, ndegree = 0), "validation", "degree_out_of_range")
  expect_pgp_error(assortative_mate_correlation(bad, trait(c(0.1, 0.2, 0.3)), permutations = -1),
                   "parameter", "parameter_out_of_range")
  expect_pgp_error(pedigree(bad), "validation", "duplicate_id")
  built <- pedigree(small_pedigree())
  expect_pgp_error(assortative_mate_correlation(built, trait(c(0.1, 0.2))), "validation", "trait_length_mismatch")
  expect_pgp_error(pafgrs_prepare(built, probands = 99), "validation", "unknown_proband")
  expect_pgp_error(pedigree(built), "usage")
  expect_pgp_error(pafgrs_prepare(42), "usage")
})

test_that("a pedigree that did not survive serialization, or another object, is a usage error", {
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path))
  saveRDS(pedigree(small_pedigree()), path)
  restored <- readRDS(path)
  err <- expect_pgp_error(pafgrs_prepare(restored), "usage")
  expect_match(conditionMessage(err), "saveRDS", fixed = TRUE)
  impostor <- structure(list(ptr = pafgrs_prepare(small_pedigree()), ids = 1L),
                        class = "pgphenotype_pedigree")
  err <- expect_pgp_error(assortative_mate_correlation(impostor, trait(1:7 / 7)), "usage")
  expect_match(conditionMessage(err), "must come from pedigree()", fixed = TRUE)
})

test_that("a prep outlives its pedigree, and a dropped pedigree is freed", {
  dropped <- get0("wrap__pedigrees_dropped_for_test", envir = asNamespace("pgphenotype"), inherits = FALSE)
  if (is.null(dropped)) {
    if (identical(Sys.getenv("PG_PHENOTYPE_REQUIRE_TEST_HOOKS"), "1")) {
      fail("the package was built without the test-hooks feature; run `pixi run -e r r-install`")
    }
    skip("installed without the test-hooks feature")
  }
  built <- pedigree(small_pedigree())
  prep <- pafgrs_prepare(built)
  want <- score_small(prep)
  gc()
  before <- .Call(dropped)
  rm(built)
  gc()
  expect_identical(.Call(dropped), before + 1)
  expect_identical(score_small(prep), want)
})
