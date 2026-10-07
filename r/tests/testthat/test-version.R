test_that("the binding's version is the package version", {
  expect_identical(pgphenotype_version(), as.character(utils::packageVersion("pgphenotype")))
})

test_that("pg_core_rev is a full git commit hash", {
  expect_match(pg_core_rev(), "^[0-9a-f]{40}$")
})
