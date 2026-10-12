# Every native call that computes in the pool hands free heap pages back
# first (src/rust/src/crossing.rs).

memory_trait <- function() trait(c(0.1, 0.9, 0.4, 0.2, 0.3, 0.6, 0.5))

# Each crossing native routine, as a public call of it.  A score is called
# with a prep built outside the counted call.
crossing <- list(
  .native_pedigree_new = function(prep) pedigree(small_pedigree()),
  .native_pafgrs_prepare = function(prep) pafgrs_prepare(small_pedigree()),
  .native_pafgrs_score_univariate = function(prep) score_small(prep),
  .native_pafgrs_score_bivariate = function(prep) score_small_pair(prep, h2 = c(0.4, 0.3), rg = 0.2),
  .native_assortative_mate_correlation = function(prep) {
    assortative_mate_correlation(small_pedigree(), memory_trait(), permutations = 0)
  },
  .native_regression_parent_offspring = function(prep) {
    regression_parent_offspring(small_pedigree(), memory_trait())
  },
  .native_relative_pairs_correlation = function(prep) {
    relative_pairs_correlation(small_pedigree(), memory_trait())
  }
)

# Native routines that do not compute in the pool, and so do not release.
not_crossing <- c(
  ".native_configure_threads", ".native_correlation_tetrachoric_pairs",
  ".native_correlation_tetrachoric_table", ".native_pafgrs_check_cip",
  ".native_pafgrs_cip_aalen_johansen", ".native_pafgrs_prep_info", ".native_pedigree_align",
  ".native_pedigree_rows", ".native_thread_budget", ".native_trait_kinds"
)

test_that("every native routine is classified as crossing or not", {
  routines <- ls(asNamespace("pgphenotype"), pattern = "^\\.native_", all.names = TRUE)
  expect_setequal(routines, c(names(crossing), not_crossing))
})

test_that("a crossing releases free memory once", {
  releases <- get0("wrap__releases_for_test", envir = asNamespace("pgphenotype"), inherits = FALSE)
  if (is.null(releases)) {
    if (identical(Sys.getenv("PG_PHENOTYPE_REQUIRE_TEST_HOOKS"), "1")) {
      fail("the package was built without the test-hooks feature; run `pixi run -e r r-install`")
    }
    skip("installed without the test-hooks feature")
  }
  prep <- pafgrs_prepare(small_pedigree())
  for (name in names(crossing)) {
    before <- .Call(releases)
    crossing[[name]](prep)
    expect_identical(.Call(releases), before + 1, info = name)
  }
})
