test_that("the budget defaults to 1", {
  out <- run_rscript(c("Sys.unsetenv('PG_PHENOTYPE_THREADS')", "cat(thread_budget())"))
  expect_identical(out, "1")
})

test_that("PG_PHENOTYPE_THREADS sets the budget", {
  expect_identical(run_rscript("cat(thread_budget())", env = "PG_PHENOTYPE_THREADS=3"), "3")
})

test_that("configure_threads wins over the environment and commits on first use", {
  out <- run_rscript(c(
    "configure_threads(2)",
    "cat(thread_budget(), '')",
    "configure_threads(2)",
    "e <- tryCatch(configure_threads(4), pgphenotype_thread_conflict_error = function(e) e)",
    "cat(class(e)[1], e$code, e$fields$configured, e$fields$requested)"
  ), env = "PG_PHENOTYPE_THREADS=3")
  expect_identical(out, "2 pgphenotype_thread_conflict_error thread_pool_conflict 2 4")
})

test_that("the first native call commits the budget", {
  out <- run_rscript(c(
    "invisible(pafgrs_prepare(data.frame(id = 1:2, mother = NA, father = NA)))",
    "e <- tryCatch(configure_threads(4), pgphenotype_thread_conflict_error = function(e) 'conflict')",
    "cat(e)"
  ), env = "PG_PHENOTYPE_THREADS=2")
  expect_identical(out, "conflict")
})

test_that("configuring again before first use replaces the value", {
  out <- run_rscript(c("configure_threads(2)", "configure_threads(5)", "cat(thread_budget())"))
  expect_identical(out, "5")
})

test_that("a bad budget is a usage error", {
  for (n in list(0, 1.5, -1, NA, "2", c(1, 2), TRUE, 2^31)) {
    expect_pgp_error(configure_threads(n), "usage")
  }
  for (raw in c("", "two", "0", "3000000000")) {
    out <- run_rscript(
      "e <- tryCatch(thread_budget(), pgphenotype_usage_error = function(e) 'usage'); cat(e)",
      env = paste0("PG_PHENOTYPE_THREADS=", raw)
    )
    expect_identical(out, "usage", info = raw)
  }
})

test_that("scores are identical across thread budgets", {
  code <- c(
    "ped <- utils::read.csv(Sys.getenv('PGP_PED'))",
    "t <- utils::read.csv(Sys.getenv('PGP_TRAITS'), colClasses = 'character')",
    "table <- pafgrs_cip(c(0, 40, 80), c(0, 0.05, 0.1))",
    "t1 <- trait(as.numeric(t$trait1))",
    "t2 <- trait(as.numeric(t$trait2))",
    "ages <- list(as.numeric(t$age1), as.numeric(t$age2))",
    "prep <- pafgrs_prepare(ped, ndegree = 3)",
    "s <- pafgrs_score_univariate(prep, t1, ages[[1]], table, h2 = 0.5)",
    "b <- pafgrs_score_bivariate(prep, list(t1, t2), ages, list(table, table), h2 = c(0.5, 0.3), rg = 0.4)",
    "cat(thread_budget(), sprintf('%a', c(s$est, s$var, b$est1, b$est2, b$var1, b$var2, b$cov12)))"
  )
  env <- c(paste0("PGP_PED=", test_path("fixtures", "pedigree_deep.csv")),
           paste0("PGP_TRAITS=", test_path("fixtures", "traits_deep.csv")))
  one <- run_rscript(code, env = c(env, "PG_PHENOTYPE_THREADS=1"))
  four <- run_rscript(code, env = c(env, "PG_PHENOTYPE_THREADS=4"))
  expect_null(attr(one, "status"))
  expect_match(one, "^1 ")
  expect_match(four, "^4 ")
  expect_identical(sub("^1 ", "", one), sub("^4 ", "", four))
})
