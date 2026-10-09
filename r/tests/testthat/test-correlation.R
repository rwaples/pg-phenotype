# correlation_tetrachoric(): the same Rust core as the Python package's
# pg_phenotype.correlation.tetrachoric, so values agree bit for bit.

test_that("a table gives Python's numbers bit for bit", {
  # Python: tetrachoric(table=[[40, 10], [15, 35]]); then the Newton-plateau table.
  cases <- list(
    list(table = matrix(c(40, 15, 10, 35), 2),
         want = c("0x1.6cea8b0193234p-1", "0x1.891a9b50c5677p-4", "0x1.e18a6d9e17f6ep-2", "0x1.b5d63d7fe6259p-1")),
    list(table = matrix(c(9891, 1, 104, 4), 2),
         want = c("0x1.9805c3d5e7ac7p-1", "0x1.703d965afe986p-4", "0x1.1599d1ef6f522p-1", "0x1.d5c2359bd43ecp-1"))
  )
  for (case in cases) {
    r <- correlation_tetrachoric(table = case$table)
    expect_s3_class(r, "pgphenotype_tetrachoric")
    expect_identical(c(r$value, r$se, r$ci), as.numeric(case$want))
    expect_identical(r$table, list(as.integer(case$table[1, ]), as.integer(case$table[2, ])))
    expect_false(r$boundary)
    expect_null(r$reason)
    expect_identical(r$ci_method, "sandwich")
  }
})

test_that("pairs count into the table, missing pairs dropped", {
  x <- c(1, 0, 1, NA, 1, 0, 1, 0)
  y <- c(TRUE, FALSE, FALSE, TRUE, NA, FALSE, TRUE, TRUE)
  r <- correlation_tetrachoric(x, y)
  expect_identical(r$table, list(c(2L, 1L), c(1L, 2L)))
  expect_identical(c(r$n, r$n_dropped), c(6L, 2L))
  expect_identical(r$value, correlation_tetrachoric(table = matrix(c(2, 1, 1, 2), 2))$value)
})

test_that("undefined and boundary fits carry reasons", {
  r <- correlation_tetrachoric(table = matrix(c(3, 0, 4, 0), 2))
  expect_identical(r$reason, "constant_margin")
  expect_null(r$value)
  expect_identical(r$se_unavailable_reason, "constant_margin")
  r <- correlation_tetrachoric(table = matrix(0, 2, 2))
  expect_identical(r$reason, "no_complete_pairs")
  r <- correlation_tetrachoric(table = diag(10, 2))
  expect_true(r$boundary)
  expect_null(r$se)
  expect_identical(r$se_unavailable_reason, "boundary")
  expect_output(print(r), "pgphenotype_tetrachoric")
  big <- correlation_tetrachoric(table = matrix(c(3e9, 1e9, 1e9, 3e9), 2))
  expect_output(print(big), "n 8000000000:")
})

test_that("bad input is a classed error", {
  err <- expect_pgp_error(correlation_tetrachoric(table = matrix(c(1, -3, 2, 4), 2)), "validation", "invalid_table")
  expect_identical(err$fields, list(field = "table", row = 2, column = 1, value = -3))
  expect_pgp_error(correlation_tetrachoric(table = matrix(c(1.5, 1, 1, 1), 2)), "validation", "invalid_table")
  expect_pgp_error(correlation_tetrachoric(table = matrix(c(2^53 + 2, 1, 1, 1), 2)), "validation", "invalid_table")
  err <- expect_pgp_error(correlation_tetrachoric(c(0, 1, 1), c(0, 1, 7)), "validation", "invalid_trait_value")
  expect_identical(err$fields, list(field = "y", kind = "binary", position = 3, value = 7))
  expect_pgp_error(correlation_tetrachoric(c(0, 1), c(0, 1, 1)), "validation", "pair_length_mismatch")
  expect_pgp_error(correlation_tetrachoric(table = 1:4), "usage")
  expect_pgp_error(correlation_tetrachoric(c(0, 1)), "usage")
  expect_pgp_error(correlation_tetrachoric(c(0, 1), c(0, 1), table = diag(2)), "usage")
  expect_pgp_error(correlation_tetrachoric(c("a", "b"), c(0, 1)), "usage")
})
