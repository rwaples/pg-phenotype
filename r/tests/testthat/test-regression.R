# regression_parent_offspring(): the same Rust core as the Python package's
# pg_phenotype.regression.parent_offspring, so values agree bit for bit.

ped <- data.frame(
  id = 1:15,
  mother = c(rep(NA, 8), 1, 3, 3, 5, 5, 7, 7),
  father = c(rep(NA, 8), 2, 4, 4, 6, 6, 8, 8)
)
v <- c(0.0, 1.0, 0.2, 0.6, 1.4, 1.0, -0.5, 0.1, 0.7, 0.1, 0.9, 1.5, 1.0, -0.6, 0.3)
stratum <- c(rep(0, 9), rep(1, 5), NA)

fields <- function(f) c(f$slope, f$intercept, f$r, f$se, f$se_ols, f$p, f$ci)

test_that("fits give Python's numbers bit for bit", {
  # Python: parent_offspring(ped, Trait(v), stratum=stratum), as hex floats.
  r <- regression_parent_offspring(ped, trait(v), stratum = stratum)
  expect_s3_class(r, "pgphenotype_parent_offspring")
  expect_identical(fields(r$pooled$midparent), as.numeric(c(
    "0x1.ff4250c39cb63p-1", "0x1.61e1530fa2584p-4", "0x1.aa7c92adc0a9bp-1", "0x1.52892f0fb7e61p-5",
    "0x1.2fbf2d41b1e9bp-2", "0x1.e0267f4d737dfp-427", "0x1.d5ca068a27eeap-1", "0x1.145d4d7e88beep+0"
  )))
  expect_identical(fields(r$pooled$mother), as.numeric(c(
    "0x1.60cacd7dca9c5p-1", "0x1.5cc25a4337809p-2", "0x1.98c06634ac104p-1", "0x1.67687251e3daap-4",
    "0x1.dc0b62191eb60p-3", "0x1.254e74822584dp-48", "0x1.08bd25c322d44p-1", "0x1.b8d8753872646p-1"
  )))
  s1 <- r$strata[[2L]]
  expect_identical(s1$stratum, 1)
  expect_identical(fields(s1$midparent), as.numeric(c(
    "0x1.3c71c71c71c72p+0", "-0x1.4b17e4b17e4b4p-3", "0x1.ca246b1862834p-1", "0x1.4702e0cb3d15cp-3",
    "0x1.6c9e3e0546933p-2", "0x1.6200962d7d83bp-47", "0x1.d8a804e007e60p-1", "0x1.8c8f8bc8df9b4p+0"
  )))
  expect_identical(c(r$pooled$midparent$n, r$pooled$midparent$n_mate_networks), c(7L, 4L))
  expect_identical(c(s1$midparent$n, s1$midparent$n_mate_networks), c(5L, 3L))
  expect_identical(r$pooled$midparent$dropped$parent_not_in_pedigree, 8L)
  expect_identical(r$n_unknown_stratum, 1L)
  expect_identical(vapply(r$strata, `[[`, 0, "stratum"), c(0, 1))
})

test_that("a pedigree() gives what its columns give", {
  a <- regression_parent_offspring(pedigree(ped), trait(v))
  b <- regression_parent_offspring(ped, trait(v))
  expect_identical(a$pooled, b$pooled)
  expect_null(a$strata)
})

test_that("undefined fits carry reasons", {
  r <- regression_parent_offspring(ped, trait(rep(NA_real_, 15), kind = "continuous"))
  f <- r$pooled$father
  expect_null(f$slope)
  expect_identical(f$reason, "no_complete_children")
  expect_identical(f$p_unavailable_reason, "no_complete_children")
})

test_that("invalid input raises the core's conditions", {
  expect_error(regression_parent_offspring(ped, trait(letters[1:15])), class = "pgphenotype_validation_error")
  expect_error(regression_parent_offspring(ped, trait(v[-1])), class = "pgphenotype_validation_error")
  expect_error(regression_parent_offspring(ped, v), class = "pgphenotype_usage_error")
})
