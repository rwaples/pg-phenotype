test_that("an ordered factor is ordinal, coded in its level order", {
  t <- trait(factor(c("severe", "mild", NA), levels = c("mild", "moderate", "severe"), ordered = TRUE))
  expect_identical(t$kind, "ordinal")
  expect_identical(t$values, c(2, 0, NA))
  expect_identical(t$levels, c("mild", "moderate", "severe"))
  expect_identical(trait(factor(c("b", "a"), ordered = TRUE), kind = "categorical")$kind, "categorical")
})

test_that("an unordered factor is categorical, coded in its level order, unused levels kept", {
  t <- trait(factor(c("lo", "hi", NA, "lo"), levels = c("lo", "mid", "hi")))
  expect_identical(t$kind, "categorical")
  expect_identical(t$values, c(0, 2, NA, 0))
  expect_identical(t$levels, c("lo", "mid", "hi"))
  expect_identical(trait(factor(c("b", "a")), kind = "ordinal")$kind, "ordinal")
})

test_that("an explicit kind is kept and integer codes pass", {
  expect_identical(trait(c(0, 1, 1), kind = "continuous")$kind, "continuous")
  t <- trait(c(3L, NA, 1L), kind = "categorical")
  expect_identical(t$values, c(3, NA, 1))
  expect_null(t$levels)
})

test_that("bad inputs are classed errors", {
  err <- expect_pgp_error(trait(c(2, 0.5, 1.5), kind = "ordinal"), "validation", "invalid_trait_value")
  expect_identical(err$fields, list(field = "trait", position = 2, value = 0.5))
  expect_match(conditionMessage(err), "trait[2] = 0.5", fixed = TRUE)
  expect_pgp_error(trait(1:3), "validation", "ambiguous_trait_kind")
  expect_pgp_error(trait(1, kind = c("binary", "ordinal")), "usage")
  expect_pgp_error(trait(1, kind = 1), "usage")
  expect_pgp_error(trait(list(1, 2)), "usage")
  expect_pgp_error(trait(1i), "usage")
})

test_that("character labels sort by code point under any collation", {
  old <- Sys.getlocale("LC_COLLATE")
  on.exit(Sys.setlocale("LC_COLLATE", old))
  if (!nzchar(suppressWarnings(Sys.setlocale("LC_COLLATE", "en_US.UTF-8")))) skip("no en_US.UTF-8 locale")
  expect_identical(trait(c("b", "B", "a", "_"))$levels, c("B", "_", "a", "b"))
})
