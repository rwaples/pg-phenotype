# pafgrs_cip_aalen_johansen(): a CIP table from ages and survival status (issue #4).

test_that("the hand example steps at each case age", {
  # t=1: Y=5, a case: 1/5.  t=2: a death.  t=4: Y=2: += 3/5 * 1/2.  t=5: Y=1: += 3/10.
  table <- pafgrs_cip_aalen_johansen(c(1, 2, 3, 4, 5), c(1, 2, 0, 1, 1))
  expect_s3_class(table, "pgphenotype_pafgrs_cip")
  expect_identical(table$interpolation, "step")
  expect_identical(table$ages, c(1, 4, 5))
  expect_equal(table$cip, c(0.2, 0.5, 0.8), tolerance = 1e-15)
  expect_identical(table$prevalence, table$cip[3])
  expect_output(print(table), "<pgphenotype_pafgrs_cip> step, 3 ages from 1 to 5", fixed = TRUE)
})

test_that("it matches survfit with late entry, ties and a competing death", {
  skip_if_not_installed("survival")
  set.seed(4)
  n <- 400
  entry <- sample(c(0, 0, 20), n, replace = TRUE)
  age <- entry + ceiling(stats::rexp(n, 1 / 30))
  status <- sample(0:2, n, replace = TRUE, prob = c(0.5, 0.3, 0.2))
  table <- pafgrs_cip_aalen_johansen(age, status, entry)
  fit <- survival::survfit(
    survival::Surv(entry, age, factor(status, 0:2)) ~ 1,
    id = seq_len(n), timefix = FALSE
  )
  grid <- sort(unique(age[status == 1]))
  expect_identical(table$ages, grid)
  expect_equal(table$cip, fit$pstate[match(grid, fit$time), 2], tolerance = 1e-12)
})

test_that("invalid input names the first offending person, 1-based", {
  err <- expect_pgp_error(pafgrs_cip_aalen_johansen(1, c(1, 0)), "validation", "pair_length_mismatch")
  expect_identical(err$fields$columns, c("age", "status"))
  err <- expect_pgp_error(pafgrs_cip_aalen_johansen(c(1, NA), c(1, 0)), "validation", "invalid_age")
  expect_identical(err$fields$position, 2)
  err <- expect_pgp_error(pafgrs_cip_aalen_johansen(c(1, 2), c(1, 3)), "validation", "invalid_status")
  expect_identical(err$fields$position, 2)
  expect_match(conditionMessage(err), "status[2]", fixed = TRUE)
  err <- expect_pgp_error(
    pafgrs_cip_aalen_johansen(c(1, 2), c(1, 2), entry = c(0, 2)), "validation", "entry_not_before_age"
  )
  expect_identical(err$fields$position, 2)
  expect_match(conditionMessage(err), "row 2", fixed = TRUE)
  err <- expect_pgp_error(pafgrs_cip_aalen_johansen(c(1, 2), c(0, 2)), "validation", "cip_not_estimable")
  expect_identical(err$fields$reason, "zero_prevalence")
  err <- expect_pgp_error(pafgrs_cip_aalen_johansen(c(1, 2), c(1, 1)), "validation", "cip_not_estimable")
  expect_identical(err$fields$reason, "prevalence_one")
  expect_pgp_error(pafgrs_cip_aalen_johansen("1", 1), "usage")
})

test_that("a step table scores a control by the last step", {
  ped <- data.frame(id = 1:3, mother = c(NA, NA, 1L), father = c(NA, NA, 2L))
  prep <- pafgrs_prepare(ped, ndegree = 1L, probands = 3L)
  dx <- trait(c(0, 1, NA), kind = "binary")
  est <- function(cip) pafgrs_score_univariate(prep, dx, age = c(55, 40, NA), cip = cip, h2 = 0.5)$est
  step <- est(pafgrs_cip(c(20, 60), c(0.05, 0.1), interpolation = "step"))
  # A linear table flat from 20 to 55 gives the mother the same w (0.5) and K.
  expect_identical(step, est(pafgrs_cip(c(20, 55, 60), c(0.05, 0.05, 0.1))))
  expect_false(identical(step, est(pafgrs_cip(c(20, 60), c(0.05, 0.1)))))
  expect_error(pafgrs_cip(c(0, 1), c(0, 0.1), interpolation = "cubic"), "should be one of")
})
