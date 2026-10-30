# relative_pairs_correlation(): the same Rust core as the Python package's
# pg_phenotype.relative_pairs.correlation, so values agree bit for bit.

ped <- data.frame(
  id = 1:9,
  mother = c(NA, NA, 1, NA, 3, 3, 3, 3, NA),
  father = c(NA, NA, 2, NA, 4, 4, 4, 9, NA)
)
v <- c(0.3, -1.2, 0.8, 2.0, 1.5, -0.4, 0.9, 0.1, 1.1)
b <- c(0, 1, 1, 0, 1, 1, 0, 1, NA)
cats <- list("FS", "MHS", PO = c("MO", "FO"), "GP")

values <- function(r, estimator) {
  vapply(r$cells, function(cell) {
    e <- Filter(function(e) e$estimator == estimator, cell$crude)[[1L]]
    e$value
  }, 0)
}

test_that("continuous cells give Python's numbers bit for bit", {
  # Python: correlation(ped, Trait(v, kind="continuous"), categories=cats), as hex floats.
  r <- relative_pairs_correlation(ped, trait(v, kind = "continuous"), categories = cats)
  expect_s3_class(r, "pgphenotype_relative_pairs")
  expect_identical(vapply(r$cells, `[[`, "", "category"), c("FS", "MHS", "PO", "GP"))
  expect_identical(values(r, "pearson"), as.numeric(c(
    "-0x1.0000000000000p-1", "-0x1.a084980cf87f2p-3", "-0x1.0118cded12cebp-4", "0x0p+0"
  )))
  expect_identical(vapply(r$cells, `[[`, 0L, "n"), c(3L, 3L, 10L, 8L))
  po <- r$cells[[3L]]
  expect_identical(po$members, list("MO", "FO"))
  expect_identical(po$roles, list("offspring", "parent"))
  expect_true(r$cells[[1L]]$symmetric)
  expect_null(r$cells[[1L]]$roles)
  expect_identical(r$settings$quantization_exponents, list(42))
  expect_identical(r$settings$quantization_steps, 2^-42)
  expect_identical(r$metadata$pedigree_graph_core_rev, pg_core_rev())
})

test_that("binary cells give Python's numbers, Concordance and tables", {
  r <- relative_pairs_correlation(ped, trait(b, kind = "binary"), categories = cats)
  expect_identical(values(r, "tetrachoric"), as.numeric(c(
    "-0x1.fdbe2c6c552bep-1", "-0x1.f51f34b6c14c3p-1", "0x1.ab19ceaed4bc7p-4", "0x0p+0"
  )))
  expect_identical(values(r, "phi"), as.numeric(c(
    "-0x1.ffffffffffffep-2", "-0x1.9999999999998p-3", "0x1.e990cdad55ed2p-5", "0x0p+0"
  )))
  fs <- r$cells[[1L]]
  expect_identical(fs$concordance, c(0L, 2L, 1L))
  expect_null(fs$table)
  expect_true(fs$crude[[1L]]$primary)
  expect_true(fs$crude[[1L]]$boundary)
  po <- r$cells[[3L]]
  expect_identical(po$table, list(c(1L, 1L), c(3L, 4L)))
  expect_identical(c(po$n, po$n_dropped), c(9L, 1L))
})

test_that("se = TRUE adds each estimate's SE and leaves the values alone", {
  plain <- relative_pairs_correlation(ped, trait(b, kind = "binary"), categories = cats)
  r <- relative_pairs_correlation(ped, trait(b, kind = "binary"), categories = cats, se = TRUE)
  expect_null(plain$cells[[1L]]$crude[[1L]]$se)
  expect_false("se_reason" %in% names(plain$cells[[1L]]$crude[[1L]]))
  expect_identical(values(r, "tetrachoric"), values(plain, "tetrachoric"))
  tet <- r$cells[[1L]]$crude[[1L]]
  expect_null(tet$se)
  expect_identical(tet$se_reason, "boundary")
  reasons <- unlist(lapply(r$cells, function(cell) lapply(cell$crude, `[[`, "se_reason")))
  expect_true(all(reasons %in% c(
    "no_estimate", "boundary", "sandwich_undefined", "degenerate_meat", "negative_meat"
  )))
  expect_error(relative_pairs_correlation(ped, trait(v), se = NA))
  expect_identical(c(plain$settings$se, r$settings$se), c(FALSE, TRUE))
})

test_that("covariance = TRUE gives the estimates' covariance, its diagonal each SE squared, within_person's too", {
  y <- c(1.0, 0.2, -0.5, 0.7, 0.4, 1.3, -0.8, 0.6, 0.4)
  plain <- relative_pairs_correlation(ped, list(trait(v), trait(y)), categories = cats, se = TRUE)
  r <- relative_pairs_correlation(ped, list(trait(v), trait(y)), categories = cats, se = TRUE, covariance = TRUE)
  expect_null(plain$covariance)
  expect_identical(c(plain$settings$covariance, r$settings$covariance), c(FALSE, TRUE))
  expect_identical(r$cells, plain$cells)
  expect_null(r$covariance$phi)
  expect_null(r$covariance$stratified)
  m <- r$covariance$pooled
  labels <- c(sprintf("%s:%d:%d", rep(c("FS", "MHS"), each = 3L), c(1L, 2L, 1L), c(1L, 2L, 2L)),
              sprintf("%s:%d:%d", rep(c("PO", "GP"), each = 4L), c(1L, 1L, 2L, 2L), c(1L, 2L, 1L, 2L)),
              "within_person:1:2")
  expect_identical(dimnames(m), list(labels, labels))
  expect_true(isSymmetric(unname(m)))
  cells <- c(r$cells, list(r$within_person))
  se <- vapply(cells, function(cell) if (is.null(cell$crude[[1L]]$se)) NaN else cell$crude[[1L]]$se, 0)
  expect_identical(unname(sqrt(diag(m))), se)
  none <- is.nan(se)
  expect_identical(unname(is.nan(m)), outer(none, none, "|"))
  expect_identical(plain$within_person$crude[[1L]]$se_reason, "within_person")
  expect_false(identical(r$within_person$crude[[1L]]$se_reason, "within_person"))
  binary <- relative_pairs_correlation(ped, trait(b, kind = "binary"), categories = cats, stratum = c(0, 0, 1, NA, 2, 2, 2, 2, 1),
                                       se = TRUE, covariance = TRUE)
  expect_identical(dim(binary$covariance$phi), c(4L, 4L))
  expect_identical(dim(binary$covariance$stratified), c(4L, 4L))
  expect_identical(rownames(binary$covariance$phi), c("FS:1:1", "MHS:1:1", "PO:1:1", "GP:1:1"))
  expect_pgp_error(relative_pairs_correlation(ped, trait(v), covariance = TRUE), "usage")
  expect_error(relative_pairs_correlation(ped, trait(v), se = TRUE, covariance = NA))
})

test_that("influence = TRUE gives each cell's entries and influences, the meat over them each SE squared", {
  # Three generations: 30 founder couples with 3 children each, then 44 couples of children of two
  # neighbouring families with 2 children each; every fifth of those has the external father 9999.
  kids <- 61:150
  mothers2 <- kids[seq(1, 88, by = 2)]
  fathers2 <- ifelse(seq_along(mothers2) %% 5 == 0, 9999, kids[seq(4, 91, by = 2)])
  ped <- data.frame(
    id = c(1:150, 151:238),
    mother = c(rep(NA, 60), rep(seq(1, 59, by = 2), each = 3), rep(mothers2, each = 2)),
    father = c(rep(NA, 60), rep(seq(2, 60, by = 2), each = 3), rep(fathers2, each = 2))
  )
  set.seed(1)
  family <- rnorm(238)[ifelse(is.na(ped$mother), ped$id, ped$mother)]
  x <- family + rnorm(238)
  y <- family + rnorm(238)
  x[sample(238, 20)] <- NA
  st <- c(rep(0, 60), rep(1, 90), rep(2, 88))
  rebuilt <- function(entries, phi) {
    a <- entries$first_id
    b <- entries$second_id
    m <- ifelse(is.na(entries$mrca_id), paste("pair", pmin(a, b), pmax(a, b)), entries$mrca_id)
    sq <- function(key, w) sum(tapply(w, key, sum)^2)
    sqrt(sq(m, phi) + sq(c(a, b), c(phi, phi)) - sq(paste(c(a, b), c(m, m)), c(phi, phi)))
  }
  for (traits in list(trait(x), list(trait(x), trait(y)), trait(as.numeric(x > 0), kind = "binary"))) {
    plain <- relative_pairs_correlation(ped, traits, stratum = st, se = TRUE)
    r <- relative_pairs_correlation(ped, traits, stratum = st, se = TRUE, influence = TRUE)
    expect_identical(c(plain$settings$influence, r$settings$influence), c(FALSE, TRUE))
    expect_null(plain$cells[[1L]]$entries)
    checked <- 0L
    for (i in seq_along(r$cells)) {
      cell <- r$cells[[i]]
      n <- if (!is.null(cell$n_entries)) cell$n_entries else if (cell$symmetric) 2L * cell$n else cell$n
      expect_s3_class(cell$entries, "data.frame")
      expect_identical(nrow(cell$entries), as.integer(n))
      expect_identical(names(cell$entries), c("first_id", "second_id", "mrca_id"))
      estimates <- c(cell$crude, unlist(lapply(cell$strata, `[[`, "crude"), recursive = FALSE),
                     list(cell$stratified$estimate))
      for (e in estimates) {
        if (is.null(e$influence)) {
          expect_true(e$se_reason %in% c("no_estimate", "boundary", "sandwich_undefined"))
          next
        }
        expect_length(e$influence, n)
        if (!is.null(e$se)) {
          expect_equal(rebuilt(cell$entries, e$influence), e$se, tolerance = 1e-9)
          checked <- checked + 1L
        }
      }
      strip <- function(e) e[setdiff(names(e), "influence")]
      cell$entries <- NULL
      cell$crude <- lapply(cell$crude, strip)
      cell$strata <- lapply(cell$strata, function(s) {
        s$crude <- lapply(s$crude, strip)
        s
      })
      cell$stratified$estimate <- strip(cell$stratified$estimate)
      expect_identical(cell, plain$cells[[i]])
    }
    expect_gt(checked, 10L)
  }
  # The paternal half-sibs through the external father 9999 have him as their MRCA.
  expect_true(any(unlist(lapply(r$cells, function(cell) cell$entries$mrca_id)) == 9999))
  expect_pgp_error(relative_pairs_correlation(ped, trait(x), influence = TRUE), "usage")
  expect_error(relative_pairs_correlation(ped, trait(x), se = TRUE, influence = NA))
})

test_that("an external MRCA id past the id storage widens mrca_id instead of truncating", {
  # Integer row ids, and a double father column whose external id is past R's integer range.
  big <- 4294967297
  ped <- data.frame(id = 1:5, mother = c(NA, NA, 1L, 2L, 1L), father = c(NA, NA, big, big, 2))
  r <- relative_pairs_correlation(ped, trait(c(0.1, 0.5, 0.9, -0.3, 1.2)), categories = c("PHS", "MHS"),
                                  se = TRUE, influence = TRUE)
  phs <- r$cells[[1L]]$entries
  expect_type(phs$first_id, "integer")
  expect_type(phs$mrca_id, "double")
  expect_identical(phs$mrca_id, c(big, big))
  expect_type(r$cells[[2L]]$entries$mrca_id, "integer")
})

test_that("print shows each estimate's SE, or why it has none", {
  plain <- relative_pairs_correlation(ped, trait(v), categories = c("FS", "FO"))
  r <- relative_pairs_correlation(ped, trait(v), categories = c("FS", "FO"), se = TRUE)
  expect_false(any(grepl("se", capture.output(print(plain)), fixed = TRUE)))
  expect_output(print(r), "FS, n 3: pearson -0.5000 \\(no se: degenerate_meat\\)")
  expect_output(print(r), sprintf("FO, n 5: pearson -0.0663 \\(se %.4f\\)", r$cells[[2L]]$crude[[1L]]$se))
  binary <- relative_pairs_correlation(ped, trait(b, kind = "binary"), categories = cats, se = TRUE)
  expect_output(print(binary), "tetrachoric -?[0-9.]+ \\(no se: boundary\\)")
})

test_that("codes as a character vector, a pedigree() and the defaults", {
  a <- relative_pairs_correlation(pedigree(ped), trait(v), categories = c("FS", "MHS"))
  b <- relative_pairs_correlation(ped, trait(v), categories = list("FS", "MHS"))
  expect_identical(a$cells, b$cells)
  d <- relative_pairs_correlation(ped, trait(v))
  expect_identical(
    vapply(d$cells, `[[`, "", "category"),
    c("MZ", "MO", "FO", "FS", "MHS", "PHS", "GP", "Av", "GGP", "HAv", "GAv", "1C")
  )
  mz <- d$cells[[1L]]
  expect_identical(mz$n, 0L)
  expect_null(mz$crude[[1L]]$value)
  expect_identical(mz$crude[[1L]]$reason, "no_complete_pairs")
  expect_output(print(d), "pgphenotype_relative_pairs")
})

test_that("invalid input raises the core's conditions", {
  expect_pgp_error(relative_pairs_correlation(ped, trait(v[-1])), "validation", "trait_length_mismatch")
  expect_pgp_error(
    relative_pairs_correlation(ped, trait(c(0, 1, 2, 0, 1, 1, 0, 1, 0), kind = "ordinal")),
    "validation", "unsupported_trait_kind"
  )
  # Fields and messages count entries from 1, as R does.
  err <- expect_pgp_error(relative_pairs_correlation(ped, trait(v), categories = "XX"), "validation", "unknown_category")
  expect_identical(err$fields$position, 1)
  expect_match(conditionMessage(err), "categories[1]", fixed = TRUE)
  err <- expect_pgp_error(
    relative_pairs_correlation(ped, trait(v), categories = list("MO", PO = c("MO", "FO"))),
    "validation", "duplicate_category"
  )
  expect_identical(err$fields$positions, c(1, 2))
  expect_identical(err$fields$entries, c("MO", "PO"))
  expect_match(conditionMessage(err), "\"MO\" (categories[1]) and \"PO\" (categories[2])", fixed = TRUE)
  err <- expect_pgp_error(
    relative_pairs_correlation(ped, trait(v), categories = list(G = c("FS", "FS"))),
    "validation", "duplicate_category"
  )
  expect_match(conditionMessage(err), "listed twice in group \"G\" (categories[1])", fixed = TRUE)
  err <- expect_pgp_error(
    relative_pairs_correlation(ped, trait(v), categories = list("FS", FS = "MHS")),
    "validation", "reserved_group_name"
  )
  expect_identical(err$fields$position, 2)
  expect_match(conditionMessage(err), "categories[2]", fixed = TRUE)
  err <- expect_pgp_error(
    relative_pairs_correlation(ped, trait(v), categories = list(within_person = "FS")),
    "validation", "reserved_group_name"
  )
  expect_identical(err$fields$group, "within_person")
  expect_pgp_error(
    relative_pairs_correlation(ped, trait(v), categories = list(G = c("MO", "GP"))),
    "validation", "mixed_category_group"
  )
  err <- expect_pgp_error(
    relative_pairs_correlation(ped, trait(v), categories = list(G = character())),
    "validation", "empty_category_group"
  )
  expect_identical(err$fields$position, 1)
  expect_match(conditionMessage(err), "(categories[1]) is empty", fixed = TRUE)
  err <- expect_pgp_error(
    relative_pairs_correlation(ped, trait(v), categories = list(G = "FS", G = "MHS")),
    "validation", "duplicate_entry_name"
  )
  expect_identical(err$fields$positions, c(1, 2))
  expect_match(conditionMessage(err), "categories[1] and categories[2]", fixed = TRUE)
  expect_pgp_error(relative_pairs_correlation(ped, trait(v), memory_budget_bytes = 0), "parameter",
                   "parameter_out_of_range")
  expect_pgp_error(relative_pairs_correlation(ped, trait(v), memory_budget_bytes = 16), "resource",
                   "memory_budget_exceeded")
  expect_pgp_error(relative_pairs_correlation(ped, trait(v), categories = list(c("FS", "MHS"))), "usage")
  expect_pgp_error(relative_pairs_correlation(ped, trait(v), categories = c(PO = "MO")), "usage")
  expect_pgp_error(relative_pairs_correlation(ped, v), "usage")
})

# Six families of a mother, a father and three children.
ped2 <- data.frame(
  id = 1:30,
  mother = rep(c(NA, NA, 1, 1, 1), 6) + rep(0:5 * 5, each = 5),
  father = rep(c(NA, NA, 2, 2, 2), 6) + rep(0:5 * 5, each = 5)
)
v2 <- c(0.034, 1.36, 1.225, -0.51, -0.298, -0.527, 0.57, NA, 0.747, -1.847, 1.567, -0.096, 0.68, -0.137,
        -0.379, 0.463, 0.825, -0.203, -0.153, 0.686, NA, -1.514, 0.395, -0.671, -1.92, -0.814, -0.468,
        -1.193, -1.492, 0.037)
b2 <- c(1, 1, 1, 0, 1, 1, 0, NA, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, NA, 0, 1, 1, 0, 0, 1, 1, 1, 1)
s2 <- c(0, NA, 10, 11, 10, 1, 2, 11, 10, 11, 0, 2, 10, NA, 10, 1, NA, 11, 10, 11, 0, 2, 10, 11, 10, 1, 2,
        11, 10, 11)
cats2 <- list("FS", PO = c("MO", "FO"))

stratum_cells <- function(cell, field) {
  vapply(cell$strata, function(s) if (is.null(s[[field]])) NA_real_ else as.numeric(s[[field]]), 0)
}

test_that("continuous strata give Python's stratum cells, counts and stratified value", {
  # Python: correlation(ped2, Trait(v2), categories=cats2, stratum=s2), as hex floats.
  r <- relative_pairs_correlation(ped2, trait(v2, kind = "continuous"), categories = cats2, stratum = s2)
  fs <- r$cells[[1L]]
  po <- r$cells[[2L]]
  expect_identical(vapply(fs$strata, `[[`, 0, "stratum"), c(0, 1, 2, 10, 11))
  expect_identical(stratum_cells(fs, "n"), c(0, 0, 0, 3, 2))
  expect_identical(stratum_cells(po, "n"), c(0, 0, 0, 16, 13))
  expect_identical(stratum_cells(po, "n_dropped"), c(0, 0, 0, 2, 3))
  pearson <- function(cell) vapply(cell$strata[4:5], function(s) s$crude[[1L]]$value, 0)
  expect_identical(pearson(fs), as.numeric(c("-0x1.d81f961343d8fp-2", "-0x1.0d9c3b9377e2dp-2")))
  expect_identical(pearson(po), as.numeric(c("0x1.a51165dd9fe32p-2", "0x1.1d575299462a0p-2")))
  expect_identical(c(fs$n, fs$n_cross_stratum, fs$n_unknown_stratum), c(16L, 9L, 2L))
  expect_identical(c(po$n, po$n_cross_stratum, po$n_unknown_stratum), c(31L, 0L, 2L))
  expect_identical(c(fs$stratified$n, fs$stratified$n_degenerate_stratum), c(14L, 0L))
  expect_identical(fs$stratified$estimate$value, as.numeric("-0x1.6efb048cfb7d5p-3"))
  expect_identical(c(po$stratified$n, po$stratified$n_degenerate_stratum), c(23L, 0L))
  expect_identical(po$stratified$estimate$value, as.numeric("0x1.bfdbbc96bad35p-3"))
  expect_identical(po$stratified$estimate$estimator, "pearson")
  expect_output(print(r), "stratified, n 23: pearson 0.2187")
})

test_that("binary strata give Python's stratum cells, counts and stratified value", {
  r <- relative_pairs_correlation(ped2, trait(b2, kind = "binary"), categories = cats2, stratum = s2)
  fs <- r$cells[[1L]]
  po <- r$cells[[2L]]
  expect_identical(fs$strata[[4L]]$concordance, c(0L, 2L, 1L))
  expect_identical(po$strata[[5L]]$table, list(c(1L, 3L), c(5L, 4L)))
  tetrachoric <- function(cell) vapply(cell$strata[4:5], function(s) s$crude[[1L]]$value, 0)
  expect_identical(tetrachoric(fs), as.numeric(c("-0x1.fdbe2c6c552bep-1", "-0x1.f91a6899db45ap-1")))
  expect_identical(tetrachoric(po), as.numeric(c("0x1.b19f97bade4c9p-3", "-0x1.d65e03659939ap-2")))
  expect_identical(fs$stratified$estimate$value, as.numeric("-0x1.fc84f4e74c706p-1"))
  expect_true(fs$stratified$estimate$boundary)
  expect_identical(c(po$stratified$n, po$stratified$n_degenerate_stratum), c(18L, 5L))
  expect_identical(po$stratified$estimate$value, as.numeric("0x1.aca8d90b44adbp-5"))
  expect_false(po$stratified$estimate$boundary)
})

test_that("without stratum the stratum fields are empty, and its length is checked", {
  r <- relative_pairs_correlation(ped2, trait(v2), categories = "FS")
  fs <- r$cells[[1L]]
  expect_null(fs$strata)
  expect_null(fs$stratified)
  expect_identical(c(fs$n_cross_stratum, fs$n_unknown_stratum), c(0L, 0L))
  err <- expect_pgp_error(
    relative_pairs_correlation(ped2, trait(v2), categories = "XX", stratum = s2[-1]),
    "validation", "stratum_length_mismatch"
  )
  expect_identical(err$fields$actual_length, 29)
  expect_pgp_error(
    relative_pairs_correlation(ped2, trait(v2[-1]), categories = "XX", stratum = s2[-1]),
    "validation", "trait_length_mismatch"
  )
})

w2 <- c(0.51, NA, 0.92, -0.33, 0.14, -0.81, 0.27, 0.66, NA, -1.12, 0.98, 0.05, 0.43, -0.29, -0.61, 0.38, NA,
        -0.47, 0.12, 0.71, 0.36, -0.95, 0.58, -0.22, -1.31, -0.44, 0.09, -0.73, -1.05, 0.19)
c2 <- c(0, 1, 1, 1, NA, 1, 0, 1, 0, 1, 0, 1, 1, NA, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, NA, 1, 1, 0, 1)

test_that("two continuous traits give Python's cross cells and within_person", {
  # Python: correlation(ped2, [Trait(v2), Trait(w2)], categories=cats2, stratum=s2), as hex floats;
  # R counts traits from 1.
  r <- relative_pairs_correlation(ped2, list(trait(v2), trait(w2)), categories = cats2, stratum = s2)
  key <- vapply(r$cells, function(cell) sprintf("%s %d %d", cell$category, cell$first_trait, cell$second_trait), "")
  expect_identical(key, c("FS 1 1", "FS 2 2", "FS 1 2", "PO 1 1", "PO 1 2", "PO 2 1", "PO 2 2"))
  fs12 <- r$cells[[3L]]
  expect_identical(c(fs12$n, fs12$n_dropped, fs12$n_entries), c(18L, 0L, 33L))
  expect_identical(fs12$crude[[1L]]$value, as.numeric("-0x1.891a1c4f9a85bp-3"))
  expect_identical(c(fs12$stratified$n, fs12$stratified$n_degenerate_stratum), c(16L, 0L))
  expect_identical(fs12$stratified$estimate$value, as.numeric("-0x1.b17ea0c977b49p-3"))
  expect_null(r$cells[[1L]]$n_entries)
  po21 <- r$cells[[6L]]
  expect_identical(c(po21$n, po21$n_dropped), c(31L, 5L))
  expect_identical(po21$crude[[1L]]$value, as.numeric("0x1.0957ac0120fd9p-2"))
  expect_identical(po21$roles, list("offspring", "parent"))
  one <- relative_pairs_correlation(ped2, trait(v2), categories = cats2, stratum = s2)
  expect_identical(r$cells[[1L]], one$cells[[1L]])
  expect_null(one$within_person)
  w <- r$within_person
  expect_identical(w$category, "within_person")
  expect_identical(c(w$first_trait, w$second_trait), c(1L, 2L))
  expect_identical(c(w$n, w$n_dropped, w$n_cross_stratum, w$n_unknown_stratum), c(25L, 5L, 0L, 1L))
  expect_identical(w$crude[[1L]]$value, as.numeric("0x1.dea43ca854085p-1"))
  expect_identical(stratum_cells(w, "n"), c(2, 3, 4, 8, 7))
  expect_identical(c(w$stratified$n, w$stratified$n_degenerate_stratum), c(24L, 0L))
  expect_identical(w$stratified$estimate$value, as.numeric("0x1.e27c6bd067a34p-1"))
  expect_identical(r$settings$quantization_exponents, list(42, 42))
  expect_output(print(r), "within_person 1 x 2, n 25: pearson 0.9348")
})

test_that("two binary traits give Python's cross tables and within_person", {
  traits <- list(trait(b2, kind = "binary"), trait(c2, kind = "binary"))
  r <- relative_pairs_correlation(ped2, traits, categories = cats2, stratum = s2)
  fs12 <- r$cells[[3L]]
  expect_identical(fs12$table, list(c(1L, 7L), c(10L, 12L)))
  expect_null(fs12$concordance)
  expect_identical(fs12$n_entries, 30L)
  expect_identical(vapply(fs12$crude, `[[`, 0, "value"),
                   as.numeric(c("-0x1.16d058651642bp-1", "-0x1.35ab663cda89ep-2")))
  expect_identical(r$cells[[2L]]$concordance, c(1L, 9L, 4L))
  w <- r$within_person
  expect_identical(w$table, list(c(6L, 2L), c(4L, 13L)))
  expect_identical(vapply(w$crude, `[[`, 0, "value"),
                   as.numeric(c("0x1.6c3edbb8240f5p-1", "0x1.f5dc420098bb4p-2")))
  expect_identical(c(w$stratified$n, w$stratified$n_degenerate_stratum), c(19L, 4L))
  expect_identical(w$stratified$estimate$value, as.numeric("0x1.c89b1cb6f0c78p-1"))
  expect_null(r$settings$quantization_exponents)
})

test_that("traits are a trait() or a list of one or two of one kind", {
  a <- relative_pairs_correlation(ped, trait(v), categories = "FS")
  expect_identical(a, relative_pairs_correlation(ped, list(trait(v)), categories = "FS"))
  expect_identical(c(a$cells[[1L]]$first_trait, a$cells[[1L]]$second_trait), c(1L, 1L))
  expect_null(a$within_person)
  expect_pgp_error(relative_pairs_correlation(ped, list(), categories = "FS"), "validation", "trait_count")
  expect_pgp_error(relative_pairs_correlation(ped, list(trait(v), trait(v), trait(v))), "validation", "trait_count")
  err <- expect_pgp_error(
    relative_pairs_correlation(ped, list(trait(v), trait(b, kind = "binary"))),
    "validation", "unsupported_trait_kind"
  )
  expect_identical(err$fields$kinds, c("continuous", "binary"))
  expect_identical(
    conditionMessage(err),
    "the traits are continuous and binary: pass two traits of one kind [unsupported_trait_kind]"
  )
  expect_pgp_error(relative_pairs_correlation(ped, list(trait(v), v)), "usage")
})

