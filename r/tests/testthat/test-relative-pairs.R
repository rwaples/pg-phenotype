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
  expect_identical(r$settings$quantization_exponent, 42)
  expect_identical(r$settings$quantization_step, 2^-42)
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
