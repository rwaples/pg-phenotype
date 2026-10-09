# assortative_mate_correlation() against the Python package on six pedsum
# goldens (tools/r_parity.py).  Both run the same Rust core, so every leaf of
# the result must agree exactly: counts, reasons and flags, and floats bit for
# bit.

am_cases <- read_fixture("am_cases.csv")

# One row per leaf, as tools/r_parity.py flattens the Python result: list
# positions from 0, names joined by dots, length-2 vectors (a CI, a table
# row) indexed like lists.
flatten <- function(x, path = "") {
  join <- function(key) if (path == "") key else paste0(path, ".", key)
  if (is.null(x)) {
    return(list(list(path = path, value = NULL)))
  }
  if (is.list(x)) {
    keys <- names(x)
    out <- list()
    for (i in seq_along(x)) {
      key <- if (!is.null(keys) && nzchar(keys[i])) join(keys[i]) else sprintf("%s[%d]", path, i - 1L)
      if (identical(key, "settings.threads")) next
      out <- c(out, flatten(x[i][[1L]], key))
    }
    return(out)
  }
  if (length(x) > 1L) {
    return(do.call(c, lapply(seq_along(x), function(i) flatten(x[[i]], sprintf("%s[%d]", path, i - 1L)))))
  }
  list(list(path = path, value = x))
}

run_case <- function(case) {
  data <- read_fixture(paste0("am_", case$name, ".csv"))
  kinds <- strsplit(case$kinds, " ", fixed = TRUE)[[1L]]
  traits <- lapply(seq_along(kinds), function(k) trait(data[[paste0("t", k - 1L)]], kind = kinds[k]))
  stratum <- if (isTRUE(as.logical(case$stratified))) data$stratum else NULL
  assortative_mate_correlation(
    data[c("id", "mother", "father")], traits, stratum = stratum, permutations = case$permutations,
    bootstrap = case$bootstrap, seed = case$seed, min_stratum_networks = case$min_stratum_networks
  )
}

test_that("results equal the Python package's on every golden case, leaf for leaf", {
  expect_identical(nrow(am_cases), 6L)
  for (i in seq_len(nrow(am_cases))) {
    case <- am_cases[i, ]
    got <- run_case(case)
    got$metadata <- NULL
    leaves <- flatten(unclass(got))
    paths <- vapply(leaves, `[[`, "", "path")
    want <- utils::read.csv(test_path("fixtures", paste0("am_expected_", case$name, ".csv")),
                            colClasses = "character", na.strings = character())
    expect_setequal(paths, want$path)
    by_path <- stats::setNames(lapply(leaves, `[[`, "value"), paths)
    for (j in seq_len(nrow(want))) {
      path <- want$path[j]
      value <- by_path[[path]]
      info <- paste(case$name, path)
      switch(want$kind[j],
        null = expect_null(value, label = info),
        bool = expect_identical(value, want$value[j] == "TRUE", label = info),
        int = expect_identical(as.double(value), as.double(want$value[j]), label = info),
        float = expect_identical(value, as.numeric(want$value[j]), label = info),
        str = expect_identical(value, want$value[j], label = info)
      )
    }
  }
})

ped <- data.frame(
  id = 1:12,
  mother = c(rep(NA, 8), 1, 2, 3, 4),
  father = c(rep(NA, 8), 5, 6, 7, 8)
)
x <- trait(c(0.1, 1.2, -0.3, 0.8, 0.3, 1.1, -0.6, 0.2, rep(NA, 4)))

test_that("one trait gives one cell, no within-person block, and the method description", {
  r <- assortative_mate_correlation(ped, x, permutations = 49)
  expect_s3_class(r, "pgphenotype_mate_correlation")
  expect_length(r$cells, 1L)
  expect_null(r$within_person)
  expect_identical(r$cells[[1L]]$crude[[1L]]$estimator, "pearson")
  expect_identical(r$method$permutation_stop_h, 20L)
  expect_identical(r$metadata$pedigree_graph_core_rev, pg_core_rev())
  expect_output(print(r), "4 Mating Pairs")
})

test_that("input errors carry the core's codes", {
  expect_pgp_error(assortative_mate_correlation(ped, list()), "validation", "trait_count")
  expect_pgp_error(assortative_mate_correlation(ped, trait(c(1, 1, 1, NA), kind = "binary")),
                   "validation", "trait_length_mismatch")
  expect_pgp_error(assortative_mate_correlation(ped, trait(rep(c("a", "b"), 6))), "validation",
                   "unsupported_trait_kind")
  expect_pgp_error(assortative_mate_correlation(ped, trait(c(rep(1, 8), rep(NA, 4)), kind = "binary")),
                   "validation", "constant_trait")
  expect_pgp_error(assortative_mate_correlation(ped, trait(rep(NA_real_, 12), kind = "continuous")),
                   "validation", "all_missing_trait")
  sparse <- trait(c(10, 20, 40, rep(NA, 9)), kind = "ordinal")
  expect_pgp_error(assortative_mate_correlation(ped, sparse), "validation", "sparse_ordinal_codes")
  unused <- trait(factor(c(rep(c("lo", "hi"), 4), rep(NA, 4)), levels = c("lo", "mid", "hi"), ordered = TRUE))
  expect_pgp_error(assortative_mate_correlation(ped, unused), "validation", "unused_level")
  err <- expect_pgp_error(assortative_mate_correlation(ped, trait(c(0, 2, rep(1, 10)), kind = "binary")),
                          "validation", "invalid_trait_value")
  expect_identical(err$fields$position, 2)
  expect_pgp_error(assortative_mate_correlation(ped, x, stratum = 1:3), "validation", "stratum_length_mismatch")
  expect_pgp_error(assortative_mate_correlation(ped, x, permutations = -1), "parameter", "parameter_out_of_range")
  expect_pgp_error(assortative_mate_correlation(ped, x, stratum = rep(1, 12), min_stratum_networks = 0),
                   "parameter", "parameter_out_of_range")
  expect_pgp_error(assortative_mate_correlation(ped, x, seed = 0.5), "usage")
  # Outside int64 is a parameter error, as in Python; inside, a whole double is a seed.
  err <- expect_pgp_error(assortative_mate_correlation(ped, x, seed = 2^63), "parameter", "parameter_out_of_range")
  expect_identical(err$fields$name, "seed")
  expect_s3_class(assortative_mate_correlation(ped, x, seed = 2^60, permutations = 9), "pgphenotype_mate_correlation")
  # A count past R's integer range comes back as a double, not wrapped negative.
  wide <- assortative_mate_correlation(ped, x, stratum = rep(1, 12), min_stratum_networks = 3e9)
  expect_identical(wide$settings$min_stratum_networks, 3e9)
  # A stratum label is coerced as an id is, and its error says which value.
  err <- expect_pgp_error(assortative_mate_correlation(ped, x, stratum = c(1.5, rep(1, 11))),
                          "validation", "invalid_integer_value")
  expect_identical(err$fields, list(field = "stratum", position = 1, value = 1.5))
})
