# The Python package's results on the same inputs (tools/r_parity.py).  Both
# run the same Rust core, so scores must agree bit for bit.

pedigrees <- list(small = read_fixture("pedigree_small.csv"), deep = read_fixture("pedigree_deep.csv"))
# The deep pedigree's missing parents become NA, the small one keeps -1.
for (col in c("mother", "father", "twin")) pedigrees$deep[[col]][pedigrees$deep[[col]] == -1L] <- NA
cips <- lapply(1:2, function(k) {
  table <- read_fixture(paste0("cip", k, ".csv"))
  pafgrs_cip(table$ages, table$cip)
})
cases <- read_fixture("cases.csv")

fixture_inputs <- function(name) {
  t <- read_fixture(paste0("traits_", name, ".csv"))
  list(traits = lapply(1:2, function(k) trait(t[[paste0("trait", k)]])),
       ages = lapply(1:2, function(k) t[[paste0("age", k)]]))
}
inputs <- lapply(c(small = "small", deep = "deep"), fixture_inputs)

# `as_pedigree` is identity for the columns path, or pedigree().
score_case <- function(case, as_pedigree = identity) {
  probands <- if (is.na(case$probands) || case$probands == "") NULL else
    as.integer(strsplit(case$probands, " ", fixed = TRUE)[[1L]])
  prep <- pafgrs_prepare(as_pedigree(pedigrees[[case$pedigree]]), ndegree = case$ndegree, probands = probands)
  x <- inputs[[case$pedigree]]
  if (case$kind == "uni") {
    return(pafgrs_score_univariate(prep, x$traits[[1L]], x$ages[[1L]], cips[[1L]], h2 = case$h2_1))
  }
  rho <- if (is.na(case$rho_within)) NULL else case$rho_within
  pafgrs_score_bivariate(prep, x$traits, x$ages, cips, h2 = c(case$h2_1, case$h2_2), rg = case$rg,
                         rho_within = rho)
}

test_that("scores equal the Python package's on every fixture case", {
  expect_identical(nrow(cases), 10L)
  for (i in seq_len(nrow(cases))) {
    case <- cases[i, ]
    got <- score_case(case)
    expect_identical(score_case(case, pedigree), got, info = case$case)
    want <- read_fixture(paste0("expected_", case$case, ".csv"))
    expect_identical(names(got), names(want), info = case$case)
    for (col in names(want)) {
      expect_identical(got[[col]], want[[col]], info = paste(case$case, col))
    }
    meta <- attr(got, "metadata")
    expect_identical(meta$threshold, as.numeric(strsplit(case$threshold, " ")[[1L]]), info = case$case)
    expect_identical(meta$controls_without_age,
                     as.integer(strsplit(as.character(case$controls_without_age), " ")[[1L]]),
                     info = case$case)
    if (case$kind == "biv") expect_identical(meta$rho_within, case$rho_within_used, info = case$case)
  }
})

# Space-separated fixture tokens, "NA" as a missing value.
tokens <- function(x) {
  out <- if (x == "") character() else strsplit(x, " ", fixed = TRUE)[[1L]]
  out[out == "NA"] <- NA_character_
  out
}

trait_input <- function(type, values) {
  x <- tokens(values)
  switch(type,
    logical = as.logical(x),
    integer = as.integer(x),
    double = as.numeric(x),
    character = x
  )
}

test_that("trait() makes of each input what the Python package's Trait does", {
  trait_cases <- utils::read.csv(test_path("fixtures", "trait_cases.csv"), colClasses = "character",
                                 na.strings = character())
  expect_identical(nrow(trait_cases), 15L)
  for (i in seq_len(nrow(trait_cases))) {
    case <- trait_cases[i, ]
    kind <- if (case$kind == "NA") NULL else case$kind
    input <- trait_input(case$type, case$values)
    if (case$code != "NA") {
      err <- expect_pgp_error(trait(input, kind), "validation", case$code)
      expect_identical(sort(names(err$fields), method = "radix"), tokens(case$field_names), info = case$case)
      if (case$position != "NA") {
        expect_identical(err$fields$position, as.numeric(case$position) + 1, info = case$case)
        expect_identical(err$fields$value, as.numeric(case$error_value), info = case$case)
      }
      next
    }
    got <- trait(input, kind)
    expect_identical(got$kind, case$out_kind, info = case$case)
    expect_identical(got$values, as.numeric(tokens(case$out_values)), info = case$case)
    levels <- if (case$levels == "NA") NULL else tokens(case$levels)
    expect_identical(got$levels, levels, info = case$case)
  }
})
