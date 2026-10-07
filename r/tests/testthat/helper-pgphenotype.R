expect_pgp_error <- function(expr, class, code = NULL) {
  err <- expect_error(expr, class = paste0("pgphenotype_", class, "_error"))
  if (!is.null(code)) expect_identical(err$code, code)
  invisible(err)
}

# A fixture CSV with hex floats (tools/r_parity.py) read back exactly.
read_fixture <- function(name) {
  raw <- utils::read.csv(test_path("fixtures", name), colClasses = "character", na.strings = "NA")
  as.data.frame(lapply(raw, function(col) {
    hex <- any(grepl("0x", col, fixed = TRUE)) && !any(grepl(" ", col, fixed = TRUE))
    if (hex) as.numeric(col) else utils::type.convert(col, as.is = TRUE)
  }))
}

small_pedigree <- function() {
  data.frame(
    id = c(10L, 11L, 12L, 13L, 14L, 15L, 16L),
    mother = c(NA, NA, 10L, 10L, NA, 12L, 12L),
    father = c(NA, NA, 11L, 11L, NA, 14L, 14L)
  )
}

small_cip <- function() pafgrs_cip(c(0, 40, 80), c(0, 0.05, 0.1))

# A binary trait's statuses and ages on n rows.
small_case <- function(n = 7L) {
  list(trait = trait(rep_len(c(1, 0, 0, NA), n)), age = rep_len(c(50, 60, NA, 30), n))
}

score_small <- function(prep, case = small_case(), h2 = 0.4) {
  pafgrs_score_univariate(prep, case$trait, case$age, small_cip(), h2)
}

score_small_pair <- function(prep, case = small_case(), ...) {
  pafgrs_score_bivariate(prep, list(case$trait, case$trait), list(case$age, case$age),
                         list(small_cip(), small_cip()), ...)
}

# bit64::integer64 without bit64: a double vector holding each id's int64 bits.
as_integer64 <- function(x) {
  bits <- vapply(x, function(v) readBin(writeBin(c(v, 0L), raw(), endian = "little"), "double",
                                        endian = "little"), double(1))
  structure(bits, class = "integer64")
}
