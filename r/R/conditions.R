# Failures come back from Rust as data (src/rust/src/errors.rs); this is the
# one place they become R conditions.  Rust never raises into R.

.pgp_signal <- function(class, code, message, fields, call) {
  stop(structure(
    class = c(paste0("pgphenotype_", class, "_error"), "pgphenotype_error", "error", "condition"),
    list(
      message = if (is.na(code)) message else paste0(message, " [", code, "]"),
      call = call,
      code = code,
      fields = fields
    )
  ))
}

.pgp_call <- function(value, call = sys.call(-1L)) {
  if (!inherits(value, "pgphenotype_native_error")) {
    return(value)
  }
  .pgp_signal(value$class, value$code, value$message, value$fields, call)
}

# A usage error raised on the R side, for arguments R checks itself.
.pgp_usage <- function(message, call = sys.call(-1L)) {
  .pgp_signal("usage", NA_character_, message, list(), call)
}

# One number, as the double the binding takes; NA passes for the core to judge.
.pgp_number <- function(x, name, call = sys.call(-1L)) {
  if (!is.numeric(x) || length(x) != 1L) .pgp_usage(sprintf("`%s` must be a single number", name), call)
  as.double(x)
}

# A numeric or logical vector as doubles, NA kept.
.pgp_doubles <- function(x, name, call = sys.call(-1L)) {
  if (!(is.numeric(x) || is.logical(x))) {
    .pgp_usage(sprintf("`%s` must be a numeric or logical vector", name), call)
  }
  as.double(x)
}
