#' Package version
#'
#' The version of the compiled Rust binding, which equals the package version.
#'
#' @return A string such as `"0.1.0"`.
#' @export
#' @examples
#' pgphenotype_version()
pgphenotype_version <- function() .Call(wrap__pgphenotype_version)

#' Linked pedigree-graph revision
#'
#' The git revision of pedigree-graph-core that this build links for pedigree
#' validation and kinship.
#'
#' @return A 40-character hexadecimal commit hash.
#' @export
#' @examples
#' pg_core_rev()
pg_core_rev <- function() .Call(wrap__pg_core_rev)
