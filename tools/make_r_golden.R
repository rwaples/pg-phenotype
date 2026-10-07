# Score PA-FGRS problems with BioPsyk/PAFGRS's own pa_fgrs (est_liab.R).
#
#   Rscript tools/make_r_golden.R <PAFGRS checkout> <problems.txt> <out.txt>
#
# problems.txt: the problem count, then per problem `n thr`, n statuses,
# n weights, and the (n+1)^2 covariance row-major, all whitespace-separated.
# out.txt: one `est var` line per problem, %.17g.
args <- commandArgs(trailingOnly = TRUE)
source(file.path(args[1], "R", "est_liab.R"))
x <- scan(args[2], what = double(), quiet = TRUE)
k <- 1
take <- function(m) { v <- x[k:(k + m - 1)]; k <<- k + m; v }
n_problems <- take(1)
out <- character(n_problems)
for (p in seq_len(n_problems)) {
  head <- take(2); n <- head[1]; thr <- head[2]
  status <- take(n); w <- take(n)
  covmat <- matrix(take((n + 1)^2), n + 1, byrow = TRUE)
  r <- pa_fgrs(rel_status = status, thr = thr, rel_w = w, covmat = covmat)
  out[p] <- sprintf("%.17g %.17g", r[1], r[2])
}
writeLines(out, args[3])
