# survfit's Aalen-Johansen CIF of cause 1 at each case age, per case (issue #4).
#
#   Rscript tools/survival_aj.R <dir>
#
# <dir> holds one <i>.csv per case (columns entry, age, status as C99 hex
# floats; entry empty for no late entry) and writes <i>.out: the case ages
# and survfit's cause-1 probability at each, in hex.  Driven by
# tools/make_survival_aj_golden.py.
suppressPackageStartupMessages(library(survival))
dir <- commandArgs(trailingOnly = TRUE)[1]
for (f in list.files(dir, pattern = "\\.csv$", full.names = TRUE)) {
  d <- read.csv(f, colClasses = "character")
  age <- as.numeric(d$age)
  status <- factor(as.integer(d$status), levels = 0:2, labels = c("censored", "case", "death"))
  if (all(d$entry == "")) {
    fit <- survfit(Surv(age, status) ~ 1, timefix = FALSE)
  } else {
    entry <- as.numeric(d$entry)
    # Surv needs entry < age; a row censored at entry is at risk nowhere.
    keep <- age > entry
    fit <- survfit(Surv(entry[keep], age[keep], status[keep]) ~ 1, id = seq_len(sum(keep)), timefix = FALSE)
  }
  grid <- sort(unique(age[status == "case"]))
  cif <- fit$pstate[match(grid, fit$time), match("case", fit$states)]
  writeLines(sprintf("%a,%a", grid, cif), sub("\\.csv$", ".out", f))
}
cat(sprintf("R %s, survival %s\n", getRversion(), packageVersion("survival")))
