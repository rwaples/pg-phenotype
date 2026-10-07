# Run R code in a child Rscript with the package attached; returns its output
# lines, with a "status" attribute when the child failed.
run_rscript <- function(code, env = character(), timeout = 60) {
  script <- tempfile(fileext = ".R")
  on.exit(unlink(script))
  writeLines(c("library(pgphenotype)", code), script)
  system2(file.path(R.home("bin"), "Rscript"), script, stdout = TRUE, stderr = TRUE, env = env,
          timeout = timeout)
}
