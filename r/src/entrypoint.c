// The routine registration extendr generates, under the name R looks for.
void R_init_pgphenotype_extendr(void *dll);

void R_init_pgphenotype(void *dll) {
    R_init_pgphenotype_extendr(dll);
}
