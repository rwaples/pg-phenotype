# Relative-Pair Correlation: standard errors when Relative Pairs overlap

Research note for pg-phenotype's planned Relative-Pair Correlation (issue #1, ADR 0009) (per relationship category; intraclass
double-entry for symmetric categories; two-step tetrachoric + phi for binary traits). Date: 2026-10-10.

Scope: SE methods for familial/intraclass correlations from pedigrees with overlapping pairs; dyadic and
multiway clustering; cluster choice per category; tetrachoric SE in family data; streaming-sum feasibility.

Verification legend. **[read]** = I read the primary text (paper PDF, manual, or source code) in this session.
**[abstract]** = only the abstract (PubMed/Crossref/Semantic Scholar) was read. **[meta]** = only the
bibliographic record was checked (Crossref); the content claims attributed to it are UNVERIFIED.
**[derived]** = my own algebra, checked numerically where stated (scratch scripts,
not kept). Every bibliographic field below was checked against
Crossref or PubMed in this session unless marked otherwise.

---

## Bottom line

1. **No finite cluster contains every dependence in a connected pedigree.** Under a polygenic model, two
   relative pairs have correlated influence functions whenever their members are related. The covariance
   shrinks with the products of the correlations involved. The question is which terms are large enough to
   matter, not where the dependence stops.
2. **A shared person gives a first-order term; disjoint pairs give a second-order term.** For a centred,
   standardised normal trait with the Pearson influence function (IF) φ = u_a u_b − (r/2)(u_a² + u_b²):
   - two pairs sharing member a, with corr(b, c) = g: Cov = g − r²/2 − 2r²g + r⁴ + r²g²/2. This is linear in g.
   - two disjoint pairs with every cross-correlation equal to ρ': Cov = 2ρ'²(1 − r)². This is quadratic.

   [derived; Monte Carlo agreement to 3 decimals at r = 0.25 and r = 0.5, `isserlis.py`]
3. **Sibling categories (MZ, FS, MHS, PHS).** Clustering by the parents' **Mate Network** is a partition that
   contains every same-person dependence and every same-sibship dependence. It nests dyadic clustering. It
   omits only second-order cousin-sibship terms.
4. **Lineal and collateral categories (MO, FO, GP, Av, 1C).** A person appears in pairs that belong to
   different Mate Networks: as the child in one pair and the parent in another, or on the maternal and the
   paternal side. Clustering by the Mate Network of the most recent common ancestor (MRCA) couple therefore
   misses first-order terms. Example at h² = 0.5: the term linking (c, m) with (m, gm) is about 9% of a
   pair's variance per link, the same order as the within-mother term it does capture (about 22%). Use the
   **union of MRCA-Mate-Network clustering and person-level (dyadic) clustering**. It is computed by
   inclusion–exclusion (CGM 2011; ASA 2015). [derived]
5. **The meat can be streamed.** The meat needs per-key sums, not per-pair data. For Pearson, each key needs
   4 numbers (symmetric case) or 6 (asymmetric case). For the two-step tetrachoric, each key needs 3–4 cell
   counts, because the IF depends only on the cell. The keys are cluster g, person i, and (i, g). The meat is
   a quadratic form in those sums, evaluated after the global estimates are known.
6. **Double entry trap.** In the symmetrised table, the two entries of one unordered pair are perfectly
   dependent. The current "every pair its own cluster" tetrachoric SE must treat the **unordered pair** as the
   minimum cluster. If it treats each entry as its own cluster, the variance is too small by a factor of 2
   (the SE by √2). [derived]
7. **The literature on this exact design is thin.** I found no published precedent for a double-entry
   tetrachoric with a pair-clustered or dyadic sandwich on one giant pedigree. FCOR (S.A.G.E.) is the
   closest established tool. It uses a model-based plug-in variance with pedigrees as independent units,
   which needs many pedigrees. Validate any choice against the empirical SD across simACE replicates.

---

## Q1. Established SE methods for familial / intraclass correlations

### S.A.G.E. FCOR: Keen & Elston (2003) and Mathew, Song & Elston (2011)

Citations:
- Keen KJ, Elston RC. "Robust asymptotic sampling theory for correlations in pedigrees." *Statistics in
  Medicine* 2003;22(20):3229–3247. doi:10.1002/sim.1559. PMID 14518025. [abstract]
- Mathew G, Song Y, Elston R. "Interval Estimation of Familial Correlations from Pedigrees." *Statistical
  Applications in Genetics and Molecular Biology* 2011;10(1). doi:10.2202/1544-6115.1573. The article number
  is not verified. The FCOR manual spells the first author "Matthew"; Crossref gives "Mathew". [abstract]

What the Keen & Elston abstract says:
- Earlier asymptotic theory for correlations "among different members of nuclear families … is applicable
  only if the nuclear families are independent."
- They "derive the asymptotic sampling distribution of correlations … among arbitrary pairs of members in
  extended families for the Pearson product-moment estimator with generalized weights … without assuming
  normality."

What the Mathew et al. abstract says:
- It defines the two extreme weights. **Pair-wise** weights give each pair the same weight. **Uniform**
  weights give each pedigree the same weight.
- It derives a "best weighted average" of the two, with its SE, from quadratic models.

FCOR manual [read; `elstonsage/sageDoc@6f3ebad:user/SAGE_Chap_FCOR.lyx`]:
- The estimator is a weighted Pearson, r_xy = Σ wᵢ(xᵢ−x̄)(yᵢ−ȳ) / √(Σ wᵢ(xᵢ−x̄)² Σ wᵢ(yᵢ−ȳ)²) (unnumbered).
- The pair vectors "are not assumed to be independent … the structure of the pairwise correlations among
  them is known via the pedigree structure."
- Worked example (manual lines 244–252): grandmother–grandson pairs sharing a grandmother "for which an
  accounting must be made". "Moreover, a sibling correlation will also need to be accommodated", and so will
  a cousin correlation.
- SE (line 301): "a second-order Taylor series expansion and replacing all correlation parameters with
  their respective estimates."
- If a required correlation is not estimable, it is replaced by zero ("robust" mode) or the SE is suppressed
  ("conservative" mode).
- Restriction (line 77): "pedigrees with no marriage rings or loops."

FCOR source [read; `elstonsage/sageCore@b1b19b9`, `trunk/src/c++/fcor/`]:
- The asymptotic variance sums over **pairs of pairs within the same pedigree only**.
  `subcalM.ipp:89` returns NaN when the two members come from different pedigrees, so pedigrees are the
  independent units.
- For each pair of pairs, it looks up the estimated correlation of the relationship type linking the cross
  members (`subcalM.ipp:106`, `get_corinfo_map()[p1](mem1, mem2)`).
- It combines those lookups as `(s_xx*s_yy + s_xy*s_yx)` (`subcalM.cpp:492`). That is the
  Isserlis/normal-theory form of a cross fourth moment written in terms of correlations.
- My reading [UNVERIFIED against the paper, which I could not access]: FCOR's "distribution-free" claim
  holds for the point estimate. The variance it implements is a **model-based plug-in**: it uses no
  empirical fourth moments, and it needs every cross-relationship correlation.
- Uniform weight is `1/m` per pedigree, where m is the number of pairs of that type in the pedigree
  (`setweight.cpp:94`). With one giant pedigree, uniform weighting has no meaning.

Implication: FCOR's design fits many independent pedigrees. On one giant component the units are
independent only in the limit, and the plug-in sums over all related pairs of pairs. That is O(pairs²) in
the worst case unless it is truncated by relationship degree.

### Donner and colleagues; Karlin; Rosner

- Donner A, Koval JJ. "The estimation of intraclass correlation in the analysis of family data."
  *Biometrics* 1980;36(1):19–25. doi:10.2307/2530491. PMID 7370372. [abstract]
  - They compare the normal-theory MLE of ρ under unequal family sizes with other estimators by simulation.
  - Recommendation: use ML when ρ may be high; use ANOVA for low-to-moderate ρ, dropping families with one
    member.
- Donner A, Koval JJ. "The Large Sample Variance of an Intraclass Correlation." *Biometrika* 1980;67(3):719ff.
  doi:10.2307/2335151. [meta] I did not transcribe the closed-form normal-theory variance under unequal
  sizes (paywalled); the formula is UNVERIFIED here.
- Donner A. "A Review of Inference Procedures for the Intraclass Correlation Coefficient in the One-Way
  Random Effects Model." *International Statistical Review* 1986;54(1):67ff. doi:10.2307/1403259. [meta]
- Karlin S, Cameron EC, Williams PT. "Sibling and parent–offspring correlation estimation with variable
  family size." *PNAS* 1981;78(5):2664–2668. doi:10.1073/pnas.78.5.2664. PMC319417. [abstract]
  - "A method for estimating intrafamilial correlations under variable family sizes involving weightings of
    paired data points."
  - This is the classic pair-weighting family. Its weights run from per-pair to per-family.
- Rosner B, Donner A, Hennekens CH:
  - "Estimation of Interclass Correlation from Familial Data." *Applied Statistics* 1977;26(2):179ff.
    doi:10.2307/2347026. [meta]
  - "Significance Testing of Interclass Correlations from Familial Data." *Biometrics* 1979;35(2):461ff.
    doi:10.2307/2530348. [meta]
  - Their content, including the variable-sibship-size weights for parent–offspring correlations, is UNVERIFIED.
- Eliasziw M, Donner A. "Comparison of Recent Estimators of Interclass Correlation from Familial Data."
  *Biometrics* 1990;46(2):391ff. doi:10.2307/2531444. [meta]

Weighting and SE, in one line. Equal pair weights (the locked choice) is FCOR's PAIR_WISE. It weights a
sibship of size k by k(k−1)/2, so large sibships dominate. That costs efficiency when ρ is high, which is
the direction of Donner & Koval's abstract. It does not cost validity, provided the SE uses the same weights
and clusters correctly.

---

## Q2. Dyadic-robust and multiway clustering

Citations:
- Fafchamps M, Gubert F. "The formation of risk sharing networks." *Journal of Development Economics*
  2007;83(2):326–350. doi:10.1016/j.jdeveco.2006.05.005. [read: GPRG-WPS-037 working paper, Aug 2005, §3.2]
- Aronow PM, Samii C, Assenova VA. "Cluster–Robust Variance Estimation for Dyadic Data." *Political
  Analysis* 2015;23(4):564–577. doi:10.1093/pan/mpv018. [read: arXiv:1312.3398v8]
- Cameron AC, Gelbach JB, Miller DL. "Robust Inference With Multiway Clustering." *Journal of Business &
  Economic Statistics* 2011;29(2):238–249. doi:10.1198/jbes.2010.07136. [read: NBER t0327 (2006). The
  equation numbers below are the working paper's.]
- Tabord-Meehan M. "Inference With Dyadic Data: Asymptotic Behavior of the Dyadic-Robust t-Statistic."
  *JBES* 37(4):671–680. doi:10.1080/07350015.2017.1409630. Crossref gives online 2018. [abstract]: it
  establishes asymptotic normality of the t-statistic with the Fafchamps–Gubert estimator, plus a
  finite-sample correction.

### Dependence assumption

Fafchamps & Gubert (§3.2) allow E[u_ij u_ik] ≠ 0 and the other shared-index terms, "but E[u_ij u_km] = 0
if i ≠ j ≠ k ≠ m."

ASA (p. 4): "by random sampling of units, Cov(ε_d, ε_d′) = 0 for all dyads that do not share a member."

Both assumptions **fail in pedigrees**. Units are not sampled independently; they are relatives.

### Formula

ASA Proposition 1, eq. (3), with S(d) = the dyads sharing a member with d:

    V̂ = (X′X)⁻¹ [ Σ_d X_d X_d′ e_d² + Σ_d Σ_{d′∈S(d)} X_d X_d′′ e_d e_d′ ] (X′X)⁻¹

Proposition 2 is a computational decomposition: V̂ = Σᵢ V̂_C,i − V̂_D − (N−2)V̂₀. Here V̂_C,i clusters all
dyads containing unit i.

For a scalar estimator with per-pair IF φ_d, the same meat is [derived; identity checked numerically,
`check.py`]:

    meat_dyadic = Σ_i S_i² − Σ_d φ_d²,   S_i = Σ_{d∋i} φ_d

This holds because two distinct unordered pairs share at most one person. **It can be negative**: a 4-cycle
of pairs with alternating φ = ±1 gives −4 [derived]. CGM (2006 WP, p. 8) report the same issue for two-way
clustering.

CGM two-way: B̂ = X′(ûû′ .∗ S_GH)X with S_GH = S_G + S_H − S_G∩H (eqs. 2.13–2.15). Each term is an
ordinary one-way cluster-robust meat.

### What dyadic clustering captures and misses in pedigrees

Let ρ be the common sibling correlation. For an exchangeable normal sibship of size k [derived]:
- Var(φ) = (1−ρ²)².
- Each pair shares a member with 2(k−2) other pairs. Covariance per such link: ρ − ρ²/2 − 2ρ³ + 1.5ρ⁴.
- Each pair has C(k−2, 2) disjoint partners. Covariance per such link: 2ρ²(1−ρ)².

At ρ = 0.25 these are 0.22·Var per shared link and 0.08·Var per disjoint link. Dyadic clustering:
- captures everything for k ≤ 3, because no two pairs in a sibship of 3 are disjoint;
- misses about 4% of the sibship variance at k = 4;
- misses about 15% at k = 6.

Dyadic clustering also misses pairs linked only through relatives. Example: MO pairs (c, m) and (c′, m′)
where m and m′ are sisters. Those terms are second order.

---

## Q3. Valid cluster choices per category

### General argument

A partition into clusters g is valid when Σ over cross-cluster pairs of pairs of Cov(φ_d, φ_d′) is
negligible next to the total.

Under the normal-theory (Isserlis) approximation:
- **First-order terms come only from pairs of pairs that share a person.**
- Disjoint pairs contribute products of two correlations.
- Correlations between relatives decay roughly as kinship, that is about 2^(−meioses).
- So any clustering that contains (a) all same-person pairs of pairs and (b) all pairs of pairs within one
  nuclear sibship leaves only second-order terms between more distant relatives.

In a connected pedigree, no partition contains everything. Connected-component clustering gives G = 1. That
is useless: the meat is identically zero after centring, and the G/(G−1) factor is undefined. OpenMx hits
the same problem: its robust SE skips the one-row GREML case ("you'd end up calculating the variance of the
row derivatives for a sample of n=1 row", `OpenMx/OpenMx@a24e868:R/MxRobustSE.R:84`) [read].

### Category mapping

The Mate Network (MN) is the connected component of mating pairs linked by shared mates
(`pg_phenotype/regression.py:5`, `assortative.py:7`).

| Category | Cluster key | Nests dyadic? | What it misses |
|---|---|---|---|
| MZ, FS, MHS, PHS (all sib types) | MN of the shared parent(s). It is the same MN for both parents, because they mated. | **Yes.** All of person i's sib-type pairs share i's parents' MN. | Cousin-sibship terms, 2ρ_1C²(1−ρ)² per link. At h² = 0.5, about 0.5% of Var per link. |
| MO / FO | MN of the parent | **No.** A woman is the parent in MN(her matings) and the child in MN(her parents). | (c, m)–(m, gm) links, first order. At h² = 0.5 (r_PO = 0.25, r_GP = 0.125), about 0.094·Var per link. |
| GP | MN of the grandparent | No | Same-person links across generations |
| Av | MN of the aunt's or uncle's parents | No | The niece's maternal and paternal sides; the aunt acting as a niece |
| 1C | MN of the shared grandparents | No | Maternal and paternal cousins of the same person |

Recommendation [derived; no literature precedent found]:
- For the sib categories, use MN clustering alone.
- For the others, use the **union** relation: same MRCA-MN, or sharing a person. By inclusion–exclusion:

      meat = Σ_g Φ_g² + (Σ_i S_i² − Σ_d φ_d²) − (Σ_{(i,g)} S_{i,g}² − Σ_d φ_d²)

  Here Φ_g = Σ_{d∈g} φ_d and S_{i,g} = Σ_{d∋i, d∈g} φ_d.
- For the sib categories S_{i,g} = S_i, so this formula reduces to Σ_g Φ_g².
- The union meat may be negative, as CGM note.
- For small-sample factors, CGM (p. 8) use a separate c = G/(G−1)·… for each component.

Under H0: ρ = 0 for all relatives, every cross term vanishes and the naive SE is valid. Clustering matters
for intervals at ρ ≠ 0, and it matters more at high h² or c².

### Single giant component: alternatives from the literature

- **Window subsampling / pseudo-independent replicates.** Varin, Reid & Firth §5.1 (below) cover the n = 1
  case. J is estimated only if the data "can be grouped into pseudo-independent replicates" (Heagerty & Lele
  1998; Lumley & Heagerty 1999). Otherwise "estimation of J must be done under model assumptions", which
  means four-dimensional expectations for pairwise likelihood (the FCOR route), or Monte Carlo from the fitted
  model.
- **Network HAC.** Kojevnikov D, Marmer V, Song K. "Limit theorems for network dependent random variables."
  *Journal of Econometrics* 2021;222(2):882–908. doi:10.1016/j.jeconom.2020.05.019 [read: arXiv:1903.01059v6].
  The estimator is V̂ = Σ_s ω(s/b_n) Ω̂(s), where Ω̂(s) sums products of centred terms at network distance s
  (eqs. 4.3–4.6). The bandwidth rule is b_n = c·log n / log(avg.deg ∨ (1+ε)) (eq. 4.7). With genealogical
  distance between pairs as the metric, this is the principled version of "cluster by relatives". Nobody has
  applied it to pedigrees as far as I found (thin).
- Further references:
  - Heagerty PJ, Lele SR. *JASA* 1998;93(443):1099–1111. doi:10.1080/01621459.1998.10473771. [meta]
  - Lumley T, Heagerty P. *JRSS-B* 1999;61(2):459–477. doi:10.1111/1467-9868.00187. [meta]

---

## Q4. Tetrachoric SE in family data

Citation: Varin C, Reid N, Firth D. "An overview of composite likelihood methods." *Statistica Sinica*
2011;21(1):5–42. [read: journal PDF]

**The framework fits directly.** A double-entry tetrachoric with one pooled threshold is a pairwise
(composite) likelihood. Its variance is the inverse **Godambe information**, G = H J⁻¹ H (VRF eq. 2.4).
Section 5.1 gives:
- Ĥ = −(1/n) Σ ∇u(θ̂; yᵢ) or Ĥ = (1/n) ΣΣ u u^T;
- Ĵ = (1/n) Σ u(θ̂; yᵢ) u(θ̂; yᵢ)^T over **independent clusters i**, or the jackknife. "Zhao and Joe (2005)
  use var_jack for … maximum pairwise likelihood estimators with clustered data." Zhao Y, Joe H, *Canadian
  Journal of Statistics* 2005;33(3):335–356, doi:10.1002/cjs.5540330303 [meta].

The two-step version replaces u by the threshold-corrected IF, as pg-phenotype already does
(`crates/core/src/assortative/sandwich.rs:14-53`, `threshold_correction` and `CellInfluence::at`) [read].

What applied studies do:
- **Speed D, Evans DM.** "Estimating disease heritability from complex pedigrees allowing for ascertainment
  and covariates." *AJHG* 2024;111(4):680–690. doi:10.1016/j.ajhg.2024.02.010. [read via PMC11023822]
  - TetraHer maximises a pairwise "approximate joint likelihood" and "computes the variance of the estimate by
    inverting the minus second derivative of the log likelihood". That is the naive Hessian, with no
    sandwich.
  - The tetrachoric comparator came from R polycor, with no SE method stated.
- **Skoglund C, Tiger A, Rück C, Petrovic P, et al.** "Familial risk and heritability of diagnosed borderline
  personality disorder: a register study of the Swedish population." *Molecular Psychiatry* 26:999–1008.
  doi:10.1038/s41380-019-0442-0. Crossref gives online 2019; the author list after 4 is not verified.
  [read via PMC7910208]
  - Pairs were double-entered ("each relative pair was included twice").
  - For the hazard ratios they used a "cluster robust sandwich estimator to adjust for non-independence
    between relative pairs".
  - For tetrachoric correlations and the twin/sibling models they **subsampled one pair per family** ("the
    sibling pair with the shortest time between births").
  - This is the common registry workaround: it restores independence at a cost in power.
- **OpenMx** (Mx tradition): the family is one data row (FIML on the multivariate normal), and families are
  independent. `imxRobustSE` is a sandwich over **row gradients**, so families are the clusters
  (`R/MxRobustSE.R:153-163`) [read]. There is no pair double entry.
- Olsson U. "Maximum likelihood estimation of the polychoric correlation coefficient." *Psychometrika*
  1979;44(4):443–460. doi:10.1007/bf02296207. [meta] This is the two-step/ML polychoric. Its SEs assume
  independent observations (UNVERIFIED detail).
- Falconer DS. *Annals of Human Genetics* 1965;29(1):51–76. doi:10.1111/j.1469-1809.1965.tb00500.x. [meta]
- Other pairwise-likelihood work, by title only [meta]:
  - Renard D, Molenberghs G, Geys H. *Computational Statistics & Data Analysis* 2004;44(4):649–667.
    doi:10.1016/s0167-9473(02)00263-3.
  - Kuk AYC, Nott DJ. *Statistics & Probability Letters* 2000;47(4):329–335. doi:10.1016/s0167-7152(99)00174-1.
  - le Cessie S, van Houwelingen JC. *Applied Statistics* 1994;43(1):95ff. doi:10.2307/2986114.

**I found no precedent for a double-entry tetrachoric with a pair-clustered or dyadic sandwich on a
connected pedigree.** The literature is thin on this point.

---

## Q5. Can the SE be built from streaming per-cluster sums?

**Yes.** The meat for any of the clusterings above is a sum of squares of per-key IF sums. When φ_d is an
affine function of a few per-pair statistics with globally estimated coefficients, per-key statistic sums
are enough [derived].

### Intraclass Pearson (symmetric, double-entered, pooled mean μ and SD s)

Per unordered pair:

    φ_d = u_i u_j − (r/2)(u_i² + u_j²)

with r̂ − r ≈ (1/P) Σ_d φ_d. The mean-estimation term vanishes at first order [derived; IF matches a
finite-difference check, `check.py`]. The asymmetric analogue is the standard correlation influence
function; see Devlin, Gnanadesikan & Kettenring, *Biometrika* 1975;62(3):531–545,
doi:10.1093/biomet/62.3.531 [meta; equation not checked].

Per key, keep n = #pairs, S₊ = Σ(x_i + x_j), S₂ = Σ(x_i² + x_j²), S_× = Σ x_i x_j. Then:

    Φ_key = [S_× − μS₊ + nμ² − (r/2)(S₂ − 2μS₊ + 2nμ²)] / s²
    Var(r̂) ≈ c · Σ_key Φ_key² / P²

**Asymmetric (MO, FO, GP, …):** use separate (μ_x, s_x) and (μ_y, s_y), and keep 6 sums per key: n, Σx,
Σy, Σx², Σy², Σxy.

Phi on 0/1 data is the same thing, with x² = x.

### Two-step tetrachoric

The IF is a function of the cell only. The threshold correction comes from the table margins
(`sandwich.rs:47-53`). So:

    Φ_key = Σ_c n_key(c) · IF(c)

For the symmetrised table, use per-key counts (n₀₀, n₀₁ + n₁₀, n₁₁). The existing formula
√(n/(n−1) Σ n_c IF_c²) is the special case where n_key is a unit vector.

**With double entry, the unordered pair is the minimum key.** Its two entries add, and
IF(0,1) = IF(1,0) under the symmetric table. Treating each entry as its own cluster makes the variance too
small by a factor of 2 [derived].

### Implementation shape

1. Pass 1: compute the global sums the engine already produces, then r̂ (or ρ̂ and the threshold).
2. While pairs stream, accumulate key → small-vector hashmaps for g (Mate Network), i (person), and (i, g).
   Pairs do not need to arrive grouped by key.
3. After the stream, evaluate the quadratic forms with the global coefficients. Equivalently, accumulate
   Σ_key m_key m_key^T as 4×4 or 6×6 matrices, but only when each key is complete when it is added.

Memory per category is O(#persons + #clusters + #(person, cluster)). For person keys at 5M rows that is
about 5M × 4–6 float64 values. Process categories one at a time if needed.

Numerics: accumulate in float64. Shift x by a provisional constant before summing, to avoid cancellation
in S₂ − 2μS₊ + 2nμ².

---

## Suggested validation (simACE)

The ground truth is the empirical SD of r̂ per category across simACE replicates. This is the Monte Carlo
estimate of J that VRF §5.1 suggests when model-based J is hard. Compare naive, dyadic, MN, and MN∪dyadic
SEs against it at several (A, C) settings. Coverage gaps should grow with h² for the PO and collateral
categories if the first-order cross-generation terms above matter.

## What I could not verify

- Full text of Keen & Elston 2003, Mathew et al. 2011, Donner & Koval 1980 (both), Donner 1986, Rosner et
  al. 1977/1979, Eliasziw & Donner 1990, Karlin et al. 1981 (abstract only), and the published Fafchamps &
  Gubert eq. 2.5 (ASA cite it as such; the working paper's text extraction did not include it).
- Whether Keen & Elston's variance is normal-free. The FCOR code computes cross fourth moments from
  correlations, which is exact under normality.
- All [derived] formulas assume a centred, standardised, multivariate-normal trait. Binary traits and the
  tetrachoric IF follow the same first-order and second-order ordering only qualitatively.
