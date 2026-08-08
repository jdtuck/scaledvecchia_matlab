[![Pipeline Status](https://github.com/jdtuck/scaledvecchia_matlab/actions/workflows/matlab.yml/badge.svg)](https://github.com/jdtuck/scaledvecchia_matlab/actions/workflows/matlab.yml)

# scaledvecchia MATLAB
MATLAB implementation of the scaled Vecchia approximation of Gaussian processes (Katzfuss et al., 2020)

## Quick start

```matlab
addpath('scaled_vecchia')

inputs = rand(2000, 8);
y      = sv_borehole(inputs);

fit  = sv_fit(y, inputs);                       % estimate
pred = sv_predict(fit, rand(500,8), 'joint', false, 'variance', true);

sv_test    % run the correctness checks
sv_demo    % likelihood-accuracy comparison + borehole emulation
```

## What the method does

A Vecchia approximation replaces the joint GP density by an ordered product of
univariate conditionals,

    p(y) ≈ ∏_i p( y_i | y_{g(i)} ),

where `g(i)` is a small set of `m` previously ordered points. Accuracy hinges
entirely on the ordering and on which points land in `g(i)`. In spatial
statistics one uses a maximin ordering and Euclidean nearest neighbours, but
for a computer experiment the input coordinates are not comparable: a run that
is close in an irrelevant input tells you nothing about the response.

The scaled Vecchia idea is to do the ordering and the neighbour search in a
*warped* input space, dividing each input by its estimated range parameter
`λ_k` so that distance measures relevance rather than raw input units. Since
the `λ_k` are themselves unknown, estimation and warping are interleaved:
order and condition using the current ranges, take a few Fisher-scoring steps
on the covariance parameters, rescale, repeat.

The covariance is the anisotropic ("scaled-dimension") Matérn

    K(x,x') = σ² · M_ν( sqrt( Σ_k (x_k − x'_k)² / λ_k² ) ) + σ²τ²·1{x = x'},

The smoothness ν can be fixed — 0.5, 1.5, 2.5 and 3.5 have closed forms and
need no Bessel calls — or estimated along with everything else. The nugget `τ²`
is relative to the variance. Mean coefficients are profiled out by GLS inside
the approximation.

The parameter vector is `[variance, range_1..range_d, smoothness, nugget]`,
matching GpGp's `matern_scaledim` ordering.

## Files

| file | role |
|---|---|
| `sv_fit.m` | main entry point: interleaved scaling and Fisher scoring |
| `sv_predict.m` | joint prediction and conditional simulation, or pointwise prediction with variances |
| `sv_loglik.m` | Vecchia loglikelihood, gradient, expected Fisher information, profiled GLS mean |
| `sv_fisher.m` | Fisher-scoring loop with ridge and backtracking line search |
| `sv_maxmin_order.m`, `sv_nn.m` | maximin ordering and ordered nearest-neighbour conditioning sets |
| `sv_covblocks.m`, `sv_covderiv.m`, `sv_matern.m` | batched covariance blocks and their derivatives |
| `sv_bchol.m`, `sv_bfsolve.m`, `sv_blastrow.m` | batched small-matrix linear algebra |
| `sv_options.m` | name/value option parsing |
| `sv_test.m` | correctness checks against brute-force GP computations |
| `sv_demo.m`, `sv_borehole.m` | demonstrations and the borehole test function |

## Key options

`sv_fit(y, inputs, 'name', value, ...)`

| option | default | meaning |
|---|---|---|
| `m` | 30 | conditioning-set size |
| `nu` | 3.5 | Matérn smoothness. A number fixes it; `'estimate'` estimates it |
| `nu_bounds` | `[0.25 6]` | box constraint applied when `nu` is estimated |
| `nugget` | 0 | relative nugget; a number fixes it, `'estimate'` estimates it |
| `trend` | `'pre'` | `'pre'`, `'zero'`, `'intercept'`, `'linear'`; or pass `X` directly |
| `scale` | `'parms'` | `'parms'` = scaled Vecchia, `'ranges'` = plain Vecchia on `[0,1]^d`, `'none'` |
| `n_est` | `min(5e3, n)` | subsample size used for parameter estimation |
| `select` | `Inf` | freeze an input once its range exceeds `select` × its observed spread |
| `vcf` | `true` | fit a held-out variance correction factor for the predictive variances |

**The default nugget is 0**, matching the reference implementation, because the
target application is a deterministic computer model. For noisy data pass
`'nugget','estimate'`.

`sv_predict(fit, inputs_pred, 'name', value, ...)`

| option | default | meaning |
|---|---|---|
| `m` | 100 | conditioning-set size for prediction |
| `joint` | `true` | joint Vecchia over observed + prediction points; needed for simulation |
| `nsims` | 0 | number of joint conditional simulations |
| `variance` | `false` | return pointwise variances (use with `joint=false` for exact ones) |

## Implementation notes

**Vectorization instead of mex.** The likelihood is a sum of `n` independent
`(m+1)`-dimensional Gaussian conditionals. Rather than looping over
observations, the blocks are stacked into `n × p × p` arrays and the Cholesky
factorization, triangular solves and back-substitution loop over the `p`
columns, so every elementary operation is vectorized across all observations at
once. Memory is bounded by chunking.

**Derivatives.** Writing `L` for the Cholesky factor of the block with the
target variable ordered last, `s = e_p'inv(L)`, `u = inv(L)r` and
`z = inv(L)(∂Σ · s')`, the score is

    ∂ℓ_i/∂θ_j = −½ z_p (1 + u_p²) + u_p (z'u),

so only the last row of `inv(L)` is needed and each parameter costs `O(p²)`
per observation rather than `O(p³)`. Because the whitened residuals are exactly
iid standard normal *under the Vecchia model*, the expected Fisher information
has the closed form

    I_jk = Σ_i ( z^j'z^k − ½ z^j_p z^k_p ),

which `sv_test` verifies against the usual trace-difference expression.

**Estimating the smoothness.** When `nu` is fixed at a half-integer the Matérn
is a polynomial times an exponential. Otherwise the general form

    M_ν(r) = 2^(1−ν)/Γ(ν) · r^ν · K_ν(r)

is evaluated in the log domain using the exponentially scaled Bessel function,
so neither the `r^ν` blow-up at small `r` nor the `exp(−r)` decay at large `r`
overflows. The range derivative still has a closed form, via the identity
`d/dx[x^ν K_ν(x)] = −x^ν K_{ν−1}(x)`. The derivative with respect to ν does
not, and is obtained by central differences in ν, as GpGp does; `h = 1e-5` is
near the optimum for this function, giving about `1e-10` absolute accuracy.
Since the Bessel evaluation dominates the cost in this branch, only the strict
upper triangle of each block is evaluated and mirrored.

Two practical consequences. Estimating ν costs roughly 3–5× more per iteration
than fixing it. And ν is bounded away from 0 and capped (default 6) because
very large smoothness makes the covariance blocks numerically singular — if
your fit lands on the upper bound, the response is smoother than the Matérn
family can usefully express and you should just fix ν at 3.5.

**Joint prediction.** The same `s` vector is a column of the sparse Vecchia
factor `U` with `inv(Σ) ≈ UU'`. Ordering the observed runs first makes `U`
block upper triangular, so

    y* | y ~ N( −inv(U_**')·U_o*'·r ,  inv(U_** U_**') ),

and both the posterior mean and exact joint samples come from sparse triangular
solves — `O(n_pred · m²)`. This is algebraically the same thing the reference R
code computes with its sequential loop, but vectorized.

## Test results

`sv_test` checks the implementation against brute-force GP computations:

```
[PASS] full-conditioning loglik equals exact                 vecchia -35.0025738753  exact -35.0025738753
[PASS] profiled beta equals exact GLS                        max diff 1.765e-14
[PASS] m=10 Vecchia loglik is finite and close to exact      m=10 -35.9468  exact -35.0026
[PASS] gradient matches finite differences                   max rel diff 3.264e-08
[PASS] Fisher information matches analytic formula           rel Frobenius diff 4.733e-09
[PASS] Fisher information rows for the smoothness agree      rel diff 1.803e-09
[PASS] Fisher information for log-variance equals n/2        I_11 = 30.000000, n/2 = 30.0
[PASS] joint prediction mean equals exact kriging            max diff 2.087e-14
[PASS] pointwise prediction mean equals exact kriging        max diff 2.354e-14
[PASS] pointwise prediction variance equals exact            max diff 1.318e-15
[PASS] joint simulation reproduces the predictive covariance  rel Frobenius diff 0.032
[PASS] Bessel Matern matches the half-integer closed forms   max diff 3.430e-11 (abs on f, rel on g)
[PASS] smoothness score matches finite differences           analytic 7.81287  fd 7.81287
[PASS] estimated smoothness recovers the truth               nu_hat 1.485 (true 1.5), ranges [0.272 0.527]
```

The last three cover the smoothness: the general Bessel form reproduces the
half-integer closed forms to `3e-11`, its score matches finite differences, and
fitting data simulated with ν = 1.5 recovers ν̂ = 1.485.

`sv_demo` part 1, with data simulated from a 5-input GP whose true ranges are
`[0.1 0.2 1 10 100]`, shows why the scaling matters — same `m`, same cost, very
different approximation quality:

```
exact loglikelihood      3823.76

    m  ordering           loglik    KL to exact
   10  scaled            3042.78         780.98
   10  unscaled          1471.39        2352.37
   20  scaled            3521.82         301.94
   20  unscaled          2066.35        1757.41
   40  scaled            3733.36          90.40
   40  unscaled          2617.23        1206.52
```

Part 2 emulates the borehole function from `n = 1500` runs, recovering a range
of 1.36 for the dominant input `r_w` against 10²–10⁴ for the inert ones, with
an RMSE around 0.08 on a response with standard deviation 46. Part 3 refits
with the smoothness free and estimates ν̂ ≈ 3.0, consistent with a very smooth
deterministic response.

## Scaling to large n

The two `O(n²)` pieces are `sv_maxmin_order` and `sv_nn`. They are exact,
blocked and vectorized, which is comfortable to roughly `n = 10⁴`. Beyond that:

- estimation already defaults to a subsample of 5000 runs (`n_est`), following
  the reference implementation, so fitting stays cheap regardless of `n`;
- for prediction, swap the brute-force neighbour search in `sv_nn` and the
  helper `knn_ref` inside `sv_predict.m` for `knnsearch` (Statistics Toolbox)
  or any k-d tree — the definition of the output is unchanged;
- an approximate maximin ordering can replace `sv_maxmin_order` for the same
  reason.

The likelihood, gradient, Fisher information and prediction machinery are all
already linear in `n`.

## Differences from the reference R code

- Ordering and neighbour search are exact brute force here, rather than the
  approximate k-d-tree searches in `GPvecchia`/`FNN`. Results are the same or
  slightly better; the cost is the `O(n²)` note above.
- The variance correction factor is obtained in closed form (the mean
  standardized squared error on held-out runs), which is the exact minimizer of
  the held-out log score that the R code finds by line search.
- Joint prediction uses one sparse triangular solve instead of a sequential
  per-point loop.

## References

Katzfuss, M., Guinness, J., & Lawrence, E. (2020). Scaled Vecchia approximation for fast computer-model emulation. arXiv:2005.00386v4.