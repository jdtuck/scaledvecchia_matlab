function [C, aux] = sv_covblocks(coords, parms, jitter, want_g, want_dnu)
%SV_COVBLOCKS  Anisotropic ("scaled-dimension") Matern covariance, batched.
%
%   [C,AUX] = SV_COVBLOCKS(COORDS,PARMS,JITTER,WANT_G,WANT_DNU)
%
%   COORDS  nb x p x d array.  Page i holds the p input points of the i-th
%           conditioning block, in d dimensions.
%   PARMS   [variance, range_1..range_d, smoothness, nugget].  The nugget is
%           relative: the diagonal gets variance*(nugget + jitter).
%   C       nb x p x p array of covariance matrices, one per page.
%
%   The covariance is
%       K(x,x') = sigma^2 * M_nu( sqrt( sum_k (x_k-x'_k)^2 / lambda_k^2 ) )
%                 + sigma^2 * tau^2 * 1{x = x'},
%   i.e. GpGp's matern_scaledim family.  The per-input ranges lambda_k are
%   exactly the "relevances" that the scaled Vecchia method uses to warp the
%   input space before ordering and neighbour selection.
%
%   WANT_G requests the range-derivative factor and WANT_DNU the smoothness
%   derivative; both are cached in AUX for SV_COVDERIV.  The smoothness
%   derivative has no convenient closed form and is obtained by central
%   differences in nu, as in GpGp.
%
%   When nu is not one of the four half-integers the Matern evaluation goes
%   through Bessel functions, which dominates the cost.  In that case only the
%   strict upper triangle of each block is evaluated and mirrored, halving the
%   number of Bessel calls.

if nargin < 4, want_g   = true;  end
if nargin < 5, want_dnu = false; end

[nb, p, d] = size(coords);
sig2   = parms(1);
ranges = parms(2:d+1);
nu     = parms(d+2);
nug    = parms(d+3);

D2 = zeros(nb, p, p);
for k = 1:d
    xk = coords(:,:,k) / ranges(k);
    dk = reshape(xk, [nb p 1]) - reshape(xk, [nb 1 p]);
    D2 = D2 + dk.^2;
end
r = sqrt(D2);

fast = any(nu == [0.5 1.5 2.5 3.5]);
if fast
    if want_g
        [f, g] = sv_matern(r, nu);
    else
        f = sv_matern(r, nu);
        g = [];
    end
else
    [f, g] = matern_sym(r, nu, nb, p, want_g);
end

C = sig2 * f;
for i = 1:p
    C(:,i,i) = C(:,i,i) + sig2 * (nug + jitter);
end

aux.f = f;
if want_g
    aux.g = sig2 * g;              % sigma^2 * M'(r)/r
end
if want_dnu
    h = 1e-5;
    fp = matern_sym(r, nu + h, nb, p, false);
    fm = matern_sym(r, nu - h, nb, p, false);
    aux.dfdnu = (fp - fm) / (2*h);
end
end

% -------------------------------------------------------------------------
function [f, g] = matern_sym(r, nu, nb, p, want_g)
% Evaluate the Matern correlation on the strict upper triangle of each page
% and mirror it.  M(0) = 1 on the diagonal; the diagonal of g is irrelevant
% because the range derivative multiplies it by a squared difference of zero.
persistent cache_p cache_ut cache_lt
if isempty(cache_p) || cache_p ~= p
    ut = find(triu(true(p), 1));
    [ii, jj] = ind2sub([p p], ut);
    cache_ut = ut;
    cache_lt = sub2ind([p p], jj, ii);
    cache_p  = p;
end
ut = cache_ut;  lt = cache_lt;

R = reshape(r, [nb p*p]);
F = ones(nb, p*p);
if want_g
    [fu, gu] = sv_matern(R(:,ut), nu);
    G = zeros(nb, p*p);
    G(:,ut) = gu;  G(:,lt) = gu;
    g = reshape(G, [nb p p]);
else
    fu = sv_matern(R(:,ut), nu);
    g = [];
end
F(:,ut) = fu;  F(:,lt) = fu;
f = reshape(F, [nb p p]);
end
