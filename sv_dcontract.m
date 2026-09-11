function T = sv_dcontract(coords, aux, s, parms, aidx, jitter)
%SV_DCONTRACT  Contract every parameter derivative against a vector, at once.
%
%   T = SV_DCONTRACT(COORDS,AUX,S,PARMS,AIDX,JITTER) returns an nb x p x nact
%   array whose page j is
%       ( d Sigma / d log PARMS(AIDX(j)) ) * S
%   for the batched covariance built by SV_COVBLOCKS.
%
%   The gradient and Fisher information only ever need the derivative matrices
%   contracted against a vector, never the matrices themselves.  Forming them
%   explicitly costs one full nb x p x p array per active parameter — with d
%   inputs that is d+2 arrays built, written and read back for every likelihood
%   evaluation, and it measured as the single largest cost in fitting.
%
%   For a range parameter the derivative is
%       dSigma/d(lambda_k) = -g .* (x_a - x_b)^2 / lambda_k^3 ,
%   and expanding the square makes the contraction separable:
%
%       [dSigma*s](a) = -1/lambda_k^2 * [ x_a^2 (g s)(a)
%                                         - 2 x_a (g (x s))(a)
%                                         + (g (x^2 s))(a) ] ,
%
%   (the extra factor of lambda_k coming from differentiating in the log).
%   Every term is a matrix-vector product against the *same* g, so stacking
%   the 1 + 2d right-hand sides into a single batched matrix product gets all
%   d range derivatives from one pass over g instead of 2d passes, and hands
%   the arithmetic to BLAS.
%
%   The variance and nugget derivatives are multiples of f and the identity, so
%   they cost one more product and a scaling.  The smoothness derivative has no
%   closed form and still needs its cached finite-difference array.

[nb, p, d] = size(coords);
nact = numel(aidx);

sig2   = parms(1);
ranges = parms(2:d+1);
nu     = parms(d+2);
nug    = parms(d+3);

T = zeros(nb, p, nact);

want_range = false(1, d);
for j = 1:nact
    k = aidx(j) - 1;
    if k >= 1 && k <= d, want_range(k) = true; end
end

% ---- one batched product against g covers every range parameter ----------
if any(want_range)
    ks = find(want_range);
    nk = numel(ks);
    RHS = zeros(nb, p, 1 + 2*nk);
    RHS(:,:,1) = s;
    for a = 1:nk
        xk = coords(:,:,ks(a));
        RHS(:,:,1+2*a-1) = xk .* s;
        RHS(:,:,1+2*a)   = (xk.^2) .* s;
    end
    Mg = sv_bmatvec(aux.g, RHS);
end

for j = 1:nact
    jp = aidx(j);

    if jp == 1                                    % variance
        t = sig2 * (sv_bmatvec(aux.f, s) + (nug + jitter) * s);

    elseif jp <= d+1                              % range k
        k = jp - 1;
        a = find(ks == k, 1);
        xk = coords(:,:,k);
        t = -( (xk.^2) .* Mg(:,:,1) ...
               - 2*xk .* Mg(:,:,1+2*a-1) ...
               + Mg(:,:,1+2*a) ) / ranges(k)^2;

    elseif jp == d+2                              % smoothness
        if ~isfield(aux, 'dfdnu')
            error('sv_dcontract:dnu', ...
                'smoothness derivative requested but not cached.');
        end
        t = nu * sig2 * sv_bmatvec(aux.dfdnu, s);

    elseif jp == d+3                              % nugget
        t = (sig2 * nug) * s;

    else
        error('sv_dcontract:index', 'parameter index out of range');
    end

    T(:,:,j) = t;
end
end

% -------------------------------------------------------------------------
function Y = sv_bmatvec(A, B)
%SV_BMATVEC  Batched product: page i of Y is A(i,:,:) * B(i,:,:).
%
%   A is nb x p x p, B is nb x p x q (or nb x p for q = 1).

if ndims(B) == 2
    B = reshape(B, [size(B,1) size(B,2) 1]);
end
[nb, p, ~] = size(A);
q = size(B, 3);

persistent has_page
if isempty(has_page)
    has_page = ~isempty(which('pagemtimes'));
end

if has_page
    Y = permute(pagemtimes(permute(A, [2 3 1]), permute(B, [2 3 1])), [3 1 2]);
    return
end

% Without pagemtimes, contracting with a reshape-and-sum still beats forming
% the derivative arrays, because it reads A once for all q right-hand sides.
Y = zeros(nb, p, q);
for c = 1:q
    Y(:,:,c) = sum(A .* reshape(B(:,:,c), [nb 1 p]), 3);
end
end
