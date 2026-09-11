function D2 = sv_sqdist(coords, ranges)
%SV_SQDIST  Scaled squared distances within each block, batched.
%
%   D2 = SV_SQDIST(COORDS,RANGES) takes an nb x p x d array of block
%   coordinates and returns the nb x p x p array with
%       D2(i,a,b) = sum_k ( COORDS(i,a,k) - COORDS(i,b,k) )^2 / RANGES(k)^2 .
%
%   The obvious implementation loops over the d inputs forming a full
%   nb x p x p difference array each time.  That is memory-bound: it writes and
%   re-reads d arrays of nb*p^2 doubles, which at the block sizes used for
%   prediction (p = m+1 around 100) is hundreds of megabytes of traffic per
%   chunk.  Expanding the square instead,
%
%       |x_a - x_b|^2 = |x_a|^2 + |x_b|^2 - 2 x_a'x_b ,
%
%   turns the same arithmetic into one small matrix product per block, with
%   only O(nb*p*d) input traffic, and hands it to BLAS.  This is several times
%   faster once p is large, and never slower.
%
%   PAGEMTIMES (R2020b onward) does the whole batch in one call; without it we
%   loop over blocks, which is still far better than the broadcast form.

[nb, p, d] = size(coords);

X  = coords ./ reshape(ranges, [1 1 d]);      % nb x p x d, scaled
sq = sum(X.^2, 3);                            % nb x p

Xp = permute(X, [2 3 1]);                     % p x d x nb

persistent has_page
if isempty(has_page)
    has_page = ~isempty(which('pagemtimes'));
end

if has_page
    G = pagemtimes(Xp, 'none', Xp, 'transpose');          % p x p x nb
else
    G = zeros(p, p, nb);
    for i = 1:nb
        G(:,:,i) = Xp(:,:,i) * Xp(:,:,i)';
    end
end
G = permute(G, [3 1 2]);                                  % nb x p x p

D2 = reshape(sq, [nb p 1]) + reshape(sq, [nb 1 p]) - 2*G;

% round-off can make near-coincident points come out slightly negative, and
% the diagonal must be exactly zero so that r = 0 there
D2 = max(D2, 0);
D2(sv_diag_idx(nb, p)) = 0;
end
