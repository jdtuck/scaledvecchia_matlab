function ord = sv_maxmin_order(locs)
%SV_MAXMIN_ORDER  Exact greedy maximin ordering.
%
%   ORD = SV_MAXMIN_ORDER(LOCS) orders the rows of LOCS so that each point is
%   as far as possible from all previously ordered points.  The first point is
%   the one closest to the centroid.
%
%   In the scaled Vecchia method this ordering is applied in the *scaled*
%   input space (each input divided by its estimated range parameter), which
%   is what makes the resulting conditioning sets informative for inputs of
%   very different relevance.
%
%   Cost is O(n^2 d).  Each of the n steps is one vectorized pass, so this is
%   fine up to n of order 10^4; for larger n use a subsample for estimation
%   (see the 'n_est' option of SV_FIT) or plug in an approximate maximin
%   ordering.

n = size(locs, 1);
ord = zeros(n, 1);
if n == 0, return; end

cent = mean(locs, 1);
[~, first] = min(sum((locs - cent).^2, 2));
ord(1) = first;

mind = sum((locs - locs(first,:)).^2, 2);
mind(first) = -Inf;

for i = 2:n
    [~, nxt] = max(mind);
    ord(i) = nxt;
    mind = min(mind, sum((locs - locs(nxt,:)).^2, 2));
    mind(nxt) = -Inf;
end
end
