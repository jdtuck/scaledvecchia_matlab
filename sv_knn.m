function NN = sv_knn(query, cache, m)
%SV_KNN  Nearest observed runs for each prediction point, in the scaled space.
%
%   NN = SV_KNN(QUERY, CACHE, M) returns the indices of the M nearest rows of
%   the observed design for each row of QUERY.  QUERY must already be scaled;
%   CACHE comes from SV_CACHE and supplies the scaled observed inputs, their
%   squared norms and, when Statistics Toolbox is available, a prebuilt k-d
%   tree.
%
%   With the tree this is O(n_pred * log(n) * d).  Without it we fall back to
%   the Gram identity against the cached reference matrix, which is O(n_pred *
%   n * d) but still avoids re-scaling the design and recomputing its norms on
%   every call.

nr = size(cache.ref, 1);
m  = min(m, nr);

if ~isempty(cache.searcher)
    NN = knnsearch(cache.searcher, query, 'K', m);
    return
end

nq = size(query, 1);
NN = zeros(nq, m);

persistent has_mink
if isempty(has_mink), has_mink = ~isempty(which('mink')); end

blk = max(1, floor(4e6 / max(nr,1)));
for b = 1:blk:nq
    ii = b:min(b+blk-1, nq);
    D = sum(query(ii,:).^2, 2) + cache.refsq - 2*(query(ii,:) * cache.ref');
    if has_mink
        [~, sel] = mink(D, m, 2);
    else
        [~, srt] = sort(D, 2);
        sel = srt(:,1:m);
    end
    NN(ii,:) = sel;
end
end
