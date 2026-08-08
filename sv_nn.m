function NN = sv_nn(locs, m, rows)
%SV_NN  Ordered nearest-neighbour conditioning sets.
%
%   NN = SV_NN(LOCS,M) returns an n x (M+1) matrix.  NN(i,1) = i and
%   NN(i,2:end) holds the indices of the (up to M) nearest neighbours of row i
%   among the rows 1..i-1, sorted by increasing distance.  Unused slots are 0.
%
%   NN = SV_NN(LOCS,M,ROWS) computes the conditioning sets only for the rows
%   listed in ROWS (used for prediction, where only the prediction locations
%   need Vecchia rows).  The returned matrix has numel(ROWS) rows.
%
%   LOCS must already be in the scaled space and in the Vecchia ordering.
%
%   Exact blocked brute force, O(n^2 d).  Swap in a k-d tree (knnsearch) for
%   very large n; the definition of the output is unchanged.

n = size(locs, 1);
if nargin < 3 || isempty(rows)
    rows = (1:n)';
end
rows = rows(:);
nr = numel(rows);
NN = zeros(nr, m+1);

sq = sum(locs.^2, 2);
blk = max(1, floor(4e6 / max(n,1)));

for b = 1:blk:nr
    ii = b:min(b+blk-1, nr);
    R  = rows(ii);
    nb = numel(ii);
    NN(ii,1) = R;

    maxprev = max(R) - 1;
    if maxprev < 1, continue; end

    D = sq(R) + sq(1:maxprev)' - 2*(locs(R,:) * locs(1:maxprev,:)');
    D(bsxfun(@ge, 1:maxprev, R)) = Inf;     % only predecessors are eligible

    k = min(m, maxprev);
    [Ds, srt] = sort(D, 2);
    sel = srt(:,1:k);
    sel(isinf(Ds(:,1:k))) = 0;              % fewer than k predecessors
    NN(ii,2:k+1) = sel;
end
end
