function NN = sv_nn(locs, m, rows)
%SV_NN  Ordered nearest-neighbour conditioning sets.
%
%   NN = SV_NN(LOCS,M) returns an n x (M+1) matrix.  NN(i,1) = i and
%   NN(i,2:end) holds the indices of the (up to M) nearest neighbours of row i
%   among the rows 1..i-1.  Unused slots are 0.
%
%   NN = SV_NN(LOCS,M,ROWS) computes the conditioning sets only for the rows
%   listed in ROWS (used for prediction, where only the prediction locations
%   need Vecchia rows).  The returned matrix has numel(ROWS) rows.
%
%   LOCS must already be in the scaled space and in the Vecchia ordering.
%
%   Method.  Rows are processed in blocks.  Every row of a block spanning
%   [s,e) shares the same set of "outside" candidates, namely rows 1..s-1, so
%   those distances need no per-row masking; only the small within-block
%   triangle does.  The previous implementation masked a full
%   (block x n_predecessors) matrix with Inf on every block, which cost more
%   than the distance computation itself.
%
%   When Statistics Toolbox is available the outside candidates come from a
%   KDTreeSearcher rebuilt per block, turning the dominant term from O(n^2 d)
%   into roughly O(n log(n) d).  Without it they come from a blocked
%   Gram-identity computation.  Both paths are exact and return the same
%   neighbours; SV_TEST checks them against each other.

n = size(locs, 1);
if nargin < 3 || isempty(rows)
    rows = (1:n)';
end
rows = rows(:);
nr = numel(rows);
NN = zeros(nr, m+1);
NN(:,1) = rows;
if m < 1 || nr == 0, return; end

persistent has_tree has_mink
if isempty(has_tree)
    has_tree = ~isempty(which('KDTreeSearcher')) && ~isempty(which('knnsearch'));
    has_mink = ~isempty(which('mink'));
end

sq = sum(locs.^2, 2);

% Handle rows in increasing order so "predecessors" is always a prefix.
[rs, perm] = sort(rows);
out = zeros(nr, m);

blk = 1024;
b = 1;
while b <= nr
    e  = min(b + blk - 1, nr);
    R  = rs(b:e);
    nb = numel(R);
    s  = R(1);                       % rows 1..s-1 precede every row in R

    k_out  = min(m, s-1);
    cand_i = zeros(nb, 0);
    cand_d = zeros(nb, 0);

    if k_out > 0
        Q = locs(R,:);
        if has_tree
            [idx, dst] = knnsearch(KDTreeSearcher(locs(1:s-1,:)), Q, 'K', k_out);
            cand_i = idx;
            cand_d = dst.^2;
        else
            D = sq(R) + sq(1:s-1)' - 2*(Q * locs(1:s-1,:)');
            if k_out < s-1
                if has_mink
                    [cand_d, cand_i] = mink(D, k_out, 2);
                else
                    [Ds, srt] = sort(D, 2);
                    cand_d = Ds(:,1:k_out);
                    cand_i = srt(:,1:k_out);
                end
            else
                cand_d = D;
                cand_i = repmat(1:s-1, nb, 1);
            end
        end
    end

    % Exact distances to earlier members of this same block.  Only the strict
    % lower triangle is eligible; the block is small so this mask is cheap.
    if nb > 1
        Db = sq(R) + sq(R)' - 2*(locs(R,:) * locs(R,:)');
        Db(triu(true(nb))) = Inf;
        cand_d = [cand_d, Db];
        cand_i = [cand_i, repmat(R(:)', nb, 1)];
    end

    if isempty(cand_d), b = e + 1; continue; end

    kk = min(m, size(cand_d, 2));
    if has_mink
        [dsel, take] = mink(cand_d, kk, 2);
    else
        [dsel, srt] = sort(cand_d, 2);
        dsel = dsel(:,1:kk);
        take = srt(:,1:kk);
    end
    lin = sub2ind(size(cand_i), repmat((1:nb)', 1, kk), take);
    picked = cand_i(lin);
    picked(~isfinite(dsel)) = 0;     % fewer than m predecessors exist

    out(b:e, 1:kk) = picked;
    b = e + 1;
end

NN(perm, 2:end) = out;
end
