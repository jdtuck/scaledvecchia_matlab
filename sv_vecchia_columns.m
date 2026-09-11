function [rowsI, colsI, vals] = sv_vecchia_columns(locs, NNp, parms, jitter)
%SV_VECCHIA_COLUMNS  Columns of the sparse Vecchia factor U for given rows.
%
%   Returns triplets (ROWSI,COLSI,VALS) such that sparse(...) is the block of
%   U whose columns correspond to the rows listed in NNP(:,1).  Shared by
%   SV_PREDICT and SV_PREPARE.
% Columns of the sparse Vecchia factor U for the rows listed in NNp(:,1).
d = size(locs, 2);
counts = sum(NNp > 0, 2);

% preallocate the triplet arrays: growing them inside the loop reallocates
% O(chunks^2) memory over a long prediction set
ntrip = sum(counts);
rowsI = zeros(ntrip, 1);
colsI = zeros(ntrip, 1);
vals  = zeros(ntrip, 1);
fill  = 0;

for p = unique(counts)'
    sel = find(counts == p);
    chunk = sv_chunk(p);
    for b = 1:chunk:numel(sel)
        ii = sel(b:min(b+chunk-1, numel(sel)));
        nb = numel(ii);
        IDX = NNp(ii, p:-1:1);                % target last
        lin = IDX(:);

        coords = zeros(nb, p, d);
        for k = 1:d
            coords(:,:,k) = reshape(locs(lin, k), [nb p]);
        end
        [C, ~] = sv_covblocks(coords, parms, jitter, false, false);
        [L, ok] = sv_bchol(C);
        if ~ok
            error('sv_vecchia_columns:chol', 'covariance block not positive definite; try a small nugget or jitter.');
        end
        s = sv_blastrow(L);                   % nb x p

        cols = repmat(ii(:), 1, p);           % local prediction index
        put = fill + (1:nb*p);
        rowsI(put) = IDX(:);
        colsI(put) = cols(:);
        vals(put)  = s(:);
        fill = fill + nb*p;
    end
end
end

