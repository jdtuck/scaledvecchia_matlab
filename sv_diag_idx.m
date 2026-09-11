function idx = sv_diag_idx(nb, p)
%SV_DIAG_IDX  Linear indices of the block diagonals in an nb x p x p array.
%
%   IDX = SV_DIAG_IDX(NB,P) returns an NB x P matrix of linear indices such
%   that A(IDX) addresses every A(i,a,a).
%
%   Setting the diagonal with a loop over the p columns costs p separate
%   three-dimensional indexed assignments, which the interpreter charges for
%   individually.  At the block sizes used for prediction (p around 100) that
%   loop was costing more than the Cholesky factorization it feeds.  One
%   vectorized assignment through these indices removes it.
%
%   Element (i,a,a) of an nb x p x p array sits at
%       i + (a-1)*nb + (a-1)*nb*p = i + (a-1)*nb*(p+1).
%
%   The indices depend only on the shape, so the last result is cached for the
%   repeated calls that a chunked loop makes with identical dimensions.

persistent last_nb last_p last_idx
if ~isempty(last_idx) && last_nb == nb && last_p == p
    idx = last_idx;
    return
end

idx = (1:nb)' + (0:p-1) * (nb * (p+1));

last_nb = nb; last_p = p; last_idx = idx;
end
