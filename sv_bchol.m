function [L, ok] = sv_bchol(A)
%SV_BCHOL  Cholesky factorization of many small matrices at once.
%
%   [L,OK] = SV_BCHOL(A) takes an nb x p x p array whose pages are symmetric
%   positive definite and returns the lower-triangular factors, A(i,:,:) =
%   L(i,:,:)*L(i,:,:)'.  OK is false if any page failed (non-positive pivot).
%
%   The loops run over the p columns, not over the nb pages, so the whole
%   batch is factorized with O(p^2) vectorized operations.  This is what keeps
%   the Vecchia likelihood fast in MATLAB without a mex file.

[nb, p, ~] = size(A);
L = zeros(nb, p, p);

for j = 1:p
    v = A(:,j,j);
    if j > 1
        v = v - sum(L(:,j,1:j-1).^2, 3);
    end
    if any(~(v > 0)) || any(~isfinite(v))
        L = []; ok = false; return
    end
    Ljj = sqrt(v);                      % nb x 1 x 1
    L(:,j,j) = Ljj;
    if j < p
        w = A(:,j+1:p,j);               % nb x (p-j) x 1
        if j > 1
            w = w - sum(L(:,j+1:p,1:j-1) .* L(:,j,1:j-1), 3);
        end
        L(:,j+1:p,j) = w ./ Ljj;
    end
end
ok = true;
end
