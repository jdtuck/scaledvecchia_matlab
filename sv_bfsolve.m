function X = sv_bfsolve(L, B)
%SV_BFSOLVE  Batched forward substitution: solve L*X = B page by page.
%
%   X = SV_BFSOLVE(L,B) with L an nb x p x p array of lower-triangular
%   factors and B an nb x p x q array of right-hand sides.

[nb, p, ~] = size(L);
q = size(B, 3);
X = zeros(nb, p, q);

for i = 1:p
    b = B(:,i,:);                                   % nb x 1 x q
    if i > 1
        Lrow = reshape(L(:,i,1:i-1), [nb i-1 1]);   % nb x (i-1) x 1
        b = b - sum(Lrow .* X(:,1:i-1,:), 2);
    end
    X(:,i,:) = b ./ L(:,i,i);
end
end
