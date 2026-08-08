function s = sv_blastrow(L)
%SV_BLASTROW  Last row of inv(L), batched:  solve L'*s = e_p.
%
%   S = SV_BLASTROW(L) returns an nb x p array whose i-th row is
%   e_p' * inv(L(i,:,:)).
%
%   This vector is the workhorse of the whole package.  If the block holds
%   the conditioning set first and the target variable last, then
%       s(p)     = 1/sqrt(d_i)          (d_i = conditional variance)
%       s(1:p-1) = -b_i/sqrt(d_i)       (b_i = kriging weights)
%   so S is exactly the corresponding column of the sparse Vecchia factor U
%   with inv(Sigma) ~ U*U'.  It also gives the derivative quantities used by
%   the gradient and the Fisher information.

[nb, p, ~] = size(L);
s = zeros(nb, p);
s(:,p) = 1 ./ L(:,p,p);
for i = p-1:-1:1
    acc = sum(reshape(L(:,i+1:p,i), [nb p-i]) .* s(:,i+1:p), 2);
    s(:,i) = -acc ./ L(:,i,i);
end
end
