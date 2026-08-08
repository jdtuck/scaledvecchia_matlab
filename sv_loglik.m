function [ll, grad, info, beta] = sv_loglik(parms, y, X, locs, NNarray, active, jitter, need_grad)
%SV_LOGLIK  Vecchia loglikelihood with profiled linear mean, plus gradient
%           and expected Fisher information on the log-parameter scale.
%
%   [LL,GRAD,INFO,BETA] = SV_LOGLIK(PARMS,Y,X,LOCS,NNARRAY,ACTIVE,JITTER,NEED_GRAD)
%
%   Y, X, LOCS and NNARRAY must all already be in the Vecchia ordering.
%   PARMS = [variance, range_1..range_d, smoothness, nugget] on the natural
%   scale.  ACTIVE is a logical vector over those d+3 parameters; GRAD and INFO are
%   returned only for the active ones and are taken with respect to
%   log(PARMS(ACTIVE)).
%
%   Method
%   ------
%   Vecchia's approximation replaces the joint density by
%       prod_i p( y_i | y_{g(i)} ),
%   with g(i) the conditioning set in row i of NNARRAY.  For each i we form
%   the (p x p) covariance of [y_{g(i)}, y_i] with the *target last*, take its
%   Cholesky factor L, and use
%       log p(y_i|y_g) = -0.5*log(2*pi) - log L(p,p) - 0.5*( inv(L)*r )_p^2 .
%   Everything is quadratic in (y,X), so the GLS estimate of the mean
%   coefficients can be profiled out at the end from the last entries alone.
%
%   Derivatives.  With s = e_p'*inv(L), z = inv(L)*(dSigma*s') and u = inv(L)*r,
%       d(loglik_i)/d(theta_j) = -0.5*z_p*(1 + u_p^2) + u_p*(z'*u),
%   and, because the whitened residuals are iid standard normal under the
%   Vecchia model, the expected Fisher information is exactly
%       I_jk = sum_i ( z^j' z^k - 0.5 * z^j_p z^k_p ).
%   Only the last row of inv(L) is needed, so each parameter costs O(p^2) per
%   observation rather than O(p^3).

if nargin < 8, need_grad = true; end

[n, d]  = size(locs);
q       = size(X, 2);
npar    = d + 3;
aidx    = find(active(:)');
nact    = numel(aidx);

grad = zeros(nact, 1);
info = zeros(nact, nact);
beta = zeros(q, 1);

% ---- per-observation storage -------------------------------------------
A     = zeros(n, 1);          % (inv(L)*y_sub)_p
Bm    = zeros(n, q);          % (inv(L)*X_sub)_p
logLp = zeros(n, 1);
if need_grad
    P = zeros(n, nact);       % z_p
    G = zeros(n, nact);       % z' * inv(L)*y_sub
    H = zeros(n, nact, q);    % z' * inv(L)*X_sub
    ZtZ  = zeros(nact);
    ZpZp = zeros(nact);
end

counts = sum(NNarray > 0, 2);           % block size p for each row
psizes = unique(counts)';

for p = psizes
    rowsp = find(counts == p);
    % keep memory for the nb x p x p arrays bounded
    chunk = max(1, floor(2e6 / (p*p)));
    for b = 1:chunk:numel(rowsp)
        R  = rowsp(b:min(b+chunk-1, numel(rowsp)));
        nb = numel(R);

        % conditioning set with the TARGET LAST
        IDX = NNarray(R, p:-1:1);
        lin = IDX(:);

        coords = zeros(nb, p, d);
        for k = 1:d
            coords(:,:,k) = reshape(locs(lin, k), [nb p]);
        end

        RHS = zeros(nb, p, 1+q);
        RHS(:,:,1) = reshape(y(lin), [nb p]);
        for c = 1:q
            RHS(:,:,1+c) = reshape(X(lin, c), [nb p]);
        end

        [C, aux] = sv_covblocks(coords, parms, jitter, need_grad, need_grad && active(d+2));
        [L, ok]  = sv_bchol(C);
        if ~ok
            ll = -Inf; grad = zeros(nact,1); info = eye(nact); return
        end

        U = sv_bfsolve(L, RHS);
        A(R)     = U(:,p,1);
        Bm(R,:)  = reshape(U(:,p,2:end), [nb q]);
        logLp(R) = log(L(:,p,p));

        if need_grad
            s = sv_blastrow(L);                       % nb x p
            Z = zeros(nb, p, nact);
            for jj = 1:nact
                j  = aidx(jj);
                dS = sv_covderiv(coords, aux, j, parms, jitter) * parms(j);
                t  = reshape(sum(dS .* reshape(s, [nb 1 p]), 3), [nb p]);
                z  = reshape(sv_bfsolve(L, reshape(t, [nb p 1])), [nb p]);
                Z(:,:,jj) = z;
                P(R,jj) = z(:,p);
                G(R,jj) = sum(z .* U(:,:,1), 2);
                for c = 1:q
                    H(R,jj,c) = sum(z .* U(:,:,1+c), 2);
                end
            end
            Zm   = reshape(Z, [nb*p, nact]);
            ZtZ  = ZtZ  + Zm' * Zm;
            Zp   = reshape(Z(:,p,:), [nb nact]);
            ZpZp = ZpZp + Zp' * Zp;
        end
    end
end

% ---- profiled GLS mean coefficients ------------------------------------
if q > 0
    XSX  = Bm' * Bm;
    beta = XSX \ (Bm' * A);
    a    = A - Bm * beta;
else
    a = A;
end

ll = -0.5*n*log(2*pi) - sum(logLp) - 0.5*(a' * a);
if ~isfinite(ll), ll = -Inf; end

if need_grad
    for jj = 1:nact
        Hj = reshape(H(:,jj,:), [n q]);
        if q > 0
            gj = G(:,jj) - Hj * beta;
        else
            gj = G(:,jj);
        end
        grad(jj) = -0.5 * sum(P(:,jj) .* (1 + a.^2)) + a' * gj;
    end
    info = ZtZ - 0.5 * ZpZp;
    info = 0.5 * (info + info');
end
end
