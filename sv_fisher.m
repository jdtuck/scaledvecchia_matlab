function [parms, conv, ll, beta, nit] = sv_fisher(parms, y, X, locs, NN, active, jitter, maxit, tol, verbose, lo, hi)
%SV_FISHER  Fisher scoring for the Vecchia loglikelihood.
%
%   Maximizes the Vecchia loglikelihood over log(PARMS(ACTIVE)) using the
%   expected Fisher information as the curvature matrix, following Guinness
%   (2021), "Gaussian process learning via Fisher scoring of Vecchia's
%   approximation".  The step is capped in the log scale, clipped to the box
%   [LO,HI] on the natural scale, and backtracked until the loglikelihood
%   improves.  Convergence is declared when the Newton
%   decrement |grad'*step| falls below TOL.

aidx = find(active(:)');
nact = numel(aidx);
npar = numel(parms);
if nargin < 11 || isempty(lo), lo = zeros(1, npar);    end
if nargin < 12 || isempty(hi), hi = inf(1, npar);      end
conv = false;
nit  = 0;

[ll, g, I, beta] = sv_loglik(parms, y, X, locs, NN, active, jitter, true);

for it = 1:maxit
    nit = it;

    % --- Newton/Fisher step, with a ridge if the information is ill-conditioned
    step = [];
    ridge = 0;
    for att = 1:12
        M = I + ridge*diag(max(diag(I), 1e-8)) + 1e-10*eye(nact);
        [Rc, flag] = chol(M);
        if flag == 0
            step = Rc \ (Rc' \ g);
            break
        end
        if ridge == 0, ridge = 1e-4; else, ridge = 10*ridge; end
    end
    if isempty(step)
        step = g / max(norm(g), 1);     % fall back to (scaled) gradient ascent
    end

    % --- limit the step size on the log scale
    ms = mean(abs(step));
    if ms > 1, step = step / ms; end

    % --- backtracking line search
    accepted = false;
    for ls = 1:12
        cand = parms;
        cand(aidx) = parms(aidx) .* exp(step(:)');
        cand(aidx) = min(max(cand(aidx), lo(aidx)), hi(aidx));
        if all(isfinite(cand)) && all(cand(aidx) > 0)
            newll = sv_loglik(cand, y, X, locs, NN, active, jitter, false);
        else
            newll = -Inf;
        end
        if isfinite(newll) && newll > ll
            accepted = true;
            break
        end
        step = step / 2;
    end

    if ~accepted
        conv = true;                     % cannot improve any further
        break
    end

    decrement = abs(g' * step);
    parms = cand;
    [ll, g, I, beta] = sv_loglik(parms, y, X, locs, NN, active, jitter, true);

    if verbose > 1
        fprintf('    it %2d  loglik %14.5f  decrement %.3e\n', it, ll, decrement);
    end
    if decrement < tol
        conv = true;
        break
    end
end
end
