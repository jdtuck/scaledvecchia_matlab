function pred = sv_draw(prep, varargin)
%SV_DRAW  Means, variances and conditional simulations from a prepared plan.
%
%   PRED = SV_DRAW(PREP, ...) uses the plan built by SV_PREPARE.  Options:
%
%     'nsims'     number of conditional simulations            (default 0)
%     'variance'  return pointwise variances                   (default false)
%     'y'         new response vector, replacing the one the plan was built
%                 with.  The cached Vecchia weights are reused, so this costs
%                 O(n_pred * m) instead of a refactorization.
%     'beta'      new mean coefficients to go with 'y'
%
%   Output fields: .mean, and .var / .samples when requested.
%
%   Everything expensive already happened in SV_PREPARE, so a call that only
%   asks for new simulations is essentially the cost of the random numbers.
%
%   See also SV_PREPARE, SV_PREDICT.

if ~isstruct(prep) || ~isfield(prep, 'is_prep')
    error('sv_draw:prep', 'first argument must come from sv_prepare.');
end

o = sv_options(struct('nsims', 0, 'variance', false, 'y', [], 'beta', []), varargin);

beta = prep.beta;
if ~isempty(o.beta), beta = o.beta; end

% ---- recompute the residual only if the response or trend changed --------
refresh = ~isempty(o.y) || ~isempty(o.beta);
if refresh
    yy = prep.y;
    if ~isempty(o.y), yy = o.y(:); end
    if size(prep.X, 2) > 0
        resid  = yy - prep.X * beta;
        trendp = prep.Xp * beta;
    else
        resid  = yy;
        trendp = zeros(prep.np, 1);
    end
else
    trendp = prep.trendp;
end

pred = struct();

if prep.joint
    if refresh
        mu_ord = -(prep.UppT \ (prep.Uop' * resid));
    else
        mu_ord = prep.mu_ord;
    end

    pm = zeros(prep.np, 1);
    pm(prep.ordp) = mu_ord;
    pred.mean = pm + trendp;

    if o.nsims > 0
        Z = randn(prep.np, o.nsims) * sqrt(prep.vcf);
        sims_ord = mu_ord + (prep.UppT \ Z);
        S = zeros(prep.np, o.nsims);
        S(prep.ordp,:) = sims_ord;
        pred.samples = S + trendp;
        if o.variance
            pred.var = var(pred.samples, 0, 2);
        end
    elseif o.variance
        warning('sv_draw:jointvar', ...
            'joint prediction returns variances only via nsims > 0; prepare with joint=false for exact pointwise variances.');
    end

else
    p = prep.p;
    if refresh
        rsub = reshape(resid(prep.NBR), prep.np, p-1);
        mu = -sum(prep.s(:,1:p-1) .* rsub, 2) ./ prep.s(:,p);
    else
        mu = prep.mean_resid;
    end

    pred.mean = mu + trendp;
    if o.variance
        pred.var = prep.var;
    end
    if o.nsims > 0
        pred.samples = pred.mean + sqrt(prep.var) .* randn(prep.np, o.nsims);
    end
end
end
