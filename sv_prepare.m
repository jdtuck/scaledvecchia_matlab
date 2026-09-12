function prep = sv_prepare(fit, inputs_pred, varargin)
%SV_PREPARE  Precompute a reusable prediction plan for fixed input locations.
%
%   PREP = SV_PREPARE(FIT, INPUTS_PRED, ...) does all of the work that depends
%   only on the covariance parameters and the input locations: the
%   nearest-neighbour conditioning sets, the covariance blocks, their Cholesky
%   factors and the resulting Vecchia weights.  It accepts the same options as
%   SV_PREDICT.
%
%   PRED = SV_DRAW(PREP, ...) then produces means, variances and conditional
%   simulations from the plan.
%
%   Use this whenever SV_PREDICT would be called repeatedly at the same
%   INPUTS_PRED — drawing many posterior realizations, or running an MCMC in
%   which the emulator itself is fixed.  A single SV_PREDICT call recomputes
%   the neighbour search, the covariance blocks and the factorizations every
%   time, even though none of them depend on the response values or on the
%   random draws.  Splitting the work moves that cost out of the loop:
%
%       prep = sv_prepare(fit, x_new, 'm', 100);
%       for it = 1:niter
%           s = sv_draw(prep, 'nsims', 1);      % microseconds, not seconds
%           ...
%       end
%
%   'noise_free' applies to the pointwise path only.  The joint path builds
%   one Vecchia factor over the stacked observed and prediction points, whose
%   blocks carry the nugget on every diagonal entry, so a joint plan always
%   predicts a fresh noisy observation rather than the latent process.
%
%   The plan stays valid as long as FIT.PARMS and INPUTS_PRED are unchanged.
%   The response values may change freely: pass a new vector to SV_DRAW with
%   the 'y' option and the mean is recomputed from the cached weights, which
%   costs O(n_pred * m) rather than a full refactorization.  If the covariance
%   parameters themselves change, build a new plan.
%
%   See also SV_DRAW, SV_PREDICT.

[np, d] = size(inputs_pred);
n = size(fit.inputs, 1);
if d ~= size(fit.inputs, 2)
    error('sv_prepare:dim', 'prediction inputs must have %d columns.', size(fit.inputs,2));
end

o = sv_options(struct('m', 100, 'joint', true, 'X_pred', [], 'scale', [], ...
    'noise_free', true), varargin);
if isempty(o.scale)
    if isfield(fit, 'scale'), o.scale = fit.scale; else, o.scale = 'parms'; end
end

% Observed-side quantities are the same for every prediction; build them once
% if the fit does not already carry them (SV_FIT attaches them), or again if
% the parameters they were built from have moved since.
if ~isfield(fit, 'cache') || isempty(fit.cache) || ~cache_current(fit)
    fit = sv_cache(fit);
end
cache = fit.cache;

parms  = fit.parms;
jitter = 0;
if isfield(fit, 'jitter'), jitter = fit.jitter; end

prep = struct();
prep.is_prep = true;
prep.np      = np;
prep.inputs_pred = inputs_pred;   % lets callers verify a cached plan still applies
prep.m       = o.m;
prep.joint   = o.joint;
prep.parms   = parms;
prep.vcf     = 1;
if isfield(fit, 'vcf') && ~isempty(fit.vcf), prep.vcf = fit.vcf; end

% ---- trend at the prediction inputs (fixed by the locations) -------------
Xp = o.X_pred;
if isempty(Xp)
    switch lower(fit.trend)
        case 'zero',                     Xp = zeros(np, 0);
        case {'intercept','constant'},   Xp = ones(np, 1);
        case 'linear',                   Xp = [ones(np,1), inputs_pred];
        otherwise
            error('sv_prepare:Xpred', 'X_pred must be supplied for trend ''%s''.', fit.trend);
    end
end
prep.Xp   = Xp;
prep.beta = fit.beta;
prep.X    = fit.X;
prep.y    = fit.y;

resid = cache.resid;
if size(fit.X, 2) > 0
    prep.trendp = Xp * fit.beta;
else
    prep.trendp = zeros(np, 1);
end

if strcmpi(o.scale, lower(getfielddef(fit, 'scale', 'parms')))
    scales = cache.scales;
else
    scales = pred_scales(fit, o.scale, d);
end
prep.noise_free = o.noise_free;

if o.joint
    ordp = sv_maxmin_order(inputs_pred .* scales);
    locs_all = [fit.inputs; inputs_pred(ordp,:)];
    N = n + np;
    m = min(o.m, n + np - 1);
    NNp = sv_nn(locs_all .* scales, m, (n+1:N)');

    [rowsI, colsI, vals] = sv_vecchia_columns(locs_all, NNp, parms, jitter);
    Uall = sparse(rowsI, colsI, vals, N, np);

    prep.ordp = ordp;
    prep.Uop  = Uall(1:n, :);
    prep.UppT = Uall(n+1:N, :)';          % lower triangular
    prep.mu_ord = -(prep.UppT \ (prep.Uop' * resid));
else
    m = min(o.m, n);
    NNo = sv_knn(inputs_pred .* scales, cache, m);
    p = m + 1;

    S   = zeros(np, p);
    NBR = zeros(np, m);

    % The block is [neighbours ... target], so gather the neighbour coordinates
    % from fit.inputs and append the prediction point.  Concatenating the two
    % into one n+np array first would copy the whole training set on every
    % call, which dominates when np is small.
    chunk = sv_chunk(p);
    for b = 1:chunk:np
        ii = (b:min(b+chunk-1, np))';
        nb = numel(ii);
        nbi = NNo(ii,:);

        coords = zeros(nb, p, d);
        for k = 1:d
            coords(:,1:m,k) = reshape(fit.inputs(nbi, k), [nb m]);
            coords(:,p,k)   = inputs_pred(ii, k);
        end

        [C, ~] = sv_covblocks(coords, parms, jitter, false, false);
        if o.noise_free
            % Predict the latent process, not a fresh noisy observation: drop
            % the nugget from the target's own diagonal entry.
            C(:,p,p) = C(:,p,p) - parms(1)*parms(end);
        end
        [L, ok] = sv_bchol(C);
        if ~ok
            error('sv_prepare:chol', ...
                'covariance block not positive definite; try a small nugget or jitter.');
        end
        S(ii,:)   = sv_blastrow(L);
        NBR(ii,:) = nbi;
    end

    prep.s   = S;
    prep.NBR = NBR;
    prep.p   = p;
    prep.var = prep.vcf ./ S(:,p).^2;      % fixed by the locations
    prep.mean_resid = pointwise_mean(S, NBR, resid, p);
end
end

% -------------------------------------------------------------------------
function tf = cache_current(fit)
% Two small-vector comparisons: cheap next to the neighbour search, and they
% keep a hand-edited fit from predicting through a stale cache.
c = fit.cache;
tf = isfield(c, 'parms') && isequal(c.parms, fit.parms) ...
    && isfield(c, 'beta') && isequal(c.beta, fit.beta);
end

function mu = pointwise_mean(S, NBR, resid, p)
rsub = reshape(resid(NBR), size(NBR));
mu = -sum(S(:,1:p-1) .* rsub, 2) ./ S(:,p);
end

function v = getfielddef(s, f, dflt)
if isfield(s, f) && ~isempty(s.(f)), v = lower(s.(f)); else, v = dflt; end
end

function s = pred_scales(fit, scale, d)
switch lower(scale)
    case 'parms'
        s = 1 ./ fit.parms(2:d+1);
    case 'ranges'
        if isfield(fit, 'input_ranges')
            s = 1 ./ fit.input_ranges;
        else
            r = max(fit.inputs, [], 1) - min(fit.inputs, [], 1);
            r(r == 0) = 1;
            s = 1 ./ r;
        end
    case 'none'
        s = ones(1, d);
    otherwise
        error('sv_prepare:scale', 'invalid scale option ''%s''.', scale);
end
end

