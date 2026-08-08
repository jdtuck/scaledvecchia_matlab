function fit = sv_fit(inputs, y, varargin)
%SV_FIT  Scaled Vecchia estimation of an anisotropic Matern GP emulator.
%
%   FIT = SV_FIT(INPUTS, Y) fits a Gaussian-process emulator to the n model
%   runs Y observed at the n x d input matrix INPUTS, using the scaled Vecchia
%   approximation of Katzfuss, Guinness & Lawrence (2020).
%
%   FIT = SV_FIT(INPUTS, Y, 'name', value, ...) accepts:
%
%     'm'          conditioning-set size                      (default 30)
%     'nu'         Matern smoothness.  A number fixes it; 0.5, 1.5, 2.5 and
%                  3.5 use fast closed forms.  'estimate' or [] estimates it
%                  along with the other parameters, which switches the
%                  covariance to the general Bessel form and costs several
%                  times more per iteration.                  (default 3.5)
%     'nu_bounds'  box constraint used when nu is estimated  (default [.25 6])
%     'nugget'     relative nugget tau^2 (noise/variance).  A number fixes it;
%                  'estimate' or [] estimates it.             (default 0)
%                  0 is right for a deterministic computer model; use
%                  'estimate' for noisy data.
%     'trend'      'pre' (subtract the sample mean up front), 'zero',
%                  'intercept', 'linear'                      (default 'pre')
%     'X'          n x q trend matrix, overrides 'trend'
%     'select'     de-activate an input once its estimated range exceeds
%                  'select' times the observed spread of that input  (default Inf)
%     'scale'      'parms'  scale inputs by the estimated ranges  (default)
%                  'ranges' scale inputs to [0,1] (plain Vecchia)
%                  'none'   no scaling
%     'n_est'      subsample size used for estimation      (default min(5e3,n))
%     'max_it'     cap on Fisher-scoring iterations             (default 32)
%     'tol_dec'    converged when the Newton decrement < 10^-tol_dec (default 4)
%     'vcf'        estimate a predictive variance correction factor (default true)
%     'jitter'     relative diagonal jitter for stability     (default 1e-12)
%     'verbose'    0 silent, 1 outer loop, 2 also inner loop     (default 0)
%     'seed'       seed for the estimation subsample            (default [])
%
%   The returned FIT struct carries everything SV_PREDICT needs:
%     .parms   [variance, range_1..range_d, smoothness, nugget]
%     .beta    mean coefficients, .X, .y, .inputs, .nu, .m, .trend, .vcf
%     .loglik  Vecchia loglikelihood at the estimate
%
%   How it works.  The two ingredients of a Vecchia approximation, the
%   ordering of the runs and the conditioning sets, are chosen in an input
%   space where coordinate k has been divided by its range parameter
%   lambda_k, so that inputs the response barely depends on are effectively
%   collapsed and near neighbours are near in the ways that matter.  Since the
%   lambda_k are unknown, estimation and scaling are interleaved: order and
%   condition with the current ranges, take a few Fisher-scoring steps,
%   rescale, repeat.
%
%   Reference: M. Katzfuss, J. Guinness, E. Lawrence (2022), Scaled Vecchia
%   approximation for fast computer-model emulation, SIAM/ASA JUQ 10(2),
%   537-554.  arXiv:2005.00386.

y = y(:);
[n, d] = size(inputs);
if numel(y) ~= n
    error('sv_fit:size', 'y must have as many entries as inputs has rows.');
end

o = sv_options(struct( ...
    'm', 30, 'nu', 3.5, 'nugget', 0, 'trend', 'pre', 'X', [], ...
    'scale', 'parms', 'n_est', min(5e3, n), 'max_it', 32, 'tol_dec', 4, ...
    'select', Inf, ...
    'vcf', true, 'jitter', 1e-12, 'verbose', 0, 'seed', [], 'nu_bounds', [0.25 6], ...
    'var_ini', [], 'ranges_ini', []), varargin);

if ~isempty(o.seed)
    try rng(o.seed); catch, rand('state', o.seed); randn('state', o.seed); end %#ok<RAND,CTCH>
end

% ---------------- trend ---------------------------------------------------
premean = 0;
trend = o.trend;
if ~isempty(o.X)
    X = o.X;  trend = 'X';
else
    switch lower(trend)
        case 'pre'
            premean = mean(y);
            X = zeros(n, 0);
        case 'zero'
            X = zeros(n, 0);
        case {'intercept', 'constant'}
            X = ones(n, 1);
        case 'linear'
            X = [ones(n,1), inputs];
        otherwise
            error('sv_fit:trend', 'unknown trend option ''%s''.', trend);
    end
end
yc = y - premean;

% ---------------- initial parameters --------------------------------------
input_ranges = max(inputs, [], 1) - min(inputs, [], 1);
input_ranges(input_ranges == 0) = 1;

if isempty(o.ranges_ini)
    cur_ranges = 0.2 * input_ranges;
else
    cur_ranges = o.ranges_ini(:)';
end

if isempty(o.var_ini)
    if size(X,2) > 0
        cur_var = var(yc - X*(X\yc));
    else
        cur_var = var(yc);
    end
    if ~(cur_var > 0), cur_var = 1; end
else
    cur_var = o.var_ini;
end

fix_nug = true;
if isempty(o.nugget) || (ischar(o.nugget) && strcmpi(o.nugget, 'estimate'))
    fix_nug = false;
    cur_nug = 0.01;
else
    cur_nug = o.nugget;
end

fix_nu = true;
if isempty(o.nu) || (ischar(o.nu) && strcmpi(o.nu, 'estimate'))
    fix_nu = false;
    cur_nu = 3.5;
else
    cur_nu = o.nu;
end

active = true(1, d+3);
active(d+2) = ~fix_nu;
active(d+3) = ~fix_nug;

lo = zeros(1, d+3);          hi = inf(1, d+3);
lo(d+2) = o.nu_bounds(1);    hi(d+2) = o.nu_bounds(2);

% ---------------- subsample for estimation --------------------------------
if o.n_est < n
    idx = randperm(n, o.n_est);
    ye = yc(idx);  Xe = X(idx,:);  inpe = inputs(idx,:);
else
    ye = yc;  Xe = X;  inpe = inputs;
end
ne = numel(ye);
m  = min(o.m, ne-1);

% ---------------- iterate: scale -> order -> Fisher scoring ---------------
tol = 10^(-o.tol_dec);
conv = false;
maxit = 2;
ll = NaN;  beta = zeros(size(X,2), 1);

while ~conv && maxit <= o.max_it
    % an input whose range parameter has grown far beyond the spread of that
    % input contributes essentially nothing; freeze it (cf. 'select' in the
    % reference R implementation) so it stops absorbing estimation effort
    if isfinite(o.select)
        inert = cur_ranges > o.select * input_ranges;
        if all(inert)
            error('sv_fit:select', 'all inputs de-activated; increase ''select''.');
        end
        cur_ranges(inert) = 1e10 * input_ranges(inert);
        active(1 + find(inert)) = false;
    end

    scales = sv_scales(cur_ranges, input_ranges, o.scale);

    ord  = sv_maxmin_order(inpe .* scales);
    NN   = sv_nn(inpe(ord,:) .* scales, m);

    parms = [cur_var, cur_ranges, cur_nu, cur_nug];

    if maxit == 2 && isempty(o.var_ini)
        % start the variance at its profile MLE given the initial ranges
        parms(1) = sv_profile_variance(parms, ye(ord), Xe(ord,:), inpe(ord,:), NN, o.jitter);
    end

    if o.verbose > 0
        fprintf('  m=%d, maxit=%d, ranges = %s\n', m, maxit, mat2str(cur_ranges, 3));
    end

    [parms, conv, ll, beta] = sv_fisher(parms, ye(ord), Xe(ord,:), inpe(ord,:), ...
        NN, active, o.jitter, maxit, tol, o.verbose, lo, hi);

    cur_var    = parms(1);
    cur_ranges = parms(2:d+1);
    cur_nu     = parms(d+2);
    cur_nug    = parms(d+3);
    maxit = maxit * 2;
end

parms = [cur_var, cur_ranges, cur_nu, cur_nug];

% ---------------- assemble the fit object ---------------------------------
fit = struct();
fit.parms   = parms;
fit.nu      = cur_nu;
fit.m       = o.m;
fit.jitter  = o.jitter;
fit.scale   = o.scale;
fit.loglik  = ll;
fit.conv    = conv;
fit.input_ranges = input_ranges;
fit.inputs  = inputs;

if strcmpi(trend, 'pre')
    fit.trend = 'intercept';
    fit.X     = ones(n, 1);
    fit.beta  = premean;
    fit.y     = y;
else
    fit.trend = trend;
    fit.X     = X;
    fit.beta  = beta;
    fit.y     = yc;
end

% refit the mean on the full data when that is cheap enough
if o.n_est < n && n <= 2e4 && size(fit.X,2) > 0 && ~strcmpi(trend,'pre')
    scales = sv_scales(cur_ranges, input_ranges, o.scale);
    ordf = sv_maxmin_order(inputs .* scales);
    NNf  = sv_nn(inputs(ordf,:) .* scales, min(o.m, n-1));
    [llf, ~, ~, bf] = sv_loglik(parms, fit.y(ordf), fit.X(ordf,:), inputs(ordf,:), ...
        NNf, active, o.jitter, false);
    fit.beta = bf;
    fit.loglik = llf;
end

% ---------------- predictive variance correction factor -------------------
fit.vcf = 1;
if o.vcf
    fit.vcf = sv_fit_vcf(fit);
    if o.verbose > 0
        fprintf('  variance correction factor: %.4f\n', fit.vcf);
    end
end
end

% =========================================================================
function s = sv_scales(cur_ranges, input_ranges, scale)
switch lower(scale)
    case 'parms',  s = 1 ./ cur_ranges;
    case 'ranges', s = 1 ./ input_ranges;
    case 'none',   s = ones(size(cur_ranges));
    otherwise, error('sv_fit:scale', 'invalid scale option ''%s''.', scale);
end
end

function v = sv_profile_variance(parms, y, X, locs, NN, jitter)
% With C = sigma^2 * R, the profile MLE of sigma^2 is mean of the squared
% whitened residuals computed at sigma^2 = 1.
d = size(locs,2);
active = false(1, d+3);
p1 = parms; p1(1) = 1;
ll = sv_loglik(p1, y, X, locs, NN, active, jitter, false);
n = numel(y);
% ll = -n/2 log(2pi) - sum(log Lpp) - 0.5*SSQ, and sum(log Lpp) does not
% depend on the data, so recover SSQ by re-evaluating with y = 0.
ll0 = sv_loglik(p1, zeros(n,1), X, locs, NN, active, jitter, false);
ssq = -2 * (ll - ll0);
v = max(ssq / n, eps);
end

function vcf = sv_fit_vcf(fit)
% Variance correction factor: hold out part of the data, make pointwise
% predictions, and rescale the predictive variances so that the held-out log
% score is minimized.  For the log score the optimum is available in closed
% form as the mean standardized squared error.
n = size(fit.inputs, 1);
ntest = min(1000, round(n/5));
if ntest < 20, vcf = 1; return; end

idx = randperm(n, ntest);
keep = true(n,1); keep(idx) = false;

sub = fit;
sub.y = fit.y(keep);
sub.inputs = fit.inputs(keep,:);
sub.X = fit.X(keep,:);
sub.vcf = 1;

p = sv_predict(sub, fit.inputs(idx,:), 'm', min(140, sum(keep)), ...
    'joint', false, 'variance', true, 'X_pred', fit.X(idx,:));

r2 = (fit.y(idx) - p.mean).^2;
good = p.var > 0 & isfinite(p.var) & isfinite(r2);
if ~any(good), vcf = 1; return; end
vcf = mean(r2(good) ./ p.var(good));
if ~isfinite(vcf) || vcf <= 0, vcf = 1; end
end
