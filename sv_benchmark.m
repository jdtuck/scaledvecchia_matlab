function results = sv_benchmark(varargin)
%SV_BENCHMARK  Measure how fitting and prediction scale with n, d and m.
%
%   RESULTS = SV_BENCHMARK() runs the default sweeps and prints a report.
%   RESULTS = SV_BENCHMARK('name', value, ...) accepts:
%
%     'ns'        training sizes for the n sweep     (default [500 1000 2000 4000])
%     'ds'        input dimensions for the d sweep   (default [2 4 8 16])
%     'ms'        conditioning sizes for the m sweep (default [10 20 40 80])
%     'n_fixed'   n held fixed in the d and m sweeps          (default 2000)
%     'd_fixed'   d held fixed in the n and m sweeps          (default 8)
%     'm_fixed'   m held fixed in the n and d sweeps          (default 30)
%     'n_pred'    prediction points                           (default 500)
%     'm_pred'    conditioning size for prediction            (default 100)
%     'niter'     draws used to time the MCMC inner loop      (default 200)
%     'sweeps'    which sweeps to run, any of 'n','d','m','mcmc'
%                                                   (default all four)
%     'reps'      repetitions of each timing, minimum taken   (default 1)
%     'verbose'   print as it goes                            (default true)
%
%   Each sweep reports wall time and, next to it, the empirical exponent from a
%   log-log fit against the swept quantity.  The exponent is the useful number:
%   it says which term dominates at the size you care about, and it is what
%   changes when an implementation detail changes, whereas absolute times move
%   with the machine and the BLAS.
%
%   What to expect.  Fitting has two parts with different exponents.  The
%   likelihood, its gradient and its Fisher information are linear in n and in
%   the number of active parameters, and cubic in m through the per-block
%   factorization.  The maximin ordering and the neighbour search are
%   quadratic in n (the latter drops toward n*log(n) when Statistics Toolbox
%   supplies a k-d tree).  So the measured exponent for the fit sits between 1
%   and 2 and drifts upward as n grows and the quadratic terms take over.
%
%   Prediction is linear in the number of prediction points and cubic in
%   m_pred.  The 'mcmc' sweep separates the one-shot cost from the repeated
%   cost, which is the distinction that matters for calibration: SV_PREPARE
%   pays the setup once and SV_DRAW is what runs in the loop.
%
%   Example:
%       r = sv_benchmark('ns', [1000 2000 4000 8000], 'sweeps', {'n'});
%
%   See also SV_FIT, SV_PREDICT, SV_PREPARE, SV_DRAW.

o = sv_options(struct( ...
    'ns', [500 1000 2000 4000], 'ds', [2 4 8 16], 'ms', [10 20 40 80], ...
    'n_fixed', 2000, 'd_fixed', 8, 'm_fixed', 30, ...
    'n_pred', 500, 'm_pred', 100, 'niter', 200, ...
    'sweeps', {{'n','d','m','mcmc'}}, 'reps', 1, 'verbose', true, ...
    'seed', 1), varargin);

if ischar(o.sweeps), o.sweeps = {o.sweeps}; end

results = struct();
results.info = env_info();

if o.verbose
    fprintf('\n=== scaled Vecchia scaling benchmark ===\n');
    fprintf('%s\n', results.info.line);
end

if any(strcmp(o.sweeps, 'n'))
    results.n = sweep_n(o);
end
if any(strcmp(o.sweeps, 'd'))
    results.d = sweep_d(o);
end
if any(strcmp(o.sweeps, 'm'))
    results.m = sweep_m(o);
end
if any(strcmp(o.sweeps, 'mcmc'))
    results.mcmc = sweep_mcmc(o);
end

if o.verbose
    fprintf('\n=== done ===\n\n');
end
if nargout == 0
    clear results
end
end

% =========================================================================
function r = sweep_n(o)
if o.verbose
    fprintf('\n-- scaling in n (d = %d, m = %d, n_pred = %d, m_pred = %d)\n', ...
        o.d_fixed, o.m_fixed, o.n_pred, o.m_pred);
    header({'n', 'fit (s)', 'predict (s)', 'fit/n (ms)'});
end
k = numel(o.ns);
r = struct('n', o.ns, 'fit', zeros(1,k), 'predict', zeros(1,k));
for i = 1:k
    [X, y] = make_data(o.ns(i), o.d_fixed, o.seed);
    Xt = rand(o.n_pred, o.d_fixed);
    [r.fit(i), fit] = time_fit(X, y, o.m_fixed, o.reps);
    r.predict(i) = time_pred(fit, Xt, o.m_pred, o.reps);
    if o.verbose
        row({o.ns(i), r.fit(i), r.predict(i), 1000*r.fit(i)/o.ns(i)});
    end
end
r.fit_exponent     = loglog_slope(o.ns, r.fit);
r.predict_exponent = loglog_slope(o.ns, r.predict);
if o.verbose
    fprintf('   empirical exponent: fit n^%.2f, predict n^%.2f\n', ...
        r.fit_exponent, r.predict_exponent);
    fprintf('   (fit mixes O(n) likelihood work with O(n^2) ordering/neighbours)\n');
end
end

% -------------------------------------------------------------------------
function r = sweep_d(o)
if o.verbose
    fprintf('\n-- scaling in d (n = %d, m = %d)\n', o.n_fixed, o.m_fixed);
    header({'d', 'fit (s)', 'predict (s)', 'n active parms'});
end
k = numel(o.ds);
r = struct('d', o.ds, 'fit', zeros(1,k), 'predict', zeros(1,k));
for i = 1:k
    [X, y] = make_data(o.n_fixed, o.ds(i), o.seed);
    Xt = rand(o.n_pred, o.ds(i));
    [r.fit(i), fit] = time_fit(X, y, o.m_fixed, o.reps);
    r.predict(i) = time_pred(fit, Xt, o.m_pred, o.reps);
    if o.verbose
        row({o.ds(i), r.fit(i), r.predict(i), o.ds(i)+1}, '%18d');
    end
end
r.fit_exponent = loglog_slope(o.ds, r.fit);
if o.verbose
    fprintf('   empirical exponent: fit d^%.2f\n', r.fit_exponent);
    fprintf('   (one extra range parameter per input, each adding a gradient term)\n');
end
end

% -------------------------------------------------------------------------
function r = sweep_m(o)
if o.verbose
    fprintf('\n-- scaling in m (n = %d, d = %d)\n', o.n_fixed, o.d_fixed);
    header({'m', 'fit (s)', 'predict (s)', 'fit/m^3 (us)'});
end
k = numel(o.ms);
r = struct('m', o.ms, 'fit', zeros(1,k), 'predict', zeros(1,k));
[X, y] = make_data(o.n_fixed, o.d_fixed, o.seed);
Xt = rand(o.n_pred, o.d_fixed);
for i = 1:k
    [r.fit(i), fit] = time_fit(X, y, o.ms(i), o.reps);
    r.predict(i) = time_pred(fit, Xt, o.ms(i), o.reps);
    if o.verbose
        row({o.ms(i), r.fit(i), r.predict(i), 1e6*r.fit(i)/o.ms(i)^3});
    end
end
r.fit_exponent     = loglog_slope(o.ms, r.fit);
r.predict_exponent = loglog_slope(o.ms, r.predict);
if o.verbose
    fprintf('   empirical exponent: fit m^%.2f, predict m^%.2f\n', ...
        r.fit_exponent, r.predict_exponent);
    fprintf('   (each block is (m+1)x(m+1); factorization is cubic, setup quadratic)\n');
end
end

% -------------------------------------------------------------------------
function r = sweep_mcmc(o)
% The calibration question: what does one more iteration cost?
if o.verbose
    fprintf('\n-- repeated prediction (n = %d, d = %d, m_pred = %d)\n', ...
        o.n_fixed, o.d_fixed, o.m_pred);
end
[X, y] = make_data(o.n_fixed, o.d_fixed, o.seed);
[~, fit] = time_fit(X, y, o.m_fixed, 1);

r = struct();

% (a) fixed prediction inputs: prepare once, draw in the loop
Xt = rand(o.n_pred, o.d_fixed);
t = tic; p = sv_predict(fit, Xt, 'm', o.m_pred, 'joint', false, 'variance', true); %#ok<NASGU>
r.predict_once = toc(t);
t = tic; prep = sv_prepare(fit, Xt, 'm', o.m_pred, 'joint', false); r.prepare = toc(t);
t = tic;
for it = 1:o.niter
    q = sv_draw(prep, 'nsims', 1, 'variance', true); %#ok<NASGU>
end
r.draw = toc(t)/o.niter;

% (b) new inputs each iteration, as in calibration
npts = 1;
Q = rand(o.niter*npts, o.d_fixed);
t = tic;
for it = 1:o.niter
    p = sv_predict(fit, Q((it-1)*npts+1:it*npts,:), 'm', o.m_pred, ...
        'joint', false, 'variance', true); %#ok<NASGU>
end
r.new_inputs = toc(t)/o.niter;

if o.verbose
    fprintf('  %-22s %14s %10s  %s\n', 'pattern', 'per iteration', 'setup', 'note');
    fprintf('  %s\n', repmat('-', 1, 72));
    fprintf('  %-22s %14s %10s  %s\n', 'sv_predict (one-shot)', ...
        sprintf('%.2f ms', 1000*r.predict_once), '-', 'full rebuild every call');
    fprintf('  %-22s %14s %10s  %s\n', 'sv_prepare + sv_draw', ...
        sprintf('%.4f ms', 1000*r.draw), sprintf('%.2f s', r.prepare), ...
        'fixed inputs; setup paid once');
    fprintf('  %-22s %14s %10s  %s\n', 'new inputs each iter', ...
        sprintf('%.2f ms', 1000*r.new_inputs), '-', ...
        'calibration; uses the fit-time cache');
    fprintf('   speedup from prepare/draw at fixed inputs: %.0fx\n', ...
        r.predict_once / r.draw);
end
end

% =========================================================================
function [X, y] = make_data(n, d, seed)
set_seed(seed);
X = rand(n, d);
if d >= 8
    y = sv_borehole(X(:,1:8));
else
    % a smooth surface with unequal input relevance, so the ranges differ
    w = 1 ./ (1:d);
    y = sin(3*X*w') + 0.5*(X*w').^2;
end
end

function [t, fit] = time_fit(X, y, m, reps)
best = Inf;
for r = 1:reps
    tt = tic;
    fit = sv_fit(X, y, 'm', m, 'nu', 3.5, 'nugget', 0, 'vcf', false);
    best = min(best, toc(tt));
end
if isobject(fit) || (isstruct(fit) && isfield(fit, 'model'))
    fit = fit.model;      % unwrap sv_model
end
t = best;
end

function t = time_pred(fit, Xt, m, reps)
best = Inf;
for r = 1:reps
    tt = tic;
    p = sv_predict(fit, Xt, 'm', m, 'joint', false, 'variance', true); %#ok<NASGU>
    best = min(best, toc(tt));
end
t = best;
end

function b = loglog_slope(x, t)
% least-squares slope of log(time) on log(x): the empirical exponent
good = t > 0 & isfinite(t);
if sum(good) < 2, b = NaN; return; end
lx = log(x(good)); lt = log(t(good));
b = sum((lx - mean(lx)) .* (lt - mean(lt))) / sum((lx - mean(lx)).^2);
end

function header(c)
fprintf('  %-10s %12s %14s %18s\n', c{1}, c{2}, c{3}, c{4});
fprintf('  %s\n', repmat('-', 1, 58));
end

function row(c, last_fmt)
if nargin < 2, last_fmt = '%18.3f'; end
fprintf(['  %-10d %12.2f %14.2f ' last_fmt '\n'], c{1}, c{2}, c{3}, c{4});
end

function set_seed(s)
try
    rng(s);
catch
    rand('state', s); randn('state', s);   %#ok<RAND>
end
end

function info = env_info()
info = struct();
has_tree = ~isempty(which('KDTreeSearcher'));
has_page = ~isempty(which('pagemtimes'));
has_mink = ~isempty(which('mink'));
if exist('OCTAVE_VERSION', 'builtin')
    plat = sprintf('Octave %s', version());
else
    plat = sprintf('MATLAB %s', version('-release'));
end
info.has_tree = has_tree;
info.has_page = has_page;
info.line = sprintf('%s | k-d tree: %s | pagemtimes: %s | mink: %s', ...
    plat, yesno(has_tree), yesno(has_page), yesno(has_mink));
end

function s = yesno(b)
if b, s = 'yes'; else, s = 'no'; end
end
