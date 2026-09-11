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
%     'reps'      repetitions of each timing, minimum taken   (default 3)
%     'warmup'    run an untimed fit and predict first         (default true)
%     'min_time'  repeat short timings until they exceed this  (default 0.05 s)
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
    'sweeps', {{'n','d','m','mcmc'}}, 'reps', 3, 'verbose', true, ...
    'warmup', true, 'min_time', 0.05, 'seed', 1), varargin);

if ischar(o.sweeps), o.sweeps = {o.sweeps}; end

results = struct();
results.info = env_info();

sweep_warn(o);

if o.verbose
    fprintf('\n=== scaled Vecchia scaling benchmark ===\n');
    fprintf('%s\n', results.info.line);
end

% MATLAB compiles each function on first execution.  Without a warm-up that
% cost lands entirely on the first row of the first sweep, which made fitting
% look sublinear in n and prediction look like it got *faster* with more
% training data.  Both are impossible; both were this.
if o.warmup
    if o.verbose, fprintf('warming up...\n'); end
    warm_up();
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
    header({'n', 'fit (s)', 'grad eval (ms)', 'predict (ms)'});
end
k = numel(o.ns);
r = struct('n', o.ns, 'fit', zeros(1,k), 'kernel', zeros(1,k), 'predict', zeros(1,k));
for i = 1:k
    [X, y] = make_data(o.ns(i), o.d_fixed, o.seed);
    Xt = rand(o.n_pred, o.d_fixed);
    [r.fit(i), fit] = time_fit(X, y, o.m_fixed, o.reps);
    r.kernel(i)  = time_kernel(X, y, o.m_fixed, o.reps, o.min_time);
    r.predict(i) = time_pred(fit, Xt, o.m_pred, o.reps, o.min_time);
    if o.verbose
        row({o.ns(i), r.fit(i), 1000*r.kernel(i), 1000*r.predict(i)});
    end
end
r.fit_exponent     = loglog_slope(o.ns, r.fit);
r.kernel_exponent  = loglog_slope(o.ns, r.kernel);
r.predict_exponent = loglog_slope(o.ns, r.predict);
if o.verbose
    fprintf('   exponents: grad eval n^%.2f | predict n^%.2f | whole fit n^%.2f\n', ...
        r.kernel_exponent, r.predict_exponent, r.fit_exponent);
    fprintf('   (grad eval is the clean one: linear in n plus O(n^2) ordering\n');
    fprintf('    and neighbour work.  predict should be flat -- it depends on\n');
    fprintf('    n_pred and m_pred, not on how much training data there is.\n');
    fprintf('    The whole-fit column also carries the optimizer iteration count.)\n');
end
end

% -------------------------------------------------------------------------
function r = sweep_d(o)
if o.verbose
    fprintf('\n-- scaling in d (n = %d, m = %d)\n', o.n_fixed, o.m_fixed);
    header({'d', 'fit (s)', 'grad eval (ms)', 'predict (ms)'});
end
k = numel(o.ds);
r = struct('d', o.ds, 'fit', zeros(1,k), 'kernel', zeros(1,k), 'predict', zeros(1,k));
for i = 1:k
    [X, y] = make_data(o.n_fixed, o.ds(i), o.seed);
    Xt = rand(o.n_pred, o.ds(i));
    [r.fit(i), fit] = time_fit(X, y, o.m_fixed, o.reps);
    r.kernel(i)  = time_kernel(X, y, o.m_fixed, o.reps, o.min_time);
    r.predict(i) = time_pred(fit, Xt, o.m_pred, o.reps, o.min_time);
    if o.verbose
        row({o.ds(i), r.fit(i), 1000*r.kernel(i), 1000*r.predict(i)});
    end
end
r.fit_exponent    = loglog_slope(o.ds, r.fit);
r.kernel_exponent = loglog_slope(o.ds, r.kernel);
if o.verbose
    fprintf('   exponents: grad eval d^%.2f | whole fit d^%.2f\n', ...
        r.kernel_exponent, r.fit_exponent);
    fprintf('   (d+1 active parameters, so the gradient term is linear in d;\n');
    fprintf('    distances are linear in d too, but that part is now one BLAS call)\n');
end
end

% -------------------------------------------------------------------------
function r = sweep_m(o)
if o.verbose
    fprintf('\n-- scaling in m (n = %d, d = %d)\n', o.n_fixed, o.d_fixed);
    header({'m', 'fit (s)', 'grad eval (ms)', 'predict (ms)'});
end
k = numel(o.ms);
r = struct('m', o.ms, 'fit', zeros(1,k), 'kernel', zeros(1,k), 'predict', zeros(1,k));
[X, y] = make_data(o.n_fixed, o.d_fixed, o.seed);
Xt = rand(o.n_pred, o.d_fixed);
for i = 1:k
    [r.fit(i), fit] = time_fit(X, y, o.ms(i), o.reps);
    r.kernel(i)  = time_kernel(X, y, o.ms(i), o.reps, o.min_time);
    r.predict(i) = time_pred(fit, Xt, o.ms(i), o.reps, o.min_time);
    if o.verbose
        row({o.ms(i), r.fit(i), 1000*r.kernel(i), 1000*r.predict(i)});
    end
end
r.fit_exponent     = loglog_slope(o.ms, r.fit);
r.kernel_exponent  = loglog_slope(o.ms, r.kernel);
r.predict_exponent = loglog_slope(o.ms, r.predict);
if o.verbose
    fprintf('   exponents: grad eval m^%.2f | predict m^%.2f | whole fit m^%.2f\n', ...
        r.kernel_exponent, r.predict_exponent, r.fit_exponent);
    fprintf('   (blocks are (m+1)x(m+1): factorization cubic, covariance setup\n');
    fprintf('    quadratic, so expect the kernel between 2 and 3 and drifting up.\n');
    fprintf('    The whole-fit column also carries the optimizer iteration count,\n');
    fprintf('    which moves with m for reasons unrelated to per-evaluation cost.)\n');
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
function sweep_warn(o)
if ~o.verbose, return; end
short = {};
if any(strcmp(o.sweeps,'n')) && numel(o.ns) < 4, short{end+1} = 'ns'; end
if any(strcmp(o.sweeps,'d')) && numel(o.ds) < 4, short{end+1} = 'ds'; end
if any(strcmp(o.sweeps,'m')) && numel(o.ms) < 4, short{end+1} = 'ms'; end
if ~isempty(short)
    fprintf(['NOTE: %s has fewer than 4 points, so the fitted exponents are\n' ...
             '      indicative only.  Fixed per-call overhead also flattens the\n' ...
             '      slope at small sizes; sweep at least a decade for a number\n' ...
             '      worth quoting.\n'], strjoin(short, ', '));
end
end

function warm_up()
% Touch every code path the sweeps will time, at negligible size.
Xw = rand(80, 3); yw = sin(3*Xw(:,1));
fw = sv_fit(Xw, yw, 'm', 8, 'nu', 3.5, 'nugget', 0, 'vcf', false);
if isobject(fw) || (isstruct(fw) && isfield(fw, 'model')), fw = fw.model; end
sv_predict(fw, rand(20,3), 'm', 10, 'joint', false, 'variance', true);
pw = sv_prepare(fw, rand(20,3), 'm', 10, 'joint', false);
sv_draw(pw, 'nsims', 1, 'variance', true);
sv_predict(fw, rand(20,3), 'm', 10, 'joint', true, 'nsims', 2);
end

function t = time_kernel(X, y, m, reps, min_time)
% One loglikelihood-with-gradient evaluation at fixed parameters.
%
% This is the quantity with a clean exponent.  Time-to-convergence also
% depends on how many Fisher-scoring steps the optimizer happens to take,
% which varies with n, d and m for reasons that have nothing to do with the
% cost of a single evaluation, so the two must be reported separately.
[n, d] = size(X);
parms  = [var(y), 0.3*ones(1,d), 3.5, 0];
active = true(1, d+3); active(d+2) = false; active(d+3) = false;
sc  = 1 ./ parms(2:d+1);
ord = sv_maxmin_order(X .* sc);
NN  = sv_nn(X(ord,:) .* sc, min(m, n-1));
yo  = y(ord); Lo = X(ord,:); Xo = ones(n,1);
t = repeat_timed(@() sv_loglik(parms, yo, Xo, Lo, NN, active, 1e-12, true), ...
                 reps, min_time);
end

function t = repeat_timed(f, reps, min_time)
% Minimum over reps, each rep looped until it clears min_time so that short
% measurements are not dominated by timer resolution.
best = Inf;
for r = 1:reps
    k = 1;
    while true
        tt = tic;
        for i = 1:k, f(); end
        el = toc(tt);
        if el >= min_time || k >= 1e6, break; end
        k = max(2*k, ceil(k * min_time / max(el, 1e-6)));
    end
    best = min(best, el/k);
end
t = best;
end

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

function t = time_pred(fit, Xt, m, reps, min_time)
t = repeat_timed(@() sv_predict(fit, Xt, 'm', m, 'joint', false, ...
    'variance', true), reps, min_time);
end

function b = loglog_slope(x, t)
% Least-squares slope of log(time) on log(x): the empirical exponent.
%
% Needs several points spanning a wide range to mean anything.  With a short
% sweep, or one that starts small, fixed per-call overhead flattens the slope
% and the exponent reads low; SWEEP_WARN says so rather than letting the
% number be quoted on its own.
good = t > 0 & isfinite(t);
if sum(good) < 2, b = NaN; return; end
lx = log(x(good)); lt = log(t(good));
b = sum((lx - mean(lx)) .* (lt - mean(lt))) / sum((lx - mean(lx)).^2);
end

function header(c)
fprintf('  %-10s %12s %14s %18s\n', c{1}, c{2}, c{3}, c{4});
fprintf('  %s\n', repmat('-', 1, 58));
end

function row(c)
fprintf('  %-10d %12.2f %14.3f %18.3f\n', c{1}, c{2}, c{3}, c{4});
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
