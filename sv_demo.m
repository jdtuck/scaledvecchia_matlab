function results = sv_demo(n, ntest, m)
%SV_DEMO  Two demonstrations of the scaled Vecchia approximation.
%
%   PART 1 reproduces the central claim of Katzfuss et al. (2020): with the
%   ordering and the conditioning sets chosen in the *scaled* input space, the
%   Vecchia loglikelihood is far closer to the exact GP loglikelihood than
%   with the usual unscaled ordering, at identical cost.  Data are simulated
%   from a known anisotropic Matern GP whose range parameters span three
%   orders of magnitude, so the exact likelihood is available for reference.
%
%   PART 2 emulates the 8-input borehole function: estimate, predict, score,
%   and draw joint conditional simulations.
%
%   RESULTS = SV_DEMO(N, NTEST, M) with defaults N=1500, NTEST=500, M=30.

if nargin < 1 || isempty(n),     n = 1500;  end
if nargin < 2 || isempty(ntest), ntest = 500; end
if nargin < 3 || isempty(m),     m = 30;    end

try rng(7); catch, rand('state',7); randn('state',7); end %#ok<CTCH>

results = struct();

% ----------------------------------------------------------------------
%  Part 1: how well does the approximation reproduce the exact likelihood?
% ----------------------------------------------------------------------
fprintf('\n--- Part 1: likelihood accuracy, simulated anisotropic GP ---\n');

d1 = 5; nu = 2.5; n1 = min(n, 1500);
parms = [1, 0.10, 0.20, 1.0, 10, 100, nu, 0];  % variance, 5 ranges, smoothness, nugget
locs = rand(n1, d1);

r2 = zeros(n1);
for k = 1:d1
    r2 = r2 + ((locs(:,k) - locs(:,k)') / parms(1+k)).^2;
end
K  = parms(1) * sv_matern(sqrt(r2), nu) + 1e-10*eye(n1);
Lf = chol(K, 'lower');
y1 = Lf * randn(n1, 1);
ll_exact = -0.5*n1*log(2*pi) - sum(log(diag(Lf))) - 0.5*sum((Lf\y1).^2);

fprintf('n = %d, d = %d, true ranges = %s\n', n1, d1, mat2str(parms(2:d1+1)));
fprintf('exact loglikelihood %12.2f\n\n', ll_exact);
fprintf('  %3s  %-10s %14s %14s\n', 'm', 'ordering', 'loglik', 'KL to exact');

act = true(1, d1+3);
tab = zeros(0,3);
for mm = [10 20 40]
    for v = {'parms', 'none'}
        if strcmp(v{1}, 'parms')
            s = 1 ./ parms(2:d1+1); lab = 'scaled';
        else
            s = ones(1, d1);        lab = 'unscaled';
        end
        ord = sv_maxmin_order(locs .* s);
        NN  = sv_nn(locs(ord,:) .* s, mm);
        ll  = sv_loglik(parms, y1(ord), zeros(n1,0), locs(ord,:), NN, act, 1e-10, false);
        fprintf('  %3d  %-10s %14.2f %14.2f\n', mm, lab, ll, ll_exact - ll);
        tab(end+1,:) = [mm, ll, ll_exact-ll]; %#ok<AGROW>
    end
end
results.likelihood_table = tab;

% ----------------------------------------------------------------------
%  Part 2: emulating the borehole function
% ----------------------------------------------------------------------
fprintf('\n--- Part 2: borehole emulation, n = %d, m = %d ---\n', n, m);

d = 8;
inputs = lhs(n, d);
y = sv_borehole(inputs);
inputs_test = rand(ntest, d);
ytest = sv_borehole(inputs_test);
fprintf('response range [%.1f, %.1f], sd %.2f\n\n', min(y), max(y), std(y));

t0 = tic;
fit = sv_fit(y, inputs, 'm', m, 'nu', 3.5, 'nugget', 0, 'scale', 'parms');
tfit = toc(t0);

t1 = tic;
p = sv_predict(fit, inputs_test, 'm', 100, 'joint', false, 'variance', true);
tpred = toc(t1);

err  = ytest - p.mean;
rmse = sqrt(mean(err.^2));
score = mean(0.5*log(2*pi*p.var) + 0.5*err.^2 ./ p.var);
cover = mean(abs(err) <= 1.96*sqrt(p.var));

fprintf('estimated variance %.4g, nugget %.3g, vcf %.3f\n', fit.parms(1), fit.parms(end), fit.vcf);
fprintf('estimated ranges   %s\n', mat2str(fit.parms(2:d+1), 3));
fprintf('(a small range means an influential input; the borehole response is\n');
fprintf(' driven mainly by input 1, the borehole radius r_w)\n\n');
fprintf('fit %.1f s, prediction at %d points %.2f s\n', tfit, ntest, tpred);
fprintf('RMSE %.4f (%.2e of the response sd), mean log score %.3f, 95%% coverage %.3f\n', ...
    rmse, rmse/std(ytest), score, cover);

t2 = tic;
ps = sv_predict(fit, inputs_test, 'm', 100, 'joint', true, 'nsims', 100);
fprintf('100 joint conditional simulations in %.2f s; ', toc(t2));
lo = quantile_(ps.samples, 0.05); hi = quantile_(ps.samples, 0.95);
fprintf('90%% simulation-interval coverage %.3f\n', mean(ytest >= lo & ytest <= hi));

% ----------------------------------------------------------------------
%  Part 3: estimating the smoothness as well
% ----------------------------------------------------------------------
fprintf('\n--- Part 3: estimating the Matern smoothness ---\n');
nsub = min(n, 800);
t3 = tic;
fitnu = sv_fit(y(1:nsub), inputs(1:nsub,:), 'm', m, 'nu', 'estimate', ...
    'nugget', 0, 'scale', 'parms', 'vcf', false);
fprintf('n = %d: nu_hat = %.3f  (fixed-nu fit used %.1f), %.1f s\n', ...
    nsub, fitnu.nu, 3.5, toc(t3));
pn = sv_predict(fitnu, inputs_test, 'm', 100, 'joint', false, 'variance', true);
fprintf('RMSE with estimated smoothness %.4f\n', sqrt(mean((ytest - pn.mean).^2)));
results.fit_nu = fitnu;

results.fit  = fit;
results.pred = p;
results.sims = ps;
results.rmse = rmse;
results.logscore = score;
fprintf('\n');

if nargout == 0
    clear results
end
end

% -------------------------------------------------------------------------
function u = lhs(n, d)
% plain Latin hypercube sample
u = zeros(n, d);
for k = 1:d
    u(:,k) = (randperm(n)' - rand(n,1)) / n;
end
end

function q = quantile_(S, p)
Ss = sort(S, 2);
ns = size(S, 2);
pos = max(1, min(ns, round(p*ns)));
q = Ss(:, pos);
end
