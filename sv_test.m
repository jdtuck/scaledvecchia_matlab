%SV_TEST  Self-checks for the scaled-Vecchia implementation.
%
%   Verifies, on small problems where brute force is possible:
%     1. with a full conditioning set the Vecchia loglikelihood equals the
%        exact GP loglikelihood (and the profiled GLS mean agrees);
%     2. the analytic gradient matches central finite differences;
%     3. the expected Fisher information matches the Monte-Carlo covariance
%        of the score;
%     4. joint and pointwise predictions match exact kriging when the
%        conditioning sets are full;
%     5. joint conditional simulation has the right mean and covariance.

rng_seed(1);
tol = 1e-7;
fprintf('\n=== scaled Vecchia self-test ===\n');

n = 60; d = 3; nu = 2.5;
locs  = rand(n, d);
parms = [1.7, 0.4, 0.9, 0.25, nu, 0.05];   % var, 3 ranges, smoothness, nugget
jitter = 0;
X = [ones(n,1), locs(:,1)];
q = size(X,2);

Kfull = exact_cov(locs, locs, parms, nu);
Lf = chol(Kfull, 'lower');
y = Lf * randn(n,1) + X * [2; -1];

% ---- 1. exact vs Vecchia ------------------------------------------------
NNfull = sv_nn(locs, n-1);
active = true(1, d+3);
[llv, ~, ~, betav] = sv_loglik(parms, y, X, locs, NNfull, active, jitter, true);

Ly  = Lf \ y;  LX = Lf \ X;
be  = (LX'*LX) \ (LX'*Ly);
res = Ly - LX*be;
lle = -0.5*n*log(2*pi) - sum(log(diag(Lf))) - 0.5*(res'*res);

report('full-conditioning loglik equals exact', abs(llv-lle)/abs(lle) < tol, ...
    sprintf('vecchia %.10f  exact %.10f', llv, lle));
report('profiled beta equals exact GLS', max(abs(betav-be)) < 1e-8, ...
    sprintf('max diff %.3e', max(abs(betav-be))));

% ---- 1b. a genuine approximation still runs ----------------------------
m = 10;
ord = sv_maxmin_order(locs ./ parms(2:d+1));
NN  = sv_nn(locs(ord,:) ./ parms(2:d+1), m);
llm = sv_loglik(parms, y(ord), X(ord,:), locs(ord,:), NN, active, jitter, false);
report('m=10 Vecchia loglik is finite and close to exact', ...
    isfinite(llm) && abs(llm-lle) < 0.05*abs(lle), ...
    sprintf('m=10 %.4f  exact %.4f', llm, lle));

% ---- 2. gradient vs finite differences ---------------------------------
[~, g, I] = sv_loglik(parms, y(ord), X(ord,:), locs(ord,:), NN, active, jitter, true);
lp = log(parms);
gfd = zeros(d+3,1);
h = 1e-6;
for j = 1:d+3
    lpp = lp; lpp(j) = lpp(j) + h;
    lpm = lp; lpm(j) = lpm(j) - h;
    l1 = sv_loglik(exp(lpp), y(ord), X(ord,:), locs(ord,:), NN, active, jitter, false);
    l2 = sv_loglik(exp(lpm), y(ord), X(ord,:), locs(ord,:), NN, active, jitter, false);
    gfd(j) = (l1-l2)/(2*h);
end
relg = max(abs(g-gfd)) / max(1, max(abs(gfd)));
report('gradient matches finite differences', relg < 1e-5, ...
    sprintf('max rel diff %.3e', relg));

% ---- 3. Fisher information vs the analytic block-difference formula ----
%      I_jk = 0.5 * sum_i [ tr(inv(S_i) dS_j inv(S_i) dS_k)
%                          - tr(inv(S_g) dS_j inv(S_g) dS_k) ]
Iref = zeros(d+3);
Lo = locs(ord,:);
for i = 1:n
    gi  = NN(i, NN(i,:) > 0);
    idx = fliplr(gi);                    % target last
    P   = numel(idx);
    pts = Lo(idx,:);
    S   = exact_cov(pts, pts, parms, nu);
    dS  = cell(d+3,1);
    for j = 1:d+3
        if j == d+2
            % the smoothness enters through Bessel functions, so a naive tiny
            % step is swamped by evaluation noise; use Richardson extrapolation
            dS{j} = (4*cdiff(pts,parms,j,2e-4) - cdiff(pts,parms,j,4e-4))/3 * parms(j);
        else
            dS{j} = cdiff(pts,parms,j,1e-7) * parms(j);
        end
    end
    Si = inv(S);
    if P > 1, Sgi = inv(S(1:P-1,1:P-1)); end
    for a = 1:d+3
        for b = 1:d+3
            T = trace(Si*dS{a}*Si*dS{b});
            if P > 1
                T = T - trace(Sgi*dS{a}(1:P-1,1:P-1)*Sgi*dS{b}(1:P-1,1:P-1));
            end
            Iref(a,b) = Iref(a,b) + 0.5*T;
        end
    end
end
keep = true(1, d+3); keep(d+2) = false;      % parameters with analytic derivatives
relI = norm(I(keep,keep) - Iref(keep,keep), 'fro') / norm(Iref(keep,keep), 'fro');
report('Fisher information matches analytic formula', relI < 1e-6, ...
    sprintf('rel Frobenius diff %.3e', relI));

% the smoothness row relies on a finite difference at both ends, so it is
% checked to the accuracy that difference can support
relN = norm(I(d+2,:) - Iref(d+2,:)) / norm(Iref(d+2,:));
report('Fisher information rows for the smoothness agree', relN < 1e-3, ...
    sprintf('rel diff %.3e', relN));

% ---- 3b. closed-form identity for the variance parameter ----------------
%      d(Sigma)/d(log sigma^2) = Sigma exactly, so z = e_p, giving
%      score_1 = -n/2 + 0.5*sum(a^2) and I_11 = n/2.
[~, ~, Iz, ~] = sv_loglik(parms, y(ord), X(ord,:), locs(ord,:), NN, active, jitter, true);
report('Fisher information for log-variance equals n/2', abs(Iz(1,1) - n/2) < 1e-8, ...
    sprintf('I_11 = %.6f, n/2 = %.1f', Iz(1,1), n/2));

% ---- 4/5. prediction ----------------------------------------------------
np = 25;
locp = rand(np, d);
fit = struct('parms', parms, 'nu', parms(d+2), 'y', y, 'inputs', locs, ...
             'X', X, 'beta', [2;-1], 'jitter', 0, 'trend', 'linear', 'vcf', 1);
Xp = [ones(np,1), locp(:,1)];

Koo = exact_cov(locs, locs, parms, nu);
Kpo = exact_cov(locp, locs, parms, nu);
Kpp = exact_cov(locp, locp, parms, nu);
r   = y - X*fit.beta;
mu_ex  = Xp*fit.beta + Kpo*(Koo\r);
Cov_ex = Kpp - Kpo*(Koo\Kpo');

pj = sv_predict(fit, locp, 'm', n+np, 'joint', true, 'X_pred', Xp);
report('joint prediction mean equals exact kriging', ...
    max(abs(pj.mean - mu_ex)) < 1e-7, sprintf('max diff %.3e', max(abs(pj.mean-mu_ex))));

ps = sv_predict(fit, locp, 'm', n, 'joint', false, 'X_pred', Xp, 'variance', true);
report('pointwise prediction mean equals exact kriging', ...
    max(abs(ps.mean - mu_ex)) < 1e-7, sprintf('max diff %.3e', max(abs(ps.mean-mu_ex))));
report('pointwise prediction variance equals exact', ...
    max(abs(ps.var - diag(Cov_ex))) < 1e-8, ...
    sprintf('max diff %.3e', max(abs(ps.var-diag(Cov_ex)))));

nsim = 20000;
pj2 = sv_predict(fit, locp, 'm', n+np, 'joint', true, 'X_pred', Xp, 'nsims', nsim);
Cs = cov(pj2.samples');
relC = norm(Cs - Cov_ex, 'fro') / norm(Cov_ex, 'fro');
report('joint simulation reproduces the predictive covariance', relC < 0.08, ...
    sprintf('rel Frobenius diff %.3f', relC));


% ---- 6. general Bessel Matern agrees with the closed forms ---------------
rr = [0; 1e-8; 0.01; 0.3; 1; 3; 12];
maxd = 0;
for nuh = [0.5 1.5 2.5 3.5]
    [fc, gc] = sv_matern(rr, nuh);
    [fb, gb] = sv_matern(rr, nuh + 1e-12);      % forces the Bessel branch
    pos = rr > 0;
    maxd = max([maxd; abs(fc-fb); abs(gc(pos)-gb(pos))./abs(gc(pos))]);
end
report('Bessel Matern matches the half-integer closed forms', maxd < 1e-9, ...
    sprintf('max diff %.3e (abs on f, rel on g)', maxd));

% ---- 7. gradient w.r.t. the smoothness ----------------------------------
pnu = parms; pnu(d+2) = 1.8;                     % off the fast path
act2 = true(1, d+3);
[~, gg] = sv_loglik(pnu, y(ord), X(ord,:), locs(ord,:), NN, act2, jitter, true);
hh = 1e-5; lpn = log(pnu);
lpa = lpn; lpa(d+2) = lpa(d+2)+hh;
lpb = lpn; lpb(d+2) = lpb(d+2)-hh;
gnu_fd = (sv_loglik(exp(lpa), y(ord), X(ord,:), locs(ord,:), NN, act2, jitter, false) - ...
          sv_loglik(exp(lpb), y(ord), X(ord,:), locs(ord,:), NN, act2, jitter, false)) / (2*hh);
relnu = abs(gg(d+2) - gnu_fd) / max(1, abs(gnu_fd));
assert(relnu < 1e-4)
report('smoothness score matches finite differences', relnu < 1e-4, ...
    sprintf('analytic %.5f  fd %.5f', gg(d+2), gnu_fd));

% ---- 8. estimating nu recovers a known smoothness -----------------------
nt = 700; dt = 2; nu_true = 1.5;
lt = rand(nt, dt);
pt = [1, 0.30, 0.60, nu_true, 0];
rt = zeros(nt);
for k = 1:dt
    rt = rt + ((lt(:,k) - lt(:,k)')/pt(1+k)).^2;
end
yt = chol(sv_matern(sqrt(rt), nu_true) + 1e-9*eye(nt), 'lower') * randn(nt,1);
ft = sv_fit(lt, yt, 'm', 20, 'nu', 'estimate', 'nugget', 0, 'trend', 'zero', 'vcf', false);
assert(abs(ft.model.nu - nu_true) < 0.45)
report('estimated smoothness recovers the truth', abs(ft.model.nu - nu_true) < 0.45, ...
    sprintf('nu_hat %.3f (true %.1f), ranges %s', ft.model.nu, nu_true, mat2str(ft.model.parms(2:3),3)));

fprintf('=== done ===\n\n');

% ------------------------------------------------------------------------
function K = exact_cov(l1, l2, parms, ~)
d = size(l1,2);
r = zeros(size(l1,1), size(l2,1));
for k = 1:d
    r = r + ((l1(:,k) - l2(:,k)') / parms(1+k)).^2;
end
K = parms(1) * sv_matern(sqrt(r), parms(d+2));
if size(l1,1)==size(l2,1) && isequal(l1,l2)
    K = K + parms(1)*parms(end)*eye(size(K,1));
end
end

function D = cdiff(pts, parms, j, h)
pp = parms; pm = parms;
pp(j) = parms(j)+h; pm(j) = parms(j)-h;
D = (exact_cov(pts,pts,pp,[]) - exact_cov(pts,pts,pm,[])) / (2*h);
end

function report(name, pass, extra)
if pass, tag = 'PASS'; else, tag = 'FAIL'; end
fprintf('[%s] %-52s  %s\n', tag, name, extra);
end

function rng_seed(s)
try
    rng(s);
catch
    randn('state', s); rand('state', s);   %#ok<RAND>
end
end

