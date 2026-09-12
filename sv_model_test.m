%SV_MODEL_TEST  Self-checks for the sv_model wrapper's two prediction paths.
%
%   Path 1: prepare once at fixed inputs, draw repeatedly.
%   Path 2: predict at inputs that change every call (calibration).
%
%   The checks that matter for a calibration chain are the reproducibility
%   ones: with an explicit 'idxSamples' the same index has to return the same
%   surrogate realization every time it is used, and a block of indices has
%   to agree with the same indices requested one at a time.
%
%   Uses the arguments block and RandStream substreams, so MATLAB only.

fprintf('\n=== sv_model self-test ===\n');

try rng(3); catch, rand('state', 3); randn('state', 3); end %#ok<CTCH>

n = 200; d = 2;
Xtr = rand(n, d);
ytr = sin(3*Xtr(:,1)) + 0.5*Xtr(:,2) + 0.01*randn(n,1);
obj = sv_fit(Xtr, ytr, 'm', 15, 'nu', 2.5, 'nugget', 0, 'vcf', false);

np = 6;
xq = rand(np, d);

% ---- path 2: predict at arbitrary inputs --------------------------------
[s1, mu1, v1] = obj.predict(xq, 'idxSamples', 7);
mreport('predict returns nsims-by-npred', isequal(size(s1), [1 np]), ...
    sprintf('size %s', mat2str(size(s1))));

s2 = obj.predict(xq, 'idxSamples', 7);
mreport('an explicit sample index is reproducible', isequal(s1, s2), ...
    'same index, same realization');

s3 = obj.predict(xq, 'idxSamples', 8);
mreport('a different index is a different realization', ...
    max(abs(s1 - s3)) > 0, 'index 7 vs 8 differ');

sblk = obj.predict(xq, 'idxSamples', [7 8]);
mreport('a block of indices matches them one at a time', ...
    isequal(size(sblk), [2 np]) && max(max(abs(sblk - [s1; s3]))) < 1e-12, ...
    'rows follow idxSamples order');

p = sv_predict(obj.model, xq, 'm', 100, 'joint', false, 'variance', true);
mreport('mean agrees with sv_predict', max(abs(mu1 - p.mean)) < 1e-12, ...
    sprintf('max diff %.3e', max(abs(mu1 - p.mean))));
mreport('variance agrees with sv_predict', max(abs(v1 - p.var)) < 1e-12, ...
    sprintf('max diff %.3e', max(abs(v1 - p.var))));

sm = obj.predict(xq, 'idxSamples', 1:2000);
rel = max(abs(std(sm, 0, 1)' ./ sqrt(v1) - 1));
mreport('frozen draws reproduce the predictive sd', rel < 0.1, ...
    sprintf('max rel sd error %.3f', rel));

a1 = obj.predict(xq, 'idxSamples', 7, 'crn', false);
a2 = obj.predict(xq, 'idxSamples', 7, 'crn', false);
mreport('crn=false draws fresh randomness', ~isequal(a1, a2), ...
    'opt-out still available');

% ---- path 1: prepare once, draw in a loop -------------------------------
obj2 = obj.prepare(xq);
sd1 = obj2.draw('idxSamples', 7);
mreport('prepare/draw equals predict for the same index', ...
    max(abs(sd1 - s1)) < 1e-12, 'the two paths agree');

sc = obj2.predict(xq, 'idxSamples', 7);
mreport('predict reuses a matching cached plan', isequal(sc, s1), ...
    'cache hit gives identical numbers');

so = obj2.predict(rand(np, d), 'idxSamples', 7);
mreport('a cached plan is not reused for other inputs', ~isequal(so, s1), ...
    'cache miss rebuilds the plan');

sj = obj2.predict(xq, 'idxSamples', 7, 'm', 50);
mreport('a cached plan is not reused for other settings', ~isempty(sj), ...
    'm=50 rebuilds rather than erroring');

% ---- legacy call shapes -------------------------------------------------
sl = obj2.draw('nsims', 3);
mreport('draw(nsims) returns nsims rows', isequal(size(sl), [3 np]), ...
    sprintf('size %s', mat2str(size(sl))));

sn = obj.predict(xq, 'nsims', 5);
mreport('predict(nsims) returns nsims rows', isequal(size(sn), [5 np]), ...
    sprintf('size %s', mat2str(size(sn))));

sa = obj2.draw('idxSamples', nan, 'nsims', 4);
mreport('idxSamples = nan still means "all of them"', ...
    isequal(size(sa), [4 np]), sprintf('size %s', mat2str(size(sa))));

threw = false;
try
    obj.draw('nsims', 1);
catch
    threw = true;
end
mreport('draw without prepare is an error', threw, ...
    'value-class prepare must be assigned');

% ---- staleness: the plan and the observed-side cache follow parms --------
obj3 = obj2;
obj3.model.parms(2) = obj3.model.parms(2) * 1.5;
sst = obj3.predict(xq, 'idxSamples', 7);
mreport('a cached plan is dropped when parms move', ~isequal(sst, s1), ...
    'prep.parms is compared against model.parms');

% Both directions, so the stamp check is shown to have teeth: a cache that
% warps one input 50x picks different neighbours at m = 5, so it changes the
% answer.  Stamped current it is trusted and the answer is wrong; stamped
% stale it is rebuilt and the answer is right.
fs = obj.model;
warp = fs.cache;
warp.scales(1) = warp.scales(1) * 50;
warp.ref = fs.inputs .* warp.scales;
warp.refsq = sum(warp.ref.^2, 2)';
warp.searcher = [];                          % force the path that reads .ref
fs.cache = warp;
ptrust = sv_predict(fs, xq, 'm', 5, 'joint', false, 'variance', true);
fs.cache.parms = fs.parms * 0.5;             % same cache, now stamped stale
prebuilt = sv_predict(fs, xq, 'm', 5, 'joint', false, 'variance', true);
pclean = sv_predict(sv_cache(obj.model), xq, 'm', 5, 'joint', false, ...
    'variance', true);
mreport('a stale observed-side cache is rebuilt', ...
    max(abs(prebuilt.mean - pclean.mean)) < 1e-12, ...
    sprintf('max diff from a clean cache %.3e', ...
    max(abs(prebuilt.mean - pclean.mean))));
mreport('a current stamp is trusted, so the check can bite', ...
    max(abs(ptrust.mean - pclean.mean)) > 1e-8, ...
    sprintf('warped cache moves the mean by %.3e', ...
    max(abs(ptrust.mean - pclean.mean))));

% ---- the calibration shape: one point, one draw -------------------------
x1 = rand(1, d);
[q1, mu1s, v1s] = obj.predict(x1, 'idxSamples', 7);
mreport('one point and one index give exactly one draw', ...
    isequal(size(q1), [1 1]) && isscalar(mu1s) && isscalar(v1s), ...
    'np = 1, nsims = 1, with mean and variance alongside');

mreport('the single-point draw is reproducible', ...
    isequal(q1, obj.predict(x1, 'idxSamples', 7)), ...
    'the calibration call is a fixed function of its input');

% One index is one standard normal, so the standardized residual carries
% across inputs.  This is the property the Metropolis ratio needs: holding
% the index fixed makes the likelihood deterministic in theta.
x2 = rand(1, d);
[q2, mu2s, v2s] = obj.predict(x2, 'idxSamples', 7);
z1 = (q1 - mu1s) / sqrt(v1s);
z2 = (q2 - mu2s) / sqrt(v2s);
mreport('one index is one standard normal across inputs', ...
    abs(z1 - z2) < 1e-12, sprintf('z = %+.6f at both inputs', z1));

fprintf('=== done ===\n\n');

% ------------------------------------------------------------------------
function mreport(name, pass, extra)
if pass, tag = 'PASS'; else, tag = 'FAIL'; end
fprintf('[%s] %-52s  %s\n', tag, name, extra);
assert(pass, name);
end
