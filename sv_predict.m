function pred = sv_predict(fit, inputs_pred, varargin)
%SV_PREDICT  Emulator predictions from a scaled Vecchia fit.
%
%   PRED = SV_PREDICT(FIT, INPUTS_PRED) returns a struct with field .mean
%   holding the posterior mean at the rows of INPUTS_PRED.
%
%   PRED = SV_PREDICT(FIT, INPUTS_PRED, 'name', value, ...) accepts:
%
%     'm'         conditioning-set size for prediction         (default 100)
%     'joint'     true  : one joint Vecchia approximation of observed and
%                         prediction points, which is what you need for
%                         conditional simulation of whole response surfaces
%                 false : each prediction point conditions on its own m
%                         nearest observed runs, giving pointwise variances
%                                                              (default true)
%     'nsims'     number of joint conditional simulations       (default 0)
%     'variance'  return pointwise variances (joint = false)    (default false)
%     'X_pred'    trend matrix at the prediction inputs; built from
%                 FIT.trend if omitted
%     'scale'     'parms' / 'ranges' / 'none', as in SV_FIT
%
%   Output fields: .mean, and .var / .samples when requested.
%
%   Joint prediction works through the sparse Vecchia factor.  Ordering the
%   observed runs first and the prediction points after them (in maximin order
%   within the scaled input space) makes the implied precision matrix of the
%   stacked vector U*U' with U upper triangular, so that
%       y_pred | y_obs  ~  N( -inv(U_pp') * U_op' * r_obs ,  inv(U_pp*U_pp') ),
%   and both the mean and exact joint samples come from sparse triangular
%   solves in O(n_pred * m^2) work.

[np, d] = size(inputs_pred);
n = size(fit.inputs, 1);
if d ~= size(fit.inputs, 2)
    error('sv_predict:dim', 'prediction inputs must have %d columns.', size(fit.inputs,2));
end

o = sv_options(struct('m', 100, 'joint', true, 'nsims', 0, ...
    'variance', false, 'X_pred', [], 'scale', []), varargin);
if isempty(o.scale)
    if isfield(fit, 'scale'), o.scale = fit.scale; else, o.scale = 'parms'; end
end

parms  = fit.parms;
jitter = 0;
if isfield(fit, 'jitter'), jitter = fit.jitter; end
vcf = 1;
if isfield(fit, 'vcf') && ~isempty(fit.vcf), vcf = fit.vcf; end

% ---- trend at the prediction inputs -------------------------------------
Xp = o.X_pred;
if isempty(Xp)
    switch lower(fit.trend)
        case 'zero',                     Xp = zeros(np, 0);
        case {'intercept','constant'},   Xp = ones(np, 1);
        case 'linear',                   Xp = [ones(np,1), inputs_pred];
        otherwise
            error('sv_predict:Xpred', 'X_pred must be supplied for trend ''%s''.', fit.trend);
    end
end

if size(fit.X, 2) > 0
    resid = fit.y - fit.X * fit.beta;
    trendp = Xp * fit.beta;
else
    resid = fit.y;
    trendp = zeros(np, 1);
end

scales = pred_scales(fit, o.scale, d);
locs_all = [fit.inputs; inputs_pred];       % prediction rows get index n + j

pred = struct();

if o.joint
    % ---- ordering: observed first, prediction points in maximin order ----
    ordp = sv_maxmin_order(inputs_pred .* scales);
    locs_all = [fit.inputs; inputs_pred(ordp,:)];
    N = n + np;

    m = min(o.m, n + np - 1);
    NNp = sv_nn(locs_all .* scales, m, (n+1:N)');

    [rowsI, colsI, vals] = vecchia_columns(locs_all, NNp, parms, jitter);
    Uall = sparse(rowsI, colsI, vals, N, np);
    Uop = Uall(1:n, :);
    Upp = Uall(n+1:N, :);

    UppT = Upp';                              % lower triangular
    mu_ord = -(UppT \ (Uop' * resid));

    pm = zeros(np, 1);
    pm(ordp) = mu_ord;
    pred.mean = pm + trendp;

    if o.nsims > 0
        Zs = randn(np, o.nsims) * sqrt(vcf);
        sims_ord = mu_ord + (UppT \ Zs);
        S = zeros(np, o.nsims);
        S(ordp,:) = sims_ord;
        pred.samples = S + trendp;
        if o.variance
            pred.var = var(pred.samples, 0, 2);
        end
    elseif o.variance
        warning('sv_predict:jointvar', ...
            'joint prediction returns variances only via nsims > 0; use joint=false for exact pointwise variances.');
    end

else
    % ---- pointwise: condition each prediction point on its m nearest runs --
    m = min(o.m, n);
    NNo = knn_ref(inputs_pred .* scales, fit.inputs .* scales, m);
    NNp = [n + (1:np)', NNo];                 % column 1 = self, then neighbours

    p = m + 1;
    mu = zeros(np, 1);
    vr = zeros(np, 1);

    chunk = max(1, floor(2e6 / (p*p)));
    for b = 1:chunk:np
        ii = (b:min(b+chunk-1, np))';
        nb = numel(ii);
        IDX = NNp(ii, p:-1:1);                % target last
        lin = IDX(:);

        coords = zeros(nb, p, d);
        for k = 1:d
            coords(:,:,k) = reshape(locs_all(lin, k), [nb p]);
        end
        [C, ~] = sv_covblocks(coords, parms, jitter, false, false);
        [L, ok] = sv_bchol(C);
        if ~ok
            error('sv_predict:chol', 'covariance block not positive definite; try a small nugget or jitter.');
        end
        s = sv_blastrow(L);

        rsub = reshape(resid(IDX(:,1:p-1)), [nb p-1]);
        mu(ii) = -sum(s(:,1:p-1) .* rsub, 2) ./ s(:,p);
        vr(ii) = vcf ./ s(:,p).^2;
    end

    pred.mean = mu + trendp;
    if o.variance
        pred.var = vr;
    end
    if o.nsims > 0
        pred.samples = pred.mean + sqrt(vr) .* randn(np, o.nsims);
    end
end
end

% =========================================================================
function [rowsI, colsI, vals] = vecchia_columns(locs, NNp, parms, jitter)
% Columns of the sparse Vecchia factor U for the rows listed in NNp(:,1).
d = size(locs, 2);
counts = sum(NNp > 0, 2);
rowsI = []; colsI = []; vals = [];

for p = unique(counts)'
    sel = find(counts == p);
    chunk = max(1, floor(2e6 / (p*p)));
    for b = 1:chunk:numel(sel)
        ii = sel(b:min(b+chunk-1, numel(sel)));
        nb = numel(ii);
        IDX = NNp(ii, p:-1:1);                % target last
        lin = IDX(:);

        coords = zeros(nb, p, d);
        for k = 1:d
            coords(:,:,k) = reshape(locs(lin, k), [nb p]);
        end
        [C, ~] = sv_covblocks(coords, parms, jitter, false, false);
        [L, ok] = sv_bchol(C);
        if ~ok
            error('sv_predict:chol', 'covariance block not positive definite; try a small nugget or jitter.');
        end
        s = sv_blastrow(L);                   % nb x p

        cols = repmat(ii(:), 1, p);           % local prediction index
        rowsI = [rowsI; IDX(:)];              %#ok<AGROW>
        colsI = [colsI; cols(:)];             %#ok<AGROW>
        vals  = [vals;  s(:)];                %#ok<AGROW>
    end
end
end

function NN = knn_ref(query, ref, m)
nq = size(query,1);
nr = size(ref,1);
m = min(m, nr);
NN = zeros(nq, m);
sq = sum(ref.^2, 2)';
blk = max(1, floor(4e6 / max(nr,1)));
for b = 1:blk:nq
    ii = b:min(b+blk-1, nq);
    D = sum(query(ii,:).^2, 2) + sq - 2*(query(ii,:) * ref');
    [~, srt] = sort(D, 2);
    NN(ii,:) = srt(:,1:m);
end
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
        error('sv_predict:scale', 'invalid scale option ''%s''.', scale);
end
end
