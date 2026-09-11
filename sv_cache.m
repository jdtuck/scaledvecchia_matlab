function fit = sv_cache(fit)
%SV_CACHE  Precompute the observed-side quantities used by every prediction.
%
%   FIT = SV_CACHE(FIT) attaches a FIT.cache struct holding everything that
%   depends only on the fitted model, not on where you predict:
%
%     .scales    1/range per input, the warp applied before neighbour search
%     .ref       the observed inputs already in that scaled space
%     .refsq     their squared norms, for the brute-force distance identity
%     .resid     y - X*beta, the residuals the kriging weights act on
%     .searcher  a KDTreeSearcher over .ref when Statistics Toolbox is present
%
%   This is the calibration case: every MCMC iteration proposes a new theta and
%   predicts at a new input, so the plan built by SV_PREPARE cannot be reused.
%   What *can* be reused is all of the above — and recomputing it per call is
%   what makes small predictions cost milliseconds instead of microseconds.
%   Scaling the n observed inputs and forming the residuals is O(n*d) work per
%   call for a prediction that may involve a single point.
%
%   SV_FIT calls this automatically, so a fitted model arrives ready.  Call it
%   again yourself after changing FIT.parms, FIT.y or FIT.beta by hand; the
%   cache is keyed to those and SV_PREDICT trusts it.
%
%   See also SV_PREPARE, SV_PREDICT.

d = size(fit.inputs, 2);

scale = 'parms';
if isfield(fit, 'scale') && ~isempty(fit.scale), scale = fit.scale; end

switch lower(scale)
    case 'parms'
        scales = 1 ./ fit.parms(2:d+1);
    case 'ranges'
        if isfield(fit, 'input_ranges') && ~isempty(fit.input_ranges)
            scales = 1 ./ fit.input_ranges;
        else
            r = max(fit.inputs, [], 1) - min(fit.inputs, [], 1);
            r(r == 0) = 1;
            scales = 1 ./ r;
        end
    case 'none'
        scales = ones(1, d);
    otherwise
        error('sv_cache:scale', 'invalid scale option ''%s''.', scale);
end

c = struct();
c.scales = scales;
c.ref    = fit.inputs .* scales;
c.refsq  = sum(c.ref.^2, 2)';

if size(fit.X, 2) > 0
    c.resid = fit.y - fit.X * fit.beta;
else
    c.resid = fit.y;
end

% A k-d tree turns the per-call neighbour search from O(n_pred*n*d) into
% roughly O(n_pred*log(n)*d).  Statistics Toolbox only; without it SV_PREPARE
% falls back to the cached brute-force quantities above.
c.searcher = [];
if ~isempty(which('KDTreeSearcher'))
    try
        c.searcher = KDTreeSearcher(c.ref);
    catch
        c.searcher = [];
    end
end

fit.cache = c;
end
