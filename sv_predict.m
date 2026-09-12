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
%     'nsims'     number of conditional simulations             (default 0)
%     'variance'  return pointwise variances (joint = false)    (default false)
%     'X_pred'    trend matrix at the prediction inputs; built from
%                 FIT.trend if omitted
%     'scale'     'parms' / 'ranges' / 'none', as in SV_FIT
%     'noise_free' predict the latent process rather than a fresh noisy
%                 observation, i.e. drop the nugget from the target's own
%                 variance (default true; irrelevant when the nugget is 0).
%                 Applies to joint = false only: the joint factor carries the
%                 nugget on every diagonal entry
%
%   Output fields: .mean, and .var / .samples when requested.
%
%   This is a convenience wrapper around SV_PREPARE followed by SV_DRAW, and
%   it repeats the whole factorization on every call.  When predicting
%   repeatedly at the same INPUTS_PRED — drawing many realizations, or running
%   an MCMC around a fixed emulator — call SV_PREPARE once and SV_DRAW in the
%   loop instead; only SV_DRAW depends on the response values and the random
%   numbers, so the expensive part leaves the loop entirely.
%
%   Joint prediction works through the sparse Vecchia factor.  Ordering the
%   observed runs first and the prediction points after them (in maximin order
%   within the scaled input space) makes the implied precision matrix of the
%   stacked vector U*U' with U upper triangular, so that
%       y_pred | y_obs  ~  N( -inv(U_pp'') * U_op'' * r_obs ,  inv(U_pp*U_pp'') ),
%   and both the mean and exact joint samples come from sparse triangular
%   solves in O(n_pred * m^2) work.
%
%   See also SV_PREPARE, SV_DRAW.

o = sv_options(struct('m', 100, 'joint', true, 'nsims', 0, ...
    'variance', false, 'X_pred', [], 'scale', [], 'noise_free', true), varargin);

prep = sv_prepare(fit, inputs_pred, 'm', o.m, 'joint', o.joint, ...
    'X_pred', o.X_pred, 'scale', o.scale, 'noise_free', o.noise_free);

pred = sv_draw(prep, 'nsims', o.nsims, 'variance', o.variance);
end
