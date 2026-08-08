function [f, g] = sv_matern(r, nu)
%SV_MATERN  Isotropic Matern correlation and scaled derivative.
%
%   [F,G] = SV_MATERN(R,NU) returns
%       F = M_nu(R)                (Matern correlation, unit range)
%       G = M_nu'(R) ./ R          (derivative divided by R)
%
%   NU in {0.5, 1.5, 2.5, 3.5} uses closed forms and avoids Bessel-function
%   evaluations, which is why those values are the fast default (and the ones
%   GpGp exposes as matern*_scaledim).  Any other NU > 0 uses the general
%   representation
%       M_nu(r) = 2^(1-nu)/gamma(nu) * r^nu * K_nu(r),
%   evaluated in the log domain with the exponentially scaled Bessel function
%   so that neither the r^nu blow-up nor the exp(-r) decay overflows.  The
%   derivative uses the identity d/dx [x^nu K_nu(x)] = -x^nu K_{nu-1}(x), so
%       M_nu'(r)/r = -2^(1-nu)/gamma(nu) * r^(nu-1) * K_{nu-1}(r).
%
%   G is the form needed for derivatives with respect to the range parameters,
%   because
%       d/d(lambda_k) M(r) = -(M'(r)/r) * (x_k - x'_k)^2 / lambda_k^3 .
%   At R = 0 the multiplying squared difference vanishes, so G is set to 0
%   there rather than to its (possibly infinite) limit.

switch nu
    case 0.5
        e = exp(-r);
        f = e;
        if nargout > 1
            g = -e ./ r;
            g(r == 0) = 0;
        end
    case 1.5
        e = exp(-r);
        f = (1 + r) .* e;
        if nargout > 1, g = -e; end
    case 2.5
        e = exp(-r);
        f = (1 + r + r.^2/3) .* e;
        if nargout > 1, g = -(1 + r) .* e / 3; end
    case 3.5
        e = exp(-r);
        f = (1 + r + 2*r.^2/5 + r.^3/15) .* e;
        if nargout > 1, g = -(3 + 3*r + r.^2) .* e / 15; end
    otherwise
        if ~isfinite(nu) || nu <= 0
            error('sv_matern:nu', 'smoothness must be a positive finite number (got %g).', nu);
        end
        f = ones(size(r));
        pos = r > 0;
        rp = r(pos);
        lc = (1-nu)*log(2) - gammaln(nu);
        % besselk(., ., 1) returns K(.)*exp(r), so log K = log(scaled) - r
        f(pos) = exp(lc + nu*log(rp) + log(besselk(nu, rp, 1)) - rp);
        if nargout > 1
            g = zeros(size(r));
            % K_{-a} = K_a, so use the absolute order for nu < 1
            g(pos) = -exp(lc + (nu-1)*log(rp) + log(besselk(abs(nu-1), rp, 1)) - rp);
        end
end
end
