function dC = sv_covderiv(coords, aux, j, parms, jitter)
%SV_COVDERIV  Derivative of the batched covariance w.r.t. parameter j.
%
%   DC = SV_COVDERIV(COORDS,AUX,J,PARMS,JITTER) returns d C / d PARMS(J) for
%   the covariance built by SV_COVBLOCKS.  Parameter order is
%   [variance, range_1..range_d, smoothness, nugget], so J = 1 is the
%   variance, J = 1+k the k-th range, J = d+2 the smoothness and J = d+3 the
%   nugget.

[nb, p, d] = size(coords);
sig2   = parms(1);
ranges = parms(2:d+1);
nug    = parms(d+3);

if j == 1                                   % d/d sigma^2
    dC = aux.f;
    for i = 1:p
        dC(:,i,i) = dC(:,i,i) + (nug + jitter);
    end
elseif j <= d+1                             % d/d lambda_k
    k  = j - 1;
    xk = coords(:,:,k);
    dk = reshape(xk, [nb p 1]) - reshape(xk, [nb 1 p]);
    dC = -aux.g .* (dk.^2) / ranges(k)^3;
elseif j == d+2                             % d/d nu
    if ~isfield(aux, 'dfdnu')
        error('sv_covderiv:dnu', ...
            'smoothness derivative requested but not cached; call sv_covblocks with want_dnu = true.');
    end
    dC = sig2 * aux.dfdnu;
elseif j == d+3                             % d/d tau^2
    dC = zeros(nb, p, p);
    for i = 1:p
        dC(:,i,i) = sig2;
    end
else
    error('sv_covderiv:index', 'parameter index out of range');
end
end
