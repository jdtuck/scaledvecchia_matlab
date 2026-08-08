function y = sv_borehole(u)
%SV_BOREHOLE  Borehole function, the standard 8-input computer-experiment test
%             case used in Katzfuss et al. (2020, Figure 3).
%
%   Y = SV_BOREHOLE(U) with U an n x 8 matrix of inputs in [0,1]^8 returns the
%   water flow rate through a borehole.  The response depends very unevenly on
%   the eight inputs, which is exactly the situation the scaled Vecchia
%   approximation is designed for.

lo = [0.05,   100,  63070,  990,  63.1,  700, 1120,  9855];
hi = [0.15, 50000, 115600, 1110, 116.0,  820, 1680, 12045];
x  = lo + u .* (hi - lo);

rw = x(:,1); r = x(:,2); Tu = x(:,3); Hu = x(:,4);
Tl = x(:,5); Hl = x(:,6); L = x(:,7); Kw = x(:,8);

lr = log(r ./ rw);
y = 2*pi*Tu .* (Hu - Hl) ./ ...
    ( lr .* (1 + 2*L.*Tu ./ (lr .* rw.^2 .* Kw) + Tu ./ Tl) );
end
