function nb = sv_chunk(p)
%SV_CHUNK  How many blocks of size p x p to process at once.
%
%   The batched routines hold several nb x p x p arrays live at the same time.
%   Making nb as large as memory allows is not the fastest choice: once the
%   working set leaves cache, every pass over it is served from main memory.
%   Capping the working set keeps the covariance build and the triangular
%   solves in cache while still amortizing interpreter overhead over many
%   blocks.
%
%   The cap is on total elements, so nb shrinks as p grows: roughly 5000
%   blocks at p = 11, 500 at p = 31, and 50 at p = 101.

max_elems = 5e5;
nb = max(1, floor(max_elems / max(p*p, 1)));
end
