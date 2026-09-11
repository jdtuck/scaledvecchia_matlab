function p = sv_page_threshold()
%SV_PAGE_THRESHOLD  Block size at which per-page LAPACK beats batched loops.
%
%   The batched routines (SV_BCHOL, SV_BFSOLVE, SV_BLASTROW) vectorize across
%   blocks and loop over the p columns.  That is the right trade while the
%   whole nb x p x p array fits comfortably in cache, which covers the small
%   conditioning sets used for fitting (m around 30).  For the larger sets used
%   for prediction (m around 100) the array no longer fits, the column loop
%   becomes memory-bound, and a per-page call into LAPACK's blocked
%   factorization is several times faster.
%
%   Measured crossover is near p = 45; below it the batched form wins by up to
%   20x at p = 6, above it LAPACK wins by 3-4x at p = 101.  The exact point
%   depends on the machine and on interpreter call overhead, so it is isolated
%   here rather than hard-coded in three places.

p = 48;
end
