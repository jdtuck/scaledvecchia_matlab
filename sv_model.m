classdef sv_model
    %SV_MODEL  Fitted scaled-Vecchia emulator with two prediction paths.
    %
    %   Wraps a fitted scaled-Vecchia emulator and exposes the two ways a
    %   calibration code needs to call it.
    %
    %   PATH 1 -- inputs fixed, draws repeated.  PREPARE once, DRAW in the
    %   loop: PREPARE does the neighbour search, the covariance blocks and
    %   the factorizations, none of which depend on the response values or
    %   on the random numbers, so the per-iteration cost drops by three
    %   orders of magnitude.
    %
    %       obj = obj.prepare(x_new);            % keep the output!
    %       for it = 1:niter
    %           s = obj.draw('idxSamples', k);   % microseconds
    %       end
    %
    %   PATH 2 -- inputs change every call, many points at once.  This is
    %   the calibration case: x_new is built from the current theta, so no
    %   plan can be reused and PREDICT builds one per call.  Pass all the
    %   rows you need at once; the per-call overhead is amortized over them.
    %
    %       s = obj.predict([x_obs, repmat(theta, nobs, 1)], 'idxSamples', k);
    %
    %   REPRODUCIBLE SURROGATE REALIZATIONS.  When an MCMC chain samples the
    %   emulator uncertainty, sample index k has to name the *same* surrogate
    %   realization every time it is used, otherwise the likelihood is a
    %   fresh random function at every evaluation and the accept ratio
    %   compares draws from different realizations.  So whenever
    %   'idxSamples' is given explicitly, the standard normals behind the
    %   draws come from a fixed random stream indexed by k (common random
    %   numbers): PREDICT(x, 'idxSamples', 7) called twice returns the same
    %   numbers, and as theta moves the realization moves smoothly with it.
    %   Pass 'crn', false for independent draws instead.  With no
    %   'idxSamples' the call just asks for NSIMS fresh draws, as before.
    %
    %   The frozen realizations are meaningful for the pointwise path
    %   (joint = false, the default here), where normal j is attached to row
    %   j of x_new: keep the rows in the same order across iterations.  With
    %   joint = true the draws are reproducible only for identical x_new,
    %   because the joint path reorders the prediction points and the
    %   ordering depends on x_new.
    %
    %   Note that joint = false draws each prediction point independently
    %   given its own conditioning set, so a multi-point call is not a
    %   correlated realization of the surrogate across those points.  Use
    %   joint = true when that correlation matters (at the cost of no exact
    %   pointwise variance, and of NOISE_FREE being ignored -- the joint
    %   factor always carries the nugget).
    %
    %   Outputs: [SAMPLES, MEAN, VAR].  SAMPLES is nsims-by-n_pred, one row
    %   per surrogate realization; MEAN and VAR are n_pred-by-1.
    %
    %   See also SV_FIT, SV_PREPARE, SV_DRAW, SV_PREDICT.

    properties
        model                % the fitted struct from sv_fit
        samples              % free slot for callers that stash draws here
        nSamples             % default number of surrogate realizations
        prep                 % cached prediction plan from sv_prepare, or empty
        prep_opts            % options the cached plan was built with
        seed                 % base seed for the frozen realizations
    end

    methods
        function obj = sv_model(model, nSamples, options)
            arguments
                model
                nSamples = 1000
                options.seed (1,1) double {mustBeInteger, mustBeNonnegative} = 1
            end
            obj.model = model;
            obj.nSamples = nSamples;
            obj.seed = options.seed;
        end

        function obj = prepare(obj, x_new, options)
            % PREPARE  Cache the prediction plan for a fixed set of inputs.
            %   OBJ = OBJ.PREPARE(X_NEW, ...) -- assign the output; sv_model
            %   is a value class, so a bare obj.prepare(x_new) is discarded.
            arguments
                obj
                x_new
                options.m = 100
                options.joint = false
                options.noise_free = true
                options.scale = []
                options.X_pred = []
            end
            obj.prep = sv_prepare(obj.model, x_new, ...
                'm', options.m, 'joint', options.joint, ...
                'noise_free', options.noise_free, ...
                'scale', options.scale, 'X_pred', options.X_pred);
            obj.prep_opts = options;
        end

        function [pred, mu, v] = draw(obj, options)
            % DRAW  Draws from the cached plan; cheap enough for a loop.
            arguments
                obj
                options.idxSamples = []
                options.nsims = []
                options.variance = true
                options.crn = []
                options.y = []
                options.beta = []
            end
            if isempty(obj.prep)
                error('sv_model:draw', ...
                    'call obj = obj.prepare(x_new) before draw().');
            end
            [idx, crn] = obj.resolve_idx(options);
            [pred, mu, v] = obj.sample_from(obj.prep, idx, crn, options);
        end

        function [pred, mu, v] = predict(obj, x_new, options)
            % PREDICT  Draws at arbitrary inputs, building the plan as needed.
            arguments
                obj
                x_new
                options.idxSamples = []
                options.nsims = []
                options.m = 100
                options.joint = false
                options.variance = true
                options.noise_free = true
                options.scale = []
                options.X_pred = []
                options.crn = []
                options.y = []
                options.beta = []
            end
            [idx, crn] = obj.resolve_idx(options);

            % Reuse the cached plan only when it was built for exactly these
            % inputs and settings -- matching sizes alone would silently
            % return predictions for the wrong locations.
            if obj.plan_matches(x_new, options)
                p = obj.prep;
            else
                p = sv_prepare(obj.model, x_new, ...
                    'm', options.m, 'joint', options.joint, ...
                    'noise_free', options.noise_free, ...
                    'scale', options.scale, 'X_pred', options.X_pred);
            end
            [pred, mu, v] = obj.sample_from(p, idx, crn, options);
        end

    end

    methods (Access = private)
        function tf = plan_matches(obj, x_new, options)
            % Is the cached plan the one this request needs?
            tf = false;
            if isempty(obj.prep) || isempty(obj.prep_opts), return; end
            tf = obj.prep.joint == options.joint ...
                && isequal(obj.prep.m, options.m) ...
                && isequal(obj.prep.noise_free, options.noise_free) ...
                && isequal(obj.prep_opts.scale, options.scale) ...
                && isequal(obj.prep_opts.X_pred, options.X_pred) ...
                && isequal(obj.prep.inputs_pred, x_new);
        end

        function [idx, crn] = resolve_idx(obj, options)
            % Which realizations to return, and whether to freeze them.
            idx = options.idxSamples;
            crn = options.crn;
            % nan has meant "all of them" here, so keep accepting it
            if isnumeric(idx) && isscalar(idx) && isnan(idx)
                idx = [];
            end
            if isempty(idx)
                ns = options.nsims;
                if isempty(ns), ns = obj.nSamples; end
                if ~(isnumeric(ns) && isscalar(ns) && ns == fix(ns) && ns >= 1)
                    error('sv_model:nsims', 'nsims must be a positive integer.');
                end
                idx = 1:ns;
                if isempty(crn), crn = false; end
            else
                if ~isnumeric(idx) || ~all(isfinite(idx(:))) ...
                        || any(idx(:) < 1) || any(idx(:) ~= fix(idx(:)))
                    error('sv_model:idxSamples', ...
                        'idxSamples must be positive integers.');
                end
                idx = reshape(idx, 1, []);
                % An explicit index names a realization, so freeze it.
                if isempty(crn), crn = true; end
            end
        end

        function [pred, mu, v] = sample_from(obj, p, idx, crn, options)
            ns = numel(idx);
            want_var = options.variance;
            if p.joint && ns < 2
                % A joint plan has no closed-form pointwise variance; the
                % Monte-Carlo estimate from a single draw is exactly 0, so
                % return nothing rather than a misleading zero.
                want_var = false;
            end
            args = {'nsims', ns, 'variance', want_var, ...
                'y', options.y, 'beta', options.beta};
            if crn
                args = [args, {'z', obj.normals(p.np, idx)}];
            end
            s = sv_draw(p, args{:});

            pred = s.samples';          % nsims-by-n_pred
            mu = s.mean;
            if isfield(s, 'var'), v = s.var; else, v = []; end
        end

        function Z = normals(obj, np, idx)
            % One fixed standard-normal vector per sample index: substream k
            % of a seeded stream, so index k always means the same draw.
            st = RandStream('Threefry', 'Seed', obj.seed);
            Z = zeros(np, numel(idx));
            for j = 1:numel(idx)
                st.Substream = idx(j);
                Z(:,j) = randn(st, np, 1);
            end
        end
    end
end
