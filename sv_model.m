classdef sv_model
    %SV_MODEL scaled vecchia model definition
    %
    %   Wraps a fitted scaled-Vecchia emulator.  For one-off predictions call
    %   PREDICT directly.  For repeated prediction at the same inputs — MCMC,
    %   or drawing many posterior realizations — call PREPARE once and then
    %   DRAW in the loop: PREPARE does the neighbour search, the covariance
    %   blocks and the factorizations, none of which depend on the response
    %   values or the random numbers, so the per-iteration cost drops by three
    %   orders of magnitude.
    %
    %       obj = obj.prepare(x_new, 'nsims', 200);
    %       for it = 1:niter
    %           s = obj.draw();
    %       end

    properties
        model
        samples
        nSamples
        prep        % cached prediction plan from sv_prepare, or empty
    end

    methods
        function obj = sv_model(model, nSamples)
            arguments
                model
                nSamples = 1000
            end
            obj.model = model;
            obj.nSamples = nSamples;
        end

        function obj = prepare(obj, x_new, options)
            % PREPARE  Cache the prediction plan for a fixed set of inputs.
            arguments
                obj
                x_new
                options.m = 100
                options.joint = false
            end
            obj.prep = sv_prepare(obj.model, x_new, ...
                'm', options.m, 'joint', options.joint);
        end

        function pred = draw(obj, options)
            % DRAW  Simulations from the cached plan; cheap enough for a loop.
            arguments
                obj
                options.idxSamples = nan;
                options.nsims = 200;
                options.variance = true
                options.y = [];
            end
            if isempty(obj.prep)
                error('sv_model:draw', ...
                    'call prepare(x_new) before draw().');
            end
            idxSamples = options.idxSamples;
            if isnan(idxSamples)
                idxSamples = 1:options.nsims;
            end
            p = sv_draw(obj.prep, 'nsims', options.nsims, ...
                'variance', options.variance, 'y', options.y);
            pred = p.samples(:,idxSamples)';
        end

        function pred = predict(obj, x_new, options)
            arguments
                obj
                x_new
                options.idxSamples = nan;
                options.m = 100
                options.joint = false
                options.variance = true
            end
            idxSamples = options.idxSamples;
            if isnan(idxSamples) 
                idxSamples = 1:obj.model.nSamples;
                nsims = obj.nSamples;
            else
                nsims = length(idxSamples);
            end

            % reuse the cached plan when it matches this request
            % Reuse the cached plan only when it was built for exactly these
            % inputs and settings -- matching sizes alone would silently
            % return predictions for the wrong locations.
            if ~isempty(obj.prep) && obj.prep.joint == options.joint ...
                    && obj.prep.m == options.m ...
                    && isequal(obj.prep.inputs_pred, x_new)
                p = sv_draw(obj.prep, 'nsims', options.nsims, ...
                    'variance', options.variance);
            else
                p = sv_predict(obj.model, x_new, 'm', options.m, 'nsims', nsims, 'joint', options.joint, 'variance', options.variance);
            end
            pred = p.samples(:,idxSamples)';
        end
    end
end
