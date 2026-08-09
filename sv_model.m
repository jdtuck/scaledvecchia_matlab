classdef sv_model
    %SV_MODEL scaled vecchia model definition

    properties
        model
        samples
    end

    methods
        function obj = sv_model(model)
            obj.model = model;
        end

        function pred = predict(obj, x_new, options)
            arguments
                obj
                x_new
                options.idxSamples = nan;
                options.m = 100
                options.nsims = 200;
                options.joint = false
                options.variance = true
            end
            idxSamples = options.idxSamples;
            if isnan(idxSamples) 
                idxSamples = 1:options.nsims;
            end

            p = sv_predict(obj.model, x_new, 'm', options.m, 'nsims', options.nsims, 'joint', options.joint, 'variance', options.variance);
            pred = p.samples(:,idxSamples)';
        end
    end
end