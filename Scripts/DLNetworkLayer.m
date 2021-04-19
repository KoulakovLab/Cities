classdef DLNetworkLayer < matlab.mixin.Copyable
    %DLNETWORKLAYER is a basic class for deep learning network layer
    %
    %   Sergey Shuvaev, 2016. sshuvaev@cshl.edu
    
    properties
        type        %Layer type ( 'input' | 'conv' | 'maxpool' | 'full' |
                    % | 'target' )
        indim       %Incoming activations array size
        outdim      %Outgoing activations array size
        wdim        %Weight array size (4D for 'conv' | 2D for 'full' |
                    % | 0D for 'maxpool')
        nltype      %Nonlinearity type ( 'relu' | 'leakyrelu' | 'sigmoid' |
                    % | 'linear' | 'arbitrary' )
        nlfun       %Nonlinear function handle
        nlfunprime  %Nonlinear function derivative handle
        initgain    %Scale factor for Xavier initialization
        w           %Weights
        b           %Bias
        y           %Activations
        delta       %Errors
        gw          %Objective gradient (weights part)
        gb          %Objective gradient (bias part)
        vw          %Momentum-related gradient (weights part)
        vb          %Momentum-related gradient (bias part)
        CI          %Max pooling regions
        MI          %Max pooling regional maxima indices
        undefined   %Shows whether all necessary properties are defined
    end
    
    methods
        
        function obj = DLNetworkLayer(type, indim, outdim, wdim, nltype, opt)
            
            %General definitions
            
            obj.type = type;
            obj.indim = indim;
            obj.outdim = outdim;
            obj.wdim = wdim;
            obj.nltype = nltype;
            obj.nlfun = [];
            obj.nlfunprime = [];
            obj.initgain = 1;
            obj.w = [];
            obj.b = [];
            obj.y = [];
            obj.delta = [];
            obj.gw = [];
            obj.gb = [];
            obj.vw = 0;
            obj.vb = 0;
            obj.CI = [];
            obj.MI = [];
            obj.undefined = isempty(indim) || isempty(outdim) || isempty(wdim);
            
            %Nonlinearity definition
            
            if isempty(nltype)
                obj.nlfun = [];
                obj.nlfunprime = [];
            else
                switch(nltype)
                    case 'relu'
                        obj.nlfun = @(x) (x > 0) .* x;
                        obj.nlfunprime = @(y) (y > 0); %y is already nlfun(x)
                        obj.initgain = sqrt(2);
                    case 'leakyrelu'
                        obj.nlfun = @(x) (x > 0) .* x + (x <= 0) .* 0.2 .* x;
                        obj.nlfunprime = @(y) (y > 0) + (y <= 0) .* 0.2;
                        obj.initgain = sqrt(2 / (1 + 0.2 ^ 2));
                    case 'sigmoid'
                        obj.nlfun = @(x) (1 ./ (1 + exp(-x)));
                        obj.nlfunprime = @(y) (y .* (1 - y));
                        obj.initgain = 1;
                    case 'linear'
                        obj.nlfun = @(x) x;
                        obj.nlfunprime = @(y) 1;
                        obj.initgain = 1;
                    otherwise
                        obj.nlfun = opt{1}; %user-defined
                        obj.nlfunprime = opt{2};
                        obj.initgain = opt(3);
                end
            end
            
            %Layer type specific definitions
            
            switch(obj.type)
                case 'full'
                    if ~obj.undefined
                        obj.initFull;
                    end
                case 'conv'
                    if ~obj.undefined
                        obj.initConv;
                    end
                case 'maxpool'
                    if ~obj.undefined
                        obj.initMaxpool;
                    end
                case 'softmax'
                    obj.y = zeros(obj.outdim);
                    obj.delta = zeros(obj.outdim);
                case 'target'
                    obj.y = zeros(obj.outdim);
                case 'input'
                    obj.y = zeros(obj.outdim);
                otherwise
                    error('This type of layer is not defined.')
            end
        end
        
        %Initialization functions
        
        function initFull(obj)
            obj.w = (2 * rand(obj.wdim) - 1) .* ...
                4 * obj.initgain * sqrt(6 / (obj.wdim(1) + obj.wdim(2)));
            obj.b = (2 * rand(obj.wdim(1), 1) - 1) .* ...
                4 * obj.initgain * sqrt(6 / (obj.wdim(1) + obj.wdim(2)));
            obj.gw = zeros(obj.wdim);
            obj.gb = zeros(obj.wdim(1), 1);
            obj.vw = zeros(obj.wdim);
            obj.vb = zeros(obj.wdim(1), 1);
            obj.y = zeros(obj.outdim);
            obj.delta = zeros(obj.outdim);
        end
        
        function initConv(obj)
            obj.w = (2 * rand(obj.wdim) - 1) .* ...
                4 * obj.initgain * sqrt(6 / (obj.wdim(1) * ...
                obj.wdim(2) * (obj.wdim(3) + obj.wdim(4))));
            obj.b = (2 * rand(1, obj.wdim(4)) - 1) .* ...
                4 * obj.initgain * sqrt(6 / (obj.wdim(1) * ...
                obj.wdim(2) * (obj.wdim(3) + obj.wdim(4))));
            obj.gw = zeros(obj.wdim);
            obj.gb = zeros(1, obj.wdim(4));
            obj.vw = zeros(obj.wdim);
            obj.vb = zeros(1, obj.wdim(4));
            obj.y = zeros(obj.outdim);
            obj.delta = zeros(obj.outdim);
        end
        
        function initMaxpool(obj)
            obj.w = obj.wdim;
            obj.y = zeros(obj.outdim);
            obj.delta = zeros(obj.outdim);
            
            OX = (1 : obj.w(1) : (obj.indim(1) - obj.w(1) + 1))'; %Max pooling origins
            OY = (0 : obj.w(2) : (obj.indim(2) - obj.w(2) + 1) - 1) * obj.indim(1);
            OZ = (0 : obj.indim(3) - 1) * obj.indim(1) * obj.indim(2);
            OI = OX * ones(size(OY)) + ones(size(OX)) * OY;
            OI = OI(:); %in-plane
            OI = OI * ones(size(OZ)) + ones(size(OI)) * OZ;
            OI = OI(:); %total
            
            MX = (0 : obj.w(1) - 1)'; %Max pooling mask
            MY = (0 : obj.w(2) - 1) * obj.indim(1);
            MI2 = MX * ones(size(MY)) + ones(size(MX)) * MY;
            MI2 = (MI2(:))';
            
            obj.CI = OI * ones(size(MI2)) + ones(size(OI)) * MI2; %Max pooling clusters
            clear OX OY OZ OI MX MY MI;
        end
        
        %Input-output
        
        function setInput(obj, image)
            obj.y = image;
        end
        
        function setTarget(obj, label)
            obj.y = 0 * obj.y;
            obj.y(label + 1) = 1;
        end
        
        function label = getLabel(obj)
            [~, label] = max(obj.y);
            label = label - 1;
        end
        
        function strip(obj)
            obj.y = [];
            obj.delta = [];
            obj.gw = [];
            obj.gb = [];
            obj.vw = [];
            obj.vb = [];
            obj.CI = [];
            obj.MI = [];
        end
        
        %Defining remaining parameters if needed. Nothing interesting beyond this point
        
        function defineConv(obj, pOutdim, nIndim)
            if isempty(obj.indim)
                obj.indim = pOutdim;
            end
            
            if isempty(obj.outdim)
                obj.outdim = nIndim;
            end
            
            if isempty(obj.indim) + isempty(obj.outdim) + isempty(obj.wdim) > 1
                error('Not enough data.');
            end
            
            if isempty(obj.indim)
                obj.indim(1 : 2) = obj.outdim(1 : 2) + obj.wdim(1 : 2) - 1;
                obj.indim(3) = obj.wdim(3);
            end
            
            if isempty(obj.outdim)
                obj.outdim(1 : 2) = obj.indim(1 : 2) - obj.wdim(1 : 2) + 1;
                obj.outdim(3) = obj.wdim(4);
            end
            
            if isempty(obj.wdim)
                obj.wdim(1 : 2) = obj.indim(1 : 2) - obj.outdim(1 : 2) + 1;
                obj.wdim(3) = obj.indim(3);
                obj.wdim(4) = obj.outdim(3);
            end
        end
        
        function defineFull(obj, pOutdim, nIndim)
            if isempty(obj.indim)
                obj.indim = pOutdim;
            end
            
            if isempty(obj.outdim)
                obj.outdim = nIndim;
            end
            
            if isempty(obj.indim) || (isempty(obj.outdim) && isempty(obj.wdim))
                error('Not enough data.');
            end
            
            if isempty(obj.outdim)
                obj.outdim(2 : 3) = [1 1];
                obj.outdim(1) = obj.wdim(1);
            end
            
            if isempty(obj.wdim)
                obj.wdim(2) = obj.indim(1) * obj.indim(2) * obj.indim(3);
                obj.wdim(1) = obj.outdim(1);
            end
        end
        
        function defineMaxpool(obj, pOutdim, nIndim)
            if isempty(obj.indim)
                obj.indim = pOutdim;
            end
            
            if isempty(obj.outdim)
                obj.outdim = nIndim;
            end
            
            if isempty(obj.indim) || (isempty(obj.outdim) && isempty(obj.wdim))
                error('Not enough data.');
            end
            
            if isempty(obj.outdim)
                obj.outdim(1 : 2) = floor(obj.indim(1 : 2) ./ obj.wdim(1 : 2));
                obj.outdim(3) = obj.indim(3);
            end
            
            if isempty(obj.wdim)
                obj.wdim(1 : 2) = floor(obj.indim(1 : 2) ./ obj.outdim(1 : 2));
            end
        end
    end
end
