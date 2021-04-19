function stepForward(N, l)
%STEPFORWARD performs forward propagation for the current network layer
%   N is the network (array of pointers to structs)
%   l is the layer to perform the forward step
%
%   Sergey Shuvaev, 2016. sshuvaev@cshl.edu

% input layer do notinhg

if l == 1, return, end

cL = N(l);      % current layer. okay since N is a reference array
pL = N(l - 1);  % previous layer
nL = N(l + 1);  % next layer

switch(cL.type)
    case 'full'
        
        if cL.undefined %definition on the first call (if needed)
            cL.defineFull(pL.outdim, nL.indim);
            cL.initFull;
            cL.undefined = 0;
        end
        
        cL.y = cL.w * pL.y(:) + cL.b; %applying weights to the inputs
        cL.y = cL.nlfun(cL.y); %nonlinearity
        
    case 'conv'
        
        if cL.undefined %definition on the first call (if needed)
            cL.defineConv(pL.outdim, nL.indim);
            cL.initConv;
            cL.undefined = 0;
        end
        
        for j = 1 : size(cL.w, 4) %convolution with each kernel
            cL.y(:, :, j) = convn(pL.y, cL.w(:, :, end : -1 : 1, j), 'valid') + cL.b(j);
        end
        
        cL.y = cL.nlfun(cL.y); %nonlinearity
        
    case 'maxpool'
        
        if cL.undefined %definition on the first call (if needed)
            cL.defineMaxpool(pL.outdim, nL.indim);
            cL.initMaxpool;
            cL.undefined = 0;
        end
        
        [~, cL.MI] = max(pL.y(cL.CI), [], 2); %maxpool origins
        cL.MI = cL.CI((1 : size(cL.CI, 1))' + size(cL.CI, 1) * (cL.MI - 1));
        cL.y(:) = pL.y(cL.MI); %pass maximum values
        
    case 'softmax'
        
        cL.y = exp(pL.y);
        cL.y = cL.y ./ sum(cL.y(:)); %normalization
        
    case 'input'
        
        warning('stepForward is called for an input layer')
        
    case 'target'
        
        warning('stepForward is called for a target layer')
end
end
