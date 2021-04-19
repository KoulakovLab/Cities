function stepBackward(N, l, LRATE, MOMENTUM, WDECAY, UPDATE)
%STEPBACKWARD performs back propagation for the current network layer
%   N is the network (array of pointers to structs)
%   l is the layer to perform the forward step
%   LRATE is the current learning rate
%   MOMENTUM (optional parameter)
%   WDECAY (optional parameter) - weight decay (as a fraction of LRATE)
%   UPDATE (optional parameter) - (0) accumulate gradient | (1) update the
%                                 weights (training in batches)
%
%   Sergey Shuvaev, 2016. sshuvaev@cshl.edu

% input layer do notinhg

if l == 1, return, end

if nargin < 5
    MOMENTUM = 0;   % momentum (typical value: 0.9)
    WDECAY = 0;     % weight decay (typical value: 1e-3
end
if nargin < 6
    UPDATE = 1;     % default: update the weights
end
cL = N(l);      % current layer. okay since N is a reference array
pL = N(l - 1);  % previous layer (next for backprop)


switch(cL.type)
    case 'full'
        
        if l ~= 2 %delta
            pL.delta(:) = (cL.w' * cL.delta(:)) .* cL.nlfunprime(pL.y(:));
        end
        cL.gw = cL.gw + cL.delta(:) * pL.y(:)'; %gragient (weight part)
        cL.gb = cL.gb + cL.delta(:);            %gradient (bias part)
        
        if UPDATE
            cL.vw = MOMENTUM * cL.vw - WDECAY * LRATE * cL.w + LRATE * cL.gw;
            cL.vb = MOMENTUM * cL.vb - WDECAY * LRATE * cL.b + LRATE * cL.gb;

            cL.w = cL.w - cL.vw; %Stochastic gradient descent
            cL.b = cL.b - cL.vb;
            
            cL.gw = 0 * cL.gw;
            cL.gb = 0 * cL.gb;
        end
        
    case 'conv'
        
        pL.delta = 0 * pL.delta;
        
        for j = 1 : size(cL.w, 4)
            
            if l ~= 2
                pL.delta = pL.delta + convn(rot90(cL.w(:, :, :, j), 2), ... %delta
                    cL.delta(:, :, j)) .* cL.nlfunprime(pL.y);
            end
            
            cL.gw(:, :, :, j) = cL.gw(:, :, :, j) + ... %gradient (weight part)
                convn(pL.y, rot90(cL.delta(:, :, j), 2), 'valid');
            
            cnv_bias = cL.delta(:, :, j);
            cL.gb(j) = cL.gb(j) + sum(cnv_bias(:)); %gradient (bias part)
        end
        
        if UPDATE
            cL.vw = MOMENTUM * cL.vw - WDECAY * LRATE * cL.w + LRATE * cL.gw;
            cL.vb = MOMENTUM * cL.vb - WDECAY * LRATE * cL.b + LRATE * cL.gb;

            cL.w = cL.w - cL.vw; %Stochastic gradient descent
            cL.b = cL.b - cL.vb;
            
            cL.gw = 0 * cL.gw;
            cL.gb = 0 * cL.gb;
        end
        
    case 'maxpool'
        
        pL.delta = pL.delta * 0; %only maxpool origins should be nonzero
        pL.delta(cL.MI) = cL.delta(:);
        %pL.delta(cL.CI) = repmat(cL.delta(:), 1, size(cL.CI, 2)); %upsampling to the entire region
        
    case 'softmax'
        
        pL.delta = cL.delta; %pass delta
        
    case 'input'
        
        warning('stepForward is called for an input layer')
        
    case 'target'
        
        pL.delta = pL.y - cL.y; %Loss function calculation (elementwise)
end
end
