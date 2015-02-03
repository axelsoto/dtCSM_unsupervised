function [C,dC] = cmm_r_grad_noDiagonal(x, X, DL, P, network, v, lambda)

%--------------------------------------------------------------------------
%CMM_R_GRAD computes the CMM gradient (Pearson's r correlation) by the
%network weights, including a regularization
%
%   [C,dC] = cmm_grad(x,X,network,v)
%
%   x        -> previous solution
%   X        -> data
%   DL       -> not used
%   P        -> pairwise input similarities
%   network  -> corresponding (cell) neural network
%   v        -> degrees of freedom of the student-t distribution
%   lambda   -> regularization parameter
%
%--------------------------------------------------------------------------

% This file was edited from code provided by:
% Laurens van der Maaten
% University California, San Diego / Delft University of Technology
% 
% The original code is part of the Matlab Toolbox for Dimensionality 
% Reduction v0.7.2b. which may be obtained at: 
% http://homepage.tudelft.nl/19j49

% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author(s).

% (C) Ryan Kiros, 2011
% Dalhousie University
% (C) Axel Soto, 2013
% Dalhousie University 


    % Initalize some variables
    n = size(X, 1);
    no_layers = length(network);

    % Update the network to store the new solution
    ii = 1;
    for i=1:length(network)
        network{i}.W = reshape(x(ii:ii - 1 + numel(network{i}.W)), size(network{i}.W)); 
        ii = ii + numel(network{i}.W);
        network{i}.bias_upW = reshape(x(ii:ii - 1 + numel(network{i}.bias_upW)), size(network{i}.bias_upW));
        ii = ii + numel(network{i}.bias_upW);
    end
    
    % Run the data through the network
    activations = cell(1, no_layers + 1);
    activations{1} = [X ones(n, 1)];
    for i=1:no_layers - 1
        activations{i + 1} = [1 ./ (1 + exp(-(activations{i} * [network{i}.W; network{i}.bias_upW]))) ones(n, 1)];
    end
    activations{end} = activations{end - 1} * [network{end}.W; network{end}.bias_upW];

    % Compute the Q-values
    d = squareform(pdist(activations{end}));
    num = 1 + d.^2 / v;
    ex = -((v + 1) / 2);
    Q = (num .^ ex)';%%no ' 
    Q(1:n+1:end) = [];%% was 0 originally
    Z = sum(Q(:));
    Q = Q ./ Z;                                                         
    Q = max(Q, eps);
    
        
    % Compute data correlation 
    Pm = P - mean(P(:));
    Qm = Q - mean(Q(:));
    pnumP = sum(sum(Pm .* Qm));
    qnumP = sum(sum(Pm.^2)) * sum(sum(Qm.^2));
    Cp = -pnumP / sqrt(qnumP);
    
    % Compute the cost function
    C = (1 - lambda) * Cp;
    
    % Compute the derivatives for the data correlation
    IxP = zeros(size(activations{end}));
    dQP = Pm / sqrt(qnumP) - pnumP * qnumP ^ (-3 / 2) * sum(Pm(:).^2) .* Qm;
    dD = 2 .* ex .* num .^ (ex - 1) .* (d / v) .* ((1 / Z) + num .^ ex * (1 / Z^2));
    dD(1:n+1:end) = [];
    dQPdD = dQP .* dD;
    for i=1:n
        tmp = bsxfun(@rdivide, bsxfun(@minus, activations{end}(i,:), activations{end}), d(i,:)');
        tmp(isnan(tmp)) = 0;
        tmp(i:n:end)=[];%%
        tmp = reshape(tmp,n-1,size(activations{end},2));%%
        %IxP(i,:) = -2 * sum(dQPdD(i,:) * tmp,1);
        IxP(i,:) = -2 * sum(dQPdD(i*(n-1)-(n-2):i*(n-1)) * tmp,1);%%
    end
    
    % Full derivative
    Ix = (1 - lambda) * IxP;

    
    % Compute gradients 
    dW = cell(1, no_layers);
    db = cell(1, no_layers);
    for i=no_layers:-1:1

        % Compute update    
        delta = activations{i}' * Ix;
        dW{i} = delta(1:end - 1,:);
        db{i} = delta(end,:);

        % Backpropagate error
        if i > 1
            Ix = (Ix * [network{i}.W; network{i}.bias_upW]') .* activations{i} .* (1 - activations{i});
            Ix = Ix(:,1:end - 1);
        end
    end
    
    % Convert gradient information
    dC = repmat(0, [numel(x) 1]);
    ii = 1;
    for i=1:no_layers
        dC(ii:ii - 1 + numel(dW{i})) = dW{i}(:); 
        ii = ii + numel(dW{i});
        dC(ii:ii - 1 + numel(db{i})) = db{i}(:); 
        ii = ii + numel(db{i});
    end 
    
end

