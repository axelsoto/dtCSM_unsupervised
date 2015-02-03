function network = tcmm(train_X, layers, training, type)

%--------------------------------------------------------------------------
%TCMM computes the deep t-distributed subspace mapping pretraining.
%
%   network = tcmm(train_X, layers, training)
%
%   train_X   -> training data
%   layers    -> network architecture
%   training  -> type of RBM training (CD1 or PCD)
%   type      -> data type (0: [0,1], 1: real values)
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


    if ~exist('training', 'var') || isempty(training)
        training = 'CD1';
    end

    % Pretrain the network
    origX = train_X;
    no_layers = length(layers);
    network = cell(1, no_layers);
    for i=1:no_layers

        % Print progress
        disp(['Training layer ' num2str(i) ' (size ' num2str(size(train_X, 2)) ' -> ' num2str(layers(i)) ')...']);
        
        if i == 1
            
            % First layer, type 1 is real valued units
            if type == 1
                network{i} = train_real_rbm(train_X, layers(i));
            else
                network{i} = train_rbm(train_X, layers(i));
            end
            train_X = 1 ./ (1 + exp(-(bsxfun(@plus, train_X * network{i}.W, network{i}.bias_upW))));
         
        elseif i ~= no_layers
          
            % Train layer using binary units for middle layers
            if strcmp(training, 'CD1')
                network{i} = train_rbm(train_X, layers(i));
            elseif strcmp(training, 'None')
                v = size(train_X, 2);
                network{i}.W = randn(v, layers(i)) * 0.1;
                network{i}.bias_upW = zeros(1, layers(i));
                network{i}.bias_downW = zeros(1, v);
            else
                error('Unknown training procedure.');
            end
                
            % Transform data using learned weights
            train_X = 1 ./ (1 + exp(-(bsxfun(@plus, train_X * network{i}.W, network{i}.bias_upW))));
        else
            % Train layer using Gaussian hidden units for last layer
            if ~strcmp(training, 'None')
                network{i} = train_lin_rbm(train_X, layers(i));
            else
                v = size(train_X, 2);
                network{i}.W = randn(v, layers(i)) * 0.1;
                network{i}.bias_upW = zeros(1, layers(i));
                network{i}.bias_downW = zeros(1, v);
            end
        end
    end
