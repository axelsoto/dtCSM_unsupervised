function [network,filename] = cmm_r_backprop_selfDistsRemoved(network, train_X, max_iter, v, dim, hDimDist, batch_size)

%--------------------------------------------------------------------------
%CMM_R_BACKPROP_selfDistsRemoved finetunes the given network with backpropagation using the
%dtCSM gradient and removing self distances for the computation of the correlation
%
%   [network,err] = cmm_r_backprop_selfDistsRemoved(network, train_X,
%    max_iter, v, dim, hDimDist, batch_size)
%
%   network       -> corresponding (cell) neural network
%   train_X       -> training data
%   max_iter      -> number of finetuning iterations
%   v             -> degrees of freedom of the student-t distribution   
%   **dim**       -> output dimensionality (obsolete)
%   hDimDist      -> distance used for high-dimensional instances
%   batch_size    -> size of the batch for the fine-tuning of the network
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

% Axel Soto, 2013
% Dalhousie University


	%setting parameters
    n = size(train_X, 1);

    if ~exist('max_iter', 'var') || isempty(max_iter)
        max_iter = 30;        
    end
    if ~exist('v', 'var') || isempty(v)
        v = length(network{end}.bias_upW) - 1;
    end
    
    if ~exist('batch_size','var') || isempty(batch_size)
        batch_size = min([fix(n/4) n]);
    end
    
    % Initialize some variables
    ind = 1:n;
	lambda = 0;
    err = zeros(max_iter, 1);
    curX = cell(floor(n ./ batch_size), 1);
    P = cell(floor(n ./ batch_size), 1);
	perplexity = 30;
        
    
    % Precompute joint probabilities for all batches
    disp('Precomputing P-values...');
    i = 1;
    
    nAux = 0; 
    for batch=1:batch_size:n          
            %%% Prepare batches
            curX{i} = double(train_X(ind(batch:min([batch + batch_size - 1 n])),:));
                        
            %%% Remove duplicates in the batch for better convergence
            allDists = squareform(pdist(curX{i},'cosine'));
            [q,w] = find(allDists==0);
            [curX{i},~] =  removeRepeated(curX{i},q,w);
            nAux = nAux + size(curX{i}, 1);
            
			if (strcmp(hDimDist,'gaussian'))
				%If we want to use gaussian densities for the high-dimensional distances
				P{i} = x2p(curX{i}, perplexity, 1e-5);
				%P{i} = bsxfun(@minus,1, P{i});
				%P{i} = bsxfun(@minus,1, abs(P{i}));
                
               
				P{i} = (P{i} + P{i}') / 2;
		
			elseif (strcmp(hDimDist,'cosine'))
				%If we want to use  cosine and a transformation
				%based on Student's T (other distances can be used too) for the high dimensional space
				P{i} = squareform(pdist(curX{i},'cosine'));
				P{i} = (1 + (P{i}.^2) ./ v) .^ -((v + 1) / 2)';%%
			else
				disp('Distance not recognized');
			end
            
            
            %Remove elements in the diagonal (making them 0 slows convergence a lot!)
            P{i}(1:size(curX{i},1)+1:end) = [];
            %Changing NaNs by 0 
            P{i}(isnan(P{i})) = 0;
            
			
			
            %Make P_ij{i} a probability
            P{i} = P{i} ./ sum(P{i}(:));                                              
            P{i} = max(P{i}, eps);
            i = i + 1;
    end
    

    % Run the optimization
    for iter=1:max_iter
        
        % Run for all mini-batches
        disp(['Iteration ' num2str(iter) '...']);
        b = 1;
        for batch=1:batch_size:n
            if batch + batch_size - 1 <= n
                
                % Construct current solution
                x = [];
                for i=1:length(network)
                    x = [x; network{i}.W(:); network{i}.bias_upW(:)];
                end

                % Perform conjugate gradient using three linesearches
                x = minimize_cmm(x, 'cmm_r_grad_noDiagonal', 3, curX{b}, [], P{b}, network, v, lambda);
                b = b + 1;                
                
                % Store new solution
                ii = 1;
                for i=1:length(network)
                    network{i}.W = reshape(x(ii:ii - 1 + numel(network{i}.W)), size(network{i}.W)); 
                    ii = ii + numel(network{i}.W);
                    network{i}.bias_upW = reshape(x(ii:ii - 1 + numel(network{i}.bias_upW)), size(network{i}.bias_upW));
                    ii = ii + numel(network{i}.bias_upW);
                end
            end
        end
        
        % Estimate the correlations (using the first batch only)
		activations = run_data_through_network(network, curX{1});
		Q = (1 + (squareform(pdist(activations)).^2) ./ v) .^ -((v + 1) / 2)';
		Q(1:size(curX{1},1)+1:end) = [];
		
		Q = Q ./ sum(Q(:));
		Q = max(Q, eps);
		
		Qm = Q - mean(Q(:));
		
		Pm = P{1} - mean(P{1}(:));

		pnumP = sum(sum(Pm .* Qm));
		qnumP = sqrt(sum(sum(Pm.^2)) * sum(sum(Qm.^2)));
		Cp = pnumP / qnumP;
		

		disp(['t-CMM data correlation: ' num2str(Cp)]);
            
        % Visualize if 2 dimensions
        %if (mod(iter,5)==0)&& (size(mappedX,2) == 2) 
        %    clf;
		%    Construct the new mapping
		%	 mappedX = run_data_through_network(network, train_X);
        %    asrs_plot(...)
        %    drawnow
		%	figure;
        %end
    end
end
