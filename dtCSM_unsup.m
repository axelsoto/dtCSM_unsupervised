function [mappedData,mapping]=dtCSM_unsup(train, architecture, DR_DIM,iter,hDimDist,realData,batchSize)

% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author(s).

% (C) Ryan Kiros, 2012
% Dalhousie University 
% (C) Axel Soto, 2013
% Dalhousie University 

%----------------------------------------------------
%INPUT
%train: instances x features matrix of training data
%architecture: number of nodes per layer. e.g. [500,100,50,2] (including last layer)
%DR_DIM: output dimensionality
%iter: number of iterations for finetuning
%hDimDist: distance used for high-dimensional instances
%----------------------------------------------------
%OUTPUT
%mappedData: matrix of mapped data (instances x DR_DIM)
%mapping: trained weights for the network

cd tCMM

%Pretrain the network
if (realData)
    %Standardize the data
    tr=train;
    train= bsxfun(@minus, train, mean(train));
    train = bsxfun(@rdivide, train, std(tr)); 
    network2 = tcmm(train, architecture, 'CD1', 1);
else
    network2 = tcmm(train, architecture, 'CD1', 0);
end
    

%If pretraining needs to be reused
save('network2.mat','network2');
%load('network2.mat');


% Fine-tune all data
network_f = cmm_r_backprop_selfDistsRemoved(network2, train,  ...
    iter, DR_DIM - 1, DR_DIM, hDimDist, batchSize);
    
% Run the data through the network
mappedData = run_data_through_network(network_f, train);
mapping=network_f;

cd ..