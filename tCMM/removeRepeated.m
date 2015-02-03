function [matrix, matrix2] = removeRepeated(matrix,rowIndices,colIndices,matrix2)
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author(s).

% (C) Axel Soto, 2013
% Dalhousie University 


toRemove=[];
for ind=1:length(rowIndices)
    if rowIndices(ind)>colIndices(ind)
        toRemove=[toRemove;rowIndices(ind)];
    end
end

matrix(toRemove,:)=[];
if exist('matrix2','var')
	matrix2(toRemove,:)=[];
else
	matrix2=[];
end