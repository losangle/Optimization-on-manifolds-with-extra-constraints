%%
%---------------------------------K means------------------------------------
close all; clear all; clc;
specifier.matlabversion = 0; %0 if older than 2015 1 otherwise

data_table = {'wine.csv', 13, 3;
              'ecoli.csv', 5, 8};


n_repeat = 4;   % Number of repeat experiment

for repeat = 1: n_repeat
    
    for dataset = 1 : length(data_table)
        filename = data_table{dataset, 1};
        rank = data_table{dataset, 3};
        
        %_______Set up data______
        data = csvread(filename);
        D = data*data.';

        %________Experiment_____
        options.maxOuterIter = 10000000;
        options.maxtime = 3600;
        options.minstepsize = 1e-10;
        
        % Only do mini-sum-max for low dimensional data
        specifier.ind = [0, 1, 1, 1, 1];

        
        result = clientconstraint_stiefel_Kmeans(D, rank, options, specifier);
        result = result(:);
        param = [dataset; repeat];
        outputdata = [result; param]';
        
        filename = sprintf('zz_KM.dat');
        dlmwrite(filename, outputdata, 'delimiter', ',', 'precision', 16, '-append');
    end
    
end
    
    