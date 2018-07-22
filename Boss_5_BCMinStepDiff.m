%%
%-----------------------Balanced Cut Minstep Difference--------------------
close all; clear all; clc;
specifier.matlabversion = 0; %0 if older than 2015 1 otherwise

dim_set = [50, 200, 500, 1000, 2000, 5000]; %dimension of the Adjacency Matrix
density_set = [0.005, 0.01, 0.02, 0.04, 0.08]; %density of the Adjacency Matrix 
n_repeat = 4;   %Number of repeat on same set of data
rank = 2;     %Graph Bisection
seed_size = 5; %fixed seed size for BA
prob_ER = 0.5; %probability of connecting an edge in ER graph.


for repeat = 1 : n_repeat
    
    for dim = dim_set
        
        for density = density_set

            %_______Set up data______
            mlink = ceil(density * dim);
            L = powerlawgraph(seed_size, prob_ER, dim, mlink);
            
            %________Experiment_____
            options.maxOuterIter = 100000000;
            options.maxtime = 3600;
            options.minstepsize = 1e-7;
            
            %Only do mini-sum-max for low dimensional data
            if dim == dim_set(1)
                specifier.ind = ones(5,1);
            else
                specifier.ind = [0, 1, 1, 1, 1];
            end
            
            result = clientconstraint_oblique_balancedcut(L, rank, options, specifier);
            result = result(:);
            param = [dim; density; repeat];
            outputdata = [result; param]';
            
            filename = sprintf('zz_BC_Dim%dMSe7.dat', dim);
            dlmwrite(filename, outputdata, 'delimiter', ',', 'precision', 16, '-append');
        end
        
    end
    
end