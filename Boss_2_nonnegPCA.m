%%
%---------------------------------PCA------------------------------------

close all; clc; clear all;
specifier.matlabversion = 0; %0 if older than 2015 1 otherwise

dim_set = [10, 50, 200, 500, 1000, 2000];  % Dimension of "the Cov Matrix"
snrset = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]; % Signal Strength
deltaset = [0.1, 0.3, 0.7, 0.9];           % Sparsity
rank = 1;                                  % Rank of BM Relaxation. 1 if we don't.
n_repeat = 4;                              % Number of repeat experiment

for repeat = 1: n_repeat
    
    for dim = dim_set
        
        for snr = snrset
            
            for delta = deltaset

                %_______Set up data______
                T = dim;
                samplesize = floor(delta*dim);
                S = randsample(dim, samplesize);
                v = zeros(dim,1);
                v(S) = 1/sqrt(samplesize);
                X = sqrt(snr) * v * (v.');
                Z = randn(dim)/sqrt(T);
                for ii = 1: dim
                    Z(ii,ii) = randn * 2/sqrt(T);
                end
                X = X+Z;

                %________Experiment_____
                options.maxOuterIter = 10000000;
                options.maxtime = 3600;
                options.minstepsize = 1e-10;
                
                %Only do mini-sum-max for low dimensional data
                if dim == dim_set(1)
                    specifier.ind = ones(5,1);
                else
                    specifier.ind = [0, 1, 1, 1, 1];
                end

                result = clientconstraint_sphere_nonnegativePCA(X, rank, options, specifier);
                result = result(:);
                param = [dim; snr; delta; repeat];
                outputdata = [result; param]';
                
                filename = sprintf('zz_NNPCA_Dim%d.dat', dim);
                dlmwrite(filename, outputdata, 'delimiter', ',', 'precision', 16, '-append');
            end
        end
    end
end

