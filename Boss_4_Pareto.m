    %%
%------------------------------Pareto-PCA------------------------------------

close all; clc; clear all;
specifier.matlabversion = 0; %0 if older than 2015 1 otherwise

dim = 100;
T = dim;
n_repeat = 100;
snr = 0.5;
delta = 0.3;
rank = 1;


%_______Set up data______
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
for repeat = 1:n_repeat
    
    %________Experiment_____
    options.maxOuterIter = 10000000;
    options.maxtime = 3600;
    options.minstepsize = 1e-10;
    
    specifier.ind = [0, 1, 1, 1, 1];
    
    result = clientconstraint_sphere_nonnegativePCA(X, rank, options, specifier);
    result = result(:);
    param = [snr; delta; dim; T; repeat];
    outputdata = [result; param]';

    filename = sprintf('zz_NNPCA_Pareto_Dim%d.dat', dim);
    dlmwrite(filename, outputdata, 'delimiter', ',', 'precision', 16, '-append');
end

