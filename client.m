dim = 1000;
A = randn(dim,dim);
A = A + A.';
cost = @(x) (x'*A*x);
grad = @(x) 2*A*x;

% Create the problem structure.
manifold = spherefactory(dim);
problem.M = manifold;

% Define the problem cost function and its Euclidean gradient.
problem.cost  = cost;
problem.egrad = grad;

%Set options
options.linesearchVersion = 1;
options.memory = 10;

xCur = problem.M.rand();
profile clear;
profile on;

[x, cost, info, options] = bfgsCautious(problem,xCur,options);



[x, cost, info, options] = bfgsManifold(problem,xCur,options);
%     Display some statistics.
    figure;
    semilogy([info.iter], [info.gradnorm], '.-');
    xlabel('Iteration number - BFGSManifold');
    ylabel('Norm of the gradient of f');


[x, cost, info, options] = bfgsIsometric(problem,xCur,options);

%     Display some statistics.
    figure;
    semilogy([info.iter], [info.gradnorm], '.-');
    xlabel('Iteration number - BFGSIsometric');
    ylabel('Norm of the gradient of f');
    
    
% [x, cost, info, options] = trustregions(problem);
%     Display some statistics.
%     figure;
%     semilogy([info.iter], [info.gradnorm], '.-');
%     xlabel('Iteration number - TrustRegion');
%     ylabel('Norm of the gradient of f');

profile off;
profile report


%% Euclidean Case
dim = 2;
cost = @(x) (1-x(1))^2+100*(x(2)-x(1)^2)^2;
grad = @(x) [-2*(1-x(1))+200*(x(2)-x(1)^2)*(-2*x(1));200*(x(2)-x(1)^2)];

euclidean(cost,grad,dim)
