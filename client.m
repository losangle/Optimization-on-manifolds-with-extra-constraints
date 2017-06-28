clear all, close all, clc;
tests = 10;
for graphs = 1 : tests
    figure
    n = 4;
    for numtest = 0 : n-1
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
        xCur = problem.M.rand();
        options.memory = 15;
        %Set options
        options.linesearchVersion = 1; %BFGS LineSearch

        [gradnorms, alphas,time] = bfgsClean(problem, xCur, options);
        subplot(n,4,1+numtest*4)
        semilogy(gradnorms)
        xlbl = sprintf('Iter LS, time %f sec', time);
        xlabel(xlbl);
        ylabel('Norm');

        subplot(n,4,2+numtest*4)
        semilogy(alphas)
        xlabel('Iter LS');
        ylabel('alpha');

        options.linesearchVersion = 2;

        [gradnorms, alphas, time] = bfgsClean(problem, xCur, options);

        subplot(n,4,3+numtest*4)
        semilogy(gradnorms)
        xlbl = sprintf('Iter Qi, time %f sec', time);
        xlabel(xlbl);
        ylabel('Norm ');

        subplot(n,4,4+numtest*4)
        semilogy(alphas)
        xlabel('Iter Qi');
        ylabel('alpha');
    end
    
    filename = sprintf('AutoSave %d', graphs);
    print('-fillpage',filename,'-dpdf');
end

%% 
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
options.memory = 30;

xCur = problem.M.rand();


profile clear;
profile on;
% 
%  [x, cost, info, options] = bfgsCautious(problem,xCur,options);
% % 
% % 
% % 
% % [x, cost, info, options] = bfgsManifold(problem,xCur,options);
% % %     Display some statistics.
%     figure;
%     semilogy([info.iter], [info.gradnorm], '.-');
%     xlabel('Iteration number - BFGSManifold');
%     ylabel('Norm of the gradient of f');

% 
bfgsClean(problem,xCur,options);
% cacheSave(problem,xCur,options);
% 
% [x, cost, info, options] = bfgsCautious(problem,xCur,options);
% %     Display some statistics.
%     figure;
%     semilogy([info.iter], [info.gradnorm], '.-');
%     xlabel('Iteration number - BFGSIsometric');
%     ylabel('Norm of the gradient of f');
    
    
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


manifold = euclideanfactory(dim);
problem.M = manifold;

% Define the problem cost function and its Euclidean gradient.
problem.cost  = cost;
problem.egrad = grad;

euclidean(cost,grad,dim)
[x, cost, info, options] = bfgsCautious(problem);

figure;
semilogy([info.iter], [info.gradnorm], '.-');
xlabel('Iteration number - TrustRegion');
ylabel('Norm of the gradient of f');