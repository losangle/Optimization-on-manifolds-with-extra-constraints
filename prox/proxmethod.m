function [X, cost, stats, options] =  proxmethod(problem0, x0, options)

    lambda = 1e-5;
    pivot = x0;
    
%     while (lambda <= 10)
    for iter = 1 : 100
        costFun = @(X) getCost(problem0, X)+ problem0.regcost(X, pivot, lambda);
        gradFun = @(X) getGradient(problem0, X) + problem0.reggrad(X, pivot, lambda);
        problem.cost = costFun;
        problem.grad = gradFun;
        problem.M = problem0.M;
%         checkgradient(problem);
%         yo = 1;
        [x0, cost, info, options] = rlbfgs(problem, x0, options);
%         lambda = lambda*1.01;
        pivot = x0;
    end
    
    X = x0;
    cost = cost;
    stats = info;
end