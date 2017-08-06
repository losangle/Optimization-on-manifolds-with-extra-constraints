% problem0.ineq_constraint_cost is a cell of function handle 
% problem0.ineq_constraint_grad is ................................
% such that all inequalities are >=0.

function xfinal = alm(problem0, x0, options)
    M = problem0.M;
    mu = 0.05;
    xCur = x0;
    totalineq = length(problem0.ineq_constraint_cost);
    lambdas = ones(totalineq, 1);
    for totaliter = 1:10
        costfun = @(X) cost_alm(X, problem0, mu, lambdas);
        gradfun = @(X) grad_alm(X, problem0, mu, lambdas);
        problem.cost = costfun;
        problem.grad = gradfun;
        problem.M = M;
        options = [];
        
        [xCur, cost, info, options] = rerealization(problem, xCur, options);
        
        for iterineq = 1: totalineq
            costhandler = problem0.ineq_constraint_cost{iterineq};
            cost_iter = costhandler(xCur);
            lambdas(iterineq) = max(lambdas(iterineq) - cost_iter/mu, 0);
        end
        
        mu = mu/2;
    end

    xfinal = xCur;
    
    function val = cost_alm(x, problem, u, lambdas)
        val = getCost(problem, x);
        for numineq = 1: length(problem.ineq_constraint_cost)
            costhandle = problem.ineq_constraint_cost{numineq};
            cost_numineq = costhandle(x);
            if cost_numineq - u * lambdas(numineq) <= 0
                val = val - lambdas(numineq) * cost_numineq + cost_numineq^2/(2*u);
            else
                val = val - u * lambdas(numineq)/2;
            end
        end
    end

    function val = grad_alm(x, problem, u, lambdas)
        val = getGradient(problem, x);
        for numineq = 1: length(problem.ineq_constraint_cost)
            costhandle = problem.ineq_constraint_cost{numineq};
            cost_numineq = costhandle(x);
            if (cost_numineq - u * lambdas(numineq) <= 0)
                gradhandle = problem.ineq_constraint_grad{numineq};
                constraint_grad = gradhandle(x);
                constraint_grad = problem.M.egrad2rgrad(x, constraint_grad);
                val = problem.M.lincomb(x, 1, val, cost_numineq/u - lambdas(numineq), constraint_grad);
            end
        end
    end

end


