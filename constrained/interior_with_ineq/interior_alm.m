% problem0.ineq_constraint_cost is a cell of function handle 
% problem0.ineq_constraint_grad is ................................
% such that all inequalities are >=0.

function alm(problem0, x0, options)
    CONSTMAX = 1e200;
    
    numconstraint = length(problem0.ineq_constraint_cost);
    
    M1 = problem0.M;
    M2 = euclideanfactory(numconstraint);
    M = cells(1,2);
    M{1} = M1;
    M{2} = M2;
    
    xCur = cells(1,2);
    xCur{1} = x0;
    xCur{2} = ones(numconstraint, 1);
    
    mu = 0.05;
    
    lambdas = ones(numconstraint, 1);
    
    for iter = 1:30
        costfun = @(X) cost_alm(X, problem0, mu, lambdas);
        gradfun = @(X) grad_alm(X, problem0, mu, lambdas);
        newproblem.cost = costfun;
        newproblem.grad = gradfun;
        newproblem.M = M;
        options = [];
        
        [xCur, cost, info, options] = rlbfgsinfeasible(newproblem, xCur, options);
        lambdas = lambdas - 
        
        mu = mu/2;
    end

    xfinal = xCur;
    
    function val = cost_alm(x, problem, u, lambdas)
        val = getCost(problem, x{1});
        w = x{2};
        for numdim = 1: length(w)
            if w(numdim) <= 0
                val = CONSTMAX;
                return;
            end
        end
        val = val - u*(sum(log(w)));
        for numineq = 1: length(problem.ineq_constraint_cost)
            costhandle = problem.ineq_constraint_cost{numineq};
            val = val - lambdas(numineq)*(costhandle(x) - w(numineq));
        end
    end

    function valgrad = grad_alm(x, problem, u, lambdas)
        numconstraints = length(problem.ineq_constraint_cost);
        valgrad = cell(1,2);
        valgrad{1} = getGradient(problem, x{1});
        w = x{2};
        
        if sum(w <= 0) > 0
            valgrad = NaN;
            fprintf('Try to fetch grad at Infeasible point\n');
            return;
        end
        valgrad{2} = -u./w;
        valgrad{2} = valgrad{2} + lambdas;
        
        for numineq = 1: numconstraints
                gradhandle = problem.ineq_constraint_grad{numineq};
                constraint_grad = gradhandle(x{1});
                constraint_grad = problem.M.egrad2rgrad(x{1}, constraint_grad);
                valgrad{1} = problem.M.lincomb(x, 1, valgrad{1}, -lambdas(numineq), constraint_grad);
        end
    end

end


