% problem0.ineq_constraint_cost is a cell of function handle 
% problem0.ineq_constraint_grad is ................................
% such that all inequalities are >=0.
function xfinal = exactpenalty(problem0, x0, options)
    M = problem0.M;
    mu = 1e-2;
    xCur = x0;
    for iter = 1:20
        costfun = @(X) cost_exactpenalty(X, problem0, 1/mu);
        gradfun = @(X) grad_exactpenalty(X, problem0, 1/mu);
        problem.cost = costfun;
        problem.grad = gradfun;
        problem.M = M;
        
%         [xCur, cost, info, options] = rerealization(problem, xCur, options);
        [xCur, cost, info, options] = rlbfgs(problem, xCur, options);
        
%         u1 = [1.2;0];
%         u2 = [0; 1.2];
%         surfprofile(problem, [0;0], u1, u2);
%         hold on
%         plot3(xCur(1)/1.2, xCur(2)/1.2, cost_exactpenalty(xCur, problem0, 1/mu), 'r*');
%         hold off
        
        mu = mu/2;
    end
    
    xfinal = xCur;

    function val = cost_exactpenalty(x, problem, u)
        val = getCost(problem, x);
        for numineq = 1: length(problem.ineq_constraint_cost)
            costhandle = problem.ineq_constraint_cost{numineq};
            val = val + u*max(0, -costhandle(x));
        end
    end

    function val = grad_exactpenalty(x, problem, u)
        val = getGradient(problem, x);
        for numineq = 1: length(problem.ineq_constraint_cost)
            costhandle = problem.ineq_constraint_cost{numineq};
            if (costhandle(x) < 0)
                gradhandle = problem.ineq_constraint_grad{numineq};
                constraint_grad = gradhandle(x);
                constraint_grad = problem.M.egrad2rgrad(x, constraint_grad);
                val = problem.M.lincomb(x, 1, val, -u, constraint_grad);
            end
        end
    end

end


