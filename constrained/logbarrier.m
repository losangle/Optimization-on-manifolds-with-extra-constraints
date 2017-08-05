% problem0.ineq_constraint_cost is a cell of function handle 
% problem0.ineq_constraint_grad is ................................
% such that all inequalities are >=0.

function logbarrier(problem0, x0, options)
    CONSTMAX = 1e200;
    M = problem0.M;
    mu = 0.05;
    xCur = x0;
    for iter = 1:30
        costfun = @(X) cost_logbarrier(X, problem0, mu);
        gradfun = @(X) grad_logbarrier(X, problem0, mu);
        problem.cost = costfun;
        problem.grad = gradfun;
        problem.M = M;
        options = [];
        
        [xCur, cost, info, options] = rlbfgs(problem, xCur, options);
        
%         u1 = [0.999999;0];
%         u2 = [0; 0.999999];
%         surfprofile(problem, [0;0], u1, u2);
%         hold on
%         plot3(xCur(1), xCur(2), cost_logbarrier(xCur, problem0, mu), 'r*');
%         hold off

        mu = mu/2;
    end

    xfinal = xCur;
    
    function val = cost_logbarrier(x, problem, u)
        val = getCost(problem, x);
        for numineq = 1: length(problem.ineq_constraint_cost)
            costhandle = problem.ineq_constraint_cost{numineq};
            cost_numineq = costhandle(x);
            if cost_numineq <= 0
                val = CONSTMAX;
                break;
            else
                val = val - u*log(cost_numineq);
            end
        end
    end

    function val = grad_logbarrier(x, problem, u)
        val = getGradient(problem, x);
        for numineq = 1: length(problem.ineq_constraint_cost)
            costhandle = problem.ineq_constraint_cost{numineq};
            cost_numineq = costhandle(x);
            if (cost_numineq <= 0)
                val = NaN;
                break;
            else
                gradhandle = problem.ineq_constraint_grad{numineq};
                constraint_grad = gradhandle(x);
                constraint_grad = problem.M.egrad2rgrad(x, constraint_grad);
                val = problem.M.lincomb(x, 1, val, -u/cost_numineq, constraint_grad);
            end
        end
    end

end


