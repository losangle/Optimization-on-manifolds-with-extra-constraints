function [xfinal,info] = exactpenaltyViaSmoothinglqh (problem0, x0, options)

    condet = constraintsdetail(problem0);
    
    %Outer Loop Setting
    localdefaults.rho = 1;
    localdefaults.thetarho = 0.3;
    localdefaults.maxOuterIter = 300;
    localdefaults.numOuterItertgn = 30;
    localdefaults.startingepsilon = 1e-1;
    localdefaults.endingepsilon = 1e-6;
    %Inner Loop Setting
    localdefaults.maxInnerIter = 200;
    localdefaults.startingtolgradnorm = 1e-3;
    localdefaults.endingtolgradnorm = 1e-6;

    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    tolgradnorm = options.startingtolgradnorm;
    thetatolgradnorm = nthroot(options.endingtolgradnorm/options.startingtolgradnorm, options.numOuterItertgn);
    theta_epsilon = nthroot(options.endingepsilon/options.startingepsilon, options.numOuterItertgn);
    
    M = problem0.M;
    xCur = x0;
    xPrev = xCur;
    epsilon = options.startingepsilon;
    rho = options.rho;
    
    OuterIter = 0;
    stats = savestats(x0);
    info(1) = stats;
    info(min(10000, options.maxOuterIter+1)).iter = [];

    totaltime = tic();
    
    for OuterIter = 1 : options.maxOuterIter
        timetic = tic();
        fprintf('Iteration: %d     ', OuterIter);
        
        costfun = @(X) cost_exactpenalty(X, problem0, rho);
        gradfun = @(X) grad_exactpenalty(X, problem0, rho);
        problem.cost = costfun;
        problem.grad = gradfun;
        problem.M = M;
        
        inneroptions.tolgradnorm = tolgradnorm;
        inneroptions.verbosity = 0;
        inneroptions.maxiter = options.maxInnerIter;
        inneroptions.minstepsize = options.minstepsize;
        
        [xCur, cost, innerInfo, Oldinneroptions] = rlbfgs(problem, xCur, inneroptions);
        
        %Save stats
        stats = savestats(xCur);
        info(OuterIter+1) = stats;

        
        if stats.maxviolation > epsilon
            rho = rho/options.thetarho;
        end
        
        epsilon  = max(options.endingepsilon, theta_epsilon * epsilon);
        tolgradnorm = max(options.endingtolgradnorm, tolgradnorm * thetatolgradnorm);
        
        fprintf('FroNormStepDiff: %.16e\n', norm(xPrev-xCur, 'fro'))
        if norm(xPrev-xCur, 'fro') < options.minstepsize && tolgradnorm <= options.endingtolgradnorm
            break;
        end
        xPrev = xCur;
        
        if toc(totaltime) > options.maxtime
            break;
        end
        
    end
    
    info = info(1: OuterIter+1);
    
    xfinal = xCur;
    
    function stats = savestats(x)
        stats.iter = OuterIter;
        if stats.iter == 0
            stats.time = 0;
        else
            stats.time = info(OuterIter).time + toc(timetic);
        end
        [maxviolation, meanviolation, costCur] = evaluation(problem0, x, condet);
        stats.maxviolation = maxviolation;
        stats.meanviolation = meanviolation;
        stats.cost = costCur;
    end
    

    function val = cost_exactpenalty(x, problem, rho)
        val = getCost(problem, x);
        % Adding ineq constraint cost
        if condet.has_ineq_cost
            for numineq = 1: condet.n_ineq_constraint_cost
                costhandle = problem.ineq_constraint_cost{numineq};
                cost_at_x = costhandle(x);
                if cost_at_x <= 0
                    cost_at_x = 0;
                elseif cost_at_x > epsilon
                        cost_at_x = cost_at_x - epsilon/2;
                else
                    cost_at_x = cost_at_x^2/(2*epsilon);
                end
                val = val + rho*cost_at_x;
            end
        end
        %Eq constratint cost
        if condet.has_eq_cost
            for numeq = 1 : condet.n_eq_constraint_cost
                costhandle = problem.eq_constraint_cost{numeq};
                cost_at_x = costhandle(x);
                val = val + rho * sqrt(cost_at_x^2 + epsilon^2);
            end
        end
    end

    function val = grad_exactpenalty(x, problem, u)
        val = getGradient(problem, x);
        if condet.has_ineq_cost
            for numineq = 1: condet.n_ineq_constraint_cost
                costhandle = problem.ineq_constraint_cost{numineq};            
                cost_at_x = costhandle(x);
                if cost_at_x >= 0
                    gradhandle = problem.ineq_constraint_grad{numineq};
                    constraint_grad = gradhandle(x);
                    constraint_grad = problem.M.egrad2rgrad(x, constraint_grad);
                    if cost_at_x >= epsilon
                        coef = 1;
                    else
                        coef = cost_at_x/epsilon;
                    end
                    val = problem.M.lincomb(x, 1, val, u * coef, constraint_grad);
                end
            end
        end
        if condet.has_eq_cost
            for numineq = 1: condet.n_eq_constraint_cost
                costhandle = problem.eq_constraint_cost{numineq};            
                cost_at_x = costhandle(x);
                gradhandle = problem.eq_constraint_grad{numineq};
                constraint_grad = gradhandle(x);
                constraint_grad = problem.M.egrad2rgrad(x, constraint_grad);
                coef = cost_at_x/sqrt(cost_at_x^2+epsilon^2);
                val = problem.M.lincomb(x, 1, val, u * coef, constraint_grad);
            end 
        end
    end

end