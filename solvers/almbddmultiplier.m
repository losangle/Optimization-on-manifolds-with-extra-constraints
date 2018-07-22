function [xfinal, info] = almbddmultiplier(problem0, x0, options)

    condet = constraintsdetail(problem0);
    
    %Outer Loop Setting
    localdefaults.rho = 1;
    localdefaults.lambdas = ones(condet.n_ineq_constraint_cost, 1);
    localdefaults.gammas = ones(condet.n_eq_constraint_cost, 1);
    localdefaults.bound = 20;
    localdefaults.tau = 0.8;
    localdefaults.thetarho = 0.3;
    localdefaults.maxOuterIter = 300;
    localdefaults.numOuterItertgn = 30;
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
    
    lambdas = options.lambdas;
    gammas = options.gammas;
    rho = options.rho;
    oldacc = Inf;
    M = problem0.M;
    xCur = x0;
    xPrev = xCur;
    OuterIter = 0;
    
    stats = savestats(x0);
    info(1) = stats;
    info(min(10000, options.maxOuterIter+1)).iter = [];

    totaltime = tic();
    
    for OuterIter = 1 : options.maxOuterIter
        timetic = tic();
        fprintf('Iteration: %d     ', OuterIter);
        costfun = @(X) cost_alm(X, problem0, rho, lambdas, gammas);
        gradfun = @(X) grad_alm(X, problem0, rho, lambdas, gammas);
        problem.cost = costfun;
        problem.grad = gradfun;
        problem.M = M;
        inneroptions.tolgradnorm = tolgradnorm;
        inneroptions.verbosity = 0;
        inneroptions.maxiter = options.maxInnerIter;
        inneroptions.minstepsize = options.minstepsize;
         
        [xCur, cost, innerinfo, Oldinneroptions] = rlbfgs(problem, xCur, inneroptions);
        
        %Update Multipliers
        newacc = 0;
        for iterineq = 1 : condet.n_ineq_constraint_cost
            costhandler = problem0.ineq_constraint_cost{iterineq};
            cost_iter = costhandler(xCur);
            newacc = max(newacc, abs(max(-lambdas(iterineq)/rho, cost_iter)));
            lambdas(iterineq) = min(options.bound, max(lambdas(iterineq) + rho * cost_iter, 0));
        end
        
        for itereq = 1 : condet.n_eq_constraint_cost
            costhandler = problem0.eq_constraint_cost{itereq};
            cost_iter = costhandler(xCur);
            newacc = max(newacc, abs(cost_iter));
            gammas(itereq) = min(options.bound, max(-options.bound, gammas(itereq) + rho * cost_iter));
        end
        
        if OuterIter == 1 || newacc > options.tau * oldacc
            rho = rho/options.thetarho;
        end
        oldacc = newacc;
        tolgradnorm = max(options.endingtolgradnorm, tolgradnorm * thetatolgradnorm);
        
        %Save stats
        stats = savestats(xCur);
        info(OuterIter+1) = stats;
        
        fprintf('FroNormStepDiff: %.16e\n', norm(xPrev-xCur,'fro'))
        if norm(xPrev-xCur, 'fro') < options.minstepsize && tolgradnorm <= options.endingtolgradnorm
            break;
        end
        if toc(totaltime) > options.maxtime
            break;
        end
        
        xPrev = xCur;
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
    
    function val = cost_alm(x, problem, rho, lambdas, gammas)
        val = getCost(problem, x);
        if condet.has_ineq_cost
            for numineq = 1: condet.n_ineq_constraint_cost
                costhandle = problem.ineq_constraint_cost{numineq};
                cost_numineq = costhandle(x);
                val = val + (rho/2) * (max(0, lambdas(numineq)/rho + cost_numineq)^2);
            end
        end
        
        if condet.has_eq_cost
            for numeq = 1: condet.n_eq_constraint_cost
                costhandle = problem.eq_constraint_cost{numeq};
                cost_numeq = costhandle(x);
                val = val + (rho/2) * (gammas(numeq)/rho + cost_numeq)^2;
            end
        end
    end

    function val = grad_alm(x, problem, rho, lambdas, gammas)
        val = getGradient(problem, x);
        if condet.has_ineq_cost
            for numineq = 1: condet.n_ineq_constraint_cost
                costhandle = problem.ineq_constraint_cost{numineq};
                cost_numineq = costhandle(x);
                if (cost_numineq + lambdas(numineq)/rho > 0)
                    gradhandle = problem.ineq_constraint_grad{numineq};
                    constraint_grad = gradhandle(x);
                    constraint_grad = problem.M.egrad2rgrad(x, constraint_grad);
                    val = problem.M.lincomb(x, 1, val, cost_numineq * rho + lambdas(numineq), constraint_grad);
                end
            end
        end
        
        if condet.has_eq_cost
            for numeq = 1:condet.n_eq_constraint_cost
                costhandle = problem.eq_constraint_cost{numeq};
                cost_numeq = costhandle(x);
                gradhandle = problem.eq_constraint_grad{numeq};
                constraint_grad = gradhandle(x);
                constraint_grad = problem.M.egrad2rgrad(x, constraint_grad);
                val = problem.M.lincomb(x, 1, val, cost_numeq * rho + gammas(numeq), constraint_grad);
            end
        end
    end

end


