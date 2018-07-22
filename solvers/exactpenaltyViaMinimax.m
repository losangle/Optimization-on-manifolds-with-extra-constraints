function [xfinal,info] = exactpenaltyViaMinimax (problem0, x0, options)

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
    localdefaults.maxNumVecInConvhull = 2000;
    
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    tolgradnorm = options.startingtolgradnorm;
    thetatolgradnorm = nthroot(options.endingtolgradnorm/options.startingtolgradnorm, options.numOuterItertgn);
    theta_epsilon = nthroot(options.endingepsilon/options.startingepsilon, options.numOuterItertgn);
    
    rho = options.rho;
    epsilon = options.startingepsilon;
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
        fprintf('Iteration: %d    ', OuterIter);
        costfun = @(X) cost_exactpenalty(X, problem0, rho);
        gradfun = @(X) grad_exactpenalty(X, problem0, rho);
        problem.cost = costfun;
        problem.grad = gradfun;
        problem.M = M;
        problem.subgrad = @(X, discre, P_operator) subgrad_exactpenalty(X, problem0, rho, discre, P_operator);
        
        inneroptions.tolgradnorm = tolgradnorm;
        inneroptions.verbosity = 0;
        inneroptions.maxiter = options.maxInnerIter;
        inneroptions.minstepsize = options.minstepsize;
        
        options.maxiter =  200;
        [xCur, cost, innerinfo, Oldinneroptions] = bfgsnonsmoothminimax(problem, xCur, inneroptions);

        %Save stats
        stats = savestats(xCur);
        info(OuterIter+1) = stats;
        
        if stats.maxviolation > epsilon
            rho = rho/options.thetarho;
        end
        
        epsilon  = max(options.endingepsilon, theta_epsilon * epsilon);
        tolgradnorm = max(options.endingtolgradnorm, tolgradnorm * thetatolgradnorm);
        
        fprintf('FroNormStepDiff: %.16e\n', norm(xPrev-xCur,'fro'))
        if norm(xPrev-xCur, 'fro') < options.minstepsize && tolgradnorm <= options.endingtolgradnorm && epsilon <=options.endingepsilon
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
    
    function val = cost_exactpenalty(x, problem, rho)
        val = getCost(problem, x);
        % Adding ineq constraint cost
        if condet.has_ineq_cost
            for numineq = 1 : condet.n_ineq_constraint_cost
                costhandle = problem.ineq_constraint_cost{numineq};
                cost_at_x = costhandle(x);
                val = val + rho * max(0, cost_at_x);
            end
        end
        %Eq constratint cost
        if condet.has_eq_cost
            for numeq = 1 : condet.n_eq_constraint_cost
                costhandle = problem.eq_constraint_cost{numeq};
                cost_at_x = costhandle(x);
                val = val + rho * abs(cost_at_x);
            end
        end
    end

    function val = grad_exactpenalty(x, problem, rho)
        val = getGradient(problem, x);
        if condet.has_ineq_cost
            for numineq = 1 : condet.n_ineq_constraint_cost
                costhandle = problem.ineq_constraint_cost{numineq};
                if (costhandle(x) > 0)
                    gradhandle = problem.ineq_constraint_grad{numineq};
                    constraint_grad = gradhandle(x);
                    constraint_grad = problem.M.egrad2rgrad(x, constraint_grad);
                    val = problem.M.lincomb(x, 1, val, rho, constraint_grad);
                end
            end
        end
        if condet.has_eq_cost
            for numeq = 1 : condet.n_eq_constraint_cost
                costhandle = problem.eq_constraint_cost{numeq};
                cost_at_x = costhandle(x);
                gradhandle = problem.eq_constraint_grad{numeq};
                constraint_grad = gradhandle(x);
                constraint_grad = problem.M.egrad2rgrad(x, constraint_grad);
                val = problem.M.lincomb(x, 1, val, rho*sign(cost_at_x), constraint_grad);
            end
        end
    end

    %Tol for Pnorm is not set.
    function val = subgrad_exactpenalty(x, problem, rho, discrepency, P_operator)
        N_mustPutVecs = 0;
        N_ineqVecs = 0;
        N_eqVecs = 0;
        mustPutVecs = cell(1, condet.n_ineq_constraint_cost + condet.n_eq_constraint_cost + 1);
        ineqVecs = cell(1, condet.n_ineq_constraint_cost);
        eqVecs = cell (1, condet.n_eq_constraint_cost);
        
        N_mustPutVecs = N_mustPutVecs + 1;
        mustPutVecs{N_mustPutVecs} = getGradient(problem, x);
        
        if condet.has_ineq_cost
            for numineq = 1: condet.n_ineq_constraint_cost
                costhandle = problem.ineq_constraint_cost{numineq};
                cost_at_x = costhandle(x);
                gradhandle = problem.ineq_constraint_grad{numineq};
                constraint_grad = gradhandle(x);
                constraint_grad = problem.M.egrad2rgrad(x, constraint_grad);
                if (abs(cost_at_x) < discrepency) && (N_ineqVecs+N_eqVecs < options.maxNumVecInConvhull)
                    N_ineqVecs = N_ineqVecs + 1;
                    ineqVecs{N_ineqVecs} = problem.M.lincomb(x, rho, constraint_grad);
                else
                    if (cost_at_x > 0)
                        N_mustPutVecs = N_mustPutVecs + 1;
                        mustPutVecs{N_mustPutVecs} = problem.M.lincomb(x, rho, constraint_grad);
                    end
                end
            end
        end
        
        if condet.has_eq_cost
            for numeq = 1: condet.n_eq_constraint_cost
                costhandle = problem.eq_constraint_cost{numeq};
                cost_at_x = costhandle(x);
                gradhandle = problem.eq_constraint_grad{numeq};
                constraint_grad = gradhandle(x);
                constraint_grad = problem.M.egrad2rgrad(x, constraint_grad);
                
                if (abs(cost_at_x) < discrepency/2) && (N_ineqVecs+N_eqVecs < options.maxNumVecInConvhull)
                    N_eqVecs = N_eqVecs + 1;
                    eqVecs{N_eqVecs} = problem.M.lincomb(x, rho * sign(cost_at_x), constraint_grad);
                else
                    N_mustPutVecs = N_mustPutVecs + 1;
                    mustPutVecs{N_mustPutVecs} = problem.M.lincomb(x, rho*sign(cost_at_x), constraint_grad);
                end
            end
        end

        val = lincomb(problem.M, x, mustPutVecs(1:N_mustPutVecs), ones(N_mustPutVecs,1));
        if (N_ineqVecs + N_eqVecs ~= 0)
            [u_norm, coeffs, val, nonposdef] = smallestinconvexhullpnormconstrained(problem.M, x, val, ineqVecs(1:N_ineqVecs), eqVecs(1:N_eqVecs), P_operator);
            %catch this!
            if nonposdef
                val = NaN;
            end
        end
    end

end


