function [x, cost, info, options] = rlbfgsinfeasible(problem, x0, options)


    % Verify that the problem description is sufficient for the solver.
    if ~canGetCost(problem)
        warning('manopt:getCost', ...
            'No cost provided. The algorithm will likely abort.');
    end
    if ~canGetGradient(problem) && ~canGetApproxGradient(problem)
        % Note: we do not give a warning if an approximate gradient is
        % explicitly given in the problem description, as in that case the user
        % seems to be aware of the issue.
        warning('manopt:getGradient:approx', ...
               ['No gradient provided. Using an FD approximation instead (slow).\n' ...
                'This algorithm is not designed to work with inexact gradient: behavior not guaranteed.\n' ...
                'It may be necessary to increase options.tolgradnorm.\n' ...
                'To disable this warning: warning(''off'', ''manopt:getGradient:approx'')']);
        problem.approxgrad = approxgradientFD(problem);
    end
    
    % Local defaults for the program
    localdefaults.minstepsize = 1e-10;
    localdefaults.maxiter = 1000;
    localdefaults.tolgradnorm = 1e-6;
    localdefaults.memory = 30;
    localdefaults.strict_inc_func = @(x) x;
    localdefaults.ls_max_steps  = 25;
    localdefaults.storedepth = 30;
    localdefaults.linesearch = @linesearch_hint;
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    % To make sure memory in range [0, Inf)
    options.memory = max(options.memory, 0);
    if options.memory == Inf
        if isinf(options.maxiter)
            options.memory = 10000;
            warning('rlbfgs:memory',['options.memory and options.maxiter'...
                'are both Inf. This might be too greedy. '...
                'options.memory is now limited to 10000']);
        else
            options.memory = options.maxiter;
        end
    end
    
    M = problem.M;
    
    % Create a random starting point if no starting point
    % is provided.
    if ~exist('x0', 'var')|| isempty(x0)
        xCur = M.rand(); 
    else
        xCur = x0;
    end
    
    timetic = tic();
    
    % Create a store database and get a key for the current x
    storedb = StoreDB(options.storedepth);
    key = storedb.getNewKey();
    
    % __________Initialization of variables______________
    % Number of iterations since the last restart
    k = 0;  
    % Number of total iteration in BFGS
    iter = 0; 
    % Saves step vectors that points to x_{t+1} from x_{t}
    % for t in range (max(0, iter - min{k, options.memory}), iter].
    % That is, saves up to options.memory number of most 
    % current step vectors. 
    % However, the implementation below does not need stepvectors 
    % in their respective tangent spaces at x_{t}'s, but rather, having 
    % them transported to the most current point's tangent space by vector tranport.
    % For detail of the requirement on the the vector tranport, see the reference. 
    % In implementation, those step vectors are iteratively 
    % transported to most current point's tangent space after every iteration.
    % So at every iteration, it will have this list of vectors in tangent plane
    % of current point.
    sHistory = cell(1, options.memory);
    % Saves the difference between gradient of x_{t+1} and the
    % gradient of x_{t} by transported to x_{t+1}'s tangent space.
    % where t is in range (max(0, iter - min{k, options.memory}), iter].
    % That is, saves up to options.memory number of most 
    % current gradient differences.
    % The implementation process is similar to sHistory.
    yHistory = cell(1, options.memory);
    % rhoHistory{t} is the innerproduct of sHistory{t} and yHistory{t}
    rhoHistory = cell(1, options.memory);
    % Scaling of direction given by getDirection for acceptable step
    alpha = 1; 
    % Scaling of initial matrix, Barzilai-Borwein.
    scaleFactor = 1;
    % Norm of the step
    stepsize = 1;
    % Boolean for whether the step is accepted by Cautious update check
    accepted = 1;
    
    [xCurCost, xCurGradient] = getCostGrad(problem, xCur, storedb, key);
    xCurGradNorm = M.norm(xCur, xCurGradient);
    lsstats = [];
    %A variable to control restarting scheme, see comment below.
    ultimatum = 0;
    
    % Save stats in a struct array info, and preallocate.
    stats = savestats();
    info(1) = stats;
    info(min(10000, options.maxiter+1)).iter = [];
    
    if options.verbosity >= 2
        fprintf(' iter\t               cost val\t                 grad. norm\t        alpha \n');
    end
    
    while (1)
        %------------------------ROUTINE----------------------------

        % Display iteration information
        if options.verbosity >= 2
        %_______Print Information and stop information________
        fprintf('%5d\t%+.16e\t%.8e\t %.4e\n', iter, xCurCost, xCurGradNorm, alpha);
        end
        
        % Start timing this iteration
        timetic = tic();
        
        % Run standard stopping criterion checks
        [stop, reason] = stoppingcriterion(problem, xCur, options, ...
            info, iter+1);
        
        % If none triggered, run specific stopping criterion check
        if ~stop 
            if stats.stepsize < options.minstepsize
                % To avoid infinite loop and to push the search further
                % in case BFGS approximation of Hessian is off towards
                % the end, we erase the memory by setting k = 0;
                % In this way, it starts off like a steepest descent.
                % If even steepest descent does not work, then it is 
                % hopeless and we will terminate.
                if ultimatum == 0
                    if (options.verbosity >= 2)
                        fprintf(['stepsize is too small, restart the bfgs procedure' ...
                            'with the current point\n']);
                    end
                    k = 0;
                    ultimatum = 1;
                else
                    stop = true;
                    reason = sprintf(['Last stepsize smaller than minimum '  ...
                        'allowed; options.minstepsize = %g.'], ...
                        options.minstepsize);
                end
            else
                ultimatum = 0;
            end
        end  
        
        if stop
            if options.verbosity >= 1
                fprintf([reason '\n']);
            end
            break;
        end

        %--------------------Get Direction-----------------------

        p = getDirection(M, xCur, xCurGradient, sHistory,...
            yHistory, rhoHistory, scaleFactor, min(k, options.memory));

        %--------------------Line Search--------------------------
        [stepsize, xNext, newkey, lsstats] = ...
            linesearch_hint(problem, xCur, p, xCurCost, M.inner(xCur,xCurGradient,p), options, storedb, key);
        
        alpha = stepsize/M.norm(xCur, p);
        step = M.lincomb(xCur, alpha, p);
        
        
        %----------------Updating the next iteration---------------
        [xNextCost, xNextGradient] = getCostGrad(problem, xNext, storedb, newkey);
        sk = M.transp(xCur, xNext, step);
        yk = M.lincomb(xNext, 1, xNextGradient,...
            -1, M.transp(xCur, xNext, xCurGradient));

        inner_sk_yk = M.inner(xNext, yk, sk);
        inner_sk_sk = M.inner(xNext, sk, sk);
        % If cautious step is not accepted, then we do no take the
        % current sk, yk into account. Otherwise, we record it 
        % and use it in approximating hessian.
        % sk, yk are maintained in the most recent point's 
        % tangent space by transport.
        if inner_sk_sk ~= 0 && (inner_sk_yk / inner_sk_sk)>= options.strict_inc_func(xCurGradNorm)
            accepted = 1;
            rhok = 1/inner_sk_yk;
            scaleFactor = inner_sk_yk / M.inner(xNext, yk, yk);
%             scaleFactor = 1;
            if (k>= options.memory)
                % sk and yk are saved from 1 to the end
                % with the most currently recorded to the 
                % rightmost hand side of the cells that are
                % occupied. When memory is full, do a shift
                % so that the rightmost is earliest and replace
                % it with the most recent sk, yk.
                for  i = 2:options.memory
                    sHistory{i} = M.transp(xCur, xNext, sHistory{i});
                    yHistory{i} = M.transp(xCur, xNext, yHistory{i});
                end
                if options.memory > 1
                sHistory = sHistory([2:end, 1]);
                yHistory = yHistory([2:end, 1]);
                rhoHistory = rhoHistory([2:end 1]);
                end
                if options.memory > 0
                    sHistory{options.memory} = sk;
                    yHistory{options.memory} = yk;
                    rhoHistory{options.memory} = rhok;
                end
            else
                for  i = 1:k
                    sHistory{i} = M.transp(xCur, xNext, sHistory{i});
                    yHistory{i} = M.transp(xCur, xNext, yHistory{i});
                end
                sHistory{k+1} = sk;
                yHistory{k+1} = yk;
                rhoHistory{k+1} = rhok;
            end
            k = k+1;
        else
            accepted = 0;
            for  i = 1:min(k, options.memory)
                sHistory{i} = M.transp(xCur, xNext, sHistory{i});
                yHistory{i} = M.transp(xCur, xNext, yHistory{i});
            end
        end
        iter = iter + 1;
        xCur = xNext;
        key = newkey;
        xCurGradient = xNextGradient;
        xCurGradNorm = M.norm(xNext, xNextGradient);
        xCurCost = xNextCost;
        
        % Make sure we don't use too much memory for the store database
        storedb.purge();
        
        
        % Log statistics for freshly executed iteration
        stats = savestats();
        info(iter+1) = stats; 
        
    end

    info = info(1:iter+1);
    x = xCur;
    cost = xCurCost;

    if options.verbosity >= 1
        fprintf('Total time is %f [s] (excludes statsfun)\n', ...
                info(end).time);
    end

    % Routine in charge of collecting the current iteration stats
    function stats = savestats()
        stats.iter = iter;
        stats.cost = xCurCost;
        stats.gradnorm = xCurGradNorm;
        if iter == 0
            stats.stepsize = NaN;
            stats.accepted = NaN;
            stats.time = toc(timetic);
        else
            stats.stepsize = stepsize;
            stats.time = info(iter).time + toc(timetic);
            stats.accepted = accepted;
        end
        stats.linesearch = lsstats;
        stats = applyStatsfun(problem, xCur, storedb, key, options, stats);
    end

end

% BFGS step, see Wen's paper for details. This functon basically takes in
% a vector g, and operate inverse approximate Hessian P on it to get 
% Pg, and take negative of it. Due to isometric transport and locking condition
% (see paper), this implementation operates in tangent spaces of the most
% recent point instead of transport this vector iteratively backwards to operate 
% in tangent planes of previous points. Notice that these two conditions are hard
% or expensive to enforce. However, in practice, there is no observed difference
% in them, if your problem requires isotransp, it may be good
% to replace transp with isotransp. There are built in isotransp
% for spherefactory and obliquefactory
function dir = getDirection(M, xCur, xCurGradient, sHistory, yHistory, rhoHistory, scaleFactor, k)
    q = xCurGradient;
    inner_s_q = zeros(1, k);
    for i = k : -1 : 1
        inner_s_q(1, i) = rhoHistory{i} * M.inner(xCur, sHistory{i}, q);
        q = M.lincomb(xCur, 1, q, -inner_s_q(1, i), yHistory{i});
    end
    r = M.lincomb(xCur, scaleFactor, q);
    for i = 1 : k
         omega = rhoHistory{i} * M.inner(xCur, yHistory{i}, r);
         r = M.lincomb(xCur, 1, r, inner_s_q(1, i)-omega, sHistory{i});
    end
    dir = M.lincomb(xCur, -1, r);
end
