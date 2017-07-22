function [x, cost, info, options] = bfgsnonsmoothwen(problem, x0, options)

    % Local defaults for the program
    localdefaults.minstepsize = 1e-10;
    localdefaults.maxiter = 10000;
    localdefaults.tolgradnorm = 1e-6;
    localdefaults.memory = 30;
    localdefaults.storedepth = 30;
    localdefaults.linesearch = @linesearch_hint;
    localdefaults.c1 = 0.001;
    localdefaults.c2 = 0.5;
    localdefaults.theta_epsilon = 0.1;
    localdefaults.theta_delta = 0.1;
    localdefaults.delta = 1e-4;
    localdefaults.epsilon = 1e-2;
    localdefaults.ys_over_yy_bound = 1e-4;
    localdefaults.ys_over_ss_bound = 1e-4;
    localdefaults.lsmaxcounter = 50;
    
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

    sHistory = cell(1, options.memory);
    yHistory = cell(1, options.memory);
    rhoHistory = cell(1, options.memory);
    alpha = 1; 
    scaleFactor = 1;
    stepsize = 1;
    accepted = 1;
   
    
    P_operator = @(v) P_operate(M, xCur, v, sHistory, yHistory, rhoHistory, scaleFactor, min(k, options.memory));
    xCurGradient = problem.subgrad(xCur, options.epsilon, P_operator);
    xCurCost = getCost(problem, xCur);
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
        
        if (stop && stop~= 2) || xCurGradNorm == 0 || options.epsilon <=1e-7
            if options.verbosity >= 1
                fprintf([reason '\n']);
            end
            break;
        end
        
        if xCurGradNorm^2 <= options.delta
            options.epsilon = options.epsilon * options.theta_epsilon;
            options.delta = options.delta * options.theta_delta;
            fprintf('epsilon is now %.16e', options.epsilon);
            P_operator = @(v) P_operate(M, xCur, v, sHistory, yHistory, rhoHistory, scaleFactor, min(k, options.memory));
            xCurGradient = problem.subgrad(xCur, options.epsilon, P_operator);
            xCurGradNorm = M.norm(xCur, xCurGradient);
            continue;
        end

        %--------------------Get Direction-----------------------
        if (isnan(xCurGradient))
            k = 0;
            P_operator = @(v) P_operate(M, xCur, v, sHistory, yHistory, rhoHistory, scaleFactor, min(k, options.memory));
            xCurGradient = problem.subgrad(xCur, options.epsilon, P_operator);
            xCurGradNorm = M.norm(xCur, xCurGradient);
        end
        Pg = P_operate(M, xCur, xCurGradient, sHistory,...
            yHistory, rhoHistory, scaleFactor, min(k, options.memory));
        
        p = M.lincomb(xCur, -1, Pg);

        %--------------------Line Search--------------------------
        dir_derivative = M.inner(xCur, p, xCurGradient);
        
        [xNext, xNextCost, alpha, fail, lsiters] = linesearchnonsmooth(problem, M, xCur, p,...
            xCurCost, dir_derivative, options.c1, options.c2, options.lsmaxcounter);
        if fail
            k = 0;
            P_operator = @(v) P_operate(M, xCur, v, sHistory, yHistory, rhoHistory, scaleFactor, min(k, options.memory));
            xCurGradient = problem.subgrad(xCur, options.epsilon, P_operator);
            xCurGradNorm = M.norm(xCur, xCurGradient);
            continue;
        end
        
        newkey = storedb.getNewKey();
        step = M.lincomb(xCur, alpha, p);
        stepsize = M.norm(xCur, step);
        
        
        %----------------Updating the next iteration---------------
        sk = M.transp(xCur, xNext, step);
        yk = M.lincomb(xNext, 1, getGradient(problem, xNext),...
            -1, M.transp(xCur, xNext, xCurGradient));
        ys_over_yy = M.inner(xNext, yk, sk)/M.inner(xNext, yk, yk);
        sk = M.lincomb(xNext, 1, sk, max(0, options.ys_over_yy_bound - ys_over_yy), yk);

        inner_sk_yk = M.inner(xNext, yk, sk);
        inner_sk_sk = M.inner(xNext, sk, sk);
        if inner_sk_yk/inner_sk_sk >= options.ys_over_ss_bound
            accepted = 1;
            rhok = 1/inner_sk_yk;
            scaleFactor = inner_sk_yk / M.inner(xNext, yk, yk);
            if (k>= options.memory)
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
            k = 0;
        end
        
        
        P_operator = @(v) P_operate(M, xNext, v, sHistory, yHistory, rhoHistory, scaleFactor, min(k, options.memory));
        xNextGradient = problem.subgrad(xNext, options.epsilon, P_operator);
        xNextCost = getCost(problem, xNext);
        xNextGradNorm = M.norm(xNext, xNextGradient);
        
        
        iter = iter + 1;
        xCur = xNext;
        key = newkey;
        xCurGradient = xNextGradient;
        xCurGradNorm = xNextGradNorm;
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
        stats.alpha = alpha;
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
function dir = P_operate(M, xCur, xCurGradient, sHistory, yHistory, rhoHistory, scaleFactor, k)
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
    dir = r;
end


function [xNext, costNext, t, fail, lsiters] = linesearchnonsmooth(problem, M, xCur, d, f0, df0, c1, c2, max_counter)
    if M.inner(xCur, getGradient(problem,xCur), d) >=0
        fprintf('LS failure by wrong direction');
        t = 1;
        fail = 1;
        costNext = inf;
        lsiters = -1;
        xNext = xCur;
        return
    end
    alpha = 0;
    fail = 0;
    beta = inf;
    t = 1;
    counter = max_counter;
    while counter > 0
        xNext = M.retr(xCur, d, t);
        if (getCost(problem, xNext) > f0 + df0*c1*t)
            beta = t;
        elseif diffretractionOblique(problem, M, t, d, xCur, xNext) < c2*df0
            alpha = t;
        else
            break;
        end
        if (isinf(beta))
            t = alpha*2;
        else
            t = (alpha+beta)/2;
        end
        counter = counter - 1;
    end
    if counter == 0
        fprintf('Failed LS \n');
        fail = 1;
    end
    costNext = getCost(problem, xNext);
    lsiters = max_counter - counter + 1;
end


function slope = diffretractionOblique(problem, M, alpha, p, xCur, xNext)
    [n, m] = size(p);
    diffretr = zeros(n, m);
    for i = 1 : m
        d = p(:, i);
        dInner = d.' * d;
        diffretr(:,i) = (d-alpha*dInner*xCur(:, i)) /sqrt((1+dInner * alpha^2)^3);
    end
    %Can be optimized.
    slope = M.inner(xNext, getGradient(problem, xNext), diffretr);
end