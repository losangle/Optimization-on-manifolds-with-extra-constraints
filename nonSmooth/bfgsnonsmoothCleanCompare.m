function [stats, finalX] = bfgsnonsmoothCleanCompare(problem, x, options)
    
    timetic = tic();
    M = problem.M;

    if ~exist('x','var')
        xCur = M.rand();
    else 
        xCur = x;
    end
    
    localdefaults.minstepsize = 1e-50;
    localdefaults.maxiter = 10000;
    localdefaults.tolgradnorm = 1e-4;  %iterimgradnorm that is used during discrepency < maxdiscrepency
    localdefaults.finalgradnorm = 1e-10;
    localdefaults.memory = 30;
    localdefaults.c1 = 0.0001; 
    localdefaults.c2 = 0.5;
    localdefaults.discrepency = 1e-4;
    localdefaults.discrepencydownscalefactor = 1e-1; 
    localdefaults.maxdiscrepency = 1e-10;
    localdefaults.lsmaxcounter = 50;
    localdefaults.lambda = 0.001;
    localdefaults.LAMBDA = 0.001;
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
   
    xCurGradient = getGradient(problem, xCur);
%     xCurGradient = problem.gradAlt(xCur, options.discrepency);
    xCurGradNorm = M.norm(xCur, xCurGradient);
    xCurCost = getCost(problem, xCur);
    
    k = 0;
    iter = 0;
    sHistory = cell(1, options.memory);
    yHistory = cell(1, options.memory);
    rhoHistory = cell(1, options.memory);
    alpha = 1;
    scaleFactor = 1;
    stepsize = 1;
    existsAssumedoptX = exist('options','var') && ~isempty(options) && exist('options.assumedoptX', 'var');
    lsiters = 1;
    ultimatum = 0;
    pushforward = 0;
    
    
    stats.gradnorms = zeros(1,options.maxiter);
    stats.alphas = zeros(1,options.maxiter);
    stats.stepsizes = zeros(1,options.maxiter);
    stats.costs = zeros(1,options.maxiter);
    stats.xHistory = cell(1,options.maxiter);
    if existsAssumedoptX
        stats.distToAssumedOptX = zeros(1,options.maxiter);
    end
    
    savestats();

    fprintf(' iter\t               cost val\t    grad. norm\t   lsiters\n');

    
    while (1)
        
        if pushforward == 1
            if options.discrepency > options.maxdiscrepency
                options.discrepency = options.discrepency*options.discrepencydownscalefactor;
                if options.discrepency < options.maxdiscrepency
                    options.tolgradnorm = options.finalgradnorm;
                end
                fprintf('current discrepency is %.16e \n', options.discrepency);
                pushforward = 0;
                k = 0;
                sHistory = cell(1, options.memory);
                yHistory = cell(1, options.memory);
                rhoHistory = cell(1, options.memory);
                alpha = 1;
                scaleFactor = stepsize * 2/xCurGradNorm; %Need to reconsider
                xCurGradient = problem.gradAlt(xCur, options.discrepency);
                xCurGradNorm = M.norm(xCur, xCurGradient);
                xCurCost = getCost(problem, xCur);
                stepsize = 1;
                continue;
            else
                break;
            end
        end
        if ultimatum == 0
            %_______Print Information and stop information________
            fprintf('%5d\t%+.16e\t%.8e\t %d\n', iter, xCurCost, xCurGradNorm, lsiters);
        end
        
        if (xCurGradNorm < options.tolgradnorm)
            fprintf('Target Reached\n');
            pushforward = 1;
            continue;
        end
        if (stepsize <= options.minstepsize)
            fprintf('Stepsize too small\n')
            pushforward = 1;
            continue;
        end
        if (iter > options.maxiter)
            fprintf('maxiter reached\n')
            pushforward = 1;
            continue;
        end

        %_______Get Direction___________________________
% 
%        p = getDirection(M, xCur, xCurGradient, sHistory,...
%             yHistory, rhoHistory, scaleFactor, min(k, options.memory));
        
%         p = -getGradient(problem, xCur);

        [g, p] = descent(problem, M, xCur, xCurCost, options.tolgradnorm, options.c1, options.discrepency, sHistory, yHistory, rhoHistory, 1, min(k, options.memory));
        p = M.lincomb(xCur, p, scaleFactor);
        
        %_______Line Search____________________________
        dir_derivative = M.inner(xCur,xCurGradient,p);
        if  dir_derivative> 0
            fprintf('directionderivative IS POSITIVE\n');
        end

        [xNextCost, alpha, fail, lsiters] = linesearchnonsmooth(problem, M, xCur, p, xCurCost, dir_derivative, options.c1, options.c2, options.lsmaxcounter);
        step = M.lincomb(xCur, alpha, p);
        newstepsize = M.norm(xCur, step);
        if fail == 1 || newstepsize < 1e-14
            if ultimatum == 1
                fprintf('Even descent direction does not help us now\n');
                pushforward = 1;
                continue;
            else
                k = 0;
                scaleFactor = stepsize*2/xCurGradNorm;
                ultimatum = 1;
                continue;
            end
        else
            ultimatum = 0;
        end
        stepsize = newstepsize;
        xNext = M.retr(xCur, step, 1);
        
        %_______Updating the next iteration_______________
        xNextGradient = problem.reallygrad(xNext);
%         xNextGradient = problem.gradAlt(xNext, options.discrepency);        
        
        sk = M.transp(xCur, xNext, step);
        yk = M.lincomb(xNext, 1, xNextGradient,...
            -1, M.transp(xCur, xNext, g));
        
        inner_sk_yk = M.inner(xNext, yk, sk);
        sk = M.lincomb(xCur, 1, sk, max(0, options.LAMBDA-inner_sk_yk/M.inner(xNext,yk,yk)), yk);
        
        if (inner_sk_yk /M.inner(xNext, sk, sk))> options.lambda
            rhok = 1/inner_sk_yk;
            scaleFactor = inner_sk_yk / M.inner(xNext, yk, yk);
            if (k>= options.memory)
                for  i = 2:options.memory
                    sHistory{i} = M.transp(xCur, xNext, sHistory{i});
                    yHistory{i} = M.transp(xCur, xNext, yHistory{i});
                end
                sHistory = sHistory([2:end 1]);
                sHistory{options.memory} = sk;
                yHistory = yHistory([2:end 1]);
                yHistory{options.memory} = yk;
                rhoHistory = rhoHistory([2:end 1]);
                rhoHistory{options.memory} = rhok;
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
            k = 0;
        end

        iter = iter + 1;
        xCur = xNext;
        xCurGradient = xNextGradient;
        xCurGradNorm = M.norm(xCur, xNextGradient);
        xCurCost = xNextCost;
      
        savestats()
    end
    
    stats.gradnorms = stats.gradnorms(1,1:iter+1);
    stats.alphas = stats.alphas(1,1:iter+1);
    stats.costs = stats.costs(1,1:iter+1);
    stats.stepsizes = stats.stepsizes(1,1:iter+1);
    stats.xHistory= stats.xHistory(1,1:iter+1);
    stats.time = toc(timetic);
    if existsAssumedoptX
        stats.distToAssumedOptX = stats.distToAssumedOptX(1, 1:iter+1);
    end
    finalX = xCur;
    
    function savestats()
        stats.gradnorms(1, iter+1)= xCurGradNorm;
        stats.alphas(1, iter+1) = alpha;
        stats.stepsizes(1, iter+1) = stepsize;
        stats.costs(1, iter+1) = xCurCost;
        if existsAssumedoptX
            stats.distToAssumedOptX(1, iter+1) = M.dist(xCur, options.assumedoptX);
        end
        stats.xHistory{iter+1} = xCur;
    end

end

function v = increasing(problem, M, xCur, xCurCost, p, a, b, c, gPg)
    normp = M.norm(xCur, p);
    t = b/normp;
    b = b/normp;
    a = a/normp;
    counter = 40;
    while(counter > 0)
        xNext = M.retr(xCur, p, t);
        v = problem.reallygrad(xNext);
        p_to_xNext = M.transp(xCur, xNext, p);
%         beta_tp = normp/M.norm(xNext, p_to_xNext);
        beta_tp = 1;
        if (M.inner(xCur, v, p_to_xNext)/beta_tp + c*gPg < 0)
            t = (a+b)/2;
            if h(b) > h(t)
                a = t;
            else
                b = t;
            end
        else
            break;
        end
        counter = counter - 1;
    end
    if counter == 0
        fprintf('h increasing bad! \n');
    end
    
    v = M.lincomb(xCur, 1/beta_tp, M.transp(xNext, xCur, v));
    
    function val = h(scale)
        val = getCost(problem, M.retr(xCur, p, scale)) - xCurCost + c*scale*gPg;
    end
end

function [g, p] = descent(problem, M, xCur, xCurCost, delta, c, epsilon, sHistory, yHistory, rhoHistory, scaleFactor, k)
    tol = 1e-6;
    vecsetNumMax = 50;
    vecsetNum = 1;
    vecset = cell(1, vecsetNumMax);
    vecsetP = cell(1, vecsetNumMax);
    grammat = zeros(vecsetNumMax, vecsetNumMax);
    u = M.randvec(xCur);
    u = M.lincomb(xCur, epsilon/(2*M.norm(xCur, u)), u);
    xNext = M.retr(xCur, u);
    v = problem.reallygrad(xNext);
    vecset{vecsetNum} = M.transp(xNext, xCur, v);
    vecsetP{vecsetNum} = getDirection(M, xCur, v, sHistory, yHistory, rhoHistory, scaleFactor, k);
    
    while(vecsetNum <= vecsetNumMax)
        %Update grammatrix
        for loop = 1: vecsetNum
            val = M.inner(xCur, vecset{loop}, vecsetP{vecsetNum});
            grammat(loop, vecsetNum) = val;
            grammat(vecsetNum, loop) = val;
        end
        
        N = vecsetNum;
        G = grammat(1:vecsetNum, 1:vecsetNum);
        opts = optimoptions('quadprog', 'Display', 'off', 'OptimalityTolerance', tol, 'ConstraintTolerance', tol);
        [s_opt, cost_opt] ...
            = quadprog(G, zeros(N, 1),     ...  % objective (squared norm)
            [], [],             ...  % inequalities (none)
            ones(1, N), 1,      ...  % equality (sum to 1)
            zeros(N, 1),        ...  % lower bounds (s_i >= 0)
            ones(N, 1),         ...  % upper bounds (s_i <= 1)
            [],                 ...  % we do not specify an initial guess
            opts);
        
        g_norm_P = real(sqrt(2*cost_opt));
        coeffs = s_opt;
        g = lincomb(M, xCur, vecset(1, 1:vecsetNum), coeffs);
        
        p = M.lincomb(xCur, -1, getDirection(M, xCur, g, sHistory, yHistory, rhoHistory, scaleFactor, k));
        
        if M.inner(xCur, g, g) < delta
            fprintf('less than delta\n');
            break;
        end
        
        normp = M.norm(xCur, p);
        if getCost(problem, M.retr(xCur, p, epsilon/normp)) - xCurCost <= -c*epsilon*(g_norm_P^2)/normp
            fprintf('vecsetNum is :%d\n', vecsetNum);
            break;
        end
        
        v = increasing(problem, M, xCur, xCurCost, p, 0, epsilon, c, g_norm_P^2);
        vecsetNum = vecsetNum + 1;
        vecset{vecsetNum} = v;
        vecsetP{vecsetNum} = getDirection(M, xCur, v, sHistory, yHistory, rhoHistory, scaleFactor, k);
    end
    
end

function dir = getDirection(M, xCur, xCurGradient, sHistory, yHistory, rhoHistory, scaleFactor, k)
    q = xCurGradient;
    inner_s_q = cell(1, k);
    for i = k : -1: 1
        inner_s_q{i} = rhoHistory{i}*M.inner(xCur, sHistory{i},q);
        q = M.lincomb(xCur, 1, q, -inner_s_q{i}, yHistory{i});
    end
    r = M.lincomb(xCur, scaleFactor, q);
    for i = 1: k
         omega = rhoHistory{i}*M.inner(xCur, yHistory{i}, r);
         r = M.lincomb(xCur, 1, r, inner_s_q{i}-omega, sHistory{i});
    end
%     dir = M.lincomb(xCur, -1, r);
    dir = r;
end


function [costNext, t, fail, lsiters] = linesearchnonsmooth(problem, M, xCur, d, f0, df0, c1, c2, max_counter)
%     df0 = M.inner(xCur, problem.reallygrad(xCur), d);
%     if df0 >=0
%         fprintf('LS failure by wrong direction');
%         t = 1;
%         fail = 1;
%         costNext = inf;
%         lsiters = -1;
%         return
%     end
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
    slope = M.inner(xNext, problem.reallygrad(xNext), diffretr);
%     slope = M.inner(xNext, getGradient(problem, xNext), diffretr);
end
