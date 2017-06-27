function  [x, cost, info, options] = bfgsCautious(problem, x, options)

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
            'It may be necessary to increase options.tolgradnorm.\n' ...
            'To disable this warning: warning(''off'', ''manopt:getGradient:approx'')']);
        problem.approxgrad = approxgradientFD(problem);
    end
    
    % Set local defaults here
    localdefaults.minstepsize = 1e-10;
    localdefaults.maxiter = 1000;
    localdefaults.tolgradnorm = 1e-6;
    localdefaults.c1 = 0.0001;
    localdefaults.c2 = 0.9;
    localdefaults.amax = 1000;
    localdefaults.memory = 10;
    localdefaults.linesearchVersion = 1; %0 is Strong Wolfe. 1 is Armijo.
    localdefaults.restart = 1; %1 = TRUE;
    options.debug = 0;
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    timetic = tic();
    
    % If no initial point x is given by the user, generate one at random.
    if ~exist('x', 'var') || isempty(x)
        xCur = problem.M.rand();
    else
        xCur = x;
    end
    
    % Create a store database and get a key for the current x
    storedb = StoreDB(options.storedepth);
    key = storedb.getNewKey();
    
    % Compute objective-related quantities for x
    [cost, grad] = getCostGrad(problem, xCur, storedb, key);
    gradnorm = problem.M.norm(xCur, grad);
    
    % Iteration counter.
    % At any point, iter is the number of fully executed iterations so far.
    iter = 0;
    
    % Save stats in a struct array info, and preallocate.
    stats = savestats();
    info(1) = stats;
    info(min(10000, options.maxiter+1)).iter = [];
    
    if options.verbosity >= 2
        fprintf(' iter\t               cost val\t    grad. norm\n');
    end
    
    
    %BFGS initialization
    newgrad = grad;
    
    k = 0;
    sHistory = cell(1,options.memory); %represents x_k+1 - x_k at T_x_k+1
    yHistory = cell(1,options.memory); %represents df_k+1 - df_k
    RhoHistory = cell(1,options.memory); %represents x's.
    
    M = problem.M;
    M.retr = @M.exp;
    
    alpha = 1; %Initial Alpha to start with
    scaleFactor = 1; %Initial Scale Factor
    
    while true
%------------------------ROUTINE----------------------------

        % Display iteration information
        if options.verbosity >= 2
            fprintf('%5d\t%+.16e\t%.8e\n', iter, cost, gradnorm);
        end
        
        % Start timing this iteration
        timetic = tic();
        
        % Run standard stopping criterion checks
        [stop, reason] = stoppingcriterion(problem, xCur, options, ...
            info, iter+1);
        
        % If none triggered, run specific stopping criterion check
        if ~stop && stats.stepsize < options.minstepsize
            stop = true;
            reason = sprintf(['Last stepsize smaller than minimum '  ...
                'allowed; options.minstepsize = %g.'], ...
                options.minstepsize);
        end
        
        if stop
            if options.verbosity >= 1
                fprintf([reason '\n']);
            end
            break;
        end
        
%%%% -----------obtain the direction for line search----------------
        xCurGradient = newgrad;
        xCurGradientNorm = M.norm(xCur, xCurGradient);
        
        p = direction(M, sHistory,yHistory,RhoHistory,...
                xCur,xCurGradient,min(k,options.memory), scaleFactor);

        %DEBUG only
        %negdir = M.lincomb(xCur, -1 ,getGradient(problem, xCur));
        
        
%%%% -----------%Get the stepsize (Default to 1)----------------        
        InnerProd_p_xCurGradient = M.inner(xCur,p,xCurGradient);
        disp(InnerProd_p_xCurGradient/(M.norm(xCur,p)*M.norm(xCur,xCurGradient)));
        
        if options.linesearchVersion == 0
            alpha = linesearchWolfe(problem,M,xCur,p,options.c1,options.c2,options.amax);
            costNext = getCost(problem, M.retr(xCur, p ,alpha));
        else
            if options.restart == 1
                if (InnerProd_p_xCurGradient > 0)
                    k = 0;
                    fprintf('restart\n');
                    continue;
                end
            end
            [costNext,changeDir,alpha] = linesearchv2(problem,M,xCur,p,InnerProd_p_xCurGradient,alpha);
            %stop the algorithm as step can not be taken. 
            if changeDir == 1
                %Stop
                alpha = 0;
                %Restart
%                 k = 0;
%                 continue;
            end
        end
        newkey = storedb.getNewKey();
        lsstats = [];        
  

% Use of Original LineSearch
%         disp(InnerProd_p_xCurGradient)
%         [stepsize, xNext, newkey, lsstats] = linesearch(problem, xCur, M.lincomb(xCur, 1/M.norm(xCur,p), p), cost, InnerProd_p_xCurGradient);
%         newgrad = getGradient(problem,xNext);
%         step = M.lincomb(xCur, stepsize, p);
%         sk = M.transp(xCur,xNext, step);
%         beta = M.norm(xCur, step) / M.norm(xNext, sk);
%         sk = M.lincomb(xNext, beta, sk);
%         xCurGradient_TS_to_xNext = M.transp(xCur, xNext, xCurGradient);
%         isometricScale = xCurGradientNorm/ M.norm(xNext, xCurGradient_TS_to_xNext);
%         yk = M.lincomb(xNext, 1/beta, newgrad,...
%             -isometricScale, xCurGradient_TS_to_xNext);
%         costNext = getCost(problem, xNext);
        
%%%% -------------------------%Update------------------------  
        xNext = M.retr(xCur,p,alpha); %!! CAN WE USE RETR HERE?
        newgrad = getGradient(problem,xNext);
        sk = M.transp(xCur,xNext,M.lincomb(xCur, alpha, p));
        beta = M.norm(xCur, M.lincomb(xCur, alpha, p)) / M.norm(xNext, sk);
        fprintf('Beta = %f',beta);
        sk = M.lincomb(xNext, beta, sk);
        xCurGradient_TS_to_xNext = M.transp(xCur, xNext, xCurGradient);
        isometricScale = xCurGradientNorm/ M.norm(xNext, xCurGradient_TS_to_xNext);
        yk = M.lincomb(xNext, 1/beta, newgrad,...
            -isometricScale, xCurGradient_TS_to_xNext);
        
        
        %DEBUG only
        if options.debug == 1
            fprintf('alpha is %f \n', alpha);
            fprintf('Check if p is descent direction: %f\n',...
                sign(M.inner(xCur,p,getGradient(problem,xCur))))
            checkWolfe(problem,M,xCur,p,options.c1,options.c2,alpha);
            checkCurvatureCur(problem,M,xCur,alpha,p);
            checkCurvatureNext(M,xNext,sk,yk);
        end
        
        innerProduct_sk_yk = M.inner(xNext, sk, yk);
        if (innerProduct_sk_yk/M.inner(xNext,sk,sk) >= xCurGradientNorm)
            if (k>=options.memory)
                for i = 2: options.memory 
                    temp = M.transp(xCur,xNext,sHistory{i});
                    sHistory{i} = M.lincomb(xNext, M.norm(xCur, sHistory{i})/M.norm(xNext, temp), temp);
                    temp = M.transp(xCur,xNext,yHistory{i});
                    yHistory{i} = M.lincomb(xNext, M.norm(xCur, yHistory{i})/M.norm(xNext, temp), temp);
                end
                sHistory = sHistory([2:end 1]); %the most recent vector is on the right
                sHistory{options.memory} = sk;
                yHistory = yHistory([2:end 1]); %the most recent vector is on the right
                yHistory{options.memory} = yk;
                RhoHistory = RhoHistory([2:end 1]); %the most recent vector is on the right
                RhoHistory{options.memory} = 1/innerProduct_sk_yk;
                k = k+1;
            else
                for i = 1: k
                    temp = M.transp(xCur,xNext,sHistory{i});
                    sHistory{i} = M.lincomb(xNext, M.norm(xCur, sHistory{i})/M.norm(xNext, temp), temp);
                    temp = M.transp(xCur,xNext,yHistory{i});
                    yHistory{i} = M.lincomb(xNext, M.norm(xCur, yHistory{i})/M.norm(xNext, temp), temp);
                end
                sHistory{k+1} = sk;
                yHistory{k+1} = yk;
                RhoHistory{k+1} = 1/innerProduct_sk_yk;
                k = k+1;
            end
            scaleFactor = innerProduct_sk_yk/M.inner(xNext,yk,yk);
        else 
            for i = 1 : min(k,options.memory)
                    temp = M.transp(xCur,xNext,sHistory{i});
                    sHistory{i} = M.lincomb(xNext, M.norm(xCur, sHistory{i})/M.norm(xNext, temp), temp);
                    temp = M.transp(xCur,xNext,yHistory{i});
                    yHistory{i} = M.lincomb(xNext, M.norm(xCur, yHistory{i})/M.norm(xNext, temp), temp);   
            end 
            scaleFactor = scaleFactor; %Does not change
            k = k; % Does not increase.
        end
        % Compute the new cost-related quantities for x
        newgradnorm = problem.M.norm(xNext, newgrad);
        
        % Make sure we don't use too much memory for the store database
        storedb.purge();
        
        % Transfer iterate info        
        xCur = xNext;
        key = newkey;
        cost = costNext;
        grad = newgrad;
        gradnorm = newgradnorm;
        %TODO
        stepsize = M.norm(xCur,p)*alpha;
        
        % iter is the number of iterations we have accomplished.
        iter = iter + 1;
        
        % Log statistics for freshly executed iteration
        stats = savestats();
        info(iter+1) = stats; 
        
    end
    
    
    %Return the needed info.
    x = xCur;
    cost = getCost(problem,xCur);
    
    info = info(1:iter+1);

    if options.verbosity >= 1
        fprintf('Total time is %f [s] (excludes statsfun)\n', ...
                info(end).time);
    end

    % Routine in charge of collecting the current iteration stats
    function stats = savestats()
        stats.iter = iter;
        stats.cost = cost;
        stats.gradnorm = gradnorm;
        if iter == 0
            stats.stepsize = NaN;
            stats.time = toc(timetic);
            stats.linesearch = [];
        else
            stats.stepsize = stepsize;
            stats.time = info(iter).time + toc(timetic);
            stats.linesearch = lsstats;
        end
        stats = applyStatsfun(problem, xCur, storedb, key, options, stats);
    end
end

%Check if <sk,yk> > 0 at the current point
function checkCurvatureCur(problem,M,xCur,alpha,p)
    sk = M.lincomb(xCur, alpha, p);
    xNext = M.retr(xCur,p,alpha);
    yk = M.lincomb(xCur, 1, M.transp(xNext,xCur,getGradient(problem,xNext)), ...
        -1, getGradient(problem,xCur));
    if (M.inner(xCur,sk,yk) < 0)
        fprintf('<sk,yk> is negative at xCur with val %f\n', M.inner(xCur,sk,yk));
    end
end

%Check if <sk,yk> > 0 at the next point
function checkCurvatureNext(M,xNext,sk,yk)
    if (M.inner(xNext,sk,yk) < 0)
        fprintf('<sk,yk> is negative at xNext with val %f\n', M.inner(xNext,sk,yk));
    end
end

%Check if Wolfe condition is satisfied.
function correct = checkWolfe(problem,M,x,p,c1,c2,alpha)
    xnew = M.retr(x,p,alpha);
    costAtxNew = getCost(problem,xnew);
    costAtx = getCost(problem,x);
    gradientAtx = getGradient(problem,x);
    gradientAtxNew = getGradient(problem,xnew);
    correct = 1;
    if (costAtxNew-costAtx)>...
            c1*alpha*M.inner(x,gradientAtx,p)
        fprintf('Wolfe Cond 1:Armijo is violated\n')
        correct = 0;
    end
    if (abs(M.inner(xnew,M.transp(x,xnew,p),gradientAtxNew)) >...
            -c2*M.inner(x,p,gradientAtx))
        correct = 0;
        fprintf('Wolfe Cond 2: flat gradient is violated\n')
        fprintf('     newgrad is %f\n',M.inner(xnew,M.transp(x,xnew,p),gradientAtxNew));
        fprintf('     oldgrad is %f\n',-c2*M.inner(x,p,gradientAtx));
    end
    if correct == 1
        fprintf('Wolfe is correct\n')
    end
end

% TODO: unroll the function

%Iteratively it returns the search direction based on memory.
function dir = direction(M, sHistory, yHistory, rhoHistory, ...
                                xCur, xCurGrad, iter, scaleFactor)
    q = xCurGrad;
    epsTempList = cell(1,iter);
    for i = iter : -1 : 1
        epsTempList{i} = rhoHistory{i} * M.inner(xCur, sHistory{i}, q);
        q = M.lincomb(xCur, 1, q, -epsTempList{i}, yHistory{i});
    end
    r = M.lincomb(xCur, scaleFactor, q);
    for i = 1 : iter
        omega = rhoHistory{i} * M.inner(xCur, yHistory{i}, r);
        r = M.lincomb(xCur, 1, r, epsTempList{i} - omega, sHistory{i});
    end
    dir = M.lincomb(xCur, -1, r);
end

%This version follows Qi et al, 2010
function [costNext,changeDir,alpha] = linesearchv2(problem, M, x, p, pTGradAtx, alphaprev)
    %For bedugging. Shows phi(alpha)
%     n = 1000;
%     steps = linspace(-10,10,n);
%     costs = zeros(1,n);
%     for i = 1:n
%         costs(1,i) = getCost(problem,M.retr(x,p,steps(i)));
%     end
%     figure
%     plot(steps,costs);
%     xlabel('x')

    alpha = alphaprev;
    changeDir = 0;
    %DEBUG
%     fprintf('c = %.16e\n',c);   
    costAtx = getCost(problem,x);
    while (getCost(problem,M.exp(x,p,2*alpha))-costAtx < alpha*pTGradAtx)
        alpha = 2*alpha;
    end
    %DEBUG
%     fprintf('alpha after first is %.16e\n', alpha);
    % Exp can't be retraction here.
    costNext = getCost(problem,M.exp(x,p,alpha));
    diff = costNext - costAtx;
    while (diff>= 0.5*alpha*pTGradAtx)
        if (diff == 0)
            changeDir = 1;
            l = logspace(-15,1,500);
            break;
        end
        alpha = 0.5 * alpha;
        costNext = getCost(problem,M.exp(x,p,alpha));
        diff = costNext - costAtx;
%         %DEBUG
%         fprintf('alpha now is %.16e\n', alpha);
%         costNext = getCost(problem,M.exp(x,p,alpha));
%         fprintf('ERROR%.16e\n',costNext); 
%         fprintf('ERROR%.16e\n',costAtx); 
%         fprintf('ERROR%.16e\n',costNext-costAtx); 
%         fprintf('0.5*alpha*c has value  :   %.16e\n',0.5*alpha*c);
%         if (alpha <1e-12)
%             t = 1; % to make a breakpoint
%             disp('Search the other direction');
%             disp(newalpha);
%         end
    end
    fprintf('alpha = %.16e\n',alpha);    
end


%This part follows Nocedal p59-60 for strong Wolfe conditions.
function alpha = linesearchWolfe(problem,M,x,p,c1,c2,amax)
    %For bedugging. Shows phi(alpha)
%     n = 1000;
%     steps = linspace(-10,10,n);
%     costs = zeros(1,n);
%     for i = 1:n
%         costs(1,i) = getCost(problem,M.retr(x,p,steps(i)));
%     end
%     figure
%     plot(steps,costs);
%     xlabel('x')

    aprev = 0;
    acur = 1;
    i = 1;
    gradAtZero = M.inner(x,getGradient(problem,x),p);
    while acur < amax
        xCur = M.retr(x,p,acur);
        if (getCost(problem,xCur)>getCost(problem,x)+c1*acur*gradAtZero)||...
                (problem.cost(xCur)>=getCost(problem,M.retr(x,p,aprev)) && i>1)
            alpha = zoom(problem,M,aprev,acur,x,p,c1,c2);
            return;
        end
        %MAYBE EXP is needed?
        gradAtCur = M.inner(xCur,getGradient(problem,xCur),M.transp(x,xCur,p));
        if (abs(gradAtCur) <= -c2*gradAtZero)
            alpha = acur;
            return;
        end
        if gradAtCur >= 0
            alpha = zoom(problem,M,acur,aprev,x,p,c1,c2);
            return;
        end
        aprev = acur;
        acur = acur * 2;
        i = i+1;
    end
    alpha = amax; %Not sure if this is right.
end

function alpha = zoom(problem,M,alo,ahi,x,p,c1,c2)
    costAtZero = getCost(problem,x);
    gradAtZero = M.inner(x,getGradient(problem,x),p);
    while abs(alo-ahi) > 1e-10
        anew = (alo+ahi)/2;
        costAtAnew = getCost(problem,M.retr(x,p,anew));
        costAtAlo = getCost(problem,M.retr(x,p,alo));
        if (costAtAnew > costAtZero +c1*anew*gradAtZero) || (costAtAnew >= costAtAlo)
            ahi = anew;
        else    
            xNew = M.retr(x,p,anew);
            gradAtAnew = M.inner(xNew,getGradient(problem,xNew),M.transp(x,xNew,p));
            if abs(gradAtAnew) <= -c2*gradAtZero
                alpha = anew;
                return
            end
            if gradAtAnew*(ahi-alo) >= 0 
                ahi = alo;
            end
            alo = anew;
        end
    end
    alpha = (alo+ahi)/2;
end