function [gradnorms, alphas, stepsizes, costs, xCur, time] = euclideannonsmoothlm(problem, x, options)
    
    timetic = tic();
    M = problem.M;

    if ~exist('x','var')|| isempty(x)
        xCur = x;
    else
        xCur = M.rand();
    end
    
    if ~exist('options','var')|| isempty(x)|| ~exist('options.memory','var')
        options.memory = 300;
    end

    xCurGradient = getGradient(problem, xCur);
    xCurGradNorm = M.norm(xCur, xCurGradient);
    xCurCost = getCost(problem, xCur);
    
    gradnorms = zeros(1,1000);
    gradnorms(1,1) = xCurGradNorm;
    alphas = zeros(1,1000);
    alphas(1,1) = 1;
    stepsizes = zeros(1,1000);
    stepsizes(1,1) = NaN;
    costs = zeros(1,1000);
    costs(1,1) = xCurCost;

    sHistory = cell(1, options.memory);
    yHistory = cell(1, options.memory);
    rhoHistory = cell(1, options.memory);
    
    k = 0;
    iter = 0;
    alpha = 1;
    scaleFactor = 1;
    stepsize = 1;

    fprintf(' iter\t               cost val\t    grad. norm\t   alpha \t  stepsize \n');

    while (1)
        %_______Print Information and stop information________
        fprintf('%5d\t%+.16e\t%.8e\t%f\t%e\n', iter, xCurCost, xCurGradNorm, alpha, stepsize);

        if (stepsize <= 1e-10)
            fprintf('Stepsize too small\n')
            break;
        end

        %_______Get Direction___________________________

        p = - getDirection(M, xCur, xCurGradient, sHistory, yHistory, rhoHistory, scaleFactor, min(options.memory,k));

        %_______Line Search____________________________
%         
        [xNext, xNextCost, alpha] = linesearchnonsmooth(problem, M, xCur, p, xCurCost, M.inner(xCur,xCurGradient,p));
        step = M.lincomb(xCur, alpha, p);
        stepsize = M.norm(xCur, step);
        
        %_______Updating the next iteration_______________
        xNextGradient = getGradient(problem, xNext);
        sk = step; 
        yk = xNextGradient - xCurGradient; 

        inner_sk_yk = M.inner(xNext, yk, sk);
        rhok = 1/inner_sk_yk;
        scaleFactor = inner_sk_yk / M.inner(xNext, yk, yk);
        if (k>= options.memory)
            sHistory = sHistory([2:end 1]);
            sHistory{options.memory} = sk;
            yHistory = yHistory([2:end 1]);
            yHistory{options.memory} = yk;
            rhoHistory = rhoHistory([2:end 1]);
            rhoHistory{options.memory} = rhok;
        else
            sHistory{k+1} = sk;
            yHistory{k+1} = yk;
            rhoHistory{k+1} = rhok;
        end
        k = k+1;
       
        %No need transport.
        
        iter = iter + 1;
        xCur = xNext;
        xCurGradient = xNextGradient;
        xCurGradNorm = M.norm(xCur, xNextGradient);
        xCurCost = xNextCost;
        
        gradnorms(1,iter+1)= xCurGradNorm;
        alphas(1,iter+1) = alpha;
        stepsizes(1, iter+1) = stepsize;
        costs(1,iter+1) = xCurCost;
    end
    
    gradnorms = gradnorms(1,1:iter+1);
    alphas = alphas(1,1:iter+1);
    stepsizes = stepsizes(1, 1:iter+1);
    costs = costs(1, 1:iter+1);
    time = toc(timetic);
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
         omega = rhoHistory{i}*M.inner(xCur, yHistory{i},r);
         r = M.lincomb(xCur, 1, r, inner_s_q{i}-omega, sHistory{i});
    end
    dir = M.lincomb(xCur, 1, r);
end
% 
% function dir = getDirection(M, xCur, curGrad, sHistory, yHistory, rhoHistory, scaleFactor, k)
%     if (k ~= 0)
%         sk = sHistory{k};
%         yk = yHistory{k};
%         rouk = rhoHistory{k};
%         temp = curGrad - rouk*(sk.'*curGrad)*yk;
%         temp = getDirection(M, xCur, temp, sHistory, yHistory, rhoHistory, scaleFactor, k-1);
%         temp = temp - rouk*(yk.'*temp)*sk;
%         dir = temp + rouk*(sk.'*curGrad)*sk;
%     else
%         dir = curGrad * scaleFactor;
%     end
% end


function [xNext, costNext, t] = linesearchnonsmooth(problem, M, xCur, d, f0, df0)
    alpha = 0;
    beta = inf;
    t = 1;
    c1 = 0.01; %need adjust
    c2 = 0.9; %need adjust.
    counter = 100;
    while counter > 0
        xNext = M.retr(xCur, d, t);
        if (getCost(problem, xNext) > f0 + df0*c1*t)
            beta = t;
        elseif M.inner(xNext, getGradient(problem, xNext), d) < c2*df0
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
    costNext = getCost(problem, xNext);
end
