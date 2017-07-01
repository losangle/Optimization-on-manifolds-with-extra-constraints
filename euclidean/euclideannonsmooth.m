function [gradnorms, alphas, stepsizes, costs, xCur, time] = euclideannonsmooth(problem, x, options)
    
    timetic = tic();
    M = problem.M;
    dim = M.dim();

    if ~exist('x','var')|| isempty(x)
        xCur = x;
    else
        xCur = M.rand();
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


    Hcore = eye(dim);
    Hside  = zeros(dim,dim);
    H = Hcore;
    
    k = 0;
    iter = 0;
    alpha = 1;
    scaleFactor = 1;
    stepsize = 1;

    fprintf(' iter\t               cost val\t    grad. norm\t   alpha \n');

    while (1)
        %_______Print Information and stop information________
        fprintf('%5d\t%+.16e\t%.8e\t%f\n', iter, xCurCost, xCurGradNorm, alpha);

        if (stepsize <= 1e-10)
            fprintf('Stepsize too small\n')
            break;
        end

        %_______Get Direction___________________________

        p = -H*xCurGradient;

        %_______Line Search____________________________
%         
        [xNext, xNextCost, alpha] = linesearchnonsmooth(problem, M, xCur, p, xCurCost, M.inner(xCur,xCurGradient,p));
        
        step = M.lincomb(xCur, alpha, p);
        stepsize = M.norm(xCur, step);
        
%         Normal LS
%         [stepsize, xNext, newkey, lsstats] =linesearch(problem, xCur, p, xCurCost, M.inner(xCur,xCurGradient,p));
%         xNextCost = getCost(problem, xNext);
%         alpha = stepsize/M.norm(xCur,p);
%         step = M.lincomb(xCur, alpha, p);
        
        %_______Updating the next iteration_______________
        xNextGradient = getGradient(problem, xNext);
        sk = step; 
        yk = xNextGradient - xCurGradient; 

        inner_sk_yk = (sk.'*yk);
        rhok = 1/inner_sk_yk;
        scaleFactor = inner_sk_yk / (yk.'*yk);
        
        Hcore = (eye(dim)-(rhok)*sk*(yk.'))...
            *Hcore* (eye(dim)-(rhok)*yk*(sk.'));
        Hside = (eye(dim)-(rhok)*sk*(yk.'))...
            *Hside* (eye(dim)-(rhok)*yk*(sk.'))+...
            rhok*(sk)*(sk.');
        H = Hcore*scaleFactor + Hside;
        
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
