function [x, cost, info, options] = blockbfgs(problem, x0, options)

% Local defaults for the program
localdefaults.minstepsize = 1e-10;
localdefaults.maxiter = 1000;
localdefaults.tolgradnorm = 1e-6;
localdefaults.memory = 30;
localdefaults.strict_inc_func = @(x) x;
localdefaults.ls_max_steps  = 25;
localdefaults.storedepth = 30;
localdefaults.linesearch = @linesearch_hint;
localdefaults.maxinneriter = 1;

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
% deltaHistory
deltaHistory = cell(1, options.memory);
% Scaling of direction given by getDirection for acceptable step
alpha = 1;
% Scaling of initial matrix, Barzilai-Borwein.
scaleFactor = 1;
% Norm of the step
stepsize = 1;
% Boolean for whether the step is accepted by Cautious update check
accepted = 1;
stop = 0;

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

while ~stop
    %------------------------ROUTINE----------------------------
    
    % Start timing this iteration
    timetic = tic();
    
    sk = cell(1, options.maxinneriter);
    yk = cell(1, options.maxinneriter);
    %         xk = cell(1, options.maxinneriter);
    
    xPrev = xCur;
    for inneriter = 1:options.maxinneriter
        
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
                    workingspace(problem, M, xCur, p, xCurCost, M.inner(xCur, xCurGradient, p), options, scaleFactor, storedb, key);
                end
                
%                 stop = true;
%                 reason = sprintf(['Last stepsize smaller than minimum '  ...
%                     'allowed; options.minstepsize = %g.'], ...
%                     options.minstepsize);
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
        
        % Display iteration information
        if options.verbosity >= 2
            %_______Print Information and stop information________
            fprintf('%5d\t%+.16e\t%.8e\t %.4e\n', iter, xCurCost, xCurGradNorm, alpha);
        end
        
        %--------------------Get Direction-----------------------
%         xCurGradientAtPrev = M.transp(xCur, xPrev, xCurGradient);
        p = getDirection(M, xCur, xCurGradient, sHistory,...
            yHistory, deltaHistory, scaleFactor, min(k, options.memory), options);
%         p = M.transp(xPrev, xCur, p);
        %--------------------Line Search--------------------------
%         workingspace(problem, M, xCur, p, xCurCost, M.inner(xCur, xCurGradient, p), options, scaleFactor, storedb, key);
        
%         [stepsize, xNext, newkey, lsstats] = ...
%             linesearch_hint(problem, xCur, p, xCurCost, M.inner(xCur,xCurGradient,p), options, storedb, key);
%         
%         alpha = stepsize/M.norm(xCur, p);
%         step = M.lincomb(xCur, alpha, p);
%         sk{inneriter} = step;
%         
        [xNext, costNext, alpha, fail, lsiters] = linesearchnonsmooth(problem, M, xCur, p, xCurCost, M.inner(xCur,xCurGradient,p), 0.001, 0.6, 40);
        step = M.lincomb(xCur, alpha, p);
        stepsize = M.norm(xCur, step);
        sk{inneriter} = step;
        newkey = storedb.getNewKey();
%         
        
        for num = 1:inneriter
            sk{num} = M.transp(xCur, xNext, sk{num}); %maybe transp together will be better, but should not matter as sampling only.
        end
        
        %Transport matrix
        if (k>= options.memory)
            for  i = 1:options.memory
                for num = 1:options.maxinneriter
                    sHistory{i}{num} = M.transp(xCur, xNext, sHistory{i}{num});
                    yHistory{i}{num} = M.transp(xCur, xNext, yHistory{i}{num});
                end
            end
        else
            for  i = 1:k
                for num = 1:options.maxinneriter
                    sHistory{i}{num} = M.transp(xCur, xNext, sHistory{i}{num});
                    yHistory{i}{num} = M.transp(xCur, xNext, yHistory{i}{num});
                end
            end
        end     
        
        xCur = xNext;
        [xCurCost, xCurGradient] = getCostGrad(problem, xCur, storedb, newkey);
        
        iter = iter + 1;
        key = newkey;
        xCurGradNorm = M.norm(xCur, xCurGradient);
        
        % Make sure we don't use too much memory for the store database
        storedb.purge();
        
        % Log statistics for freshly executed iteration
        stats = savestats();
        info(iter+1) = stats;
                
    end
    
    %----------------Updating the next iteration---------------
    if ~stop
        %Get GD, equivalently difference in yks
        shrinkage = 1;
        for num = 1:options.maxinneriter
            samplepoint1 = M.retr(xCur, sk{num}, shrinkage);
            sampleGradient1 = getGradient(problem, samplepoint1);
            samplepoint2 = M.retr(xCur, sk{num}, -shrinkage);
            sampleGradient2 = getGradient(problem, samplepoint2);
            yk{num} = M.lincomb(xCur, 1/(shrinkage*2), M.transp(samplepoint1, xCur, sampleGradient1),...
                -1/(2*shrinkage), M.transp(samplepoint2, xCur, sampleGradient2));
%             yk{num} = M.lincomb(xCur, 1/(shrinkage), M.transp(samplepoint1, xCur, sampleGradient1),...
%                 -1/(shrinkage), xCurGradient);
        end

        deltak = zeros(options.maxinneriter, options.maxinneriter);
        for row = 1: options.maxinneriter
            for col = row: options.maxinneriter
                deltak(row, col) = M.inner(xCur, sk{row}, yk{col});
                deltak(col, row) = deltak(row, col);
            end
        end
        %Probably has to be done in this case
%                 [V,D,W] = eig(deltak);
%         diag_of_D = diag(D);
%         diag_of_D = max(1, diag_of_D);
%         deltak = V * diag(diag_of_D) * W;
        
        
%         smallesteig = eigs(deltak, 1, 'sm');
        if norm(deltak) > 0
            smallesteig = min(eig(deltak));
            if smallesteig < 0
                deltak = deltak - eye(options.maxinneriter) * (smallesteig -1);
            end
            deltak = inv(deltak);
        else
            deltak = eye(options.maxinneriter);
        end
        scaleFactor = 1;
%         scaleFactor = abs(M.inner(xCur, sk{options.maxinneriter}, yk{options.maxinneriter})...
%             / M.inner(xNext, yk{options.maxinneriter}, yk{options.maxinneriter}));
        
        if (k>= options.memory)
            if options.memory > 1
                sHistory = sHistory([2:end, 1]);
                yHistory = yHistory([2:end, 1]);
                deltaHistory = deltaHistory([2:end, 1]);
            end
            if options.memory > 0
                sHistory{options.memory} = sk;
                yHistory{options.memory} = yk;
                deltaHistory{options.memory} = deltak;
            end
        else
            sHistory{k+1} = sk;
            yHistory{k+1} = yk;
            deltaHistory{k+1} = deltak;
        end
        k = k+1;
        
        
    end

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
function dir = getDirection(M, xCur, xCurGradient, sHistory, yHistory, deltaHistory, scaleFactor, k, options)
    q = xCurGradient;
    inner_s_q = zeros(options.maxinneriter, k);
    for i = k : -1 : 1
        for num = 1: options.maxinneriter
            inner_s_q(num, i) = M.inner(xCur, sHistory{i}{num}, q);
        end
        inner_s_q(:, i) = deltaHistory{i}*inner_s_q(:, i);
        yblend = lincomb(M, xCur, yHistory{i}, inner_s_q(:,i)); 
        q = M.lincomb(xCur, 1, q, -1, yblend);
    end
    r = M.lincomb(xCur, scaleFactor, q); %Need to be tuned
    omega = zeros(options.maxinneriter, 1);
    for i = 1 : k
        for num = 1: options.maxinneriter
            omega(num, 1) = M.inner(xCur, yHistory{i}{num}, r);
        end
        omega(:, 1) = deltaHistory{i} * omega(:, 1);
        sblend = lincomb(M, xCur, sHistory{i}, inner_s_q(:, i) - omega);
        r = M.lincomb(xCur, 1, r, 1, sblend);
    end
    dir = M.lincomb(xCur, -1, r);
end

function workingspace(problem, M, xCur, p, xCurCost, df0, options, scaleFactor, storedb, key)
    yo = 1;
end

function [xNext, costNext, t, fail, lsiters] = linesearchnonsmooth(problem, M, xCur, d, f0, df0, c1, c2, max_counter)
%    df0 = M.inner(xCur, problem.reallygrad(xCur), d);
%     if M.inner(xCur, problem.reallygrad(xCur), d) >=0
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
%     slope = M.inner(xNext, problem.reallygrad(xNext), diffretr);
    slope = M.inner(xNext, getGradient(problem, xNext), diffretr);
end
