function [x, cost, info, options] = blockbfgv1(problem, x0, options)

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
k = 0;
iter = 0;
sHistory = cell(1, options.memory);
yHistory = cell(1, options.memory);
deltaHistory = cell(1, options.memory);
alpha = 1;
scaleFactor = 1;
stepsize = 1;
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
                if ultimatum == 0
                    if (options.verbosity >= 2)
                        fprintf(['stepsize is too small, restart the bfgs procedure' ...
                            'with the current point\n']);
                    end
                    xPrev = xCur;
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
        
        % Display iteration information
        if options.verbosity >= 2
            %_______Print Information and stop information________
            fprintf('%5d\t%+.16e\t%.8e\t %.4e\n', iter, xCurCost, xCurGradNorm, alpha);
        end
        
        %--------------------Get Direction-----------------------
        xCurGradientAtPrev = M.transp(xCur, xPrev, xCurGradient);
        p = getDirection(M, xPrev, xCurGradientAtPrev, sHistory,...
            yHistory, deltaHistory, scaleFactor, min(k, options.memory), options);
        p = M.transp(xPrev, xCur, p);
        %--------------------Line Search--------------------------
%         workingspace(problem, M, xCur, p, xCurCost, M.inner(xCur, xCurGradient, p), options, scaleFactor, storedb, key);
        
        [stepsize, xNext, newkey, lsstats] = ...
            linesearch_hint(problem, xCur, p, xCurCost, M.inner(xCur, xCurGradient,p), options, storedb, key);
        
        alpha = stepsize/M.norm(xCur, p);
        step = M.lincomb(xCur, alpha, p);
        sk{inneriter} = step;
        
        for num = 1:inneriter
            sk{num} = M.transp(xCur, xNext, sk{num}); 
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
        shrinkage = 1e1;
        for num = 1:options.maxinneriter
            samplepoint1 = M.retr(xCur, sk{num}, shrinkage);
            sampleGradient1 = getGradient(problem, samplepoint1);
            samplepoint2 = M.retr(xCur, sk{num}, -shrinkage);
            sampleGradient2 = getGradient(problem, samplepoint2);
            yk{num} = M.lincomb(xCur, 1/(shrinkage*2), M.transp(samplepoint1, xCur, sampleGradient1),...
                -1/(2*shrinkage), M.transp(samplepoint2, xCur, sampleGradient2));
        end

        deltak = zeros(options.maxinneriter, options.maxinneriter);
        indices = zeros(1, options.maxinneriter);
        targetsize = 1;
        for numsample = 1: options.maxinneriter
            indices(1, targetsize) = numsample;
            for col = 1: targetsize
                deltak(targetsize, col) = M.inner(xCur, sk{indices(1, col)}, yk{numsample});
                deltak(col, targetsize) = deltak(targetsize, col);
            end
            smallesteig = min(eig(deltak(1:targetsize, 1:targetsize)));
                if smallesteig > 0  && cond(deltak(1:targetsize, 1:targetsize)) < 10^5
                    if numsample ~= options.maxinneriter
                        targetsize = targetsize + 1;
                    end
                else
                    if numsample == options.maxinneriter
                        targetsize = targetsize -1;
                    end
                end
        end
        fprintf('targetsize = %d  \n', targetsize);
        indices = indices(1, 1: targetsize);
        sk = sk(1, indices);
        yk = yk(1, indices);
%         
        scaleFactortemp = deltak(1,1);
        if scaleFactortemp > 0
            scaleFactor = scaleFactortemp;
        end
        fprintf('scalefactor = %.16e', scaleFactor);
        
%         smallesteig = eigs(deltak, 1, 'sm');
%         smallesteig = min(eig(deltak));
%         if smallesteig < 0
%             deltak = deltak - eye(options.maxinneriter) * (smallesteig -1);
%         end
        deltak = inv(deltak(1:targetsize, 1:targetsize));

%         scaleFactor = 1;
%         scaleFactortemp = M.inner(xCur, sk{options.maxinneriter}, yk{options.maxinneriter})...
%             / M.inner(xNext, yk{options.maxinneriter}, yk{options.maxinneriter});

        
            if (k>= options.memory)
                for  i = 2:options.memory
                    for num = 1:length(sHistory{i})
                        sHistory{i}{num} = M.transp(xPrev, xCur, sHistory{i}{num});
                        yHistory{i}{num} = M.transp(xPrev, xCur, yHistory{i}{num});
                    end
                end
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
                for  i = 1:k
                    for num = 1:length(sHistory{i})
                        sHistory{i}{num} = M.transp(xPrev, xCur, sHistory{i}{num});
                        yHistory{i}{num} = M.transp(xPrev, xCur, yHistory{i}{num});
                    end
                end
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

function dir = getDirection(M, xCur, xCurGradient, sHistory, yHistory, deltaHistory, scaleFactor, k, options)
    q = xCurGradient;
    inner_s_q = zeros(options.maxinneriter, k);
    for i = k : -1 : 1
        len_cell = length(sHistory{i});
        for num = 1: len_cell
            inner_s_q(num, i) = M.inner(xCur, sHistory{i}{num}, q);
        end
        inner_s_q(1: len_cell, i) = deltaHistory{i} * inner_s_q(1: len_cell, i);
        yblend = lincomb(M, xCur, yHistory{i}, inner_s_q(1: len_cell,i)); 
        q = M.lincomb(xCur, 1, q, -1, yblend);
    end
    r = M.lincomb(xCur, scaleFactor, q); %Need to be tuned
    omega = zeros(options.maxinneriter, 1);
    for i = 1 : k
        len_cell = length(sHistory{i});
        for num = 1: len_cell
            omega(num, 1) = M.inner(xCur, yHistory{i}{num}, r);
        end
        omega(1: len_cell, 1) = deltaHistory{i} * omega(1: len_cell, 1);
        sblend = lincomb(M, xCur, sHistory{i}, inner_s_q(1: len_cell, i) - omega(1: len_cell, 1));
        r = M.lincomb(xCur, 1, r, 1, sblend);
    end
    dir = M.lincomb(xCur, -1, r);
end

function workingspace(problem, M, xCur, p, xCurCost, df0, options, scaleFactor, storedb, key)
    yo = 1;
end