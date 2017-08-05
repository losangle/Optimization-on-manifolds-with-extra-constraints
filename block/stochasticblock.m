%To Do: make sure memory is in (0, Inf)
%Add time tic
%Stepsize change
%tune scaleFactor

function [x, cost, info, options] = stochasticblock(problem, x0, options)
    %ToAdd: check can get partialgradient. Check if initial point is
    %given. Check can get gradient, get cost.
    localdefaults.maxepoch = 100;
    localdefaults.batchsize = floor(sqrt(problem.ncostterm));
    localdefaults.stepsize = 0.1;
    localdefaults.memory = 30;
    localdefaults.storedepth = 30;
    localdefaults.tolgradnorm = 1e-6;
    localdefaults.verbosity = 2;
    localdefaults.blocksize = 3;
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    M.problem.M;
    
    batchsize = options.batchsize;
    maxinneriter = floor(problem.ncostterm / batchsize);
    
    
    
    storedb = StoreDB(options.storedepth);
    key = storedb.getNewKey();
    
    epoch = 0;
    k = 0;
    sHistory = cell(1, options.memory);
    yHistory = cell(1, options.memory);
    deltaHistory = cell(1, options.memory);
    scaleFactor = 1;
    
    [xCurCost, xCurGradient] = getCostGrad(problem, xCur, storedb, key);
    xCurGradNorm = M.norm(xCur, xCurGradient);
    stats = savestats();
    info(1) = stats;
    info(min(10000, options.maxiter+1)).iter = [];
    
    if options.verbosity >= 2
        fprintf(' iter\t               cost val\t                 grad. norm\t        alpha \n');
    end
    
    perm_idx = linspace(1, options.ncostterm, options.ncostterm);
    
    for epoch = 1: options.maxepoch
        perm_idx = perm_idx(randperm(options.ncostterm));
        num_elt_in_basket = 0;
        
        w = xCur;
        w_pivot = w;
        fullgrad_at_w_pivot = xCurGradient;
        sk = cell(1, options.maxinneriter);
        yk = cell(1, options.maxinneriter);
        wk = cell(1, options.maxinneriter);
        
        for inneriter = 1: maxinneriter
            start_index = (inneriter - 1) * batchsize + 1;
            end_index = batchsize* inneriter;
            idx_batch = perm_idx(start_index: end_index);
            
            partialgrad = getPartialGradient(problem, w, idx_batch);
            
            partialgrad0 = getPartialGradient(problem, xCur, idx_batch);
            
            partialgrad0_at_w_pivot = M.transp(xCur, w_pivot, partialgrad0);
            
            partialgrad_at_w_pivot = M.transp(w, w_pivot, partialgrad);
            
            correctedgrad = M.lincomb(w_pivot, 1, partialgrad_at_w_pivot, -1, partialgrad0_at_w_pivot);
            correctedgrad = M.lincomb(w_pivot, 1, correctedgrad, 1, fullgrad_at_w_pivot);
            
            dir = getDirection(M, w_pivot, correctedgrad, sHistory, yHistory, deltaHistory, scaleFactor, k, options);
            dir = M.transp(w_pivot, w, dir);
            
            wNext = M.retr(w, dir, -options.stepsize);
            num_elt_in_basket = num_elt_in_basket + 1;
            sk{num_elt_in_basket} = dir;
            wk{num_elt_in_basket} = w;
            
            
            w = wNext;
            key = storedb.getNewKey();
            
            if num_elt_in_basket >= options.blocksize || inneriter == maxinneriter
                
                % Create a new block
                for num_elt = 1: num_elt_in_basket
                    sk{num_elt} = M.transp(wk{num_elt}, w, sk{num_elt});
                end
                
                shrinkage = 1e-3;
                for num = 1: num_elt_in_basket
                    samplepoint1 = M.retr(w, sk{num}, shrinkage);
                    sampleGradient1 = getPartialGradient(problem, samplepoint1, idx_batch);
                    samplepoint2 = M.retr(w, sk{num}, -shrinkage);
                    sampleGradient2 = getPartialGradient(problem, samplepoint2, idx_batch);
                    yk{num} = M.lincomb(w, 1/(shrinkage*2), M.transp(samplepoint1, w, sampleGradient1),...
                        -1/(2*shrinkage), M.transp(samplepoint2, w, sampleGradient2));
                end
                
                deltak = zeros(num_elt_in_basket, num_elt_in_basket);
                indices = zeros(1, num_elt_in_basket);
                targetsize = 1;
                for numsample = 1: num_elt_in_basket
                    indices(1, targetsize) = numsample;
                    for col = 1: targetsize
                        deltak(targetsize, col) = M.inner(xCur, sk{indices(1, col)}, yk{numsample});
                        deltak(col, targetsize) = deltak(targetsize, col);
                    end
                    smallesteig = min(eig(deltak(1:targetsize, 1:targetsize)));
                    if smallesteig > 0  && cond(deltak(1:targetsize, 1:targetsize)) < 10^5
                        if numsample ~= num_elt_in_basket
                            targetsize = targetsize + 1;
                        end
                    else
                        if numsample == num_elt_in_basket
                            targetsize = targetsize -1;
                        end
                    end
                end
                fprintf('targetsize = %d  \n', targetsize);
                indices = indices(1, 1: targetsize);
                sk = sk(1, indices);
                yk = yk(1, indices);
                
                deltak = inv(deltak(1:targetsize, 1:targetsize));
                
                %Update block Histories
                if (k>= options.memory)
                    for  i = 2:options.memory
                        for num = 1:length(sHistory{i})
                            sHistory{i}{num} = M.transp(w_pivot, w, sHistory{i}{num});
                            yHistory{i}{num} = M.transp(w_pivot, w, yHistory{i}{num});
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
                            sHistory{i}{num} = M.transp(w_pivot, w, sHistory{i}{num});
                            yHistory{i}{num} = M.transp(w_pivot, w, yHistory{i}{num});
                        end
                    end
                    sHistory{k+1} = sk;
                    yHistory{k+1} = yk;
                    deltaHistory{k+1} = deltak;
                end
                k = k+1;
                
                w_pivot = w;
                
                num_elt_in_basket = 0;
                
                fullgrad_at_w_pivot = M.transp(xCur, w_pivot, xCurGradient);
            end
            
        end
        
        xCur = w;
        key = storedb.getNewKey();
        [xCurCost, xCurGradient] = getCostGrad(problem, xCur, storedb, key);
        xCurGradNorm = M.norm(xCur, xCurGradient);
        
        
        
    end
   
    
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