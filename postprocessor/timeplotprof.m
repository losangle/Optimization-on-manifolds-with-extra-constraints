function [newdata, fail, feasibility, optimality] = timeplotprof(data, ftol, contol)
    [nf, ns] = size(data);
    rowmaxvio = 1;
    rowcost = 2;
    rowtime = 3;
    newdata = data(rowtime, :);
    feasibility = ones(1, ns);
    optimality = ones(1, ns); 
    fail = 0; %Indicator if none of the solvers solved the problem.
    
    mincost = Inf;
    % Filter through constraint violation
    for solver = 1: ns
        if isnan(data(rowmaxvio, solver)) || data(rowmaxvio, solver) > contol
            newdata(1, solver) = NaN;
            feasibility(1, solver) = 0;
            optimality(1, solver) = 0;
        else
            mincost = min(mincost, data(rowcost, solver));
        end
    end
    
    if isinf(mincost)
        fail = 1;
        return;
    end
    
    %disp(newdata)
    
    %Filter through cost optimality
    for solver = 1: ns
        if isnan(data(rowcost, solver)) || (data(rowcost, solver) / mincost) > ftol
            newdata(1, solver) = NaN;
            optimality(1, solver) = 0;
        end
    end   
end