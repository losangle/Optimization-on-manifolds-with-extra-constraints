function prodgrad = getProductGradient(problems, xs)
    l = length(problems);
    prodgrad = cell(1, l);
    for iter = 1: l
        prodgrad{iter} = getGradient(problems{iter}, xs{iter});
    end
end