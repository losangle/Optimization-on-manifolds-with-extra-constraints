function val = innerprodProduct(Ms, xs, g1, g2)
    l = length(Ms);
    val = 0;
    for iter = 1 : l
        M = Ms{iter};
        val = val + M.inner(xs{iter}, g1{iter}, g2{iter});
    end
end