function lincombGrad = lincombProduct(Ms, xs, t1, d1, t2, d2)
    l = length(Ms);
    lincombGrad = cell(1, l);
    for iter = 1 : l
        M = Ms{iter};
        lincombGrad{iter} = M.lincomb(xs{iter}, t1, d1{iter}, t2, d2{iter});
    end
end