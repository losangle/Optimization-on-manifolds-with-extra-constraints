function xnew = retrProduct(Ms, xs, dirs, t)
    l = length(Ms);
    xnew = cell(1, l);
    for iter = 1 : l
        M = Ms{iter};
        xnew{iter} = M.retr(xs{iter}, dirs{iter}, t);
    end
end