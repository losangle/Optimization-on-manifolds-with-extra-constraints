function val = normProduct(Ms, xs, g)
    val = innerprodProduct(Ms, xs, g, g);
    val = sqrt(val);
end