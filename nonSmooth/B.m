function B
    M = spherefactory(10);
    x1 = M.rand();
    x2 = M.rand();
    v1 = M.randvec(x1);
    v2 = M.randvec(x1);
    c1 = M.inner(x1, v1, v2)
    u1 = M.isotransp(x1, x2, v1);
    u2 = M.isotransp(x1, x2, v2);
    c2 = M.inner(x2, u1, u2)
end

