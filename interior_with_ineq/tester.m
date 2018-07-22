M1 = euclideanfactory(2);
M2 = euclideanfactory(5);
x1 = M1.rand();
x2 = M2.rand();
v1 = M1.randvec(x1);
v2 = M2.randvec(x2);
w1 = M1.randvec(x1);
w2 = M2.randvec(x2);

M = cell(1, 2);
x = cell(1, 2);
v = cell(1, 2);
w = cell(1, 2);

M{1} = M1;
M{2} = M2;
x{1} = x1;
x{2} = x2;
v{1} = v1;
v{2} = v2;
w{1} = w1;
w{2} = w2;

newvec = lincombProduct(M, x, 1, v, 1, w);

newvec1 = M1.lincomb(x1, 1, v1, 1, w1);
newvec2 = M2.lincomb(x2, 1, v2, 1, w2);

what = 1;

