function B
    a = -rand(10)
%     [q,r] = qr(a);
%     diag(r)
%     q = q * diag(sign(diag(r)));
%     r = diag(sign(diag(r))) * r;
%     a - q*r

    L = tril(a, -1)
    Lt = -L.'
    a - (L+Lt)
end