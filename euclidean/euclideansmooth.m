function euclideansmooth (cost,grad,dim)
    
    %Parameter of convergence
    error = 1e-6;

    %Coefficients for Wolf condition and line search
    c1 = 0.0001;
    c2 = 0.9;

    %Parameter of Hessian update
    memory = 10;
    
    %BFGS
    x = randn(dim,1);
    k = 0;
    s = zeros(dim, memory); %represents x_k+1 - x_k
    y = zeros(dim, memory); %represents df_k+1 - df_k
    disp('here')
    iter = 1;
    hist(1) = norm(grad(x));
    while norm(grad(x)) > error
            %obtain the direction for line search
            p = -direction(s,y,grad(x),memory);

            %Get the stepsize (Default to 1)
            alpha = linesearchStrongWolfe(cost,grad,x,p,c1,c2);
            
            %Update
            sk = alpha * p;
            yk = grad(x+alpha*p) - grad(x);
            s = circshift(s,-1,2); %the most recent vector is on the right
            s(:,memory) = sk;
            y = circshift(y,-1,2);
            y(:,memory) = yk;
            x = x + alpha * p;
            
            k = k+1;
            
            disp(x)
            
            fprintf('Norm at iteration %d is %f\n', k, norm(grad(x)));
            iter = iter + 1;
            hist(iter) = norm(grad(x));
    end
    disp (cost(x))
    figure;
    semilogy(hist, '.-');
    xlabel('Iteration number - BFGS');
    ylabel('Norm of the gradient of f');
end

%Iteratively it returns the search direction based on memory.
function dir = direction(s,y,gradfk,iter)
    if (iter ~= 0)        
        sk = s(:,iter);
        yk = y(:,iter);
        %Arbitrary cutoff to avoid huge error and useless steps.
        if (abs(sk'*yk) < 1e-11)
            dir = gradfk;
            return;
        end
        rouk = 1/(sk'*yk);
        temp = gradfk - rouk*(sk'*gradfk)*yk;
        temp = direction(s,y,temp,iter-1);
        temp = temp - rouk*(yk'*temp)*sk;
        dir = temp + rouk*(sk'*gradfk)*sk;
    else
        dir = gradfk;
    end
end

%This part follows Nocedal p59-60 for strong Wolfe conditions.
function alpha = linesearchStrongWolfe(cost,grad,x,p,c1,c2)
    amax = 1000; %(trial)
    aprev = 0;
    acur = 1;
    i = 1;
    gradAtZero = grad(x)'*p;
    while acur < amax
        if (cost(x+acur*p)>cost(x)+c1*acur*(gradAtZero))||...
                (cost(x+acur*p)>=cost(x+aprev*p) && i>1)
            alpha = zoom(aprev,acur,cost,grad,x,p,c1,c2);
            return;
        end
        gradAtCur = grad(x+acur*p)'*p;
        if (abs(gradAtCur) <= -c2*gradAtZero)
            alpha = acur;
            return;
        end
        if gradAtCur >= 0
            alpha = zoom(acur,aprev,cost,grad,x,p,c1,c2);
            return;
        end
        aprev = acur;
        acur = acur * 2;
        i = i+1;
    end
    alpha = amax; %Not sure if this is right.
end

function alpha = zoom(alo,ahi,cost,grad,x,p,c1,c2)
    while abs(alo-ahi) > 1e-10
        anew = (alo+ahi)/2;
        costAtAnew = cost(x+anew*p);
        costAtZero = cost(x);
        costAtAlo = cost(x+alo*p);
        gradAtZero = grad(x)'*p;
        if (costAtAnew > costAtZero +c1*anew*gradAtZero) || (costAtAnew >= costAtAlo)
            ahi = anew;
        else    
            gradAtAnew = grad(x+anew*p)'*p;
            if abs(gradAtAnew) <= -c2*gradAtZero
                alpha = anew;
                return
            end
            if gradAtAnew*(ahi-alo) >= 0 
                ahi = alo;
            end
            alo = anew;
        end
    end
    alpha = alo;
end





