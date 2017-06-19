function bfgsManifold(problem)
    
    % Verify that the problem description is sufficient for the solver.
    if ~canGetCost(problem)
        warning('manopt:getCost', ...
            'No cost provided. The algorithm will likely abort.');
    end
    if ~canGetGradient(problem) && ~canGetApproxGradient(problem)
        % Note: we do not give a warning if an approximate gradient is
        % explicitly given in the problem description, as in that case the user
        % seems to be aware of the issue.
        warning('manopt:getGradient:approx', ...
            ['No gradient provided. Using an FD approximation instead (slow).\n' ...
            'It may be necessary to increase options.tolgradnorm.\n' ...
            'To disable this warning: warning(''off'', ''manopt:getGradient:approx'')']);
        problem.approxgrad = approxgradientFD(problem);
    end
    if ~canGetHessian(problem) && ~canGetApproxHessian(problem)
        % Note: we do not give a warning if an approximate Hessian is
        % explicitly given in the problem description, as in that case the user
        % seems to be aware of the issue.
        warning('manopt:getHessian:approx', ...
            ['No Hessian provided. Using an FD approximation instead.\n' ...
            'To disable this warning: warning(''off'', ''manopt:getHessian:approx'')']);
        problem.approxhess = approxhessianFD(problem);
    end
   
    
    %Parameter of convergence
    error = 1e-6;

    %Coefficients for Wolf condition and line search
    c1 = 0.0001;
    c2 = 0.9;
    amax = 1000;

    %Parameter of Hessian update
    memory = 10;
    
    %BFGS
    xCur = problem.M.rand(); %current point
    k = 0;
    sHistory = cell(1,memory); %represents x_k+1 - x_k at T_x_k+1
    yHistory = cell(1,memory); %represents df_k+1 - df_k
    xHistory = cell(1,memory); %represents x's.
    
    M = problem.M;
    
    while M.norm(xCur,getGradient(problem,xCur)) > error
        
            fprintf('\nNorm at start of iteration %d is %f\n', k, M.norm(xCur,getGradient(problem,xCur)));
            fprintf('Cost at start of iteration %d is %f\n', k, getCost(problem,xCur));
        
            %obtain the direction for line search
            if (k>=memory)
                negdir = direction(M, sHistory,yHistory,xHistory,...
                    xCur,getGradient(problem,xCur),memory);
%                 negdir = directiondummy(M, sHistory,yHistory,xHistory,...
%                     xCur,getGradient(problem,xCur),memory);
            else
                negdir = direction(M, sHistory,yHistory,xHistory,...
                    xCur,getGradient(problem,xCur),k);
%                 negdir = directiondummy(M, sHistory,yHistory,xHistory,...
%                     xCur,getGradient(problem,xCur),k);
            end
            p = M.mat(xCur, -M.vec(xCur,negdir));
            
            fprintf('Check if p is descent direction: %f\n',...
                M.inner(xCur,p,getGradient(problem,xCur)))
            
            %Get the stepsize (Default to 1)
            alpha = linesearch(problem,M,xCur,p,c1,c2,amax);
%            alpha = linesearchv2(problem,M,xCur,p);
            
            checkWolfe(problem,M,xCur,p,c1,c2,alpha);
            checkCurvatureCur(problem,M,xCur,alpha,p);
            
            %If step size is too small, there must be something wrong
%             if (alpha < 1e-10)
%                 fprintf('Step size is too small')
%                 return
%             end
            fprintf('alpha is %f \n', alpha);
            
            %Update
            xNext = M.retr(xCur,p,alpha); %!! CAN WE USE RETR HERE?
            sk = M.transp(xCur,xNext,M.mat(xCur,alpha*M.vec(xCur,p)));
            yk = M.mat(xNext, M.vec(xNext, getGradient(problem,xNext))...
                - M.vec(xNext,M.transp(xCur, xNext, getGradient(problem,xCur))));
            checkCurvatureNext(M,xNext,sk,yk);
            
            if (k>=20)
                sHistory = sHistory([2:end 1]); %the most recent vector is on the right
                sHistory{memory} = sk;
                yHistory = yHistory([2:end 1]); %the most recent vector is on the right
                yHistory{memory} = yk;
                xHistory = xHistory([2:end 1]); %the most recent vector is on the right
                xHistory{memory} = xCur;
                k = k+1;
            else
                k = k+1;
                sHistory{k} = sk;
                yHistory{k} = yk;
                xHistory{k} = xCur;
            end
            xCur = xNext;
            
            
    end
    
end

%Check if <sk,yk> > 0 at the current point
function checkCurvatureCur(problem,M,xCur,alpha,p)
    sk = M.mat(xCur,alpha*M.vec(xCur,p));
    xNext = M.retr(xCur,p,alpha);
    yk = M.vec(xCur,M.transp(xNext,xCur,getGradient(problem,xNext)))-...
        M.vec(xCur,getGradient(problem,xCur));
    yk = M.mat(xCur,yk);
    if (M.inner(xCur,sk,yk) < 0)
        fprintf('<sk,yk> is negative at xCur with val %f\n', M.inner(xCur,sk,yk));
    end
end

%Check if <sk,yk> > 0 at the next point
function checkCurvatureNext(M,xNext,sk,yk)
    if (M.inner(xNext,sk,yk) < 0)
        fprintf('<sk,yk> is negative at xNext with val %f\n', M.inner(xNext,sk,yk));
    end
end

%Check if Wolfe condition is satisfied.
function checkWolfe(problem,M,x,p,c1,c2,alpha)
    correct = 1;
    xnew = M.retr(x,p,alpha);
    if (getCost(problem,xnew)-getCost(problem,x))>...
            c1*alpha*M.inner(x,getGradient(problem,x),p)
        fprintf('Wolfe Cond 1:Armijo is violated\n')
        correct = 0;
    end
    if (abs(M.inner(xnew,M.transp(x,xnew,p),getGradient(problem,xnew))) >...
            -c2*M.inner(x,p,getGradient(problem,x)))
        correct = 0;
        fprintf('Wolfe Cond 2: flat gradient is violated\n')
        fprintf('     newgrad is %f\n',M.inner(xnew,M.transp(x,xnew,p),getGradient(problem,xnew)));
        fprintf('     oldgrad is %f\n',-c2*M.inner(x,p,getGradient(problem,x)));
    end
    if correct == 1
        fprintf('Wolfe is correct\n')
    end
end

%Iteratively it returns the search direction based on memory.
function dir = direction(M, sHistory,yHistory,xHistory,xCur,xgrad,iter)
    if (iter ~= 0)        
        sk = sHistory{iter};
        yk = yHistory{iter};
        xk = xHistory{iter};
        rouk = 1/(M.inner(xCur,sk,yk));
        %DEBUG
%         fprintf('Rouk is %f \n', rouk);
        tempvec = M.vec(xCur,xgrad) - rouk*M.inner(xCur,sk,xgrad)*M.vec(xCur,yk);
        temp = M.mat(xCur,tempvec);
        %transport to the previous point.
        temp = M.transp(xCur,xk,temp);
        temp = direction(M, sHistory,yHistory,xHistory,xk,...
            temp,iter-1);
        %transport the vector back
        temp = M.transp(xk,xCur,temp);
        tempvec = M.vec(xCur,temp) - rouk*M.inner(xCur,yk,temp)*M.vec(xCur,sk);
        tempvec = tempvec + rouk*M.inner(xCur,sk,xgrad)*M.vec(xCur,sk);
        dir = M.mat(xCur, tempvec);
    else
        dir = xgrad;
    end
end

%This version follows Qi et al, 2010
function alpha = linesearchv2(problem, M, x, p)
    %For bedugging. Shows phi(alpha)
%     n = 1000;
%     steps = linspace(-10,10,n);
%     costs = zeros(1,n);
%     for i = 1:n
%         costs(1,i) = getCost(problem,M.retr(x,p,steps(i)));
%     end
%     figure
%     plot(steps,costs);
%     xlabel('x')

    alpha = 1;
    c = M.inner(x,getGradient(problem,x),p);
    while (getCost(problem,M.retr(x,p,2*alpha))-getCost(problem,x) < alpha*c)
        alpha = 2*alpha;
    end
    while (getCost(problem,M.retr(x,p,alpha))-getCost(problem,x) >= 0.5*alpha*c)
        alpha = 0.5 * alpha;
    end
end


%This part follows Nocedal p59-60 for strong Wolfe conditions.
function alpha = linesearch(problem,M,x,p,c1,c2,amax)
    %For bedugging. Shows phi(alpha)
%     n = 1000;
%     steps = linspace(-10,10,n);
%     costs = zeros(1,n);
%     for i = 1:n
%         costs(1,i) = getCost(problem,M.retr(x,p,steps(i)));
%     end
%     figure
%     plot(steps,costs);
%     xlabel('x')

    aprev = 0;
    acur = 1;
    i = 1;
    gradAtZero = M.inner(x,getGradient(problem,x),p);
    while acur < amax
        xCur = M.retr(x,p,acur);
        if (getCost(problem,xCur)>getCost(problem,x)+c1*acur*gradAtZero)||...
                (problem.cost(xCur)>=getCost(problem,M.retr(x,p,aprev)) && i>1)
            alpha = zoom(problem,M,aprev,acur,x,p,c1,c2);
            return;
        end
        %MAYBE EXP is needed?
        gradAtCur = M.inner(xCur,getGradient(problem,xCur),M.transp(x,xCur,p));
        if (abs(gradAtCur) <= -c2*gradAtZero)
            alpha = acur;
            return;
        end
        if gradAtCur >= 0
            alpha = zoom(problem,M,acur,aprev,x,p,c1,c2);
            return;
        end
        aprev = acur;
        acur = acur * 2;
        i = i+1;
    end
    alpha = amax; %Not sure if this is right.
end

function alpha = zoom(problem,M,alo,ahi,x,p,c1,c2)
    costAtZero = getCost(problem,x);
    gradAtZero = M.inner(x,getGradient(problem,x),p);
    while abs(alo-ahi) > 1e-10
        anew = (alo+ahi)/2;
        costAtAnew = getCost(problem,M.retr(x,p,anew));
        costAtAlo = getCost(problem,M.retr(x,p,alo));
        if (costAtAnew > costAtZero +c1*anew*gradAtZero) || (costAtAnew >= costAtAlo)
            ahi = anew;
        else    
            xNew = M.retr(x,p,anew);
            gradAtAnew = M.inner(xNew,getGradient(problem,xNew),M.transp(x,xNew,p));
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
    alpha = (alo+ahi)/2;
end
