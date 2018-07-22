function plotperfprof
    

    % BC 
    ftol = 1.02; % the tolerance ratio of function value
    constrtol = 5e-4; % Max violation of constraint.

    filenames = ["zz_BC_Dim50.dat", "zz_BC_Dim200.dat",...
        "zz_BC_Dim500.dat", "zz_BC_Dim1000.dat", ...
        "zz_BC_Dim2000.dat", "zz_BC_Dim5000.dat"];
    filenames = ["zz_BC_Dim50MSe7.dat", "zz_BC_Dim200MSe7.dat",...
        "zz_BC_Dim500MSe7.dat", "zz_BC_Dim1000MSe7.dat", ...
        "zz_BC_Dim2000MSe7.dat", "zz_BC_Dim5000MSe7.dat"];
    titlenames = ["Dimension 50", "Dimension 200",...
        "Dimension 500", "Dimension 1000",...
        "Dimension 2000", "Dimension 5000"];
    locs = {'southeast','southeast','southeast','southeast','southeast','northwest'};
    
    
    
    %NNPCA
    filenames = ["zz_NNPCA_Dim10.dat", "zz_NNPCA_Dim50.dat",...
        "zz_NNPCA_Dim200.dat", "zz_NNPCA_Dim500.dat", ...
        "zz_NNPCA_Dim1000.dat", "zz_NNPCA_Dim2000.dat"];
    titlenames = ["Dimension 10", "Dimension 50", "Dimension 200",...
        "Dimension 500", "Dimension 1000",...
        "Dimension 2000"];
    locs = {'southeast','southeast','southeast','southeast','southeast','southeast'};
    
    
    
    startingsolver = [1 2 2 2 2 2];
    fig = figure;
    
    for plotnum = 1:6
    
        filename = filenames{plotnum};

        data = csvread(filename);
        [nrow, ~] = size(data);

        T = zeros(nrow, 5);
        for ii = 1 : nrow
            extable = data(ii, 1 : 15);
            extable = extable';
            extable = reshape(extable, [3, 5]);
            extable(extable == 0) = eps;
            [T(ii, :), ~,~,~] = timeplotprof(extable, ftol, constrtol);
        end

        subplot(3,2,plotnum) 
        perf(T(:,startingsolver(plotnum):5), 1, plotnum);
    end
    
    saveas(gcf,'plottry','epsc');
    
    function perf(T, logplot, plotnum)
        if (nargin< 2) 
            logplot = 0; 
        end
        
        colors = ['m' 'b' 'r' 'g' 'r' 'k' 'y'];
        co = [0 0 1;
              0 0.5 0;
              1 0 0;
              0 0.75 0.75;
              0.75 0 0.75;
              0.75 0.75 0;
              0.25 0.25 0.25];
        
        lines = {'--' '-' '-.' '-' '--'};
       
        markers = [' ' ' ' ' ' ' ' ' ' '.' '.'];
       
        [np,ns] = size(T);
        minperf = min(T, [], 2);
        r = zeros(np, ns);
        for p = 1: np
            r(p,:) = T(p,:)/minperf(p);
        end
        if (logplot) 
            r = log2(r); 
        end
        
        disp(r)

        max_ratio = max(max(r));
        r(find(isnan(r))) = 2*(max_ratio);
        r = sort(r);
        
        disp(r)
        
        if ns == 5
            r = circshift(r, [0,-1]);
        end
        if ns == 5
            for s = 1: ns
                [xs, ys] = stairs(r(:,s), [1 : np]/np);
                plot(xs, ys, 'LineStyle', lines{s});
                hold on;
            end
        else
            for s = 1: ns
                [xs, ys] = stairs(r(:,s), [1 : np]/np);
                plot(xs, ys, 'LineStyle', lines{s});
                hold on;
            end            
        end
        
            
        axis([ -0.1 1*(max_ratio)+0.01 0 1 ]);
        
        if ns == 5
            legend({'ALM','REPMS($Q^{lqh}$)', 'REPMS($Q^{lse}$)', 'fmincon', 'REPMSD'},'Location',locs{plotnum},'Interpreter','latex');
        else
            legend({'ALM','REPMS($Q^{lqh}$)', 'REPMS($Q^{lse}$)', 'fmincon'},'Location',locs{plotnum},'Interpreter','latex');
        end
        ylabel('Performance Profile');
        xlabel('$$\log_2\tau$','Interpreter','latex');
        title(titlenames(plotnum));
    end

    %response = fig2plotly(fig, 'filename', 'matlab-basic-line');
    %plotly_url = response.url;

end
