function datacleaner
    formatSpec = '%f';
    for i = 1:17
        formatSpec = strcat(formatSpec, '%f');
    end
    formatSpec = strcat(formatSpec, '%C');
    
    A = readtable('xaa.dat','Delimiter', ',', ...
    'Format',formatSpec);
    A = cell2mat(table2cell(A(:, 1:18)));
    
    B = readtable('xab.dat','Delimiter', ',', ...
    'Format',formatSpec);
    A = [A; cell2mat(table2cell(B(:, 1:18)))];
    
    B = readtable('xac.dat','Delimiter', ',', ...
    'Format',formatSpec);
    A = [A; cell2mat(table2cell(B(:, 1:18)))];
    
    B = readtable('xad.dat','Delimiter', ',', ...
    'Format',formatSpec);
    A = [A; cell2mat(table2cell(B(:, 1:18)))];
    
    B = readtable('xae.dat','Delimiter', ',', ...
    'Format',formatSpec);
    A = [A; cell2mat(table2cell(B(:, 1:18)))];   
    
    B = readtable('xaf.dat','Delimiter', ',', ...
    'Format',formatSpec);
    A = [A; cell2mat(table2cell(B(:, 1:18)))];
    
    B = readtable('xag.dat','Delimiter', ',', ...
    'Format',formatSpec);
    A = [A; cell2mat(table2cell(B(:, 1:18)))];
    
    B = readtable('xah.dat','Delimiter', ',', ...
    'Format',formatSpec);
    A = [A; cell2mat(table2cell(B(:, 1:18)))];
    
    B = readtable('xai.dat','Delimiter', ',', ...
    'Format',formatSpec);
    A = [A; cell2mat(table2cell(B(:, 1:18)))];   
    
    [nrow, ncol] = size(A);
    stdA = std(A);
    meanA = mean(A);
    A = (A-repmat(meanA, [nrow,1]));
    A = A./repmat(stdA,[nrow,1]);
    
    csvwrite('vehicle.csv', A);
end