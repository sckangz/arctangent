% this code implements the algorithm for CIKM 2015 paper
%'Robust subspace clustering via tighter rank approximation'
% By Zhao Kang, 08/2015, Zhao.Kang@siu.edu
% This is for motion segmentation experiment with HopKins 155 data
% this deals with HopKins 155 with X=XZ+E model for new function
clear all
close all
warning off
data = load_motion_data(1);
iter=150;
rho=2;%  Two tunable parameters
betazero=10;
rate=1.05;

type=21;% different modeling of E
alpha=2;% W cconstrct
errs = zeros(length(data), 1);
costs = zeros(length(data), 1);

for i = 1 : length(data)
    tic;
    
    X = data(i).X;
    gnd = data(i).ids;
    K = max(gnd);
    [m,n]=size(X);
    W=eye(n);
    ww=ones(n,1);
    Y1=zeros(m,n);
    Y2=zeros(n);
    E=zeros(m,n);
    J=W;
    if abs(K - 2) > 0.1 && abs(K - 3) > 0.1
        id = i; % the discarded sequqnce
    end
    beta=betazero;
    
    
    for ii=1:iter
        
        if ii>=2
            Zold=Z;
        end
        
        W=inv(eye(size(J))+X'*X)*(X'*(X-E)+J+(X'*Y1+Y2)/beta);
        gamma=W-Y2/beta;
        [ J,nw] = DC(gamma,beta/2,ww);
        
        ww=nw;
        
        [E]=errormin(Y1,X,W,rho,beta,type);
        
        Y1=Y1+beta*(X-X*W-E);
        Y2=Y2+beta*(J-W);
        beta=beta*rate;
        Z=W;
        
        if type==1
            enorm=sum(sum(abs(X-X*Z)));
        elseif type==2
            enorm=sum(sum((X-X*Z).^2));
        else
            enorm=sum(sqrt(sum((X-X*Z).^2,1)));
        end
        
        if ii>3 && norm(Z-Zold,'fro')/norm(Zold,'fro')<1e-5
            break
        end
        
        
        func(ii)=sum(atan(nw))+rho*enorm;
        
    end
    consuming_time = toc;
    
    [U s V] = svd(Z);
    s = diag(s);
    r = sum(s>1e-6);
    
    U = U(:, 1 : r);
    s = diag(s(1 : r));
    V = V(:, 1 : r);
    
    M = U * s.^(1/2);
    mm = normr(M);
    rs = mm * mm';
    L = rs.^(2 * alpha);
    
    
    actual_ids = spectral_clustering(L, K);
    
    [err]=1-AccMeasure(gnd, actual_ids);
    errs(i) = err;
    
    LK(i)=K;
    err2obj = max(0,mean(errs(LK==2)));
    mederr2obj = max(0,median(errs(LK==2)));
    err3obj = max(0,mean(errs(LK==3)));
    mederr3obj = max(0,median(errs(LK==3)));
    iters(i) = ii;
    costs(i) = consuming_time;
    disp(['i:' num2str(i) ,',seg2 err=' num2str(err2obj) ',seg3 err=' num2str(err3obj) ',seg err=' num2str(err) ',rho=' num2str(rho) ', beta=' num2str(beta) ',alpha=' num2str(alpha) ...
        ',time=' num2str(consuming_time),',iter=' num2str(ii)]);
end


disp('results of all 156 sequences:');

disp('results of all motions:');
disp(['max = ' num2str(max(errs)) ',min=' num2str(min(errs)) ...
    ',median=' num2str(median(errs)) ',mean=' num2str(mean(errs)) ...
    ',std=' num2str(std(errs)) 'average time=' num2str(mean(costs))] );

dlmwrite('allmotion.txt', [rho betazero alpha ...
    max(errs) min(errs) median(errs) mean(errs) std(errs) mean(costs)], ...
    '-append', 'delimiter', '\t', 'newline', 'pc');


errs = errs([1:id-1,id+1:end]);
costs = costs([1:id-1,id+1:end]);
disp('results of all 155 sequences:');
disp(['max = ' num2str(max(errs)) ',min=' num2str(min(errs)) ...
    ',median=' num2str(median(errs)) ',mean=' num2str(mean(errs)) ',std=' num2str(std(errs)) ', time=' num2str(mean(costs))] );
dlmwrite('motion.txt', [rho betazero alpha err2obj mederr2obj err3obj mederr3obj median(errs) mean(errs) std(errs) max(errs) min(errs) mean(iters) mean(costs) rate] , '-append', 'delimiter', '\t', 'newline', 'pc');


