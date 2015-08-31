%This code implements face clustering experiment in CIKM 2015
%paper 'Robust subspace clustering via tighter rank approximation'
% zhao kang, 08/2015, zhao.kang@siu.edu
clear all
close all
load YaleBCrop025.mat;

iter=150;
rho=1e-5 ;% two tuning parameters
betazero=1.7;
alpha=2 ;% W construct
type=1;% different modeling of E
rate=1.03;

nSet = [2 3 5 8 10];
for i = 1 : length(nSet)
    n = nSet(i);
    idx = Ind{n};
    for j = 1 : size(idx,1)
        X = [];
        for p = 1 : n
            X = [X Y(:,:,idx(j,p))];
        end
        X = mat2gray(X);
        [m,nn]=size(X);
        W=eye(nn);
        sig=ones(nn,1);
        Y1=zeros(m,nn);
        Y2=zeros(nn);
        E=zeros(m,nn);
        beta=betazero;
        J=W;
        tic;
        for ii=1:iter
            if ii>=2
                Zold=Z;
            end
            W=inv(eye(size(J))+X'*X)*(X'*(X-E)+J+(X'*Y1+Y2)/beta);
            gamma=W-Y2/beta;
            
            [ J,nw] = DC(gamma,beta/2,sig);
            
            sig=nw;
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
        
        time_cost = toc;
        
        [U ss V] = svd(Z);
        ss = diag(ss);
        r = sum(ss>1e-6);
        
        U = U(:, 1 : r);
        ss = diag(ss(1 : r));
        V = V(:, 1 : r);
        
        M = U * ss.^(1/2);
        mm = normr(M);
        rs = mm * mm';
        
        L = abs(rs).^(2 * alpha);
        
        
        actual_ids = spectral_clustering(L, n);
        [err]=1-AccMeasure(s{n}, actual_ids);
        
        disp(err);
        
        
        missrateTot{n}(j) = err;
        timeTot{n}(j) = time_cost;
        iters{n}(j) = ii;
        dlmwrite('face_detail.txt', [n rho beta alpha err time_cost ] , '-append', 'delimiter', '\t', 'newline', 'pc');
        
    end
    avgmissrate(n) = mean(missrateTot{n});
    medmissrate(n) = median(missrateTot{n});
    meancost(n) = mean(timeTot{n});
    avgiter(n) = mean(iters{n});
    dlmwrite('face.txt', [n rho betazero alpha avgmissrate(n) medmissrate(n) meancost(n) avgiter(n) rate] , '-append', 'delimiter', '\t', 'newline', 'pc');
    disp([n rho betazero alpha avgmissrate(n) medmissrate(n) meancost(n) avgiter(n)]);
end