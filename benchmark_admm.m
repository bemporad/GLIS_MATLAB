function out=test_admm
% (C) 2019 by A Bemporad, June 11, 2019
%
% Use GLIS to optimize parameters rho,alpha of ADMM to solve the QP problems
%
%     min  .5*z'*Q*z+(F*x+c)'*z
%     s.t. A*z <= b + Sx
%
% with x ranging in the box [lbx,ubx]

addpath(genpath('./glis'))

close all
rng(0) % for reproducibility of results

Ntests=20;

time0=tic;

clear out

run_bayesopt=1;
run_glis=1;

n=5; % number of variables
m=10; % number of general inequality constraints
p=3; % number of parameters in mpQP

% Generate random QP with Hessian of given condition number, as in [1, p.127]
% [1] Bierlaire, Toint, Tuyttens, "On Iterative Algorithms for Linear LS Problems
% with Bound Constraints", Linear Algebra and Its Applications, 143:111, 1991
condH=10^2; % condition number of Hessian matrix
S1=diag(exp(-log(condH)/4:log(condH)/2/(n-1):log(condH)/4));
[U,~]=qr((rand(n,n)-.5)*200);
[V,~]=qr((rand(n,n)-.5)*200);
Q=U*S1*V';
Q=Q'*Q;
A=[randn(m,n-1) -ones(m,1)]; % soft constraints to make QP always feasible
c=10*randn(n,1);
F=10*randn(n,p);
b=rand(m,1);
S=randn(m,p);
lbx=-ones(p,1);
ubx=ones(p,1);

% GLIS parameters
nvars=2;
lb=[.01;.01];
ub=[3;3];
maxevals=30;

fbest=Inf;
zbest=NaN(nvars,1);

N=2000; % number of parameter samples for multiparametric QP
X=rand(N,p).*(ones(N,1)*(ubx-lbx)')+(ones(N,1)*lbx');

% Solve QPs at sample points
quadprog_opts=optimset('quadprog');
quadprog_opts.Display='off';
FSTAR=zeros(N,1);
for i=1:N
    x=X(i,:)';
    bx=b+S*x;
    f=c+F*x;
    zstar=quadprog(Q,f,A,bx,[],[],[],[],[],quadprog_opts);
    FSTAR(i)=.5*zstar'*Q*zstar+f'*zstar;
end

fun = @(x) log(admm_eval(x,Q,c,F,A,A'*A,b,S,X,FSTAR));

clear opts
opts.maxevals=maxevals;

XXidw=zeros(0,0,0);
FFidw=zeros(0,0);
XXbo=zeros(0,0,0);
FFbo=zeros(0,0);
TTidw=zeros(0);
TTbo=zeros(0);

if Ntests>1
    bar_handle = waitbar(0,'');
end

for test=1:Ntests
    
    if Ntests>1
        waitbar((test-1)/Ntests,bar_handle,sprintf('Running test #%d/%d',test,Ntests));
    end
    
    if run_glis
        
        opts.rbf="inverse_quadratic"; % Radial Basis Functions
%         opts.rbf="idw"; % Inverse Distance Weighting
        epsil=.5;
        opts.rbf_epsil=epsil;
        
        opts.alpha=.5; % weight on variance
        opts.delta=.1; % weight on distance
        
        opts.n_initial_random=2*numel(lb);
        opts.svdtol=1e-6;
        
        %opts.globoptsol='direct';
        opts.globoptsol='pswarm';
        opts.display=1;
        
        fprintf('Running GLIS optimization:\n')
        
        t0=tic;
        [xopt1,fopt1,out1]=solve_glis(fun,lb,ub,opts);
        t1=toc(t0);
        TTidw=[TTidw;t1];
        
        nn=opts.maxevals;
        minf=zeros(nn,1);
        for i=1:nn
            minf(i)=min(out1.F(1:i));
        end
        
        XXidw(:,:,end+1)=out1.X;
        FFidw(:,end+1)=minf;
        
        if fopt1<fbest
            fbest=fopt1;
            zbest=xopt1;
        end
        
    end
    
    if isempty(which('bayesopt'))
        warning('Bayesian optimization function not found.');
        run_bayesopt=0;
    end
    
    if run_bayesopt
        opt_vars = [];
        for i=1:nvars
            opt_vars = [opt_vars optimizableVariable(sprintf('x%d',i), [lb(i),ub(i)],'Type','real')];
        end
        string='bfun=@(x) fun([';
        for i=1:nvars
            string=sprintf('%sx.x%d',string,i);
            if i<nvars
                string=[string ';'];
            end
        end
        string=sprintf('%s]'');',string);
        eval(string);
        
        t0=tic;
        results = bayesopt(bfun,opt_vars,...
            'Verbose',1,...
            'AcquisitionFunctionName','lower-confidence-bound',... % 'expected-improvement' %-plus',...
            'IsObjectiveDeterministic', true,...
            'MaxObjectiveEvaluations', opts.maxevals,...
            'MaxTime', inf,...
            'NumCoupledConstraints',0, ...
            'NumSeedPoint',10,...
            'GPActiveSetSize', 300,...
            'PlotFcn',{});
        t2=toc(t0);
        TTbo=[TTbo;t2];
        
        xopt2 = results.XAtMinObjective;
        string='xopt2=[';
        for i=1:nvars
            string=sprintf('%sxopt2.x%d',string,i);
            if i<nvars
                string=[string ';'];
            end
        end
        string=[string '];'];
        eval(string);
        
        XXbo(:,:,end+1)=table2array(results.XTrace);
        FFbo(:,end+1)=results.ObjectiveMinimumTrace;
    end
end

if Ntests>1
    close(bar_handle);
    
    % Process results
    close all
    Nv=opts.maxevals;
    Fm_idw=zeros(Nv,1);
    Fm_bo=zeros(Nv,1);
    Fm_idw_min=zeros(Nv,1);
    Fm_bo_min=zeros(Nv,1);
    Fm_idw_max=zeros(Nv,1);
    Fm_bo_max=zeros(Nv,1);
    Ntests_BO=size(FFbo,2);
    
    for i=1:Nv
        if run_glis
            Fm_idw(i)=sum(FFidw(i,:))/Ntests;
            Fm_idw_min(i)=min(FFidw(i,:));
            Fm_idw_max(i)=max(FFidw(i,:));
        end
        if run_bayesopt
            Fm_bo(i)=sum(FFbo(i,:))/Ntests_BO;
            Fm_bo_min(i)=min(FFbo(i,:));
            Fm_bo_max(i)=max(FFbo(i,:));
        end
    end
    
    c1=[ 0    0.4470    0.7410];
    c2=[ 0.8500    0.3250    0.0980];
    
    hp=plot(1:Nv,Fm_idw,'LineWidth',2.5,'Color',c2);
    hold on
    if run_bayesopt
        hp(2)=plot(1:Nv,Fm_bo,'LineWidth',2.5,'Color',c1);
    end
    
    h=patch([1:Nv,Nv:-1:1]',[Fm_idw_min;Fm_idw_max(Nv:-1:1)],c2);
    set(h,'FaceAlpha',0.3,'EdgeAlpha',0);
    if run_bayesopt
        h=patch([1:Nv,Nv:-1:1]',[Fm_bo_min;Fm_bo_max(Nv:-1:1)],c1);
        set(h,'FaceAlpha',0.3,'EdgeAlpha',0);
        plot(1:Nv,Fm_bo_min,'-','LineWidth',0.5,'Color',c1);
        plot(1:Nv,Fm_bo_max,'-','LineWidth',0.5,'Color',c1);
    end
    plot(1:Nv,Fm_idw_min,'-','LineWidth',0.5,'Color',c2);
    plot(1:Nv,Fm_idw_max,'-','LineWidth',0.5,'Color',c2);
    set(gca,'children',flipud(get(gca,'children')));
    grid
    hold off
    title('ADMM hyperparameter optimization')
    axis([1 30 -6 25])
    
    hp(end+1)=patch([1 opts.n_initial_random opts.n_initial_random 1],[-6 -6 25 25],...
        [.8 .8 .8],'FaceAlpha',0.5,'EdgeAlpha',0);
    
    if run_bayesopt
        legend(hp,'GLIS','BO','init phase');
    else
        legend(hp,'GLIS','init phase');
    end
    
    fprintf('CPU time: GLIS = %5.4f (average), %5.4f (worst-case)\n',...
        sum(TTidw)/Ntests,max(TTidw));
    fprintf('           BO = %5.4f (average), %5.4f (worst-case)\n',...
        sum(TTbo)/Ntests,max(TTbo));
end

fprintf('Total elapsed time = %5.2f\n',toc(time0));

out=struct('zbest',zbest,'fbest',fbest,'XXidw',XXidw,'FFidw',FFidw,'TTidw',TTidw,...
    'XXbo',XXbo,'FFbo',FFbo,'TTbo',TTbo);

function result=admm_eval(xx,Q,c,F,A,AA,b,S,X,FSTAR)
if min(size(xx))>1
    m=size(xx,1);
else
    m=1;
end
result=zeros(m,1);

for j=1:m
    x=xx(j,:);
    rho=x(1);
    alpha=x(2);
    maxiter=50;
    
    irho=1/rho;
    iM=Q*irho+AA;
    MA=iM\A';
    
    N=size(X,1);
    cost1=0;
    cost2=0;
    for i=1:N
        x=X(i,:)';
        bx=b+S*x;
        f=c+F*x;
        Mfrho = iM\f*irho;
        
        za=qp_admm(MA,Mfrho,A,bx,alpha,maxiter);
        fa=.5*za'*Q*za+f'*za;
        fstar=FSTAR(i);
        
        cost1=cost1+max((fa-fstar)/(1+abs(fstar)),0);
        cost2=cost2+max(max(A*za-bx,0)./(1+abs(bx)));
    end
    beta=1; % tradeoff coefficient between feasibility and optimality
    result(j)=(cost1+beta*cost2)/N;
end


function x=qp_admm(MA,Mcrho,A,b,alpha,maxiter)
% QP_ADMM Solve a strictly convex QP problem via the Alternating
% Directions Method of Multipliers (ADMM) with over-relaxation.
%
% Solves the QP problem
%
%     min  .5*x'*Q*x+c'*x
%     s.t. A*x<=b
%
% x\in R^n, b\in R^m. The algorithm run for 'maxiter' iterations.
%
% The following matrices need to be provided:
%
% MA    = inv(Q/rho+A'A)*A'
% Mcrho = M*c/rho
%
% (C) 2013 by A. Bemporad, July 4, 2013. Revised by A Bemporad, June 11, 2019

m=size(A,1);

z=zeros(m,1);
u=z;

for i=1:maxiter
    x=MA*(z-u)-Mcrho;
    Ax=A*x;
    w=alpha*Ax+(1-alpha)*z;
    z=min(w+u,b);
    u=u+w-z;
end
