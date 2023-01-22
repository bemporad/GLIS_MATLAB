function [xopt,fopt,out]=glis(f0,lb,ub,opts,g0,s0)
% GLIS Solve (GL)obal optimization problems using (I)nverse distance
% weighting and radial basis function (S)urrogates.
%
% (C) 2019 A. Bemporad, June 14, 2019
%
% [xopt,fopt,out] = glis(f,lb,ub,opts) solves the global optimization problem 
%
% min  f(x)
% s.t. lb <= x <=ub, A*x <=b, g(x)<=0
%
% using the global optimization algorithm described in [1]. The approach is
% particularly useful when f(x) is time-consuming to evaluate, as it
% attempts at minimizing the number of function evaluations.
%
% The input argument opts specifies various parameters of the algorithm:
%
% opts.maxevals:     maximum number of function evaluations
% opts.alpha:        weight on function uncertainty variance measured by IDW
% opts.delta:        weight on distance from previous samples
% opts.nsamp:        number of initial samples
% opts.useRBF:       true = use RBFs, false = use IDW interpolation
% opts.rbf:          function handle to RBF function (only used if opts.useRBF=true)
% opts.scalevars:    scale problem variables (default: true)
% opts.svdtol:       tolerance used to discard small singular values
% opts.Aineq:        matrix A defining linear inequality constraints 
% opts.bineq:        right hand side of constraints A*x <= b
% opts.g:            constraint function handle
% opts.shrink_range  if 0, disable shrinking lb and ub to bounding box of feasible set
% opts.constraint_penalty: penalty term on violation of linear inequality
%                          and nonlinear constraints
% opts.feasible_sampling: if true, initial samples are forced to be feasible 
% opts.epsDeltaF:    minimum value used to scale the IDW distance function
% opts.globoptsol:   nonlinear solver used during acquisition.
%                    interfaced solvers are:
%                      'direct' DIRECT from NLopt tool (nlopt.readthedocs.io)
%                      'pswarm' PSWarm solver v2.1 (www.norg.uminho.pt/aivaz/pswarm/)
%                      'tmw-pso' Particle Swarm Optimizer from MATLAB Global Optimization Toolbox
%                      'tmw-ga' Genetic Algorithm from MATLAB Global Optimization Toolbox
% opts.PSOiters:     number of iterations in PSO solver
% opts.PSOwarmsize:  swarm size in PSO solver
% opts.display:      verbosity level (0=minimum)
%    
% The output argument 'out' is a structure reporting the following information:
%
% out.X:  trace of all samples x at which f(x) has been evaluated
% out.F:  trace of all function evaluations f(x)
% out.W:  final set of weights (only meaningful for RBFs)
% out.M   RBF matrix (only meaningful for RBFs)
% out.xopt: best sample found during search
% out.fopt: best value found during search, fopt=f(xopt)
%
% [1] A. Bemporad, "Global optimization via inverse weighting and radial basis functions," 
% Computational Optimization and Applications, vol. 77, pp. 571–595.

%%%%%%%%%%%%%%%%%%%%%% 
% (C-GLIS)
% Note: Add features to handle unknown constraints (by M. Zhu, June 3, 2021)
%       Known constraints will be handled via penalty functions
%       For unknown constraints, here we assume that after performing the experiment, we can access the feasibility & satisfactory labels
% 
% Following are the new parameters introduced in C-GLISp
% opts.isUnknownFeasibilityConstrained: if true, unknown feasibility constraints are involved
% opts.isUnknownSatisfactionConstrained: if true, unknown satisfaction constraints are involed
% delta_E: delta for te pure IDW exploration term, \delta_E in the paper
% delta_G_default: delta for feasibility constraints, \delta_{G,default} in the paper
% delta_S_default: delta for satisfaction constraints, \delta_{S,default} in the paper
% Feasibility_unkn: feasibility labels for unknown feasibility constraints
% SatConst_unkn: satisfaction labels for unknown satisfactory constraints


% Parameter setup
[nvar,Aineq,bineq,g,isLinConstrained,isNLConstrained,...
    X,F,z,nsamp,maxevals,epsDeltaF,alpha,delta,rhoC,display,svdtol,...
    dd,d0,useRBF,rbf,M,scalevars,globoptsol,pswarm_vars,direct_vars,...
    isUnknownFeasibilityConstrained,isUnknownSatisfactionConstrained]=...
    glis_init(lb,ub,opts);

delta_E = delta; 
delta_G_default = delta; 
delta_S_default = delta/2;  

Feasibility_unkn = []; % feasibility labels (for unknown constraints only)
SatConst_unkn = []; % satisfactory labels (for unknwon constraints onl)
isfeas_seq = ones(maxevals,1); % keep track the feasibility of the decision variables (including both known and unknown constraints)
ibestseq = ones(maxevals,1); % keep track of the ibest throughout 


if scalevars
    f=@(X) f0(X.*(ones(size(X,1),1)*dd')+ones(size(X,1),1)*d0');
    g_unkn=@(X) g0(X.*(ones(size(X,1),1)*dd')+ones(size(X,1),1)*d0');
    s_unkn=@(X) s0(X.*(ones(size(X,1),1)*dd')+ones(size(X,1),1)*d0');
else
    f=f0;
    g_unkn=g0;
    s_unkn=s0;
end

if ~display
    F(1:nsamp)=f(X(1:nsamp,:)); 
else
    fprintf('Generating initial samples:\n');
    for i=1:nsamp
        F(i)=f(X(i,:));
        fprintf('N = %4d, cost = %8g. ',i,F(i));
        for j=1:nvar
            aux=X(i,j);
            if scalevars
                aux=aux*dd(j)+d0(j);
            end
            fprintf('x%d = %5.4f ',j,aux);
        end
        fprintf('\n');
    end
    fprintf('\nDone. Now starting optimizing:\n\n');
end

if isUnknownFeasibilityConstrained
    Feasibility_unkn = zeros(nsamp,1);
    for i=1:nsamp
        Feasibility_unkn(i) = g_unkn(X(i,:)) < 1e-6;
    end
    delta_G = get_deltaAdpt(X,Feasibility_unkn,delta_G_default);
else
    Feasibility_unkn(1:nsamp) = ones(nsamp,1);
    delta_G = 0;
end 

if isUnknownSatisfactionConstrained
    SatConst_unkn = zeros(nsamp,1);
    for i=1:nsamp
        SatConst_unkn(i) = s_unkn(X(i,:)) < 1e-6;
    end
    delta_S = get_deltaAdpt(X,SatConst_unkn,delta_S_default);
else
    SatConst_unkn(1:nsamp) = ones(nsamp,1);
    delta_S =0;
end
    
if useRBF
    W=get_rbf_weights(M,F,nsamp,svdtol);
else
    W=[];
end

[~,ibest] = min(F);
zbest = X(ibest,:);
fbest=Inf;
% zbest=zeros(nvar,1);
% ibest =1;

for i=1:nsamp
    isfeas=true;
    if isLinConstrained
        isfeas=isfeas && all(Aineq*X(i,:)'<=bineq);
    end
    if isNLConstrained
        isfeas=isfeas && all(g(X(i,:)')<=0);
    end
    if isUnknownFeasibilityConstrained
        isfeas=isfeas && Feasibility_unkn(i) >0;
    end
    if isUnknownSatisfactionConstrained
        isfeas=isfeas && SatConst_unkn(i) >0;
    end
    if isfeas && fbest>F(i)
        fbest=F(i);
        zbest=X(i,:)';
        ibest = i;
    end
   ibestseq(i) = ibest; 
   isfeas_seq(i) = isfeas;
end

Fmax=max(F(1:nsamp));
Fmin=min(F(1:nsamp));

N=nsamp;
while N<maxevals

    dF=max(Fmax-Fmin,epsDeltaF);
    
    if isLinConstrained || isNLConstrained 
        penalty=rhoC*dF;
    else
        penalty = 0;
    end
    if isLinConstrained && isNLConstrained
        constrpenalty=@(x) penalty*(sum(max(Aineq*x(:)-bineq,0).^2) +...
            sum(max(g(x(:)),0).^2));
    elseif isLinConstrained && ~isNLConstrained
        constrpenalty=@(x) penalty*(sum(max(Aineq*x(:)-bineq,0).^2));
    elseif ~isLinConstrained && isNLConstrained
        constrpenalty=@(x) penalty*sum(max(g(x(:)),0).^2);
    else
        constrpenalty=@(x) 0;
    end
    
    d_ibest=sum(([X(1:ibest-1,:);X(ibest+1:N,:)]-X(ibest,:)).^2,2); % exclude the ibest term in X when calculate d_ibest
    ii=find(d_ibest<1e-12,1);
    if ~isempty(ii)
        iw_ibest=0;
    else
        iw_ibest = 1/sum(1./d_ibest);
    end

    acquisition=@(x,p) facquisition(x(:)',X,F,N,alpha,delta_E,dF,W,rbf,isUnknownFeasibilityConstrained,isUnknownSatisfactionConstrained,Feasibility_unkn,SatConst_unkn,delta_G,delta_S,iw_ibest,maxevals) +...
                       constrpenalty(x(:));
    
    switch globoptsol
        case 'pswarm'
            pswarm_vars.Problem.ObjFunction= @(x) facquisition(x(:)',...
            X,F,N,alpha,delta_E,dF,W,rbf,isUnknownFeasibilityConstrained,isUnknownSatisfactionConstrained,Feasibility_unkn,SatConst_unkn,delta_G,delta_S,iw_ibest,maxevals)+...
                                                    constrpenalty(x(:));
            evalc('z=PSwarm(pswarm_vars.Problem,pswarm_vars.InitialPopulation,pswarm_vars.Options);');
            
        case 'direct'
            direct_vars.opt.min_objective = acquisition;
            zold=z;
            z=nlopt_optimize(direct_vars.opt,zold);
            z=z(:);
        
		case {'tmw-pso','tmw-ga'}
            lb2=lb;
            ub2=ub;
            if scalevars
                lb2=-ones(nvar,1);
                ub2=ones(nvar,1);
            end
            if strcmp(globoptsol,'tmw-pso')
                z=particleswarm(acquisition,nvar,lb2,ub2,pswarm_vars.Options);
            else
                z=ga(acquisition,nvar,[],[],[],[],lb2,ub2,[],pswarm_vars.Options);
            end
            z=z(:);
    end
    
    fz = f(z'); % function evaluation
    
    N=N+1;

    X(N,:)=z';
    F(N)=fz;
    
    Fmax=max(Fmax,fz);
    Fmin=min(Fmin,fz);
    
    if isUnknownFeasibilityConstrained
        Feasibility_unkn(N) = g_unkn(z) < 1e-6;
        delta_G = get_deltaAdpt(X,Feasibility_unkn,delta_G_default);
    else
        delta_G = 0;
    end
    
    if isUnknownSatisfactionConstrained
        SatConst_unkn(N) = s_unkn(z) < 1e-6;
        delta_S = get_deltaAdpt(X,SatConst_unkn,delta_S_default);
    else
        delta_S = 0;
    end
    
    isfeas=true;
    if isLinConstrained
        isfeas=isfeas && all(Aineq*z<=bineq);
    end
    if isNLConstrained
        isfeas=isfeas && all(g(z)<=0);
    end
    if isUnknownFeasibilityConstrained
        isfeas=isfeas && Feasibility_unkn(N) >0;
    end
    if isUnknownSatisfactionConstrained
        isfeas=isfeas && SatConst_unkn(N) >0;
    end
    if isfeas && fbest>fz
        fbest=fz;
        zbest=z;
        ibest=N;
    end
    
    ibestseq(N)= ibest;
    isfeas_seq(N)= isfeas;
    
    if display
        fprintf('N = %4d, cost = %8g, best = %8g. ',N,fz,fbest);
        for j=1:nvar
            aux=zbest(j);
            if scalevars
                aux=aux*dd(j)+d0(j);
            end
            fprintf('x%d = %5.4f ',j,aux);
        end
        fprintf('\n');
    end
    
    if useRBF
        % Just update last row and column of M
        for h=1:N
            mij=rbf(X(h,:),X(N,:));
            M(h,N)=mij;
            M(N,h)=mij;
        end
        W=get_rbf_weights(M,F,N,svdtol);
    end
   
end

if isfeas_seq(ibest) ==0 % for the case where no feasible optimizer is identified
    [fbest,ibest] = min(F);
    zbest = X(ibest,:);
end
xopt=zbest;
if ~isUnknownFeasibilityConstrained
    Feasibility_unkn = ones(maxevals,1);
end
if ~isUnknownSatisfactionConstrained && ~isUnknownFeasibilityConstrained
    SatConst_unkn = ones(maxevals,1);
elseif ~isUnknownSatisfactionConstrained && isUnknownFeasibilityConstrained
    SatConst_unkn = Feasibility_unkn;
end
fes_opt_unkn = Feasibility_unkn(ibest);
satConst_opt_unkn = SatConst_unkn(ibest);
feas_opt_comb = isfeas_seq(ibest);

if scalevars
    % Scale variables back
    xopt=xopt.*dd+d0;
    X=X.*(ones(N,1)*dd')+ones(N,1)*d0';
end

fopt=fbest;

if ~useRBF
    W=[];
end

out=struct('X',X,'F',F,'W',W,'M',M,'fopt',fopt,'xopt',xopt,'ibest',ibest,'ibestseq',ibestseq,...
    'Feasibility_unkn',Feasibility_unkn,'SatConst_unkn',SatConst_unkn,'fes_opt_unkn',fes_opt_unkn,'satConst_opt_unkn',satConst_opt_unkn,'isfeas_seq',isfeas_seq,'feas_opt_comb',feas_opt_comb);

%%%%%%%%%%%%%%%%%%%%%%
function [f,fhat,dhat]=facquisition(x,X,F,N,alpha,delta_E,dF,W,rbf,isUnknownFeasibilityConstrained,isUnknownSatisfactionConstrained,Feasibility_unkn,SatConst_unkn,delta_G,delta_S,iw_ibest,maxevals)
% Acquisition function to minimize to get next sample

m=size(x,1); % number of points x to evaluate the acquisition function

f=zeros(m,1);

for i=1:m
    xx=x(i,:)';
    
    d=sum((X(1:N,:)-ones(N,1)*xx').^2,2);
    ii=find(d<1e-12);
    if ~isempty(ii)
        fhat=F(ii(1));
        dhat=0;
        if isUnknownFeasibilityConstrained
            Ghat=Feasibility_unkn(ii);
        else
            Ghat=1;
        end
        if isUnknownSatisfactionConstrained
            Shat=SatConst_unkn(ii);
        else
            Shat=1;
        end
    else
        w=exp(-d)./d;
        sw=sum(w);
        
        if ~isempty(rbf)
            v=zeros(N,1);
            for j=1:N
                v(j)=rbf(X(j,:),xx');
            end
            fhat=v'*W;
        else
            fhat=sum(F(1:N).*w)/sw;
        end
        if maxevals <= 30
            dhat=delta_E*atan(1/sum(1./d))*2/pi*dF+...
                alpha*sqrt(sum(w.*(F(1:N)-fhat).^2)/sw);  
        else
            dhat=delta_E*((1-N/maxevals)*atan((1/sum(1./d))/iw_ibest)+ N/maxevals *atan(1/sum(1./d)))*2/pi*dF+...
                alpha*sqrt(sum(w.*(F(1:N)-fhat).^2)/sw); 
        end

        if isUnknownFeasibilityConstrained
            Ghat=sum(Feasibility_unkn(1:N)'*w)/sw;
        else
            Ghat = 1;
        end  

        if isUnknownSatisfactionConstrained
            Shat=sum(SatConst_unkn(1:N)'*w)/sw;
        else
            Shat = 1;
        end
    end
    
    f(i)=fhat-dhat+(delta_G*(1-Ghat)+delta_S*(1-Shat))*dF;
end


%%%%%%%%%%%%%%%%%%%%%%
function W=get_rbf_weights(M,F,NX,svdtol)
% Solve M*W = F using SVD
        
[U,S,V]=svd(M(1:NX,1:NX));
dS=diag(S);
ns=find(dS>=svdtol,1,'last');
W=V(:,1:ns)*diag(1./dS(1:ns))*U(:,1:ns)'*F(1:NX);

%%%%%%%%%%%%%%%%%%%%%%
function delta_adpt = get_deltaAdpt(X,constraint_set,delta_const_default)
% Adaptively tune the hyperparameter delta_G and delta_S for the feasibility and satisfaction term in the acquisition function
% For both terms, their delta is tuned via leave-one-out cross validation using IDW interpolation as a prediction method

ind = size(constraint_set,1);
sqr_error_feas = zeros(ind,1);
for i= 1:ind
    xx = X(i,:);
    Xi = [X(1:i-1,:); X(i + 1:ind,:)];
    const_classifier_i = [constraint_set(1:i-1);constraint_set(i+1:ind)];
    Feas_xx = constraint_set(i);
    d = sum((Xi - xx).^2,2);
    w = exp(-d)./d;
    sw = sum(w);
    ghat = sum(const_classifier_i'* w) / sw;
    sqr_error_feas(i) = (ghat-Feas_xx)^2;
end

std_feas = min(1,(sum(sqr_error_feas)/(ind-1))^(1/2));
delta_adpt = (1-std_feas) *delta_const_default;
