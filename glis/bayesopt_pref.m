function [xopt,out]=bayesopt_pref(pref0,lb,ub,opts)
% Active preference learning via Bayesian optimization.
%
% [xopt,out] = bayesopt_pref(pref,lb,ub,opts) solves the active
% preference learning problem
%
% find x such that pref(x,y) <= 0 for all x,y in X,
% X = {x: lb <= x <=ub, A*x <=b, g(x)<=0}
% 
% where pref(x,y) = -1 if x "better than" y
%                    0 if x "as good as" y
%                    1 if x "worse than" y
%
% A special case is to solve the global optimization problem
%
% min  f(x)
% s.t. lb <= x <=ub, A*x <=b, g(x)<=0
%
% based only on comparisons between function values
%
% pref(x,y) = -1 if f(x1) <= f(x2) - tol
%           =  0 if |f(x1)-f(x2)| <= tol
%           =  1 if f(x1) >= f(x2) + tol      
%
% where tol is the threshold deciding the outcome of the comparison,
% i.e., comparison is "even" if |f(x1)-f(x2)| <= tol
%
% opts is a structure with parameters used by the optimization algorithm.
% Type "help glis_pref" for more details.
%
% The output argument 'out' is a structure reporting the following information:
%       xopt: best value of the input x
%       out:  structure with tested points and observed preferences
%
% (C) 2019 D. Piga, July 5, 2019
%     Revised by A. Bemporad, September 22, 2019
%     Revised by M. Zhu, June 9, 2021 

opts.useRBF=true;
opts.scalevars=false;
pref=pref0;

opts.alpha=0; % dummy
opts.rbf=@(x1,x2) 0; % dummy

[nvar,Aineq,bineq,g,isLinConstrained,isNLConstrained,...
    X,F,~,nsamp,maxevals,epsDeltaF,~,~,rhoC,display]=glis_init(lb,ub,opts);
N=nsamp;

if ~isfield(opts,'RBFcalibrate') || isempty(opts.RBFcalibrate)
    RBFcalibrate=false;
else
    RBFcalibrate=opts.RBFcalibrate;
end
if ~isfield(opts,'RBFcalibrationSteps') || isempty(opts.RBFcalibrationSteps)
    RBFcalibrationSteps=nsamp+round([0;(maxevals-nsamp)/4;
    (maxevals-nsamp)/2;3*(maxevals-nsamp)/4]);
else
    RBFcalibrationSteps=opts.RBFcalibrationSteps;
end

% Fills in initial preference vectors and find best initial guess
zbest=X(1,:);
ibest=1;
I=[]; % I(i,1:2)=[h k] if F(h)<F(k)-comparetol
Feasibility_unkn = [];
SatConst_unkn = [];  
isfeas_seq = ones(maxevals,1); 
ibestseq = ones(maxevals,1); % keep track of the ibest throughout

if isfield(opts,'isUnknownFeasibilityConstrained')
    isUnknownFeasibilityConstrained= opts.isUnknownFeasibilityConstrained;
else
    isUnknownFeasibilityConstrained=0;
end

if isfield(opts,'isUnknownSatisfactionConstrained')
    isUnknownSatisfactionConstrained= opts.isUnknownSatisfactionConstrained;
else
    isUnknownSatisfactionConstrained=0;
end

M=0;
for i=2:nsamp
    if i ==2
        if isUnknownSatisfactionConstrained && isUnknownFeasibilityConstrained % when has both unknown feasibility and satisfactory constraints
            [prefi,fesi,fesbest,satconsti,satconstbest]=pref(X(i,:),zbest'); 
            SatConst_unkn = [SatConst_unkn;satconstbest]; 
            Feasibility_unkn=[Feasibility_unkn;fesbest];
        elseif ~isUnknownSatisfactionConstrained && isUnknownFeasibilityConstrained % when only has unknown feasibility constraints
            [prefi,fesi,fesbest]=pref(X(i,:),zbest'); 
            Feasibility_unkn=[Feasibility_unkn;fesbest];
        elseif isUnknownSatisfactionConstrained && ~isUnknownFeasibilityConstrained  % when only has unknown satisfactory constraints
            [prefi,satconsti,satconstbest]=pref(X(i,:),zbest');
            SatConst_unkn = [SatConst_unkn;satconstbest];
        else  % when there is no unknown constraints
           prefi=pref(X(i,:),zbest'); 
        end  
    else
        if isUnknownSatisfactionConstrained && isUnknownFeasibilityConstrained
            [prefi,fesi,~,satconsti,~]=pref(X(i,:),zbest');
        elseif ~isUnknownSatisfactionConstrained && isUnknownFeasibilityConstrained
           [prefi,fesi,~]=pref(X(i,:),zbest'); 
        elseif isUnknownSatisfactionConstrained && ~isUnknownFeasibilityConstrained
            [prefi,satconsti,~]=pref(X(i,:),zbest');
        else
            prefi=pref(X(i,:),zbest');
        end
    end
    if isUnknownFeasibilityConstrained
        Feasibility_unkn=[Feasibility_unkn;fesi]; 
    end
    if isUnknownSatisfactionConstrained
        SatConst_unkn = [SatConst_unkn;satconsti];
    end
    
    isfeas=true;
    % for known constraints (Note: no query is required for known constraints)
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
        
%     prefi=pref(X(i,:),zbest');
    if prefi==-1
        I=[I;i,ibest];
        zbest=X(i,:)';
        ibest=i;
        M=M+1;
    elseif prefi==1
        I=[I;ibest,i];
        M=M+1;
    else
        I=[I;ibest,i;i,ibest];
        M=M+2;
    end
    ibestseq(i) = ibest;
    isfeas_seq(i) = isfeas;
end

%Bopts.PSOsolver.max_iter = 200; % Maximum number of iterations in maximizing the acquisition function
Bopts.hyper_opt = 1; % set Bopts.hyper_opt = 1 if kernel hyper-parameters should be selected by maximizing the marginal likelihood

% Initial values of kernel hyper-parameters
Bopts.SE.l2 = 10;
Bopts.SE.sigmaf2 = 10;
Bopts.sigmae2 = 1; % noise variance

% Set parameters tocompute f_MAP, then used for Laplace approximation
Bopts.maxiter_fMAP = 1000;   % Maximum number of iterations for MAP optimization
Bopts.tol = 1e-3;            % Set tolerance to terminate optimization algorithm
Bopts.opt_var = 1;           % set 1 for Newton-Raphsod algorithm. Otherwise, gradient method is used
Bopts.nsearch = 10;          % number of grid points for exact line-search
Bopts.f0 = zeros(N,1);       % Initial condition for optimization (Nx1-vector)
Bopts.alpha = 0.0001;        % Regularization parameter for Kernel matrix


Bopts.M = M;
Bopts.N = N;
Bopts.nsamp = nsamp;

Bopts.nvars = nvar;

Bopts.RBFcalibrate=RBFcalibrate;
Bopts.RBFcalibrationSteps=RBFcalibrationSteps;
Bopts.display=display;

D.X=X(1:N,:);
D.Xp=I;
D.Feasibility_unkn=Feasibility_unkn;
D.SatConst_unkn=SatConst_unkn;
D.isUnknownFeasibilityConstrained=isUnknownFeasibilityConstrained;
D.isUnknownSatisfactionConstrained=isUnknownSatisfactionConstrained;
D.isLinConstrained = isLinConstrained;
D.isNLConstrained = isNLConstrained;
D.ibestseq=ibestseq;
D.isfeas_seq=isfeas_seq;
D.Aineq = Aineq;
D.bineq = bineq;
D.g = g;
D.maxevals = maxevals;

Bopts.D = D;

%%
% Optimize the hyper-parameters by maximixing the evidence via
% Bayesian optimization
if Bopts.hyper_opt == 1
    [Bopts.SE.l2, Bopts.SE.sigmaf2, Bopts.sigmae2] = BO_hyp(D,Bopts);
end

Bopts.hyper_opt = 1;



%% Active Preference Learning
Bopts.g = g; % Set here the function with constraints
Bopts.APL.MaxObjectiveEvaluations = maxevals; %Maximum number of evaluations in BO
Bopts.eta = 0.05; % used in the definition of the expected improvement (mu_x - mu_max- mu_max*costparams.opts.eta)/s_x;

if display
    fprintf('Running Active Preference Learning:\n')
end

Bopts.constraints.Aineq=Aineq;
Bopts.constraints.bineq=bineq;
Bopts.constraints.g=g;
Bopts.constraints.isLinConstrained=isLinConstrained;
Bopts.constraints.isNLConstrained=isNLConstrained;
Bopts.constraints.rhoC=rhoC;
Bopts.constraints.epsDeltaF=epsDeltaF;

[xopt, out]=ActivePreferenceLearning(pref, lb, ub, Bopts);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xopt, out] = ActivePreferenceLearning(pref, lb, ub, opts)

D = opts.D;
opts.N = size(D.X,1);   % Number of data
opts.M = size(D.Xp,1);  % Number of pairwise preferencesend

Options=PSwarm('defaults');
Options.MaxObj=200000;
Options.MaxIter=150;
Options.Size=50;

clear Problem
Problem.Variables=opts.nvars;
Problem.LB=lb;
Problem.UB=ub;
Problem.SearchType=2;
InitialPopulation=[];
Options.IPrint=0;
Options.CPTolerance=1e-1;

maxeval = opts.APL.MaxObjectiveEvaluations;
out.X = D.X;
out.Xp = D.Xp;
out.Feasibility_unkn = D.Feasibility_unkn;
out.SatConst_unkn = D.SatConst_unkn;
out.isfeas_seq = D.isfeas_seq;
out.ibestseq = D.ibestseq;


RBFcalibrate=opts.RBFcalibrate;
RBFcalibrationSteps=opts.RBFcalibrationSteps;
nsamp = opts.nsamp;

display=opts.display;

for ind = 1:maxeval-nsamp
    
    %flagEx = 0;
    N = opts.N;
    M = opts.M;
    
    % Optimize hyper-parameters through Bayesian optimization
    %if (ind== 5 || ind== 10 || ind == 30 || ind == 50 || ind == 80) && opts.hyper_opt==1 
    if opts.hyper_opt==1 && RBFcalibrate && any(N==RBFcalibrationSteps) 
        [opts.SE.l2, opts.SE.sigmaf2, opts.sigmae2] = BO_hyp(D,opts);
    end
    
    K = build_Kernel(D.X,D.X,opts);
    opts.Sigma = K+opts.alpha*eye(opts.N); % Kernel
    opts.Sigmainv = inv(opts.Sigma);
    [fMAP, gMAP, HMAP, betaMAP, LambdaMAP, L, log_pD] = compute_fMAP(D,opts);
    opts.LambdaMAP = LambdaMAP;
    
    
    D.fMAP = fMAP;
    [m_vec, s_vec] = GP_mu_s(D.X,D,opts); % Fit GP of the posterior p(f|D). Returns mean and standard deviation
    [mu_max, ind_max] = max(m_vec);
    % ind_max2(ind) = ind_max;
    
    
    %% PSO optimization
    
    cost_params.D = D;                          %Parameters for the cost function
    cost_params.opts = opts;                    %Parameters for the cost function
    cost_params.mu_max = mu_max;                %Parameters for the cost function
    
    epsDeltaF=opts.constraints.epsDeltaF;
    dF=max(max(m_vec)-min(m_vec),epsDeltaF);
    cost_params.opts.constraints.dF=dF;
    
    
    Problem.ObjFunction = @(x) -EI_fun(x,cost_params); %+penalty*sum(max(opts.g(x(:)),0).^2);;
    
    % solve problem
    evalc('xp=PSwarm(Problem,InitialPopulation,Options);');
    
    if display
        fprintf('N = %4d: x = [',ind+nsamp);
        for j=1:numel(xp)
            aux=xp(j);
            fprintf('%5.4f',aux);
            if j<numel(xp)
                fprintf(', ');
            end
        end
        fprintf(']\n');
    end

    % Extract solution
    xp = xp(:)';
    D.X(end+1,:) = xp;
    % thepref=pref(xp',D.X(ind_max,:));
    
    if D.isUnknownSatisfactionConstrained && D.isUnknownFeasibilityConstrained
        [thepref,fesN,~,satconstN,~] = pref(xp',D.X(ind_max,:)); % preference query
        
    elseif ~D.isUnknownSatisfactionConstrained && D.isUnknownFeasibilityConstrained
        [thepref,fesN,~] = pref(xp',D.X(ind_max,:)); % preference query
    elseif D.isUnknownSatisfactionConstrained && ~D.isUnknownFeasibilityConstrained
        [thepref,satconstN,~]=pref(xp',D.X(ind_max,:));
    else
        thepref = pref(xp',D.X(ind_max,:));
    end
    
    if D.isUnknownFeasibilityConstrained
        D.Feasibility_unkn(N+1) = fesN;
    end
    
    if D.isUnknownSatisfactionConstrained
        D.SatConst_unkn(N+1) = satconstN;
    end
    
    isfeas=true;
    % for known constraints
    if D.isLinConstrained
        isfeas=isfeas && all(D.Aineq*xp'<=D.bineq);
    end
    if D.isNLConstrained
        isfeas=isfeas && all(D.g(xp)<=0);
    end
    if D.isUnknownFeasibilityConstrained
        isfeas=isfeas && D.Feasibility_unkn(N+1) >0;
    end
    if D.isUnknownSatisfactionConstrained
        isfeas=isfeas && D.SatConst_unkn(N+1) >0;
    end

    if thepref==-1
        D.Xp(M+1,:) = [N+1, ind_max];
    elseif thepref==1
        D.Xp(M+1,:) = [ind_max, N+1];
    else
        D.Xp(M+1:M+2,:) = [N+1, ind_max; ind_max, N+1];
    end
    
    D.ibestseq(N+1) = ind_max;
    D.isisfeas_seq(N+1) = isfeas;
    
    opts.N = size(D.X,1);
    opts.M = size(D.Xp,1);
    opts.f0 = zeros(opts.N,1); % Initial conditions for optimize posterior distribution at the next step
    
end

out.X = D.X;
out.Xp = D.Xp;
out.I=D.Xp;

xopt = D.X(ind_max,:)';
out.ibest = ind_max;
out.ibestseq = D.ibestseq;
out.isfeas_seq = D.isfeas_seq;

if ~D.isUnknownFeasibilityConstrained
    D.Feasibility_unkn = ones(D.maxevals,1);
end
if ~D.isUnknownSatisfactionConstrained && ~D.isUnknownFeasibilityConstrained
    D.SatConst_unkn = ones(D.maxevals,1);
elseif ~D.isUnknownSatisfactionConstrained && D.isUnknownFeasibilityConstrained
    D.SatConst_unkn = D.Feasibility_unkn;
end

out.Feasibility_unkn = D.Feasibility_unkn;
out.SatConst_unkn= D.SatConst_unkn;
out.satConst_opt_unkn = D.SatConst_unkn(ind_max);
out.fes_opt_unkn = D.Feasibility_unkn(ind_max);
out.feas_opt_comb =  D.isfeas_seq(ind_max);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function EI_x = EI_fun(x,costparams)
% Expected improvement maximized in Preferential Bayesian optimization
%
% (C) 2019 D. Piga, Lugano, July 5, 2019

D = costparams.D;
opts = costparams.opts;
mu_max = costparams.mu_max;
Aineq=opts.constraints.Aineq;
bineq=opts.constraints.bineq;
g=opts.constraints.g;
isLinConstrained=opts.constraints.isLinConstrained;
isNLConstrained=opts.constraints.isNLConstrained;
rhoC=opts.constraints.rhoC;
dF=opts.constraints.dF;

[mu_x, s_x] = GP_mu_s(x',D,opts);
z = (mu_x - mu_max-0*costparams.opts.eta -abs(mu_max)*costparams.opts.eta)/s_x;
Phi_d = normcdf(z);
phi_d = normpdf(z);

if s_x > 0
    EI_x = z*Phi_d + s_x*phi_d;
else
    EI_x = 0;
end

penalty=rhoC*dF;
if isLinConstrained && isNLConstrained
    EI_x=EI_x-penalty*(sum(max(Aineq*x(:)-bineq,0).^2) +...
        sum(max(g(x(:)),0).^2));
elseif isLinConstrained && ~isNLConstrained
    EI_x=EI_x-penalty*(sum(max(Aineq*x(:)-bineq,0).^2));
elseif ~isLinConstrained && isNLConstrained
    EI_x=EI_x- penalty*sum(max(g(x(:)),0).^2);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [l2, sigmaf2, sigmae2] = BO_hyp(D,opts)
% Optimize hyper-parameters via Bayesian optimization.
%
% The kernel is defined as sigmaf2*exp(-0.5||x_i-x_j||^2/l2) and the
% optimized hyper-parameters are: sigmaf2, l2, and noise variance sigmae2
%
% Inputs:
%       D: structure with inputs x and pairwise preference Xp
%       opts: structure with parameters used by the optimization algorithm
%
% (C) 2019 D. Piga, Lugano, July 5, 2019


bfun = @(theta) BO_MAP(theta,D,opts);

%     opt_vars = [optimizableVariable('l2', [0.00001,20],'Type','real'), ...
%                     optimizableVariable('sigmaf2', [100,200],'Type','real'), ...
%                     optimizableVariable('sigmae2', [opts.sigmae2,opts.sigmae2+0.0001],'Type','real')];


opt_vars = [optimizableVariable('l2', [1e-6,10],'Type','real'), ...
    optimizableVariable('sigmaf2', [1e-6,50],'Type','real'), ...
    optimizableVariable('sigmae2', [1e-6,0.01],'Type','real')];

display=opts.display;

t0=tic;
results = bayesopt(bfun,opt_vars,...
    'Verbose',display,...
    'AcquisitionFunctionName','lower-confidence-bound',... 'expected-improvement' %-plus',...
    'IsObjectiveDeterministic', true,... % simulations with noise --> objective function is not deterministic
    'MaxObjectiveEvaluations', 10,...
    'MaxTime', inf,...
    'NumCoupledConstraints',0, ...
    'NumSeedPoint',10,...
    'GPActiveSetSize', 300,...
    'PlotFcn',{}); %@plotMinObjective});%,@plotObjectiveEvaluationTime}); %);
t2=toc(t0);

xopt3 = results.XAtMinObjective;
fopt3=results.MinObjective;


l2=xopt3.l2;
sigmaf2=xopt3.sigmaf2;
sigmae2=xopt3.sigmae2;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function EI_x = APL_acquisition(x,D,opts,mu_max)
% Define here the acquisition function

[mu_x, s_x] = GP_mu_s(x,D,opts);
z = (mu_x - mu_max)/s_x;
Phi_d = normcdf(z);
phi_d = normpdf(z);

if s_x > 0
    EI_x = (mu_x - mu_max)*Phi_d + s_x*phi_d;
else
    EI_x = 0;
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function log_pD = BO_MAP(theta,D,opts)
% Compute log of the evidence p(D|theta). Used to optimize the
% hyper-parameters via Bayesian optimization
%
% The kernel is defined as sigmaf2*exp(-0.5||x_i-x_j||^2/l2) and the
% optimized hyper-parameters are: sigmaf2, l2, and noise variance sigmae2
%
% Inputs:
%       theta: structure with hyper-parameters sigmaf2, l2, sigmae2
%       D: structure with inputs x and pairwise preference Xp
%       opts: structure with parameters used by the optimization algorithm
%
% (C) 2019 D. Piga, Lugano, July 5, 2019

opts.SE.l2 = theta.l2;
opts.SE.sigmaf2 = theta.sigmaf2;
opts.sigmae2 = theta.sigmae2;

% Compute Kernel Matrix
K = build_Kernel(D.X,D.X,opts);
N = size(D.X,1);
opts.Sigma = K+opts.alpha*eye(N); % Kernel + regularization
opts.Sigmainv = inv(opts.Sigma);
[fMAP, gMAP, HMAP, betaMAP, LambdaMAP, L, log_pD] = compute_fMAP(D,opts);
log_pD = -log_pD;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function K = build_Kernel(X1,X2,opts)

n1 = size(X1,1);
n2 = size(X2,1);

K = ones(n1,n2);
for ind1 = 1:n1
    x1 = X1(ind1,:)';
    for ind2 = 1:n2
        x2 = X2(ind2,:)';
        K(ind1,ind2) = SE_ij(x1,x2,opts);
    end
end

end


function Kij = SE_ij(x1,x2,opts)

Kij = opts.SE.sigmaf2*exp(-0.5*(x1-x2)'*(x1-x2)/opts.SE.l2);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [fMAP, gMAP, HMAP, betaMAP, LambdaMAP, L, log_pD] = compute_fMAP(D,opts)
% Compute MAPof the latent preference function f
%
% Inputs:
%       D: structure with inputs x and pairwise preference Xp
%       opts: structure with parameters used by the optimization algorithm
%
% Outputs:
%       fMAP: MAP value of the latent function f
%       gMAP: Gradient at f=fMAP
%       betaMAP: beta parameter (see paper Chu, Ghahramani, ICML, 2005, eq. 11)
%       LambdaMAP: Lambda matrix at f = fMAP (see paper Chu, Ghahramani, ICML, 2005)
%       L: -log of the posterior p(f|D) (up to constant 1/p(D))
%       log_PD: approximation of the log of the evidence p(D) (used for hyper-parameter selection)
%
% (C) 2019 D. Piga, Lugano, July 5, 2019

[N,d] = size(D.X);

%% Compute fMAP
% Initial value of f
f = opts.f0;

for i = 1:opts.maxiter_fMAP
    [grad_f, H_f, flag_c] = compute_grad_H_loss(D,f,opts);      % Compute Gradient and Hessian at f. If flag_c==1, then the cumulative is equal to zero and the Hesssian is not well approximated.
    %In this case, at that iteration, gradient method will be used instead of Newton-Raphson
    
    L = compute_log_posterior(D,f,opts);                        % Compute -log of the posterior p(f|D) (up to the constant term 1/p(D))
    
    %fprintf('Iteration: %d, Cost: %2.4f, Gradient: %2.6f \n',i,L,norm(grad_f))
    
    if opts.opt_var == 1 && flag_c == 0 % Use Newton method
        t = linsearch(D,f,H_f,grad_f,opts,flag_c);
        deltaf =  - t*(H_f\grad_f);
    else                                % Use gradient descent
        t = linsearch(D,f,H_f,grad_f,opts,flag_c);
        deltaf = -t*grad_f;
    end
    f = f + deltaf;  % Update f
    
    if norm(grad_f)<=opts.tol
        break
    end
    
    if abs(L)>=1e20
        L = 1e21;
        break
    end
    
end % for i = 1:opts.maxiter_fMAP
%%

% Extract Gradient and Hessian at the optimum
[gMAP, HMAP, flag_c,  betaMAP, LambdaMAP] = compute_grad_H_loss(D,f,opts);
fMAP = f;

if abs(L)<1e20
    L = compute_log_posterior(D,fMAP,opts);
end

log_pD =  -L -0.5*log(det(eye(N)+opts.Sigma*LambdaMAP)); %(approximation of) the log of the evidence p(D) (used for hyper-parameter tuning)


end


function [grad_f, H_f, flag_c, beta, Lambda_MAP] = compute_grad_H_loss(D,f,opts)
% Compute gradient and Hessian of the Loss w.r.t. the latent function f, at
% a given point f

flag_c = 0;
M = size(D.Xp,1);
N = size(D.X,1);

grad_f = zeros(N,1);
H_f = zeros(N,N);

for ind = 1:M
    
    v = D.Xp(ind,1);
    u = D.Xp(ind,2);
    
    den2 = 2*opts.sigmae2;
    den = sqrt(den2);
    
    z = (f(v) - f(u))/den;
    if z <-25
        flag_c = 1;
        z = -25;
    end
    
    normal = normpdf(z);
    cumulative = normcdf(z);
    
    if cumulative == 0
        flag_c = 1;
        grad1 = abs(f(v))*0.1;
        grad2 = abs(f(u))*0.1;
        grad_f(v) = grad_f(v) - grad1;
        grad_f(u) = grad_f(u) + grad2;
        
    else
        grad_f(v) = grad_f(v) - 1/den*normal/cumulative;
        grad_f(u) = grad_f(u) + 1/den*normal/cumulative;
    end
    
    
    
    
    H_f(v,u) = H_f(v,u)-1/den2*(normal^2/cumulative^2 + z*normal/cumulative);
    H_f(u,v) = H_f(v,u);
    H_f(v,v) = H_f(v,v)+1/den2*(normal^2/cumulative^2 + z*normal/cumulative);
    H_f(u,u) = H_f(u,u)+1/den2*(normal^2/cumulative^2 + z*normal/cumulative);
    
    
    
end

beta = -grad_f; %-(grad_f - (opts.Sigma\f));
Lambda_MAP = H_f; %-(H_f - opts.Sigmainv);

grad_f = grad_f + 1*(opts.Sigma\f);%zeros(N,1);
H_f = H_f + opts.Sigmainv; %zeros(N,N);

end



function L = compute_log_posterior(D,f,opts)

M = size(D.Xp,1);
L = 0.5*f'*opts.Sigmainv*f;


for ind = 1:M
    
    v = D.Xp(ind,1);
    u = D.Xp(ind,2);
    
    den2 = 2*opts.sigmae2;
    den = sqrt(den2);
    
    z = (f(v) - f(u))/den;
    normal = normpdf(z);
    cumulative = normcdf(z);
    if cumulative==0
        cumulative = 1e-55;
    end
    
    L = L - log(cumulative);
end

end


function t = linsearch(D,f,H_f,grad_f,opts, flag_c)

if opts.opt_var==0 || flag_c == 1
    tvec = linspace(0,1,opts.nsearch);
    tvec(1) = 0.01;
else
    tvec = linspace(0.2,1,opts.nsearch);
end

for ind = 1:1 %opts.nsearch
    %t = tvec(ind);
    t = 1;
    if opts.opt_var == 1 && flag_c==0
        deltaf =  - t*(H_f\grad_f);
    else
        deltaf = -t*grad_f;
    end
    fn = f + deltaf;
    
    L(ind) = compute_log_posterior(D,fn,opts);
end

[minval, indmin] = min(L);

t = tvec(indmin);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  [mu_x, s_x] = GP_mu_s(x,D,opts)
% Compute mean value standard deviation of the surrogate GP
%
% Inputs:
%     x: inputmatrix
%     D: training data
%     opts: structure with parameters used by the optimization algorithm
%
% Outputs:
%     mu: mean
%     s_x: standard deviation
%
% (C) 2019 D. Piga, Lugano, July 5, 2019


Kt =  build_Kernel(D.X,x,opts);
mu_x = Kt'*(opts.Sigma\D.fMAP);

Sigmat = build_Kernel(x,x,opts);
s2_x = Sigmat - Kt'*( (opts.Sigma+inv(opts.LambdaMAP+0.000001*eye(opts.N))) \Kt);
s_x = sqrt(diag(s2_x));

end
