function [nvar,Aineq,bineq,g,isLinConstrained,isNLConstrained,...
    X,F,z,nsamp,maxevals,epsDeltaF,alpha,delta,rhoC,display,svdtol,...
    dd,d0,useRBF,rbf,M,scalevars,globoptsol,pswarm_vars,direct_vars,...
    isUnknownFeasibilityConstrained,isUnknownSatisfactionConstrained]=glis_init(lb,ub,opts)
% Init function for GLIS.
%
% (C) 2019 A. Bemporad, June 14, 2019

nvar=numel(lb); % number of optimization variables
lb=lb(:);
ub=ub(:);

if isfield(opts,'Aineq')
    Aineq=opts.Aineq;
else
    Aineq=[];
end
if isfield(opts,'bineq')
    bineq=opts.bineq;
else
    bineq=[];
end
if (~isempty(Aineq) && isempty(bineq)) || (~isempty(bineq) && isempty(Aineq))
    error('You must specify both A and b to set linear inequality constraints');
end
isLinConstrained=~isempty(Aineq);
if isfield(opts,'g')
    g0=opts.g;
else
    g0=[];
end
isNLConstrained=~isempty(g0);

if ~isfield(opts,'shrink_range') || isempty(shrink_range)
    shrink_range=1;
else
    shrink_range=opts.shrink_range;
end

if isLinConstrained || isNLConstrained
    if isfield(opts,'constraint_penalty')
        rhoC=opts.constraint_penalty; 
    else
        rhoC=1000;
    end
else
    rhoC=NaN;
end

if isfield(opts,'feasible_sampling') && ~isempty(opts.feasible_sampling)
    feasible_sampling=opts.feasible_sampling;
else
    feasible_sampling=false;
end
if ~isLinConstrained && ~isNLConstrained
    feasible_sampling = false;
end
if isfield(opts,'epsDeltaF')
    epsDeltaF=opts.epsDeltaF;
else
    epsDeltaF=1e-4;
end
    
if isfield(opts,'scalevars')
    scalevars=opts.scalevars;
else
    scalevars=true;
end

if scalevars
    % Rescale problem variables in [-1,1]
    dd=(ub-lb)/2;
    d0=(ub+lb)/2;
    lb=-ones(nvar,1);
    ub=ones(nvar,1);
    
    if isLinConstrained
        bineq=bineq-Aineq*d0;
        Aineq=Aineq*diag(dd);
    end
    if isNLConstrained
        g=@(x) g0(x(:).*dd(:)+d0(:));
    else
        g=g0;
    end
else
    g=g0;
    dd=ones(nvar,1);
    d0=zeros(nvar,1);
end

globoptsol=opts.globoptsol;

pswarm_vars=[];
direct_vars=[];
switch globoptsol
    case 'pswarm'
        if isempty(which('PSwarm.m'))
            error('PSwarm package not found, please download from http://www.norg.uminho.pt/aivaz/pswarm/software/PSwarmM_v2_1.zip and add to MATLABPATH.');
        end
        Options=PSwarm('defaults');
        Options.MaxObj=200000;
        if isfield(opts,'PSOiters') && ~isempty(opts.PSOiters)
            Options.MaxIter=opts.PSOiters;
        else
            Options.MaxIter=200000;
        end
        if isfield(opts,'PSOwarmsize') && ~isempty(opts.PSOwarmsize)
            Options.Size=opts.PSOwarmsize;
        else
            Options.Size=50;
        end

        clear Problem
        Problem.Variables=nvar;
        Problem.LB=lb;
        Problem.UB=ub;
        Problem.SearchType=2;
        InitialPopulation=[];

        %Problem.A=Aineq; %<--- this is somehow much slower !
        %Problem.b=bineq;
        
        Options.IPrint=0;
        Options.CPTolerance=1e-1;
        
        pswarm_vars.Options=Options;
        pswarm_vars.Problem=Problem;
        pswarm_vars.InitialPopulation=InitialPopulation;
        
    case 'direct'
        if isempty(which('nlopt_optimize'))
            error('NLOPT package not found, please download from https://nlopt.readthedocs.io/ and add to MATLABPATH.');
        end
        clear opt
        opt.ftol_rel=1e-5;
        opt.ftol_abs=1e-5;
        opt.xtol_abs=1e-5*ones(nvar,1);
        opt.xtol_rel=1e-5;
        opt.verbose = 0;
        opt.maxeval=50000;
        opt.lower_bounds=lb;
        opt.upper_bounds=ub;
        opt.algorithm=NLOPT_GN_DIRECT;
        %opt.algorithm=NLOPT_GN_DIRECT_L;
        
        direct_vars.opt=opt;
    case {'tmw-pso','tmw-ga'}
        if ~license('test','GADS_Toolbox')
            error('Global Optimization Toolbox not found.');
        end
        
        if strcmp(globoptsol,'tmw-pso')
            options = optimoptions('particleswarm','Display','off');
            if ~isfield(opts,'PSOiters') || isempty(opts.PSOiters)
                options.MaxIterations=200000;
            else
                options.MaxIterations=opts.PSOiters;
            end
            if ~isfield(opts,'PSOwarmsize') || isempty(opts.PSOwarmsize)
                options.SwarmSize=50;
            else
                options.SwarmSize=opts.PSOwarmsize;
            end
        else
            options = optimoptions('ga','Display','off');
        end
        
        
        pswarm_vars.Options=options;
end

if shrink_range
    % possibly shrink lb,ub to constraints
    if ~isNLConstrained && isLinConstrained && exist('linprog','file')
        lpopts=struct('Display','None');
        for i=1:nvar
            flin=[zeros(i-1,1);1;zeros(nvar-i,1)];
            [~,aux]=linprog(flin,Aineq,bineq,[],[],lb,ub,[],lpopts);
            lb(i)=max(lb(i),aux);
            [~,aux]=linprog(-flin,Aineq,bineq,[],[],lb,ub,[],lpopts);
            ub(i)=min(ub(i),-aux);
        end
    elseif isNLConstrained || (isLinConstrained && ~exist('linprog','file'))
        if isNLConstrained
            NLpenaltyfun = @(x) sum(max(g(x(:)),0).^2);
        else
            NLpenaltyfun = @(x) 0;
        end
        if isLinConstrained
            LINpenaltyfun = @(x) sum(max(Aineq*x(:)-bineq,0).^2);
        else
            LINpenaltyfun = @(x) 0;
        end
        
        z0=zeros(nvar,1);
        for i=1:nvar
            obj_fun=@(x) x(i)+1e4*(NLpenaltyfun(x) + LINpenaltyfun(x));
            switch globoptsol
                case 'pswarm'
                    Problem.ObjFunction= @(x) obj_fun(x(:));
                    evalc('z=PSwarm(Problem,InitialPopulation,Options);');
                case 'direct'
                    opt.min_objective = obj_fun;
                    z=nlopt_optimize(opt,z0);
                case 'tmw-pso'
                    z=particleswarm(obj_fun,nvar,lb,ub,options);

            end
            lb(i)=max(lb(i),z(i));
            
            obj_fun=@(x) -x(i)+1e4*(NLpenaltyfun(x) + LINpenaltyfun(x));
            switch globoptsol
                case 'pswarm'
                    Problem.ObjFunction= @(x) obj_fun(x(:));
                    evalc('z=PSwarm(Problem,InitialPopulation,Options);');
                case 'direct'
                    opt.min_objective = obj_fun;
                    z=nlopt_optimize(opt,z0);
                case 'tmw-pso'
                    z=particleswarm(obj_fun,nvar,lb,ub,options);
            end
            ub(i)=min(ub(i),z(i));
        end
    end
end

useRBF=opts.useRBF;
if useRBF
    rbf=opts.rbf;
else
    rbf=[];
end

nsamp=opts.nsamp;
alpha=opts.alpha;
delta=opts.delta;
maxevals=opts.maxevals;
if maxevals<nsamp
    errstr='Max number of function evaluations is too low. You specified';
    error('%s maxevals = %d and nsamp = %d',errstr,maxevals,nsamp);
end

if isfield(opts,'display')
    display=opts.display;
else
    display=true;
end
if isfield(opts,'svdtol')
    svdtol=opts.svdtol;
else
    svdtol=1e-6;
end

useLHS=exist('lhsdesign','file');
if ~useLHS
    fprintf('\nLatin hypercube sampling function LHSDESIGN not available, generating random samples\n');
end

% Allocate variables
X=zeros(maxevals,nvar);
F=zeros(maxevals,1);
z=zeros(nvar,1);

% Generate initial samples
if ~feasible_sampling
    % Don't care about constraints
    if useLHS
        X(1:nsamp,:)=lhsdesign(nsamp,nvar); % Use Latin hypercube sampling instead
    else
        X(1:nsamp,:)=rand(nsamp,nvar);
    end
    X(1:nsamp,:)=X(1:nsamp,:).*(ones(nsamp,1)*(ub-lb)')+ones(nsamp,1)*lb';
else
    nn=nsamp;
    nk=0;
    while nk<nsamp
        if useLHS
            XX=lhsdesign(nn,nvar);
        else
            XX=rand(nn,nvar);
        end
        XX=XX.*(ones(nn,1)*(ub-lb)')+ones(nn,1)*lb';
        
        ii=true(nn,1);
        for i=1:nn
            if isLinConstrained
                ii(i)=all(Aineq*XX(i,:)'<=bineq);
            end
            if isNLConstrained
                ii(i)=ii(i) && all(g(XX(i,:)')<=0);
            end
        end
        nk=sum(ii);
        if nk==0
            nn=20*nn;
        elseif nk<nsamp
            nn=ceil(min(20,1.1*nsamp/nk)*nn);
        end
    end
    ii=find(ii);
    X(1:nsamp,:)=XX(ii(1:nsamp),:);
end

if useRBF
    M=zeros(maxevals,maxevals); % preallocate the entire matrix
    for i=1:nsamp
        for j=i:nsamp
            mij=rbf(X(i,:),X(j,:));
            M(i,j)=mij;
            M(j,i)=mij;
        end
    end
else
    M=[];
end

if isfield(opts,'isUnknownFeasibilityConstrained')
    isUnknownFeasibilityConstrained = opts.isUnknownFeasibilityConstrained;
else
    isUnknownFeasibilityConstrained=false;
end
if isfield(opts,'isUnknownSatisfactionConstrained')
    isUnknownSatisfactionConstrained= opts.isUnknownSatisfactionConstrained;
else
    isUnknownSatisfactionConstrained=false;
end
