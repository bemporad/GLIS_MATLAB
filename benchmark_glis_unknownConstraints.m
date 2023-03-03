% (C) 2019-2023 Alberto Bemporad, Mengjia Zhu

clear all
close all

addpath(genpath('./glis'))

TIME0=tic;

rng(0) % for repeatability

Ntests=1; % number of tests executed on the same problem


benchmark='MBC';  %Mishra's Bird function constrained


switch benchmark
    case 'MBC' % Mishra's Bird function constrained
        nvars = 2;
        lb=[-10.0, -6.5];
        ub=[-2, 0.0];
        f=@(x) sin(x(2))*exp((1-cos(x(1)))^2) + cos(x(1))*exp((1-sin(x(2)))^2) + (x(1) - x(2))^2;
        xopt0 = [-3.1302468, -1.5821422]; % unconstrained optimizer
        fopt0 = -106.7645367;  % unconstrained optimum
        
        xopt_const = [-9.3669,-1.62779]; % constrained optimizer
        fopt_const = -48.4060;  % constrained optimum
        
        comparetol=1e-4;
        Aineq=[];
        bineq=[];
        g=[];
        
        % for unknown constraints
        isUnknownFeasibilityConstrained = 1; % isUnknownFeasibilityConstrained = 1, if unknown inequality feasibility constraints exist, isUnknownFeasibilityConstrained = 0, otherwise
        isUnknownSatisfactionConstrained = 0; %  isUnknownSatisfactionConstrained = 1, if unknown satisfactory constraints exist,  isUnknownSatisfactionConstrained = 0, otherwise
        if isUnknownFeasibilityConstrained ==1
            g_unkn_fun = @(x) sum(max((x(1) + 9)^2 + (x(2) + 3)^2 - 9,0));
        else
            g_unkn_fun =@(x) 0;
        end
        
        if isUnknownSatisfactionConstrained % add the necessary eqns if relavent
            s_unkn_fun =@(x) 0;
        else
            s_unkn_fun =@(x) 0;
        end
        
        delta=1;
        maxevals=50;
        nsamp=round(maxevals/4);
        
end

opts=[];
opts.delta=delta;
opts.n_initial_random=nsamp;
opts.maxevals=maxevals;
opts.feasible_sampling=true;

epsil=1;
opts.rbf_epsil=epsil;
opts.rbf="inverse_quadratic";


opts.maxevals=maxevals;

%opts.globoptsol='direct';
opts.globoptsol='pswarm';

opts.display=1;
opts.scalevars=1;

opts.Aineq=Aineq;
opts.bineq=bineq;
opts.g=g;
opts.has_unknown_constraints = isUnknownFeasibilityConstrained;
opts.has_satisfaction_fun = isUnknownSatisfactionConstrained;
opts.constraint_penalty=1e5;
opts.alpha=delta/5;

eval_feas_ =@(x) eval_feas(x,isUnknownFeasibilityConstrained,g_unkn_fun);
eval_sat_ =@(x) eval_sat(x,isUnknownSatisfactionConstrained,s_unkn_fun);

[xbest, fbest,prob_setup] = solve_glis(f,lb,ub,opts,eval_feas_,eval_sat_);
X=prob_setup.X;
F=prob_setup.F;
    
fprintf('\nTotal CPU time: %5.1f s\n',toc(TIME0));

% Plot the level curve of the objective functions
[x1,x2]=meshgrid(lb(1):(ub(1)-lb(1))/50:ub(1),lb(2):(ub(2)-lb(2))/50:ub(2));
y=zeros(size(x1));
for i=1:size(x1,1)
    for j=1:size(x1,2)
        x=[x1(i,j);x2(i,j)];
        y(i,j)=f(x');
    end
end
axes('Units', 'normalized', 'Position', [0.05 0.05 0.9 0.9]);
contourf(x1,x2,y,50);
colormap(1-.7*bone);
hold on
if ~isempty(Aineq)
    try
        polyplot(Aineq,bineq);
    catch
        warning('To plot polyhedral constraints please install the Hybrid Toolbox for MATLAB');
    end
end

% Plot the constraints
if strcmp(benchmark,'MBC')
    th=0:.01:2*pi;
    N=numel(th);
    xg=zeros(N,1);
    yg=zeros(N,1);
    for i=1:N
        xg(i)=-9.0+sqrt(9)*cos(th(i));
        yg(i)=-3.0+sqrt(9)*sin(th(i));
    end
    patch(xg,yg,[.8 .8 .8],'FaceAlpha',0.5)
end

hold on

% Plot the unconstrained optimum
if numel(xopt0) >2
    plot(xopt0(1,:),xopt0(2,:),'d','linewidth',4,'color',[0.4940    0.1840    0.5560]);
else
    plot(xopt0(1),xopt0(2),'d','linewidth',4,'color',[0.4940    0.1840    0.5560]);
end

% plot the constrained optimum
if isUnknownFeasibilityConstrained || isUnknownSatisfactionConstrained
    plot(xopt_const(1),xopt_const(2),'s','linewidth',4,'color',[0 0.9 0.1]);
end

axis([lb(1),ub(1),lb(2),ub(2)]);

if Ntests ==1
    % plot the point tested during one run
    for i=1:nsamp
        h=plot(X(i,1),X(i,2),'x','linewidth',2,'color',[.2 .2 1]);
    end
    for i=nsamp+1:maxevals
        h=plot(X(i,1),X(i,2),'o','linewidth',1,'color',[.2 0 0]);
    end
    plot(xbest(1),xbest(2),'*','linewidth',4,'color',[.8 0 0]);
    plot(xbest(1),xbest(2),'o','linewidth',5,'color',[.8 0 0]);
    set(gcf,'Position',[130 20 950 700]);
else
    % plot the constrained optimum computed at the end of each run
    for i=1:1:Ntests
        h1=plot(xopt_Ntests(i,1),xopt_Ntests(i,2),'x','linewidth',1,'color',[.8 0 0]);
    end
end
hold off
title(benchmark)


