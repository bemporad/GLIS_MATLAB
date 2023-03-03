% Test optimization by preference learning with unknown constraint handling on benchmark problems
% Algorithms:
%   - C-GLISp
%   - C-GLIS
%   - PBO
%
% Reference code: 'test_pref_benchmarks' by A. Bemporad, September 21, 2019

% M. Zhu, June 07, 2021

clear all
close all

addpath(genpath('./glis'))

rng(0) % for repeatability

Ntests=1; % number of tests executed on the same problem

RBFcalibrate=1; % recalibrate parameters during optimization
acquisition_method=1; % acquisition method for RBF-based preference learning

% benchmarks without unknown constraints
% benchmark='1d';
% benchmark='camelsixhumps';
% benchmark='camelsixhumps-constr'; %camelsixhumps with known constraints

% 2D benchmarks used for illustration of pref with unknown constraints
% benchmark='MBC';  %Mishra's Bird function constrained
benchmark='CHC'; %CamelSixHumps function with feasibility constraints
% benchmark='CHSC'; %CamelSixHumps function with feasibility and satisfactory constraints

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
            g_unkn_fun = @(x) (x(1) + 9)^2 + (x(2) + 3)^2 - 9;
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
        
    case 'CHC' % CamelSixHumps function with feasibility constraints
        nvars = 2;
        lb=[-2;-1];
        ub=[2;1];
        f = @(x) (4-2.1*x(1).^2+x(1).^4/3).*x(1).^2+...
            x(1).*x(2)+(4*x(2).^2-4).*x(2).^2;
        xopt0 = [0.0898, -0.0898;-0.7126, 0.7126];  % unconstrained optimizers, one per column
        fopt0 = -1.0316;  % unconstrained optimum
        comparetol=1e-4;
        
        xopt_const = [0.21305, 0.57424]; %constrained optimizers
        fopt_const = -0.58445; % constrained optimum
        
        Aineq=[];
        bineq=[];
        g=[];

        % for unknown constraints
        isUnknownFeasibilityConstrained = 1; % isUnknownFeasibilityConstrained = 1, if unknown inequality feasibility constraints exist, isUnknownFeasibilityConstrained = 0, otherwise
        isUnknownSatisfactionConstrained = 0; %  isUnknownSatisfactionConstrained = 1, if unknown satisfactory constraints exist,  isUnknownSatisfactionConstrained = 0, otherwise
        if isUnknownFeasibilityConstrained ==1
            Aineq_unkn=[1.6295 1;
                          -1 4.4553;
                          -4.3023 -1;
                          -5.6905 -12.1374
                          17.6198 1];
            bineq_unkn = [3.0786; 2.7417; -1.4909; 1; 32.5198];
            g_nl_unkn=@(x) [(x(1)-0)^2+(x(2)+.1)^2-.5];
            g_unkn_fun = @(x) sum(max(Aineq_unkn*x(:)-bineq_unkn,0)) + sum(max(g_nl_unkn(x(:)),0));
        else
            g_unkn_fun =@(x) 0;
        end
        
        if isUnknownSatisfactionConstrained % add the necessary eqns if relavent
            s_unkn_fun =@(x) 0;
        else
            s_unkn_fun =@(x) 0;
        end
            
        delta=2;
        maxevals= 100;
        nsamp=round(maxevals/4);

    case 'CHSC' % CamelSixHumps function with feasibility and satisfactory constraints        
        nvars = 2;
        lb=[-2;-1];
        ub=[2;1];
        f = @(x) (4-2.1*x(1).^2+x(1).^4/3).*x(1).^2+...
            x(1).*x(2)+(4*x(2).^2-4).*x(2).^2;
        xopt0 = [0.0898, -0.0898;-0.7126, 0.7126];  % unconstrained optimizers, one per column
        fopt0 = -1.0316;  % unconstrained optimum
        comparetol=1e-4;
        
        xopt_const = [0.0781, 0.6562]; % constrained optimizers
        fopt_const = -0.9050;  % constrained optimum constrained optimizers

        
        Aineq=[];
        bineq=[];
        g=[];
        
        % for unknown constraints
        isUnknownFeasibilityConstrained = 1; % isUnknownFeasibilityConstrained = 1, if unknown inequality feasibility constraints exist, isUnknownFeasibilityConstrained = 0, otherwise
        isUnknownSatisfactionConstrained = 1; %  isUnknownSatisfactionConstrained = 1, if unknown satisfactory constraints exist,  isUnknownSatisfactionConstrained = 0, otherwise
        if isUnknownFeasibilityConstrained ==1
            g_unkn_fun = @(x) (x(1)-0)^2+(x(2)+.04)^2-.8;
        else
            g_unkn_fun =@(x) -1;
        end
        if isUnknownSatisfactionConstrained
                        Aineq_unkn=[1.6295 1;
                                      0.5 3.875;
                                      -4.3023 -4;
                                      -2 1;
                                      0.5 -1];
            bineq_unkn = [3.0786; 3.324; -1.4909; 0.5;0.5];
            s_unkn_fun =@(x) sum(max(Aineq_unkn*x(:)-bineq_unkn,0));
        else
            s_unkn_fun =@(x) 0;
        end
        delta=1;
        maxevals=50;
        nsamp=round(maxevals/4);
end

if isUnknownSatisfactionConstrained && isUnknownFeasibilityConstrained 
    pref=@(x,y) glisp_function3(x,y,f,comparetol,g_unkn_fun,s_unkn_fun);  % Include query for both unknown feasibility and satisfactory constraints besides the preference query
elseif ~isUnknownSatisfactionConstrained && isUnknownFeasibilityConstrained 
    pref=@(x,y) glisp_function2(x,y,f,comparetol,g_unkn_fun); % Inlude query for only feasibility constraints besides the preference query
elseif isUnknownSatisfactionConstrained && ~isUnknownFeasibilityConstrained
    pref=@(x,y) glisp_function2(x,y,f,comparetol,s_unkn_fun); % Inlude query for only satisfactory constraints besides the preference query
else
    pref=@(x,y) glisp_function1(x,y,f,comparetol,Aineq,bineq,g); % with only preference query
end

epsil=1;
thetas=logspace(-1,1,11);thetas=thetas(1:end-1);

sepvalue=1/maxevals;

opts=[];
opts.sepvalue=sepvalue;
opts.delta=delta;
opts.n_initial_random=nsamp;
opts.maxevals=maxevals;
opts.feasible_sampling=true;
opts.RBFcalibrate=RBFcalibrate;
opts.thetas=thetas;
opts.acquisition_method=acquisition_method;
opts.has_unknown_constraints = isUnknownFeasibilityConstrained;
opts.has_satisfaction_fun = isUnknownSatisfactionConstrained;
opts.scale_delta = true;

opts.rbf_epsil=epsil;
opts.rbf="inverse_quadratic";

opts.maxevals=maxevals;

%opts.globoptsol='direct';
opts.globoptsol='pswarm';

opts.display=1;
opts.scalevars=0;

opts.Aineq=Aineq;
opts.bineq=bineq;
opts.g=g;
opts.isUnknownFeasibilityConstrained = isUnknownFeasibilityConstrained;
opts.isUnknownSatisfactionConstrained = isUnknownSatisfactionConstrained;
opts.constraint_penalty=1e5;

eval_feas_ =@(x) eval_feas(x,isUnknownFeasibilityConstrained,g_unkn_fun);
eval_sat_ =@(x) eval_sat(x,isUnknownSatisfactionConstrained,s_unkn_fun);


if isUnknownSatisfactionConstrained && isUnknownFeasibilityConstrained 
    glisp_function3('clear');
elseif ~isUnknownSatisfactionConstrained && isUnknownFeasibilityConstrained
    glisp_function2('clear');
elseif isUnknownSatisfactionConstrained && ~isUnknownFeasibilityConstrained
    glisp_function2('clear');
else
    glisp_function1('clear');
end

[xbest,out]=solve_glisp(pref,lb,ub,opts,eval_feas_,eval_sat_);

X=out.X;
F=zeros(maxevals,1);
for i=1:maxevals
    if isUnknownSatisfactionConstrained && isUnknownFeasibilityConstrained 
        F(i)=glisp_function3('get',X(i,:)',f);
    elseif ~isUnknownSatisfactionConstrained && isUnknownFeasibilityConstrained
        F(i)=glisp_function2('get',X(i,:)',f);
    elseif isUnknownSatisfactionConstrained && ~isUnknownFeasibilityConstrained
        F(i)=glisp_function2('get',X(i,:)',f);
    else
        F(i)=glisp_function1('get',X(i,:)',f);
    end
end


% plot the fun. eval. 
figure
xbest_seq = out.X(out.ibest_seq,:);
fbest_seq = zeros(maxevals,1);
for i=1:size(out.X,1)
    fbest_seq(i) = f(xbest_seq(i,:));
end
plot(1:maxevals,fbest_seq,'--','linewidth',1.5);

hold on
if isUnknownFeasibilityConstrained || isUnknownSatisfactionConstrained || ~isempty(Aineq)||~isempty(g)
    plot(1:maxevals,fopt_const*ones(maxevals,1),'--','Color',[0,0,0],'LineWidth',1.0);
else
    plot(1:maxevals,fopt0*ones(maxevals,1),'--','Color',[0,0,0],'LineWidth',1.0);
end
hold off
title(benchmark)

figure
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

if strcmp(benchmark,'CHC') || strcmp(benchmark,'camelsixhumps-constr')
    th=0:.01:2*pi;
    N=numel(th);
    xg=zeros(N,1);
    yg=zeros(N,1);
    for i=1:N
        xg(i)=0+sqrt(.5)*cos(th(i));
        yg(i)=-.1+sqrt(.5)*sin(th(i));
    end
    patch(xg,yg,[.8 .8 .8],'FaceAlpha',0.5)
    V = [0.4104, -0.2748; 0.1934, 0.6588; 1.3286, 0.9136;...
    1.8412, 0.0783;1.9009, -0.9736];
    f = [1,2,3,4,5];
    patch('Faces',f,'Vertices',V,'FaceColor',[.8 .8 .8],'FaceAlpha',0.5)
end

if strcmp(benchmark,'CHSC') 
    th=0:.01:2*pi;
    N=numel(th);
    xg=zeros(N,1);
    yg=zeros(N,1);
    for i=1:N
        xg(i)=0+sqrt(.8)*cos(th(i));
        yg(i)=-.04+sqrt(.8)*sin(th(i));
    end
    patch(xg,yg,[.8 .8 .8],'FaceAlpha',0.5)
    V = [1.48,0.667;0.168,0.836; -0.041,0.417; 0.554, -0.223;...
    1.68,0.34];
    f = [1,2,3,4,5];
    patch('Faces',f,'Vertices',V,'FaceColor',[.8 .8 .8],'FaceAlpha',0.5)
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
hold off
title(benchmark)
    

