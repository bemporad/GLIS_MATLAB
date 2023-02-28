% Test optimization by preference learning on benchmark problems
%
% (C) 2019 A. Bemporad, September 21, 2019

clear all
close all

addpath(genpath('./glis'))

rng(0) % for repeatability

Ntests=1; % number of tests executed on the same problem

runBayesopt=0; % 0 = run preference learning based on RBF surrogates
%                1 = run preference learning based on Bayesian optimization

RBFcalibrate=1; % recalibrate parameters during optimization
acquisition_method=1; % acquisition method for RBF-based preference learning

%benchmark='1d';
%benchmark='brochu-2d';
benchmark='camelsixhumps';
%benchmark='camelsixhumps-constr';
%benchmark='sasena-constr';

switch benchmark
    case '1d'
        nvars=1;
        lb=-3;
        ub=3;
        f=@(x) (1+x.*sin(2*x).*cos(3*x)./(1+x.^2)).^2+x.^2/12+x/10;
        comparetol=1e-4;
        Aineq=[];
        bineq=[];
        g=[];
        
        delta=2;
        maxevals=25;
        nsamp=10;
        
    case 'camelsixhumps'
        %CamelSixHumps function
        nvars = 2;
        lb=[-2;-1];
        ub=[2;1];
        f = @(x) (4-2.1*x(1).^2+x(1).^4/3).*x(1).^2+...
            x(1).*x(2)+(4*x(2).^2-4).*x(2).^2;
        comparetol=1e-4;
        Aineq=[];
        bineq=[];
        g=[];
        
        maxevals=20;
        delta=1;
        nsamp=round(maxevals/3);
        
    case 'camelsixhumps-constr'
        %CamelSixHumps function with constraints
        nvars = 2;
        lb=[-2;-1];
        ub=[2;1];
        f = @(x) (4-2.1*x(1).^2+x(1).^4/3).*x(1).^2+...
            x(1).*x(2)+(4*x(2).^2-4).*x(2).^2;
        comparetol=1e-4;
        
        Aineq=[1.6295 1;
            -1 4.4553;
            -4.3023 -1;
            -5.6905 -12.1374
            17.6198 1];
        
        bineq = [3.0786; 2.7417; -1.4909; 1; 32.5198];
        
        g=@(x) [(x(1)-0)^2+(x(2)+.1)^2-.5];
        
        delta=.3;
        maxevals=50;
        nsamp=round(maxevals/3);
        
    case 'brochu-2d'
        nvars = 2;
        lb=0*ones(nvars,1);
        ub=1*ones(nvars,1);
        f=@(x) -max(sin(x(1))+x(1)/3+sin(12*x(1))+sin(x(2))+...
            x(2)/3+sin(12*x(2))-1,0);
        comparetol=2/1000;
        Aineq=[];
        bineq=[];
        g=[];
        
        delta=.5;
        maxevals=30;
        nsamp=round(maxevals/3);
        
    case 'sasena-constr'
        nvars = 2;
        lb=[0;0];
        ub=[5;5];
        f = @(x) 2+ 0.01*(x(2) - x(1).^2).^2 + ...
            (1 - x(1)).^2 + 2*(2 - x(2)).^2 + ...
            7*sin(0.5*x(1)).*sin(0.7*x(1).*x(2));
        comparetol=1e-4;
        Aineq=[];
        bineq=[];
        g=@(x) -sin(x(1)-x(2)-pi/8);
        
        delta=1;
        maxevals=25;
        nsamp=8;
end

% TODO: update from here
pref=@(x,y) glisp_function1(x,y,f,comparetol);

epsil=1;
thetas=logspace(-1,1,11);thetas=thetas(1:end-1);

sepvalue=1/maxevals;

opts=[];
opts.sepvalue=sepvalue;
opts.delta=delta;
opts.nsamp=nsamp;
opts.maxevals=maxevals;
opts.feasible_sampling=true;
opts.RBFcalibrate=RBFcalibrate;
opts.thetas=thetas;
opts.acquisition_method=acquisition_method;

opts.rbf_epsil=epsil;
opts.rbf="inverse_quadratic"; % Radial Basis Functions

opts.maxevals=maxevals;

%opts.globoptsol='direct';
opts.globoptsol='pswarm';

opts.display=1;
opts.scalevars=0;

opts.Aineq=Aineq;
opts.bineq=bineq;
opts.g=g;
opts.constraint_penalty=1e5;

if Ntests>1
    bar_handle = waitbar(0,'');
end

FF=NaN(maxevals,Ntests);
for j=1:Ntests
    
    if Ntests>1
        waitbar((j-1)/Ntests,bar_handle,sprintf('%s - running test #%d/%d',benchmark,j,Ntests));
    end
    
    glisp_function1('clear');
    if runBayesopt
        [xbest,out]=bayesopt_pref(pref,lb,ub,opts);
    else
        [xbest,out]=glisp(pref,lb,ub,opts);
    end
    X=out.X;
    F=zeros(maxevals,1);
    for i=1:maxevals
        F(i)=glisp_function1('get',X(i,:)',f);
    end
    
    F=zeros(maxevals,1);
    ff=zeros(size(X,1),1);
    for i=1:size(X,1)
        F(i)=f(X(i,:));
        ff(i)=min(F(1:i,:));
    end
    FF(:,j)=ff;
    
end
if Ntests>1
    close(bar_handle);
end

fprintf('\nTotal CPU time: %5.1f s\n',toc(TIME0));


% Find global optimum for comparison
if ~isempty(Aineq)
    Aineqfun=@(x) sum(max(Aineq*x(:)-bineq,0));
else
    Aineqfun=@(x) 0;
end
if ~isempty(g)
    gfun=@(x) sum(max(g(x(:)),0));
else
    gfun=@(x) 0;
end
latentfun=@(x) f(x) + opts.constraint_penalty*(Aineqfun(x)+gfun(x));

switch opts.globoptsol
    case 'pswarm'
        
        % Use PSO to minimize function

        Options=PSwarm('defaults');
        Options.MaxObj=1000;
        Options.Size=50;

        clear Problem
        Problem.Variables=nvars;
        Problem.LB=lb;
        Problem.UB=ub;
        Problem.SearchType=2;
        InitialPopulation=[];

        Options.IPrint=0;
        Options.CPTolerance=1e-1;
        
        pswarm_vars.Options=Options;
        pswarm_vars.Problem=Problem;
        pswarm_vars.InitialPopulation=InitialPopulation;
        pswarm_vars.Problem.ObjFunction= @(x) latentfun(x');
        evalc('xopt0=PSwarm(pswarm_vars.Problem,pswarm_vars.InitialPopulation,pswarm_vars.Options);');
        
    case 'direct'
        clear opt
        opt.ftol_rel=1e-5;
        opt.ftol_abs=1e-5;
        opt.xtol_abs=1e-5*ones(nvars,1);
        opt.xtol_rel=1e-5;
        opt.verbose = 0;
        opt.maxeval=50000;
        opt.lower_bounds=lb;
        opt.upper_bounds=ub;
        %opt.algorithm=NLOPT_GN_DIRECT;
        opt.algorithm=NLOPT_GN_DIRECT_L;
        opt.min_objective = latentfun;
        xopt0=nlopt_optimize(opt,zeros(nvars,1));
        xopt0=xopt0(:);
end
fopt0=f(xopt0(:)');

if Ntests==1
    figure
    if nvars==1
        x=(lb:(ub-lb)/1000:ub)';
        fx=f(x); % true function
        plot(x,fx);
        grid
        hold on
        plot(X,f(X),'o');
        plot(xbest,f(xbest),'*','linewidth',3.0);
        plot(xopt0,fopt0,'d','linewidth',4.0);
        hold off
        
    elseif nvars==2
        [x1,x2]=meshgrid(lb(1):(ub(1)-lb(1))/50:ub(1),lb(2):(ub(2)-lb(2))/50:ub(2));
        y=zeros(size(x1));
        for i=1:size(x1,1)
            for j=1:size(x1,2)
                x=[x1(i,j);x2(i,j)];
                y(i,j)=f(x');
            end
        end
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
        
        if strcmp(benchmark,'camelsixhumps-constr')
            th=0:.01:2*pi;
            N=numel(th);
            xg=zeros(N,1);
            yg=zeros(N,1);
            for i=1:N
                xg(i)=0+sqrt(.5)*cos(th(i));
                yg(i)=-.1+sqrt(.5)*sin(th(i));
            end
            patch(xg,yg,[.8 .8 .8],'FaceAlpha',0.5)
        end
        hold on
        plot(xopt0(1),xopt0(2),'d','linewidth',4,'color',[0.4940    0.1840    0.5560]);
        
        ax=axis;
        dd=(ax(2)-ax(1))/70;
        ibest=out.I(end,1);
        for i=1:maxevals
            h=plot(X(i,1),X(i,2),'o','linewidth',1,'color',[.2 0 0]);
        end
        plot(xbest(1),xbest(2),'*','linewidth',4,'color',[.8 0 0]);
        plot(xbest(1),xbest(2),'o','linewidth',5,'color',[.8 0 0]);
        set(gcf,'Position',[130 165 965 825]);
        hold off
    end
end

figure
Fm_idw=zeros(maxevals,1);
if Ntests>1
    Fm_idw_min=zeros(maxevals,1);
    Fm_idw_max=zeros(maxevals,1);
    
    for i=1:maxevals
        aux=sort(FF(i,:));
        Fm_idw_min(i)=aux(1);
        Fm_idw_max(i)=aux(Ntests);
        % compute median:
        if rem(Ntests,2)
            Fm_idw(i)=aux((Ntests+1)/2);
        else
            Fm_idw(i)=(aux(Ntests/2)+aux(Ntests/2+1))/2;
        end
    end
else
    for i=1:maxevals
        aux=sort(FF(1:i));
        Fm_idw(i)=aux(1);
    end
end

c1=[ 0    0.4470    0.7410];
c2=[ 0.8500    0.3250    0.0980];

plot(1:maxevals,fopt0*ones(maxevals,1),'--','Color',[0,0,0],'LineWidth',1.0);
hold on
plot(1:maxevals,Fm_idw,'LineWidth',2.5,'Color',c2);

if Ntests>1
    h=patch([1:maxevals maxevals:-1:1]',[Fm_idw_min;Fm_idw_max(maxevals:-1:1)],c2);
    set(h,'FaceAlpha',0.3,'EdgeAlpha',0);
    
    plot(1:maxevals,Fm_idw_min,'-','LineWidth',0.5,'Color',c2);
    plot(1:maxevals,Fm_idw_max,'-','LineWidth',0.5,'Color',c2);
    set(gca,'children',flipud(get(gca,'children')));
end

ax=axis;
h1=patch([0:nsamp-1,nsamp-1:-1:0]',[ax(3)*ones(1,nsamp) ax(4)*ones(1,nsamp)],[.5 .5 .5]);
set(h1,'FaceAlpha',0.2,'EdgeAlpha',0);
plot([nsamp-1,nsamp-1],ax(3:4),'LineWidth',1.0,'Color',[.5 .5 .5]);

grid
hold off
title(benchmark)

