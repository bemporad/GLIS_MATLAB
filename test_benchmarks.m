% Test global optimization based on IDW and RBF on a benchmark problem
%
% (C) 2019 A. Bemporad, June 14, 2019

clear all
close all

rng(2) % for repeatability

run_bayesopt=1;

%benchmark='ackley';
%benchmark='camelsixhumps';
benchmark='hartman6';
%benchmark='rosenbrock8';

switch benchmark
    case 'camelsixhumps'
        %CamelSixHumps function
        nvars = 2;
        lb=[-2;-1];
        ub=[2;1];
        fun = @(x) (4-2.1*x(:,1).^2+x(:,1).^4/3).*x(:,1).^2+...
            x(:,1).*x(:,2)+(4*x(:,2).^2-4).*x(:,2).^2;
        maxevals=25;
        xopt0=[0.0898 -0.0898;
            -0.7126 0.7126]; % optimizers, one per column
        fopt0=-1.0316; % optimum
        use_linear_constraints=0;
        use_nl_constraints=0;
        
    case 'hartman6'
        nvars=6;
        lb=zeros(nvars,1);
        ub=ones(nvars,1);
        alphaH = [1.0, 1.2, 3.0, 3.2]';
        AH = [10, 3, 17, 3.5, 1.7, 8;
            0.05, 10, 17, 0.1, 8, 14;
            3, 3.5, 1.7, 10, 17, 8;
            17, 8, 0.05, 10, 0.1, 14];
        PH = 10^(-4) * [1312, 1696, 5569, 124, 8283, 5886;
            2329, 4135, 8307, 3736, 1004, 9991;
            2348, 1451, 3522, 2883, 3047, 6650;
            4047, 8828, 8732, 5743, 1091, 381];
        fun=@(x) -exp(-[(x-ones(size(x,1),1)*PH(1,:)).^2*AH(1,:)', ...
            (x-ones(size(x,1),1)*PH(2,:)).^2*AH(2,:)', ...
            (x-ones(size(x,1),1)*PH(3,:)).^2*AH(3,:)', ...
            (x-ones(size(x,1),1)*PH(4,:)).^2*AH(4,:)'])*alphaH;
        fopt0=-3.32237;
        xopt0=[.20169 .150011 .476874 .275332 .311652 .6573]';
        maxevals=80;
        use_linear_constraints=0;
        use_nl_constraints=0;
        
    case 'rosenbrock8'
        nvars=8;
        lb=-30*ones(nvars,1);
        ub=30*ones(nvars,1);
        fun=@(x) 100*(x(:,2)-x(:,1).^2).^2+(x(:,1)-1).^2 + ...
            100*(x(:,3)-x(:,2).^2).^2+(x(:,2)-1).^2 + ...
            100*(x(:,4)-x(:,3).^2).^2+(x(:,3)-1).^2 + ...
            100*(x(:,5)-x(:,4).^2).^2+(x(:,4)-1).^2 + ...
            100*(x(:,6)-x(:,5).^2).^2+(x(:,5)-1).^2 + ...
            100*(x(:,7)-x(:,6).^2).^2+(x(:,6)-1).^2 + ...
            100*(x(:,8)-x(:,7).^2).^2+(x(:,7)-1).^2;
        maxevals=80;
        xopt0=ones(8,1);
        fopt0=0;
        use_linear_constraints=0;
        use_nl_constraints=0;
        
    case 'ackley'
        nvars=2;
        lb=-5*ones(nvars,1);
        ub=5*ones(nvars,1);
        fun=@(x) -20*exp(-.2*sqrt(0.5*(x(:,1).^2+x(:,2).^2)))-exp(0.5*...
            (cos(2*pi*x(:,1))+cos(2*pi*x(:,2))))+exp(1)+20;
        maxevals=60;
        xopt0=zeros(2,1);
        fopt0=0;
        use_linear_constraints=0;
        use_nl_constraints=0;
end
clear opts
opts.maxevals=maxevals;

opts.useRBF=true; % Radial Basis Functions
%opts.useRBF=false; % Inverse Distance Weighting
if opts.useRBF
    epsil=0.5;
    opts.rbf=@(x1,x2) 1/(1+epsil^2*sum((x1-x2).^2)); % inverse quadratic
end
opts.alpha=1.0/nvars; % weight on variance
opts.delta=2.0/nvars; % weight on distance

opts.nsamp=2*numel(lb);
opts.svdtol=1e-6;

opts.globoptsol='pswarm';
%opts.globoptsol='tmw-pso';
%opts.globoptsol='tmw-ga'; % this runs slower
%opts.globoptsol='direct'; % faster, but may provide more suboptimal solutions

opts.display=1;

if use_nl_constraints
    opts.g=@(x) [(x(1)-1).^2+x(2).^2-.25;
        (x(1)-0.5).^2+(x(2)-0.5).^2-.25];
end
if use_linear_constraints
    opts.Aineq=[1.6295    1.0000;
        -1.0000    4.4553;
        -4.3023   -1.0000;
        -5.6905  -12.1374;
        17.6198    1.0000];
    
    opts.bineq=[3.0786;
        2.7417;
        -1.4909;
        1.0000;
        32.5198];
end
opts.feasible_sampling=false;

fprintf('Running GLIS optimization:\n')

t0=tic;
[xopt1,fopt1,out]=glis(fun,lb,ub,opts);
t1=toc(t0);

nn=opts.maxevals;
minf=zeros(nn,1);
for i=1:nn
    minf(i)=min(out.F(1:i));
end


if nvars==2
    [x1,x2]=meshgrid(lb(1):.1:ub(1),lb(2):.1:ub(2));
    y=zeros(size(x1));
    %yg=y;
    for i=1:size(x1,1)
        for j=1:size(x1,2)
            x=[x1(i,j);x2(i,j)];
            y(i,j)=fun(x');
            %yg(i,j)=max(g(x'),0);
        end
    end
    figure
    surf(x1,x2,y);
    title(sprintf('%s function',benchmark))
    figure
    contour(x1,x2,y,50);
    hold on
    plot(out.X(:,1),out.X(:,2),'o','linewidth',1.5);
    for j=1:size(xopt0,2)
        plot(xopt0(1,j),xopt0(2,j),'dr','linewidth',2);
    end
    plot(xopt1(1),xopt1(2),'*','linewidth',2);
    
    if use_nl_constraints
        % Plot constraints defined by function opts.g
        C1=cos(0:.05:2*pi);
        S1=sin(0:.05:2*pi);
        X1=1+C1*.5;
        Y1=S1*.5;
        X2=0.5+C1*.5;
        Y2=0.5+S1*.5;
        plot(X1,Y1,X2,Y2,'linewidth',1.5);
    end
    if use_linear_constraints
        V =[0.4104 -0.2748;
            0.1934 0.6588;
            1.3286 0.9136;
            1.8412 0.0783;
            1.9009 -0.9736];
        patch(V(:,1),V(:,2),[.8 .8 .8],'FaceAlpha',0.2);
    end
    
    hold off
end

figure
if ~strcmp(benchmark,'rosenbrock8')
    plot(1:nn,minf,1:nn,ones(1,nn)*fopt0,'--','linewidth',1.5);
else
    semilogy(1:nn,minf,1:nn,ones(1,nn)*fopt0,'--','linewidth',1.5);
end
grid
title('Best function value found')
xlabel('number of function evaluations')
set(gcf,'Tag','MinFun');
drawnow

XXidw=out.X;
FFidw=minf;

if isempty(which('bayesopt'))
    warning('Bayesian optimization function not found.');
    run_bayesopt=0;
end

if run_bayesopt
    
    fprintf('Running Bayesian optimization:\n')
    
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
        'AcquisitionFunctionName','lower-confidence-bound',... 'expected-improvement' %-plus',...
        'IsObjectiveDeterministic', true,... % simulations with noise --> objective function is not deterministic
        'MaxObjectiveEvaluations', opts.maxevals,...
        'MaxTime', inf,...
        'NumCoupledConstraints',0, ...
        'NumSeedPoint',10,...
        'GPActiveSetSize', 300,...
        'PlotFcn',{}); %@plotMinObjective});%,@plotObjectiveEvaluationTime}); %);
    t2=toc(t0);
    
    xopt3 = results.XAtMinObjective;
    string='xopt3=[';
    for i=1:nvars
        string=sprintf('%sxopt3.x%d',string,i);
        if i<nvars
            string=[string ';'];
        end
    end
    string=[string '];'];
    eval(string);
    fopt3=results.MinObjective;
    
    feval3=results.NumObjectiveEvaluations;
    
    fh=findobj('Tag','MinFun');
    figure(fh);
    hold on
    plot(1:feval3,results.ObjectiveMinimumTrace,'linewidth',1.5);
    hold off
    
    if ~strcmp(benchmark,'rosenbrock8')
        ax=axis;
        patch([1 opts.nsamp opts.nsamp 1],[ax(3) ax(3) ax(4) ax(4)],...
            [.8 .8 .8],'FaceAlpha',0.5,'EdgeAlpha',0);
    end
    legend('GLIS','optimum','BO','init phase')
    drawnow
    
    XXbo=table2array(results.XTrace);
    FFbo=results.ObjectiveMinimumTrace;
else
    t2=NaN;
end

fprintf('Total elapsed time: GLIS = %5.2f, BO = %5.2f\n',t1,t2);
