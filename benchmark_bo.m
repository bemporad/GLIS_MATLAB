% Test global optimization based on IDW and RBF on a benchmark problem
%
% (C) 2019 A. Bemporad, June 14, 2019

clear all
close all

addpath(genpath('./glis'))

rng(2) % for repeatability

run_bayesopt=1;

%benchmark='ackley';
benchmark='camelsixhumps';
% benchmark='hartman6';
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
        'MaxObjectiveEvaluations', maxevals,...
        'MaxTime', inf,...
        'NumCoupledConstraints',0, ...
        'NumSeedPoint',10,...
        'GPActiveSetSize', 300,...
        'PlotFcn',{}); %@plotMinObjective});%,@plotObjectiveEvaluationTime}); %);
    t2=toc(t0);
   
else
    t2=NaN;
end

