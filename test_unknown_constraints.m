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

TIME0=tic;

rng(0) % for repeatability

Ntests=1; % number of tests executed on the same problem

% choose the solver to run
runGLIS =0;    % 0 = run PBO or GLISp, 1 = run GLIS
runGLISp=1-runGLIS; 
runBayesopt=0; % 0 = run preference learning based on RBF surrogates (GLISp) or GLIS
               % 1 = run preference learning based on Bayesian optimization (PBO)

RBFcalibrate=1; % recalibrate parameters during optimization
acquisition_method=1; % acquisition method for RBF-based preference learning

% benchmarks without unknown constraints
% benchmark='1d';
% benchmark='camelsixhumps';
% benchmark='camelsixhumps-constr'; %camelsixhumps with known constraints

% 2D benchmarks used for illustration of pref with unknown constraints
benchmark='MBC';  %Mishra's Bird function constrained
% benchmark='CHC'; %CamelSixHumps function with feasibility constraints
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
            g_nl_unkn=@(x) [(x(1)-0)^2+(x(2)+.04)^2-.8];
            g_unkn_fun = @(x) sum(max(g_nl_unkn(x(:)),0));
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
        
        
    %%%%%%%%% benchmarks w/o unknown constraints 
    case '1d'
        nvars=1;
        lb=-3;
        ub=3;
        f=@(x) (1+x.*sin(2*x).*cos(3*x)./(1+x.^2)).^2+x.^2/12+x/10;
        xopt0 = -0.956480816387759;  % unconstrained optimizers
        fopt0 = 0.279546426870577;  % unconstrained optimum
        comparetol=1e-4;
        Aineq=[];
        bineq=[];
        g=[];
        
        isUnknownFeasibilityConstrained = 0; % isUnknownFeasibilityConstrained = 1, if unknown inequality feasibility constraints exist, isUnknownFeasibilityConstrained = 0, otherwise
        isUnknownSatisfactionConstrained = 0; %  isUnknownSatisfactionConstrained = 1, if unknown satisfactory constraints exist,  isUnknownSatisfactionConstrained = 0, otherwise
        if isUnknownFeasibilityConstrained ==1 % Add unknown constraints here if exist
            g_unkn_fun = @(x) 0;
        else
            g_unkn_fun =@(x) 0;
        end
        
        if isUnknownSatisfactionConstrained % add the necessary eqns if relavent
            s_unkn_fun =@(x) 0;
        else
            s_unkn_fun =@(x) 0;
        end
        
        delta=2;
        maxevals=25;
        nsamp=10;
        
    case 'camelsixhumps' % CamelSixHumps function 
        nvars = 2;
        lb=[-2;-1];
        ub=[2;1];
        f = @(x) (4-2.1*x(1).^2+x(1).^4/3).*x(1).^2+...
            x(1).*x(2)+(4*x(2).^2-4).*x(2).^2;
        xopt0 = [0.0898, -0.0898;-0.7126, 0.7126];  % unconstrained optimizers, one per column
        fopt0 = -1.0316;  % unconstrained optimum
        comparetol=1e-4;
        
%         xopt_const = ; %constrained optimizers
%         fopt_const =; % constrained optimum
        
        Aineq=[];
        bineq=[];
        g=[];

        isUnknownFeasibilityConstrained = 0; % isUnknownFeasibilityConstrained = 1, if unknown inequality feasibility constraints exist, isUnknownFeasibilityConstrained = 0, otherwise
        isUnknownSatisfactionConstrained = 0; %  isUnknownSatisfactionConstrained = 1, if unknown satisfactory constraints exist,  isUnknownSatisfactionConstrained = 0, otherwise
        if isUnknownFeasibilityConstrained ==1 % Add unknown constraints here if exist
            g_unkn_fun = @(x) 0;
        else
            g_unkn_fun =@(x) 0;
        end
        
        if isUnknownSatisfactionConstrained % add the necessary eqns if relavent
            s_unkn_fun =@(x) 0;
        else
            s_unkn_fun =@(x) 0;
        end
            
        delta=1;
        maxevals= 20;
        nsamp=round(maxevals/3);   
        
        case 'camelsixhumps-constr'
        %CamelSixHumps function with known constraints
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
        
        Aineq=[1.6295 1;
            -1 4.4553;
            -4.3023 -1;
            -5.6905 -12.1374
            17.6198 1];
        
        bineq = [3.0786; 2.7417; -1.4909; 1; 32.5198];
        
        g=@(x) [(x(1)-0)^2+(x(2)+.1)^2-.5];
        
        isUnknownFeasibilityConstrained = 0; % isUnknownFeasibilityConstrained = 1, if unknown inequality feasibility constraints exist, isUnknownFeasibilityConstrained = 0, otherwise
        isUnknownSatisfactionConstrained = 0; %  isUnknownSatisfactionConstrained = 1, if unknown satisfactory constraints exist,  isUnknownSatisfactionConstrained = 0, otherwise
        if isUnknownFeasibilityConstrained ==1 % Add unknown constraints here if exist
            g_unkn_fun = @(x) 0;
        else
            g_unkn_fun =@(x) 0;
        end
        
        if isUnknownSatisfactionConstrained % add the necessary eqns if relavent
            s_unkn_fun =@(x) 0;
        else
            s_unkn_fun =@(x) 0;
        end
        
        delta=.3;
        maxevals=50;
        nsamp=round(maxevals/3);
end

if runGLISp
    if isUnknownSatisfactionConstrained && isUnknownFeasibilityConstrained 
        pref=@(x,y) glisp_function3(x,y,f,comparetol,g_unkn_fun,s_unkn_fun);  % Include query for both unknown feasibility and satisfactory constraints besides the preference query
    elseif ~isUnknownSatisfactionConstrained && isUnknownFeasibilityConstrained 
        pref=@(x,y) glisp_function2(x,y,f,comparetol,g_unkn_fun); % Inlude query for only feasibility constraints besides the preference query
    elseif isUnknownSatisfactionConstrained && ~isUnknownFeasibilityConstrained
        pref=@(x,y) glisp_function2(x,y,f,comparetol,s_unkn_fun); % Inlude query for only satisfactory constraints besides the preference query
    else
        pref=@(x,y) glisp_function1(x,y,f,comparetol,Aineq,bineq,g); % with only preference query
    end
end

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

opts.epsil=epsil;
%opts.rbf=@(x1,x2,epsil) exp(-(epsil^2*sum((x1-x2).^2))); % Gaussian RBF
opts.rbf=@(x1,x2,epsil) 1/(1+epsil^2*sum((x1-x2).^2)); % inverse quadratic
%opts.rbf=@(x1,x2,epsil) sqrt(1+epsil^2*sum((x1-x2).^2)); % multiquadric

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

if runGLIS
    opts.useRBF=1;
    opts.alpha=delta/5;
%     opts.delta=.5;
%         opts.maxevals=150;
    opts.rbf=@(x1,x2) opts.rbf(x1,x2,opts.epsil);
end

if Ntests>1
    bar_handle = waitbar(0,'');
end

FF=NaN(maxevals,Ntests);
xopt_Ntests = zeros(Ntests, nvars);
fopt_Ntests = zeros(Ntests, 1);
feas_unkn_Ntests = zeros(Ntests, 1);
fesseq_unkn_Ntest = zeros(Ntests, maxevals);
fes_first_unkn_Ntest = zeros(Ntests,1);
satConst_unkn_Ntests= zeros(Ntests, 1);
satConstseq_unkn_Ntest = zeros(Ntests, maxevals);
feas_comb_Ntests = zeros(Ntests,1);
feascombseq_unkn_Ntest = zeros(Ntests, maxevals);
ibest_Ntests = zeros(Ntests, 1);
ibestseq_Ntests = zeros(Ntests, maxevals);

for j=1:Ntests
    
    if Ntests>1
        waitbar((j-1)/Ntests,bar_handle,sprintf('%s - running test #%d/%d',benchmark,j,Ntests));
    end
    
    if runGLISp || runBayesopt
        if isUnknownSatisfactionConstrained && isUnknownFeasibilityConstrained 
            glisp_function3('clear');
        elseif ~isUnknownSatisfactionConstrained && isUnknownFeasibilityConstrained
            glisp_function2('clear');
        elseif isUnknownSatisfactionConstrained && ~isUnknownFeasibilityConstrained
            glisp_function2('clear');
        else
            glisp_function1('clear');
        end
        if runBayesopt
            [xbest,out]=bayesopt_pref(pref,lb,ub,opts);
        else
            [xbest,out]=glisp(pref,lb,ub,opts);
        end
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
    end 
    
    if runGLIS
        [xbest,~,out]=glis(f,lb,ub,opts,g_unkn_fun,s_unkn_fun);
        fbest=f(xbest(:)');
        X=out.X;
        F=out.F;
    end
    
    xopt_Ntests(j,:) = xbest;
    fopt_Ntests(j) = f(xbest');
    feas_unkn_Ntests(j) = out.fes_opt_unkn;
    satConst_unkn_Ntests(j) = out.satConst_opt_unkn;
    feas_comb_Ntests(j) = out.feas_opt_comb;
    ibest_Ntests(j) = out.ibest;
    ibestseq_Ntests(j,:) = out.ibestseq;
    fesseq_unkn_Ntest(j,:) = out.Feasibility_unkn;
    satConstseq_unkn_Ntest(j,:) = out.SatConst_unkn;
    feascombseq_unkn_Ntest(j,:) = out.isfeas_seq;
    fes_ind_j = find(fesseq_unkn_Ntest(j,:) >0);
    if numel(fes_ind_j) ==0
        fes_ind_j = maxevals;
    end
    fes_first_unkn_Ntest(j) = min(fes_ind_j);
    
    if runGLISp || runBayesopt
        F=zeros(maxevals,1);
        ff=zeros(size(X,1),1);
        for i = 1:maxevals 
            F(i)=f(X(i,:));
            ff(i) = F(ibestseq_Ntests(j,i));
        end
    else
        ff=zeros(size(X,1),1);
        for i = 1:maxevals 
            F(i)=f(X(i,:));
            ff(i) = F(ibestseq_Ntests(j,i));
        end
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

latentfun=@(x) f(x) + opts.constraint_penalty*(Aineqfun(x)+gfun(x)+g_unkn_fun(x)+s_unkn_fun(x));

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
        evalc('xopt0_globoptsol=PSwarm(pswarm_vars.Problem,pswarm_vars.InitialPopulation,pswarm_vars.Options);');
        
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
        xopt0_globoptsol=nlopt_optimize(opt,zeros(nvars,1));
        xopt0_globoptsol=xopt0_globoptsol(:);
end
fopt0_globoptsol=f(xopt0_globoptsol(:)');


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
end



if Ntests==1
    figure
%     Fm_iter=zeros(maxevals,1);

    c1=[ 0    0.4470    0.7410];
    c2=[ 0.8500    0.3250    0.0980];
    c3=[ 0   0.2    0.8];
    Fm_iter = FF;
    if fes_first_unkn_Ntest >1
        plot(1:fes_first_unkn_Ntest,Fm_iter(1:fes_first_unkn_Ntest),'LineWidth',2.5,'Color',c3);
        plot(fes_first_unkn_Ntest:maxevals,Fm_iter(fes_first_unkn_Ntest:maxevals),'LineWidth',2.5,'Color',c2);
    else
        plot(1:maxevals,Fm_iter,'LineWidth',2.5,'Color',c2);
    end
    hold on
    if isUnknownFeasibilityConstrained || isUnknownSatisfactionConstrained || ~isempty(Aineq)||~isempty(g)
        plot(1:maxevals,fopt_const*ones(maxevals,1),'--','Color',[0,0,0],'LineWidth',1.0);
    else
        plot(1:maxevals,fopt0*ones(maxevals,1),'--','Color',[0,0,0],'LineWidth',1.0);
    end
    hold off
    title(benchmark)
end


