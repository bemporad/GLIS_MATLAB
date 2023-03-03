% Test preference-based optimization based on RBF+IDW on
% multiobjective optimization problem.
%
% For comparison, one can also use this file to solve the benchmarks with
% Preference-based Bayesian optimization by setting run_bayesopt = 1
%
% (C) 2019 A. Bemporad, September 22, 2019
% updated by M. Zhu, March 3, 2023


clear all
close all

addpath(genpath('.././glis'))

run_bayesopt=1;
run_glisp=1;

rng(0); % reset counter (for replicability of results)

Ntests=1; % number of tests to repeat with different random seends
RBFcalibrate=1; % 1 = recalibrate epsilon param in RBF during iterations

acquisition_method=1; % 1 = IDW acquisition function, 2 = prob. improvement
maxevals=80;
delta=1; % not used when acquisition_method=2
nsamp=ceil(maxevals/3);
epsil=1;
thetas=logspace(-1,1,11);thetas=thetas(1:end-1);
comparetol=1e-4;
sepvalue=1/maxevals;

RBFcalibrationSteps=[]; % default

nz=3; % number of optimization variables in multi-objective problem

% multi-objective function
F = @(z) [2*z(1)*sin(z(2))-3*cos(z(1)*z(2));
    z(3)*(z(1)+z(2))^2;
    z(1)+z(2)+z(3)].^2;

lbz=-ones(nz,1); % upper and lower bounds on z
ubz=ones(nz,1);

clear nlpopt
nlpopt.ftol_rel=1e-5;
nlpopt.ftol_abs=1e-5;
nlpopt.xtol_abs=1e-5*ones(nz,1);
nlpopt.xtol_rel=1e-5;
nlpopt.verbose = 0;
nlpopt.maxeval=50000;
nlpopt.lower_bounds=lbz;
nlpopt.upper_bounds=ubz;
try
    nlpopt.algorithm=NLOPT_GN_DIRECT;
    %nlpopt.algorithm=NLOPT_GN_DIRECT_L;
catch
    error(sprintf('NLOPT toolbox not installed or in MATLAB path\nPlease download from http://nlopt.readthedocs.io)'));
end

nvars=numel(F(zeros(nz,1)))-1;
f=@(x) desired_pareto(x,F,lbz,ubz,nz,nvars,nlpopt);

lb=zeros(nvars,1);
ub=ones(nvars,1); % without loss of generality
Aineq=ones(1,nvars); % x(nvars+1) = 1 - sum(x(i)) >= 0
bineq=1;
g=[];

Fm=zeros(maxevals,1,3);
Fm_min=Inf(maxevals,1,3);
Fm_max=-Inf(maxevals,1,3);

FF=NaN(maxevals,Ntests,3);
XOPT=NaN(Ntests,nvars,3);
xbest=zeros(nvars,1);
fbest=Inf;

close all
c1=[0.4940    0.1840    0.5560];
c2=[0 0 0];
labels={'GLISp','PBO'};


%%%%%%%%%%%%%%%%%%%%
% Preference-based optimization

opts=[];

opts.rbf_epsil=epsil;
opts.rbf="inverse_quadratic"; % Radial Basis Functions
opts.thetas=thetas;

opts.sepvalue=sepvalue;
opts.rho=0e-3; % noise of function evaluation, impacting comparison
%opts.phi=phi;
opts.delta=delta;
opts.n_initial_random=nsamp;
% opts.stoptol=stoptol;
% opts.opttol=1e-4;
opts.maxevals=maxevals;
opts.feasible_sampling=true;
opts.acquisition_method=acquisition_method;


opts.stoptol=-1e-4; % do not stop if ||z(k)-z(k-1)||>stoptol
opts.ftol=-1e-4; % do not stop if |f(k)-f(k-1)|>ftol
opts.maxevals=maxevals;
opts.rho=0e-3; % simulated noise on function evaluations

%opts.globoptsol='direct';
opts.globoptsol='pswarm';

opts.display=1;
opts.scalevars=1;

opts.Aineq=Aineq;
opts.bineq=bineq;
opts.g=g;
opts.constraint_penalty=1000;

opts.RBFcalibrate=RBFcalibrate;
opts.RBFcalibrationSteps=RBFcalibrationSteps;

if Ntests>1
    bar_handle = waitbar(0,'');
end

pref=@(x,y) glisp_function1(x,y,f,comparetol,[],[],[]);

for test=1:Ntests
    
    if Ntests>1
        waitbar((test-1)/Ntests,bar_handle,sprintf('Running test #%d/%d',test,Ntests));
    end
    
    if run_glisp
        rng(test); % for reproducibility and same initial samples
        glisp_function1('clear');
        [xopt,outpref] = solve_glisp(pref, lb,ub,opts);
        FFF=zeros(maxevals,1);
        for i=1:maxevals
            FFF(i)=glisp_function1('get',outpref.X(i,:)',f);
        end
        FF(:,test,1)=FFF;
        XOPT(test,:,1)=xopt;
        fopt=FF(outpref.ibest,test,1);
        if fopt<fbest
            fbest=fopt;
            xbest=xopt;
        end
    end
    
    if run_bayesopt
        rng(test); % for reproducibility and same initial samples
        glisp_function1('clear');
        [xopt,outpref]=bayesopt_pref(pref,lb,ub,opts);
        for i=1:maxevals
            FFF(i)=glisp_function1('get',outpref.X(i,:)',f);
        end
        FF(:,test,2)=FFF;
        XOPT(test,:,2)=xopt;
        fopt=FF(outpref.ibest,test,2);
        if fopt<fbest
            fbest=fopt;
            xbest=xopt;
        end
    end
end

for i=1:size(FF,3)
    for j=1:maxevals
        vec=zeros(Ntests,1);
        for h=1:Ntests
            vec(h)=min(FF(1:j,h,i));
        end
        %Fm(j,i)=sum(vec)/Ntests; % mean
        aux=sort(vec);
        Fm_min(j,i)=aux(1);
        Fm_max(j,i)=aux(Ntests);
        % compute median:
        if rem(Ntests,2)
            Fm(j,i)=aux((Ntests+1)/2);
        else
            Fm(j,i)=(aux(Ntests/2)+aux(Ntests/2+1))/2;
        end
    end
end

if Ntests>1
    close(bar_handle);
end

% Plot results
figure
plot_results(maxevals,nsamp,Fm,Fm_min,Fm_max,fbest,1,2,c1,c2,labels);


function plot_results(maxevals,nsamp,Fm,Fm_min,Fm_max,fopt,i1,i2,c1,c2,labels)

plot(0:maxevals-1,Fm(1:maxevals,i1),'LineWidth',2,'Color',c1);
hold on
plot(0:maxevals-1,Fm(1:maxevals,i2),'LineWidth',2,'Color',c2);
plot(0:maxevals-1,fopt*ones(maxevals,1),'--','Color',[0.8500    0.3250    0.0980;],...
    'LineWidth',0.5);

h1=patch([0:maxevals-1,maxevals-1:-1:0]',[Fm_min(1:maxevals,i1);Fm_max(maxevals:-1:1,i1)],c1);
set(h1,'FaceAlpha',0.2,'EdgeAlpha',0);
h1=patch([0:maxevals-1,maxevals-1:-1:0]',[Fm_min(1:maxevals,i2);Fm_max(maxevals:-1:1,i2)],c2);
set(h1,'FaceAlpha',0.2,'EdgeAlpha',0);
ax=axis;
h1=patch([0:nsamp-1,nsamp-1:-1:0]',[ax(3)*ones(1,nsamp) ax(4)*ones(1,nsamp)],[.5 .5 .5]);
set(h1,'FaceAlpha',0.2,'EdgeAlpha',0);
plot([nsamp-1,nsamp-1],ax(3:4),'LineWidth',1.0,'Color',[.5 .5 .5]);

plot(0:maxevals-1,Fm_min(1:maxevals,i1),'--','LineWidth',0.5,'Color',c1);
plot(0:maxevals-1,Fm_max(1:maxevals,i1),'--','LineWidth',0.5,'Color',c1);

plot(0:maxevals-1,Fm_min(1:maxevals,i2),'--','LineWidth',0.5,'Color',c2);
plot(0:maxevals-1,Fm_max(1:maxevals,i2),'--','LineWidth',0.5,'Color',c2);

%set(gca,'children',flipud(get(gca,'children')));
grid
hold off
ax=axis;
axis([1 maxevals ax(3) ax(4)]);
title('multi-objective optimization - latent function','Interpreter','LaTeX');
legend(labels([i1 i2]));
xlabel('preference queries','Interpreter','LaTeX');
set(gcf,'Position',[70 650 780 340]);

drawnow

end

%%%%%%%%%%%%%%%%%%%%
function [y,zopt]=desired_pareto(x,F,lbz,ubz,nz,nx,nlpopt)

fun = @(z) sum([x(:);1-sum(x)].*F(z));
nlpopt.min_objective = fun;
zopt=nlopt_optimize(nlpopt,zeros(nz,1));
Fopt=F(zopt(:));
%y=(Fopt(2)-.1)^2;
y=sqrt((Fopt(1)-Fopt(2))^2+(Fopt(2)-Fopt(3))^2+(Fopt(1)-Fopt(3))^2);

end

