% (C) 2019-2023 Alberto Bemporad, Mengjia Zhu

clear all
close all

addpath(genpath('./glis'))

rng(2) % for repeatability

benchmark='camelsixhumps';


switch benchmark
    case 'camelsixhumps'
        %CamelSixHumps function
        nvars = 2;
        lb=[-2;-1];
        ub=[2;1];
        fun = @(x) (4-2.1*x(:,1).^2+x(:,1).^4/3).*x(:,1).^2+...
            x(:,1).*x(:,2)+(4*x(:,2).^2-4).*x(:,2).^2;
        maxevals=25;
        nsamp=round(maxevals/3);
        xopt0=[0.0898 -0.0898;
            -0.7126 0.7126]; % unconstrained optimizers, one per column
        fopt0=-1.0316; % unconstrained optimum
        xopt0_const = [0.21305 0.57424]; % constrained optimizer
        fopt0_const = -0.58445;  % constrained optimum
        use_linear_constraints=1;
        use_nl_constraints=1;

        if use_nl_constraints
            g=@(x) x(1)^2 + (x(2)+0.1)^2 - 0.5;
        end
        if use_linear_constraints
            Aineq=[1.6295    1.0000;
                -1.0000    4.4553;
                -4.3023   -1.0000;
                -5.6905  -12.1374;
                17.6198    1.0000];
            
            bineq=[3.0786;
                2.7417;
                -1.4909;
                1.0000;
                32.5198];
        end
        delta=1;  
end

clear opts

epsil=1;
thetas=logspace(-1,1,11);thetas=thetas(1:end-1);

sepvalue=1/maxevals;
comparetol=1e-4;
RBFcalibrate=1; % recalibrate parameters during optimization
acquisition_method=1; % acquisition method for RBF-based preference learning

opts=[];
opts.sepvalue=sepvalue;
opts.delta=delta;
opts.n_initial_random=nsamp;
opts.maxevals=maxevals;
opts.feasible_sampling=true;
opts.RBFcalibrate=RBFcalibrate;
opts.thetas=thetas;
opts.acquisition_method=acquisition_method;
opts.comparetol = comparetol;

opts.rbf_epsil=epsil;
opts.rbf="inverse_quadratic"; % Radial Basis Functions

opts.maxevals=maxevals;

%opts.globoptsol='direct';
opts.globoptsol='pswarm';

opts.display=1;
opts.scalevars=1;

opts.Aineq=Aineq;
opts.bineq=bineq;
opts.g=g;
opts.constraint_penalty=1e5;

pref=@(x,y) glisp_function1(x,y,fun,comparetol,Aineq,bineq,g);
glisp_function1('clear');

[xopt1, prob_setup] = solve_glisp(pref, lb,ub,opts);


% figures
nn=opts.maxevals;
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
    plot(prob_setup.X(1:opts.n_initial_random,1),prob_setup.X(1:opts.n_initial_random,2),'o','linewidth',1.5,'Color',[1, 0.5, 0]);
    plot(prob_setup.X(opts.n_initial_random+1:end,1),prob_setup.X(opts.n_initial_random+1:end,2),'bo','linewidth',1.5);
    for j=1:size(xopt0,2)
        plot(xopt0(1,j),xopt0(2,j),'dr','linewidth',2);
    end
    plot(xopt0_const(1),xopt0_const(2),'r*','linewidth',2);
    plot(xopt1(1),xopt1(2),'g*','linewidth',2);
    
    if use_nl_constraints
        % Plot constraints defined by function opts.g
        C1=cos(0:.05:2*pi);
        S1=sin(0:.05:2*pi);
        X1=0+C1*sqrt(.5);
        Y1=-.1+S1*sqrt(.5);
        plot(X1,Y1,'linewidth',1.5);
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
xbest_seq = prob_setup.X(prob_setup.ibest_seq,:);
fbest_seq = zeros(maxevals,1);
for i=1:size(prob_setup.X,1)
    fbest_seq(i) = fun(xbest_seq(i,:));
end
plot(1:nn,fbest_seq,1:nn,ones(1,nn)*fopt0_const,'--','linewidth',1.5);

grid
title('Best function value found')
xlabel('number of function evaluations')
set(gcf,'Tag','MinFun');
drawnow


