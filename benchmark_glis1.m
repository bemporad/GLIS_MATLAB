% (C) 2019-2023 Alberto Bemporad, Mengjia Zhu

clear all
close all

rng(2) % for repeatability

benchmark='ackley';
% benchmark='camelsixhumps';
% benchmark='hartman6';
% benchmark='rosenbrock8';


switch benchmark
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
end

clear opts
opts.maxevals=maxevals;
opts.alpha=1.0/nvars; % weight on variance
opts.delta=2.0/nvars; % weight on distance
opts.n_initial_random=2*numel(lb);
opts.svdtol=1e-6;

opts.globoptsol='pswarm';
%opts.globoptsol='tmw-pso';
%opts.globoptsol='tmw-ga'; % this runs slower
%opts.globoptsol='direct'; % faster, but may provide more suboptimal solutions
opts.display=1;
opts.scale_delta = false;
opts.feasible_sampling=true;


fprintf("Solve the problem by feeding the simulator/fun directly into the GLIS solver \n")
[xopt1, fopt1,prob_setup] = solve_glis(fun,lb,ub,opts);


fprintf("Solve the problem incrementally (i.e., provide the function evaluation at each iteration) \n")
rng(2)
x_= initialize(lb,ub,opts); % x is unscaled
for k = 1: maxevals
    f_val = fun(x_);
    [x_, prob_setup] = update_glis(f_val);
end
xopt2 = prob_setup.xbest;
fopt2 = prob_setup.fbest;


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
    plot(xopt1(1),xopt1(2),'g*','linewidth',2);
    
    hold off
end

figure
if ~strcmp(benchmark,'rosenbrock8')
    plot(1:nn,prob_setup.fbest_seq,1:nn,ones(1,nn)*fopt0,'--','linewidth',1.5);
else
    semilogy(1:nn,prob_setup.fbest_seq,1:nn,ones(1,nn)*fopt0,'--','linewidth',1.5);
end
grid
title('Best function value found')
xlabel('number of function evaluations')
set(gcf,'Tag','MinFun');
drawnow


