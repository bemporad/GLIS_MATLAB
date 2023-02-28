function [xbest,xnext] = initialize_glisp(lb,ub,opts)
% Initialize the problem
%             - obtain the initial samples to query
%             - preallocate the RBF coefficient matrix for the initial samples uf RBF surrogate is used
%  (C) 2019-2023 Alberto Bemporad, Mengjia Zhu

global prob_setup

Xs = glis_init(lb, ub, opts);

% parameters related to glisp
if ~isfield(opts,'sepvalue') || isempty(opts.sepvalue)
    prob_setup.sepvalue=1. / prob_setup.expected_max_evals;
else
    prob_setup.sepvalue=opts.sepvalue;
end

if ~isfield(opts,'RBFcalibrate') || isempty(opts.RBFcalibrate)
    prob_setup.RBFcalibrate=false;
else
    prob_setup.RBFcalibrate=opts.RBFcalibrate;
end

if prob_setup.RBFcalibrate
    if ~isfield(opts,'thetas') || isempty(opts.thetas)
        thetas=logspace(-1,1,11);thetas=thetas(1:end-1);
        prob_setup.thetas = thetas;
        prob_setup.itheta=find(abs(prob_setup.thetas-1)<=1e-14);
    else
        prob_setup.thetas=opts.thetas;
        prob_setup.itheta=find(abs(prob_setup.thetas-1)<=1e-14);
        if isempty(prob_setup.itheta)
            error('At least one element in thetas must be equal to 1');
        end
        prob_setup.thetas(prob_setup.itheta)=1;
    end
end

prob_setup.theta = prob_setup.thetas(prob_setup.itheta);
prob_setup.MM=zeros(prob_setup.n_initial_random,prob_setup.n_initial_random,numel(prob_setup.thetas));
prob_setup.iM = 0; % index denoting the portion of MM already computed
M = prob_setup.MM(1:prob_setup.n_initial_random,1:prob_setup.n_initial_random,prob_setup.itheta);
for i=1:prob_setup.n_initial_random
    for j=i:prob_setup.n_initial_random
        mij=prob_setup.rbf(Xs(i,:),Xs(j,:),prob_setup.rbf_epsil);
        M(i,j)=mij;
        M(j,i)=mij;
    end
end
prob_setup.MM(1:prob_setup.n_initial_random,1:prob_setup.n_initial_random,prob_setup.itheta) = M;

if ~isfield(opts,'RBFcalibrationSteps') || isempty(opts.RBFcalibrationSteps)
    prob_setup.RBFcalibrationSteps=prob_setup.n_initial_random+round([0;(prob_setup.expected_max_evals-prob_setup.n_initial_random)/4;
    (prob_setup.expected_max_evals-prob_setup.n_initial_random)/2;3*(prob_setup.expected_max_evals-prob_setup.n_initial_random)/4]);
else
    prob_setup.RBFcalibrationSteps=opts.RBFcalibrationSteps;
end

if ~isfield(opts,'acquisition_method') || isempty(opts.acquisition_method)
    prob_setup.acquisition_method=1;
else
    prob_setup.acquisition_method=opts.acquisition_method;
end

if prob_setup.isObjTransformed
    warning("This is preference-based optimization, argument 'obj_transform' ignored");
end

if ~prob_setup.useRBF
    error("IDW not supported in GLISp, only RBF surrogates")
end

if ~ ((prob_setup.acquisition_method==1)  || (prob_setup.acquisition_method==2))
    error("Supported acquisition methods are 1 (scaled surrogate - delta * IDW) and 2 (probability of improvement")
end

if (prob_setup.isLinConstrained || prob_setup.isNLConstrained) && ~prob_setup.feasible_sampling
    error("Must generate feasible initial samples, please set 'feasible_sampling = true'")
end

prob_setup.I = [];
prob_setup.Ieq = [];

% remove self-calibration steps smaller than n_initial_random
prob_setup.RBFcalibrationSteps = prob_setup.RBFcalibrationSteps(prob_setup.RBFcalibrationSteps >= prob_setup.n_initial_random);

xbest = prob_setup.X(1,:);
prob_setup.xbest = xbest;
prob_setup.ibest = 1;
prob_setup.ibest_seq = [prob_setup.ibest_seq;prob_setup.ibest];
xnext = prob_setup.X(2,:);
prob_setup.xnext = xnext;  % initial sample to query (unscaled)
prob_setup.iter = 2;


end
