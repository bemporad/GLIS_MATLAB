function xnext = initialize(lb,ub,opts)
% Initialize the problem
%             - obtain the initial samples to query
%             - preallocate the RBF coefficient matrix for the initial samples uf RBF surrogate is used
%  (C) 2019-2023 Alberto Bemporad, Mengjia Zhu

global prob_setup

Xs = glis_init(lb, ub, opts);
if prob_setup.useRBF
    prob_setup.M = zeros(prob_setup.n_initial_random,prob_setup.n_initial_random); % preallocate the entire matrix
    for i=1:prob_setup.n_initial_random
        for j=i:prob_setup.n_initial_random
            mij=prob_setup.rbf(Xs(i,:),Xs(j,:),prob_setup.rbf_epsil);
            prob_setup.M(i,j)=mij;
            prob_setup.M(j,i)=mij;
        end
    end
end
  
if prob_setup.has_unknown_constraints
    prob_setup.M_unkn = prob_setup.M;
end

xnext = prob_setup.X(1,:);
prob_setup.xnext = xnext;  % initial sample to query (unscaled)
prob_setup.isInitialized = true;



end