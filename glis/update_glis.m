function [xnext,prob_setup] = update_glis(f_val, feasible, satisfactory)
% - Update the relevant variables w.r.t the newly queried sample
% - And then solve the optimization problem on the updated acquisition function to obtain the next point to query

global prob_setup

if nargin <3
    feasible = true;
    satisfactory = true;

end

prob_setup.F = [prob_setup.F; f_val];
f0 = f_val;
if prob_setup.isObjTransformed
    f0 = prob_setup.obj_transform(f_val);
    prob_setup.transformed_F = [prob_setup.transformed_F; f0];
end

prob_setup.UnknownFeasible = [prob_setup.UnknownFeasible;feasible];
prob_setup.UnknownSatisfactory= [prob_setup.UnknownSatisfactory;satisfactory];

x = prob_setup.xnext; % this was computed at the previous call after n_initial_random iterations

if prob_setup.iter < prob_setup.n_initial_random
    isfeas = prob_setup.KnownFeasible(prob_setup.iter);
    prob_setup.time_opt_acquisition = [prob_setup.time_opt_acquisition;0];
    prob_setup.time_fit_surrogate = [prob_setup.time_fit_surrogate;0];
else
    isfeas = true; % actively generated samples are always feasible wrt known constraints
    prob_setup.KnownFeasible = [prob_setup.KnownFeasible; isfeas];
end
if prob_setup.has_unknown_constraints
    isfeas = isfeas && feasible;
end
if isfeas && f_val < prob_setup.fbest
    prob_setup.fbest = f_val;
    prob_setup.ibest = prob_setup.iter;
    prob_setup.xbest = x;
end

prob_setup.ibest_seq = [prob_setup.ibest_seq; prob_setup.ibest];
prob_setup.fbest_seq = [prob_setup.fbest_seq;prob_setup.fbest];
prob_setup.isfeas_seq = [prob_setup.isfeas_seq; isfeas];
ind_feas = find(prob_setup.isfeas_seq);

if isfeas
    prob_setup.Fmax = max(prob_setup.Fmax, f0);
    prob_setup.Fmin = min(prob_setup.Fmin, f0);
end

if prob_setup.display
    if isfeas
        fprintf('N = %4d, best = %8g, current = %8g, x= [ ',prob_setup.iter,prob_setup.fbest,f_val);
    else
        fprintf('N = %4d, best = %8g, current = infeasible sample, x= [ ',prob_setup.iter,prob_setup.fbest);
    end
    for j=1:prob_setup.nvar
        fprintf('%5.4f ',x(j));
        if j < prob_setup.nvar - 1
            fprintf(", ");
        end
    end
    fprintf('] \n');
end

if prob_setup.iter == prob_setup.n_initial_random
    % Possibly remove rows/columns corresponding to infeasible samples
    % This step is necessary even when Unknown constraints are not present (for the case, when feasible_sampling = False)
    prob_setup.M = prob_setup.M(ind_feas,ind_feas);
end

if prob_setup.iter >= prob_setup.n_initial_random
    Xs_all = (prob_setup.X - ones(prob_setup.iter,1)*prob_setup.d0') ./ (ones(prob_setup.iter,1)*prob_setup.dd');

    delta_E = prob_setup.delta;

    if prob_setup.has_unknown_constraints
        delta_G_default = prob_setup.delta;
        delta_G = get_delta_adpt(Xs_all, prob_setup.UnknownFeasible, delta_G_default);
    else
        delta_G = 0.;
    end
    if prob_setup.has_satisfaction_fun
        delta_S_default = prob_setup.delta / 2.;
        delta_S = get_delta_adpt(Xs_all, prob_setup.UnknownSatisfactory, delta_S_default);
    else
        delta_S = 0.;
    end

   dF = prob_setup.Fmax - prob_setup.Fmin;
   if dF == -inf  % no feasible samples found so far
        dF_ = nan;
   else
        dF_ = max(dF,prob_setup.epsDeltaF);
   end

   if prob_setup.scale_delta && (numel(ind_feas) >0)
       d_ibest=sum(([Xs_all(1:prob_setup.ibest-1,:);Xs_all(prob_setup.ibest+1:prob_setup.iter,:)]-Xs_all(prob_setup.ibest,:)).^2,2); % exclude the ibest term in X when calculate d_ibest
        ii=find(d_ibest<1e-12,1);
        if ~isempty(ii)
            iw_ibest=0;
        else
            iw_ibest = sum(1./d_ibest);
        end
   else
       iw_ibest=0;
   end

   F_all = prob_setup.F;
   F = F_all(ind_feas);  % only keeps values f(x) corresponding to feasible samples x
   Xs = Xs_all(ind_feas, :);  % RBF or IDW only defined wrt feasible samples in Xs

   % Update RBF matrix M
   if prob_setup.iter >= prob_setup.n_initial_random
       if prob_setup.useRBF && isfeas
           N = numel(ind_feas);
           prob_setup.M(N, 1:N) = 0; 
           prob_setup.M(1:N, N) = 0;
           % Just update last row and column of M
           for h = 1:N-1
               mij=prob_setup.rbf(Xs(h,:),Xs(N,:),prob_setup.rbf_epsil);
               prob_setup.M(h,N)=mij;
               prob_setup.M(N,h)=mij;
           end 
           prob_setup.M(N,N)=1.0;
       end

       if prob_setup.useRBF && prob_setup.has_unknown_constraints
           N = prob_setup.iter;
           prob_setup.M_unkn(N, 1:N) = 0;
           prob_setup.M_unkn(1:N, N) = 0;
           % Just update last row and column of M
           for h = 1:N-1
               mij=prob_setup.rbf(Xs_all(h,:),Xs_all(N,:),prob_setup.rbf_epsil);
               prob_setup.M_unkn(h,N)=mij;
               prob_setup.M_unkn(N,h)=mij;
           end 
           prob_setup.M_unkn(N,N)=1.0;
       end
   end

   tic
   if prob_setup.useRBF
       % update weights using current F and matrix M (only consider the feasible samples)
       W=get_rbf_weights(prob_setup.M, F,prob_setup.svdtol);
   else
       W = zeros(prob_setup.iter,1);
   end
   prob_setup.time_fit_surrogate = [prob_setup.time_fit_surrogate;toc];

   % Related to unknown constraints
   F_unkn = F_all;
   if prob_setup.useRBF && prob_setup.has_unknown_constraints
       ind_infeas = find(~prob_setup.isfeas_seq);
       F_unkn(ind_infeas) = ones(numel(ind_infeas),1)*prob_setup.rhoC * dF_; % for infeasible ones, penalty values are assigned to the fun. eval
       W_unkn = get_rbf_weights(prob_setup.M_unkn,F_unkn,prob_setup.svdtol); % update weights using current F and matrix M (consider all the samples)
   else
      W_unkn = zeros(prob_setup.iter,1); 
   end

  acquisition=@(xs,p) facquisition(xs(:)',Xs,F,Xs_all,F_all,prob_setup.useRBF,prob_setup.rbf,prob_setup.rbf_epsil,W,delta_E,dF_,delta_G, delta_S, ...
                                            prob_setup.scale_delta, prob_setup.iter,prob_setup.expected_max_evals, prob_setup.alpha,iw_ibest, ...
                                            prob_setup.has_unknown_constraints, prob_setup.has_satisfaction_fun,...
                                            prob_setup.UnknownFeasible, prob_setup.UnknownSatisfactory,...
                                            prob_setup.isfeas_seq, prob_setup.rhoC,...
                                            W_unkn, F_unkn) +...
                                            dF_*prob_setup.constrpenalty(xs(:));
    
  tic
    switch prob_setup.globoptsol
    case 'pswarm'
       pswarm_vars = prob_setup.pswarm_vars;
       pswarm_vars.Problem.ObjFunction= @(xs) facquisition(xs(:)',Xs,F,Xs_all,F_all,prob_setup.useRBF,prob_setup.rbf,prob_setup.rbf_epsil,W,delta_E,dF_,delta_G, delta_S, ...
                                            prob_setup.scale_delta, prob_setup.iter,prob_setup.expected_max_evals, prob_setup.alpha,iw_ibest, ...
                                            prob_setup.has_unknown_constraints, prob_setup.has_satisfaction_fun,...
                                            prob_setup.UnknownFeasible, prob_setup.UnknownSatisfactory,...
                                            prob_setup.isfeas_seq, prob_setup.rhoC,...
                                            W_unkn, F_unkn) +...
                                            dF_*prob_setup.constrpenalty(xs(:));
       evalc('z=PSwarm(pswarm_vars.Problem,pswarm_vars.InitialPopulation,pswarm_vars.Options);');
        
    case 'direct'
       direct_vars = prob_setup.direct_vars;
       direct_vars.opt.min_objective = acquisition;
       zold=prob_setup.z;
       z=nlopt_optimize(direct_vars.opt,zold);
       z=z(:);
    
    case {'tmw-pso','tmw-ga'}
       lb2=lb;
       ub2=ub;
       if scalevars
           lb2=-ones(nvar,1);
           ub2=ones(nvar,1);
       end
       if strcmp(globoptsol,'tmw-pso')
           z=particleswarm(acquisition,nvar,lb2,ub2,pswarm_vars.Options);
       else
           z=ga(acquisition,nvar,[],[],[],[],lb2,ub2,[],pswarm_vars.Options);
       end
       z=z(:);
    end

    prob_setup.time_opt_acquisition = [prob_setup.time_opt_acquisition;toc];

    xsnext = z;
    prob_setup.z = z;
    prob_setup.xnext = xsnext .* prob_setup.dd + prob_setup.d0;
    prob_setup.X = [prob_setup.X;prob_setup.xnext'];

else
    prob_setup.xnext = prob_setup.X(prob_setup.iter + 1,:)';
end

prob_setup.iter = prob_setup.iter + 1;

xnext = prob_setup.xnext';

end

%%%%%%%%%%%%%%%%%%%%%%
function W=get_rbf_weights(M,F,svdtol)
% Solve M*W = F using SVD

[U,S,V]=svd(M);
dS=diag(S);
ns=find(dS>=svdtol,1,'last');
if ~isempty(ns)
    W=V(:,1:ns)*diag(1./dS(1:ns))*U(:,1:ns)'*F;
else
    W = [];
end
end

%%%%%%%%%%%%%%%%%%%%%%
function [f,fhat,dhat]=facquisition(xs, Xs, F, Xs_all, F_all, useRBF, rbf, rbf_epsil, W, delta_E, dF,...
                 delta_G, delta_S, scale_delta, N, maxevals, alpha, iw_ibest,...
                 has_unknown_constraints, has_satisfaction_fun, UnknownFeasible, UnknownSatisfactory,...
                 isfeas_seq, constrpenalty_value, W_unkn, F_unkn)

% Acquisition function to minimize to get next sample
m=size(xs,1); % number of points x to evaluate the acquisition function
f=zeros(m,1);

for i=1:m
    xx=xs(i,:)';
   
    N_fes=size(Xs,1);
    d=sum((Xs-ones(N_fes,1)*xx').^2,2);

    if useRBF
        rbf_xs = rbf(Xs,xx',rbf_epsil);
    else
        rbf_xs = 0.0;
    end
      
    if useRBF && has_unknown_constraints
        rbf_xs_unkn = rbf(Xs_all,xx',rbf_epsil);
    else
        rbf_xs_unkn = 0.0;
    end
    
    if all(isfeas_seq) % if samples are all feasible
        d_all = d;
    else
        % to account for all X that have been sampled (including the infeasible ones,
        % since the distance info is used to estimate the probability of feasibility)
        d_all = sum((Xs_all-ones(N,1)*xx').^2,2);
    end

    ii_=find(d_all<1e-12);
    if ~isempty(ii_)
        fhat=F_all(ii_(1));
        fhat_unkn = F_unkn(ii_(1));
        if ~isfeas_seq(ii_(1))
            fhat = constrpenalty_value*dF;
        end
        dhat=0;
        if has_unknown_constraints
            Ghat=UnknownFeasible(ii_(1));
        else
            Ghat=1;
        end
        if has_satisfaction_fun
            Shat=UnknownSatisfactory(ii_(1));
        else
            Shat=1;
        end
    else
        if isempty(W)
            fhat = 0.;
            w = 0.;
            sw = 0.;
            aux = 0.;

            if has_unknown_constraints
                w_unkn = exp(-d_all)./d_all;
                sw_unkn = sum(w_unkn);
                if useRBF
                    v_infes = rbf_xs_unkn;
                    fhat_unkn = sum(v_infes.*W_unkn);
                else
                    fhat_unkn = sum(F_unkn.* w_unkn) / sw_unkn;
                end
            else
                fhat_unkn = 0.;
            end
        else
            w=exp(-d)./d;
            sw=sum(w);
            aux = 1./sum(1./d);

            if useRBF
                v = rbf_xs;
                fhat = sum(v'*W);
            else
                fhat = sum(F'*w)/sw;
            end

            if has_unknown_constraints
                w_unkn = exp(-d_all)./d_all;
                sw_unkn = sum(w_unkn);
                if useRBF
                    v_infes = rbf_xs_unkn;
                    fhat_unkn = sum(v_infes.*W_unkn);
                else
                    fhat_unkn = sum(F_unkn .* w_unkn) / sw_unkn;
                end
            else
                fhat_unkn = 0.;
            end
        end

        if all(isfeas_seq)
            w_all = w;
            sw_all = sw;
            aux_all = aux;
        else
            w_all = exp(-d_all)./d_all;
            sw_all = sum(w_all);
            aux_all = 1./sum(1./d_all);
        end

        % when considering the IDW exploration function, take all the points sampled into account
        if ~scale_delta
            dhat = delta_E * atan(aux_all);
            if ~isempty(W)
                dhat = dhat * 2/pi*dF + alpha * sqrt(sum(w.*(F-fhat).^2)/sw);
            end
        else
            dhat = delta_E * ((1-N/maxevals)*atan(aux_all * iw_ibest)+ N/maxevals *atan(aux_all));
            if ~isempty(W)
                dhat = dhat * 2/pi*dF + alpha * sqrt(sum(w.*(F-fhat).^2)/sw);
            end
        end

        if has_unknown_constraints
            Ghat = sum(UnknownFeasible'*w_all)/sw_all;
        else
            Ghat = 1;
        end 

        if has_satisfaction_fun
            Shat=sum(UnknownSatisfactory'*w_all)/sw_all;
        else
            Shat = 1;
        end
    end

    f(i)=fhat-dhat+(delta_G*(1-Ghat)+delta_S*(1-Shat))*fhat_unkn;

end

end