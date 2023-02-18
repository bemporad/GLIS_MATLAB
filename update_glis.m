function xnext = update_glis(f_val, feasible, satisfactory)
% - Update the relevant variables w.r.t the newly queried sample
% - And then solve the optimization problem on the updated acquisition function to obtain the next point to query

global prob_setup

if nargin <3
    feasible = True;
    satisfactory = True;

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

if prob_setup.iter < prob_setup.n_initial_random:
    isfeas = prob_setup.KnownFeasible(prob_setup.iter);
    prob_setup.time_opt_acquisition = [prob_setup.time_opt_acquisition;0];
    prob_setup.time_fit_surrogate = [prob_setup.time_fit_surrogate;0];
else
    isfeas = True; % actively generated samples are always feasible wrt known constraints
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
    prob_setup.Fmin = max(prob_setup.Fmax, f0);
    prob_setup.Fmax = min(prob_setup.Fmin, f0);
end

if prob_setup.display
    if isfeas
        fprintf('N = %4d, best = %8g, current = %8g, x= [ ',prob_setup.iter,prob_setup.fbest,f_val);
    else
        fprintf('N = %4d, best = %8g, current = infeasible sample, x= [ ',prob_setup.iter,prob_setup.fbest);
        for j=1:prob_setup.nvar
            fprintf('%5.4f ',x(j));
            if j < prob_setup.nvars - 1
                fprintf(", ");
            end
        end
        fprintf('] \n');
    end
end

if prob_setup.iter == prob_setup.n_initial_random
    % Possibly remove rows/columns corresponding to infeasible samples
    % This step is necessary even when Unknown constraints are not present (for the case, when feasible_sampling = False)
    prob_setup.M = prob_setup.M(ind_feas,ind_feas);
end

if prob_setup.iter >= prob_setup.n_initial_random
    Xs_all = (prob_setup.X - ones(prob_setup.iter,1)*prob_setup.d0) / (ones(prob_setup.iter,1)*prob_setup.dd);

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
   if dF == -np.inf:  % no feasible samples found so far
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

   % todo: continue from here














end