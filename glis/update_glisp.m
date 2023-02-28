function [xnext,prob_setup] = update_glisp(pref_val, feasible, satisfactory)
% Update the relevant variables w.r.t the newly queried sample
% And then solve the optimization problem on the updated acquisition function to obtain the next point to query

global prob_setup

if nargin <3
    feasible = true;
    satisfactory = true;

end

x = prob_setup.xnext;  % this was computed at the previous call after n_initial_random iterations
N = prob_setup.iter;  % current sample being examined

if prob_setup.iter < prob_setup.n_initial_random
    isfeas = prob_setup.KnownFeasible(prob_setup.iter);
    prob_setup.time_opt_acquisition = [prob_setup.time_opt_acquisition;0];
    prob_setup.time_fit_surrogate = [prob_setup.time_fit_surrogate;0];
else
    isfeas = true;  % actively generated samples are always feasible wrt known constraints
    prob_setup.KnownFeasible = [prob_setup.KnownFeasible; isfeas];
end

if prob_setup.has_unknown_constraints
    isfeas = isfeas && feasible;
end

prob_setup.isfeas_seq = [prob_setup.isfeas_seq; isfeas];
prob_setup.UnknownFeasible = [prob_setup.UnknownFeasible;feasible];
prob_setup.UnknownSatisfactory= [prob_setup.UnknownSatisfactory;satisfactory];

if pref_val == -1  % the expert has decided the preference, no matter what are unknown constraints/satisfactory values
    % update optimal solution
    prob_setup.I = [prob_setup.I;N, prob_setup.ibest];
    prob_setup.ibest = N;
    prob_setup.xbest = x;
elseif pref_val == 1
    prob_setup.I = [prob_setup.I;prob_setup.ibest, N];
else
    prob_setup.Ieq = [prob_setup.Ieq; N, prob_setup.ibest];
end
prob_setup.ibest_seq = [prob_setup.ibest_seq; prob_setup.ibest];

if prob_setup.display
    if prob_setup.ibest == N
        txt = '(***improved x!)';
    else
        txt = '(no improvement)';
    end
    fprintf("Query #%3d %s: testing x = [", N-1, txt);
    for j=1:prob_setup.nvar
        fprintf('%5.4f ',x(j));
        if j < prob_setup.nvar - 1
            fprintf(", ");
        end
    end
    fprintf('] \n');
end

if prob_setup.iter >= prob_setup.n_initial_random
    % Active sampling: prepare vector xnext to query
    Xs = (prob_setup.X - ones(prob_setup.iter,1)*prob_setup.d0') ./ (ones(prob_setup.iter,1)*prob_setup.dd');

    delta_E = prob_setup.delta;

    if prob_setup.RBFcalibrate && any(N==prob_setup.RBFcalibrationSteps)
        rbf_calibrate(Xs)
    end

    if prob_setup.has_unknown_constraints
        delta_G_default = prob_setup.delta;
        delta_G = get_delta_adpt(Xs, prob_setup.UnknownFeasible, delta_G_default);
    else
        delta_G = 0.;
    end
    if prob_setup.has_satisfaction_fun
        delta_S_default = prob_setup.delta / 2.;
        delta_S = get_delta_adpt(Xs, prob_setup.UnknownSatisfactory, delta_S_default);
    else
        delta_S = 0.;
    end

    M = prob_setup.MM(:,:,prob_setup.itheta); % current RBF matrix
    
    tic
    % update weights associated with RBF matrix M and current preference info
    W=get_rbf_weights_pref(M,N,prob_setup.I,prob_setup.Ieq,prob_setup.ibest);
    prob_setup.time_fit_surrogate = [prob_setup.time_fit_surrogate;toc];

    % Compute range of current surrogate function
    FH=M*W; % surrogate at current samples
    dF=max(max(FH)-min(FH),prob_setup.epsDeltaF);

    if prob_setup.scale_delta && (numel(ind_feas) >0)
       d_ibest=sum(([Xs(1:prob_setup.ibest-1,:);Xs(prob_setup.ibest+1:prob_setup.iter,:)]-Xs(prob_setup.ibest,:)).^2,2); % exclude the ibest term in X when calculate d_ibest
        ii=find(d_ibest<1e-12,1);
        if ~isempty(ii)
            iw_ibest=0;
        else
            iw_ibest = sum(1./d_ibest);
        end
   else
       iw_ibest=0;
   end

   acquisition=@(x,p) facquisition_pref(x(:)',Xs,N,delta_E,dF,W,prob_setup.rbf,prob_setup.rbf_epsil,prob_setup.theta,prob_setup.sepvalue,prob_setup.ibest,prob_setup.acquisition_method, ...
                                         prob_setup.has_unknown_constraints, prob_setup.has_satisfaction_fun,...
                                         prob_setup.UnknownFeasible, prob_setup.UnknownSatisfactory,...
                                         delta_G,delta_S,iw_ibest,prob_setup.expected_max_evals,prob_setup.scale_delta) + ...
                       prob_setup.constrpenalty(x(:));
    
    tic
    switch prob_setup.globoptsol
        case 'pswarm'
            pswarm_vars = prob_setup.pswarm_vars;
            pswarm_vars.Problem.ObjFunction= @(x) facquisition_pref(x(:)',...
                Xs,N,delta_E,dF,W,prob_setup.rbf,prob_setup.rbf_epsil,prob_setup.theta,prob_setup.sepvalue,prob_setup.ibest,prob_setup.acquisition_method, ...
                                         prob_setup.has_unknown_constraints, prob_setup.has_satisfaction_fun,...
                                         prob_setup.UnknownFeasible, prob_setup.UnknownSatisfactory,...
                                         delta_G,delta_S,iw_ibest,prob_setup.expected_max_evals,prob_setup.scale_delta)+prob_setup.constrpenalty(x(:));
            evalc('z=PSwarm(pswarm_vars.Problem,pswarm_vars.InitialPopulation,pswarm_vars.Options);');           
        case 'direct'
            direct_vars.opt.min_objective = acquisition;
            zold=prob_setup.z;
            z=nlopt_optimize(direct_vars.opt,zold);
            z=z(:);
    end
    prob_setup.time_opt_acquisition = [prob_setup.time_opt_acquisition;toc];

    xsnext = z;
    prob_setup.z = z;
    prob_setup.xnext = xsnext .* prob_setup.dd + prob_setup.d0;
    prob_setup.X = [prob_setup.X;prob_setup.xnext'];

    % Update RBF matrix M
    N = N+1;
    epsilth=prob_setup.rbf_epsil*prob_setup.theta;
    prob_setup.MM(N,1:N,prob_setup.itheta) = 0;
    prob_setup.MM(1:N,N,prob_setup.itheta) = 0;
    M = prob_setup.MM(:,:,prob_setup.itheta); % current RBF matrix
    for h=1:N-1
        mij=prob_setup.rbf(Xs(h,:),xsnext',epsilth);
        M(h,N)=mij;
        M(N,h)=mij;
    end
    M(N,N) = 1.0;
    prob_setup.MM(:,:,prob_setup.itheta) = M; % current RBF matrix
else
    prob_setup.xnext = prob_setup.X(prob_setup.iter + 1,:)';
end

prob_setup.iter = prob_setup.iter + 1;

xnext = prob_setup.xnext';

end



%%%%%%%%%%%%%%%%%%%%%%
function [f,fhat,dhat]=facquisition_pref(x,X,N,delta_E,dF,beta,rbf,epsil,...
    theta,sepvalue,ibest,acquisition_method,isUnknownFeasibilityConstrained,isUnknownSatisfactionConstrained,Feasibility_unkn,SatConst_unkn,delta_G,delta_S,iw_ibest,maxevals,scale_delta)
% Acquisition function to minimize to get next sample
%
% 1: a(x) = scaled surrogate + delta_E * z_N + delta_G * (1-G_hat) + delta_S * (1-S_hat)
% 2: a(x) = probability of improvement
            
m=size(x,1); % number of points x to evaluate the acquisition function

f=zeros(m,1);
v=zeros(N,1);

epsilth=epsil*theta;

for i=1:m
    xx=x(i,:)';
    
    if acquisition_method==1
        for j=1:N
            v(j)=rbf(X(j,:),xx',epsilth);
        end
        fhat=v'*beta;
        d=sum((X(1:N,:)-ones(N,1)*xx').^2,2);
        ii=find(d<1e-12,1);
        if ~isempty(ii)
            dhat=0;
            if isUnknownFeasibilityConstrained
                Ghat=Feasibility_unkn(ii);
            else
                Ghat=1;
            end
            if isUnknownSatisfactionConstrained
                Shat=SatConst_unkn(ii);
            else
                Shat=1;
            end
        else
            w=exp(-d)./d;
            sw=sum(w);
            if ~scale_delta
                dhat = delta_E *atan(1/sum(1./d)); %for comparision, used in the original GLISp and when N_max <= 30 in C-GLISp
            else
                dhat = delta_E * ((1-N/maxevals)*atan((1/sum(1./d))/iw_ibest)+ N/maxevals *atan(1/sum(1./d))); % used in C-GLISp
            end
            
            if isUnknownFeasibilityConstrained
                Ghat=sum(Feasibility_unkn(1:N)'*w)/sw;
            else
                Ghat = 1;
            end  

            if isUnknownSatisfactionConstrained
                Shat=sum(SatConst_unkn(1:N)'*w)/sw;
            else
                Shat = 1;
            end
        end
        
%         f(i)=fhat/dF-dhat; % for comparision, used in GLISp
        f(i)=fhat/dF-dhat+delta_G*(1-Ghat)+delta_S*(1-Shat); % used in C-GLISp
  
    elseif acquisition_method==2
        PHIbeta=0;
        for j=1:N
            PHIbeta=PHIbeta+(rbf(X(j,:),xx',epsilth)-rbf(X(j,:),X(ibest,:),epsilth))*beta(j);
        end
        lm1=max(PHIbeta+sepvalue,0);
        l0=max([0;PHIbeta-sepvalue;-PHIbeta-sepvalue]);
        l1=max(sepvalue-PHIbeta,0);
        c0=1;
        cm1=1;
        c1=1;
        em1=exp(-cm1*lm1);

        f(i)=-em1/(em1+exp(-c0*l0)+exp(-c1*l1));
    end
end
end

