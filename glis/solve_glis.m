function [xbest, fbest, prob_setup] = solve_glis(fun, lb,ub,opts, unknown_constraint_fun,satisfactory_fun)

global prob_setup

if nargin <6
    unknown_constraint_fun = [];
    satisfactory_fun = [];
end

tic;

prob_setup.expected_max_evals = opts.maxevals;
x = initialize(lb,ub,opts); % x is unscaled

for k = 1:prob_setup.expected_max_evals
    tic;

    % evaluate fun/performance
    f_val = fun(x);

    % evaluate unknown feasibility/satisfactory, if exist, of new x
    if prob_setup.has_unknown_constraints
        feasible = unknown_constraint_fun(x);
    else
        feasible = true;
    end
    if prob_setup.has_satisfaction_fun
        satisfactory = satisfactory_fun(x);
    else
        satisfactory = true;
    end

    prob_setup.time_fun_eval = [prob_setup.time_fun_eval;toc];

    x = update_glis(f_val, feasible, satisfactory);
    xbest = prob_setup.xbest;
    fbest = prob_setup.fbest;
end

prob_setup.X = prob_setup.X(1:end-1,:);  % it is because in prob.update, it will calculate the next point to query (the last x2 is calculated but not assessed at max_evals +1)

prob_setup.time_total = toc;


end