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

x = prob_setup.xnext;


% todo: to update from here









end