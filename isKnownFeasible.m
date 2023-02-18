function isfeas_known = isKnownFeasible(xs)
% Check the feasibility of sample xs w.r.t known constraints

global prob_setup

isfeas_known=true;
if prob_setup.isLinConstrained
    isfeas_known=isfeas_known && all(prob_setup.Aineq*xs'<=prob_setup.bineq);
end
if prob_setup.isNLConstrained
    isfeas_known=isfeas_known && all(prob_setup.g(xs')<=0);
end

end