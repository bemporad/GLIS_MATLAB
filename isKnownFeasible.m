function isfeas_known = isKnownFeasible(xs,isLinConstrained,isNLConstrained,Aineq,bineq,g)
% Check the feasibility of sample xs w.r.t known constraints

isfeas_known=true;
if isLinConstrained
    isfeas_known=isfeas_known && all(Aineq*xs'<=bineq);
end
if isNLConstrained
    isfeas_known=isfeas_known && all(g(xs')<=0);
end

end