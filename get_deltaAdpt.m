function delta_adpt = get_deltaAdpt(X,constraint_set,delta_const_default)
% Adaptively tune the hyperparameter delta_G and delta_S for the feasibility and satisfaction term in the acquisition function
% For both terms, their delta is tuned via leave-one-out cross validation using IDW interpolation as a prediction method

ind = size(constraint_set,1);
sqr_error_feas = zeros(ind,1);
for i= 1:ind
    xx = X(i,:);
    Xi = [X(1:i-1,:); X(i + 1:ind,:)];
    const_classifier_i = [constraint_set(1:i-1);constraint_set(i+1:ind)];
    Feas_xx = constraint_set(i);
    d = sum((Xi - xx).^2,2);
    w = exp(-d)./d;
    sw = sum(w);
    ghat = sum(const_classifier_i'* w) / sw;
    sqr_error_feas(i) = (ghat-Feas_xx)^2;
end

std_feas = min(1,(sum(sqr_error_feas)/(ind-1))^(1/2));
delta_adpt = (1-std_feas) *delta_const_default;

end