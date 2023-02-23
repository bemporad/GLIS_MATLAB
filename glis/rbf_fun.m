function fun = rbf_fun(rbf_type)
% GLIS - (GL)obal optimization solvers using (I)nverse distance weighting and
% radial basis function (S)urrogates.
% 
% RBF functions.
% 
% (C) 2019-2023 Alberto Bemporad, Mengjia Zhu

if rbf_type == "inverse_quadratic"
    fun = @(x1, x2, epsil) 1./(1+epsil^2*sum((x1-x2).^2,2));
elseif rbf_type == "gaussian"
    fun = @(x1,x2,epsil) exp(-(epsil^2*sum((x1-x2).^2,2)));
elseif rbf_type == "multiquadric"
    fun = @(x1,x2,epsil) sqrt(1+epsil^2*sum((x1-x2).^2,2));
elseif rbf_type == "thin_plate_spline"
    fun = @(x1,x2,epsil) 2 * sum((x1-x2).^2) * log(epsil * sum((x1-x2).^2,2));
elseif rbf_type == "linear"
    fun = @(x1,x2,epsil) epsil * sqrt(sum((x1-x2).^2,2));
elseif rbf_type == "inverse_multi_quadric"
    fun = @(x1,x2,epsil) 1./sqrt(1+epsil^2*sum((x1-x2).^2,2));
else
    msg = "Please define a valid rbf type (inverse_quadratic, gaussian, multiquadric, thin_plate_spline, linear, or inverse_multi_quadric), or define a new one in rbf.m";
    error(msg)
end


end

