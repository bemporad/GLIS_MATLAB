function satisfactory = eval_sat(x, has_syn_unknown_satfun,s_unkn_fun)
if nargin <2
    has_syn_unknown_satfun = false;
end
if ~has_syn_unknown_satfun
    satisfactory = true;
else
    satisfactory = s_unkn_fun(x) < 1.0e-6 ;
end

end