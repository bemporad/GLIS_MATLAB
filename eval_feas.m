function feasible = eval_feas(x, has_syn_unknown_const,g_unkn_fun)
if nargin <2
    has_syn_unknown_const = false;
end
if ~has_syn_unknown_const
    feasible = true;
else
    feasible = g_unkn_fun(x) < 1.0e-6 ;
end

end