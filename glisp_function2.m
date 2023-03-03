function [out,fesx,fesy,Xtest_out,Ftest_out,itest_out]=glisp_function2(x,y,fun,comparetol,g_unkn)
% Preference query function. Along with either satisfactory or feasibility
% constraints (but NOT both, if the problem include both satisfactory AND feasibility constraints,
% use 'glisp_function3.m' instead)
%
% pref=glisp_function2(x,y,fun,comparetol,g_unkn)
%
% evaluates the preference function based on fun:
%
% pref = -1 if fun(x) < fun(y) - comparetol
%         1 if fun(x) > fun(y) + comparetol
%         0 if |fun(x)-fun(y)| <= comparetol
%
% g_unkn is a handle to the function defining unknown constraints.
%
% glisp_function2('clear') resets values of fun already computed, that are
% stored to save computations of expensive functions.
%
% val = glisp_function2('get',x) returns val = fun(x), that is retrieved from
% existing values if the function has been already evaluated at x, or
% computes a new value if not.
%
% (C) 2019 by A. Bemporad, September 22, 2019

% Note: Modified to account for unknown constraints by M. Zhu, June 03, 2021

persistent Xtest Ftest itest Festest

if isa(x,'char')
    if strcmp(x,'clear')
        Xtest=[];
        Ftest=[];
        Festest=[];
        itest=0;
    elseif strcmp(x,'get')
        out=glisp_function_value(y);
    end
    return
end

xfound=0;
yfound=0;
for i=1:itest
    if ~xfound && sum(abs(Xtest(i,:)-x(:)'))<=1e-10
        xfound=1;
        fx=Ftest(i);
        fesx = Festest(i);
    end
    if ~yfound && sum(abs(Xtest(i,:)-y(:)'))<=1e-10
        yfound=1;
        fy=Ftest(i);
        fesy = Festest(i);
    end
end
if ~xfound
    fx=fun(x(:)');
    gx_unkn = g_unkn(x(:)');
    if numel(gx_unkn) >1
        if max(gx_unkn) < comparetol
            fesx = 1;
        else
            fesx = 0;
        end 
    else
        if gx_unkn < comparetol
            fesx = 1;
        else
            fesx = 0;
        end 
    end
    
    itest=itest+1;
    if itest==1
        Xtest=x(:)';
        Ftest=fx;
        Festest=fesx;
    else
        Xtest(itest,:)=x(:)';
        Ftest(itest)=fx;
        Festest(itest)=fesx;
    end
end
if ~yfound
    fy=fun(y(:)');
        gy_unkn = g_unkn(y(:)');
    if numel(gy_unkn) >1
        if max(gy_unkn) < comparetol
            fesy = 1;
        else
            fesy = 0;
        end 
    else
        if gy_unkn < comparetol
            fesy = 1;
        else
            fesy = 0;
        end 
    end
    
    itest=itest+1;
    if itest==1
        Xtest=y(:)';
        Ftest=fy;
        Festest=fesy;
    else
        Xtest(itest,:)=y(:)';
        Ftest(itest)=fy;
        Festest(itest)=fesy;
    end
end

% Make comparison
if fx<fy-comparetol
    if (fesx) || (~fesx && ~fesy)
        out=-1;
    else
        out =1;
    end
elseif fx>fy+comparetol
    if (fesy) || (~fesx && ~fesy)
        out=1;
    else
        out=-1;
    end
else
    out=0;
end

if nargout>1
    Xtest_out=Xtest;
    Ftest_out=Ftest;
    itest_out=itest;
end

    function [val,fes]=glisp_function_value(x)
        % Compute function value, from available ones if available
        %
        % (C) 2019 A. Bemporad, September 22, 2019
        
        j=0;
        while j<itest
            j=j+1;
            if sum(abs(Xtest(j,:)-x(:)'))<=1e-10
                val=Ftest(j);
                fes=Festest(j);
                return
            end
        end
        
        % Value has not been found
        val=fun(x(:)');
        g_unknx = g_unkn(x(:)');
        if numel(g_unknx) >1
            if max(g_unknx) < comparetol
                fes = 1;
            else
                fes = 0;
            end
        else
            if g_unknx < comparetol
                fes = 1;
            else
                fes = 0;
            end 
        end
        
        itest=itest+1;
        Xtest(itest,:)=x';
        Ftest(itest)=val;
        Festest(itest)=fes;
        return
    end
end