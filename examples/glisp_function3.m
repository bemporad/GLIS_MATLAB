function [out,fesx,fesy,softx,softy,Xtest_out,Ftest_out,itest_out]=glisp_function3(x,y,f,comparetol,g_unkn,s_unkn)
% Synthetic preference query function. Along with both satisfactory AND feasibility
% constraints (if the problem only include one of them, use 'glisp_function2.m' instead)
%
% This file is specific for the 'CHSC' benchmark.
% (This benchmark function includes both feasibility and satisfaction constraints)
%
% pref=glisp_function3(x,y,f,comparetol,g_unkn,s_unkn)
%
% evaluates the preference function based on f:
%
% pref = -1 if f(x) < f(y) - comparetol
%         1 if f(x) > f(y) + comparetol
%         0 if |f(x)-f(y)| <= comparetol
%
% g_unkn is a handle to the function defining unknown constraints.
% s_unkn is a handle to the function defining whether a sample is satisfactory or not.
%
% glisp_function3('clear') resets values of f already computed, that are
% stored to save computations of expensive functions.
%
% val = glisp_function3('get',x) returns val = f(x), that is retrieved from
% existing values if the function has been already evaluated at x, or
% computes a new value if not.
% Todo: make this file more general, 
% i.e., check the input, 
% e.g., check if'unknownconstraints', 'softconstratints' , etc
% current version assumes that there will be soft constraints and hard
% constraints

%Reference code: 'glisp_function' by A. Bemporad, September 22, 2019
% M.Zhu, May 31, 2021

persistent Xtest Ftest itest Festest SoftconstTest

if isa(x,'char')
    if strcmp(x,'clear')
        Xtest=[];
        Ftest=[];
        Festest=[];
        SoftconstTest =[];
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
        softx = SoftconstTest(i);
    end
    if ~yfound && sum(abs(Xtest(i,:)-y(:)'))<=1e-10
        yfound=1;
        fy=Ftest(i);
        fesy = Festest(i);
        softy = SoftconstTest(i);
    end
end
if ~xfound
    fx=f(x(:)');
    gx_unkn = g_unkn(x(:)');
    sx_unkn = s_unkn(x(:)');
    if sx_unkn < comparetol
        softx = 1;
    else
        softx = 0;
    end 
    
    if gx_unkn < comparetol
        fesx = 1;
    else
        fesx = 0;
    end 
    
    itest=itest+1;
    if itest==1
        Xtest=x(:)';
        Ftest=fx;
        Festest=fesx;
        SoftconstTest=softx;
    else
        Xtest(itest,:)=x(:)';
        Ftest(itest)=fx;
        Festest(itest)=fesx;
        SoftconstTest(itest)=softx;
    end
end
if ~yfound
    fy=f(y(:)');
    gy_unkn = g_unkn(y(:)');
    sy_unkn = s_unkn(y(:)');
    if sy_unkn < comparetol
        softy = 1;
    else
        softy = 0;
    end 

    if gy_unkn < comparetol
        fesy = 1;
    else
        fesy = 0;
    end 
    
    itest=itest+1;
    if itest==1
        Xtest=y(:)';
        Ftest=fy;
        Festest=fesy;
        SoftconstTest=softy;
    else
        Xtest(itest,:)=y(:)';
        Ftest(itest)=fy;
        Festest(itest)=fesy;
        SoftconstTest(itest)=softy;
    end
end

% Make comparison
if fx<fy-comparetol
    if (fesx && softx) || (~fesx && ~softx && ~fesy && ~softy)
        out=-1;
    else
        out =1;
    end
elseif fx>fy+comparetol
    if (fesy && softy) || (~fesx && ~softx && ~fesy && ~softy)
        out=1;
    else
        out=-1;
    end
else
    out =0;
end

if nargout>1
    Xtest_out=Xtest;
    Ftest_out=Ftest;
    itest_out=itest;
end

    function [val,fes,softconst]=glisp_function_value(x)
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
        val=f(x(:)');
        g_unkn_x = g_unkn(x(:)');
        s_unkn_x = s_unkn(x(:)');
        if s_unkn_x < comparetol
            softconst = 1;
        else
            softconst = 0;
        end 

        if g_unkn_x < comparetol
            fes = 1;
        else
            fes = 0;
        end
        
        itest=itest+1;
        Xtest(itest,:)=x';
        Ftest(itest)=val;
        Festest(itest)=fes;
        SoftconstTest(itest)=soft;
        return
    end
end