function [out,Xtest_out,Ftest_out,itest_out]=glisp_function1(x,y,f,comparetol,Aineq,bineq,g)
% Template for preference query function w/o unknown constraints
%
% pref=glisp_function1(x,y,f,comparetol,Aineq,bineq,g)
%
% evaluates the preference function based on f:
%
% pref = -1 if f(x) < f(y) - comparetol
%         1 if f(x) > f(y) + comparetol
%         0 if |f(x)-f(y)| <= comparetol
%
% for x,y in the feasible set. (Aineq, bineq) and g are optional arguments
% defining polyhedral constraints and general nonlinear constraints, respectively.
%
% glisp_function1('clear') resets values of f already computed, that are
% stored to save computations of expensive functions.
%
% val = glisp_function1('get',x) returns val = f(x), that is retrieved from
% existing values if the function has been already evaluated at x, or
% computes a new value if not.
%
% (C) 2019 by A. Bemporad, September 22, 2019
% modifed on June 06,2021 by M. Zhu 
% To incorporate known constraints in the preference assessment for numerical benchmarks

persistent Xtest Ftest itest Festest

if nargin<7
    g=[];
end
if nargin<6
    bineq=[];
end
if nargin<5
    Aineq=[];
end

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
        fes_known_x = Festest(i);
    end
    if ~yfound && sum(abs(Xtest(i,:)-y(:)'))<=1e-10
        yfound=1;
        fy=Ftest(i);
        fes_known_y = Festest(i);
    end
end
if ~xfound
    fx=f(x(:)');
    fes_known_x = feas_check(x(:),Aineq,bineq,g);
    itest=itest+1;
    if itest==1
        Xtest=x(:)';
        Ftest=fx;
        Festest=fes_known_x;
    else
        Xtest(itest,:)=x(:)';
        Ftest(itest)=fx;
        Festest(itest)=fes_known_x;
    end
end
if ~yfound
    fy=f(y(:)');
    fes_known_y = feas_check(y(:),Aineq,bineq,g);
    itest=itest+1;
    if itest==1
        Xtest=y(:)';
        Ftest=fy;
        Festest=fes_known_y;
    else
        Xtest(itest,:)=y(:)';
        Ftest(itest)=fy;
        Festest(itest)=fes_known_y;
    end
end

% Make comparison
if fx<fy-comparetol
    if (fes_known_x ==1) || (fes_known_x ==0 && fes_known_y ==0)
        out=-1;
    else
        out =1;
    end
elseif fx>fy+comparetol
    if (fes_known_y ==1) || (fes_known_x ==0 && fes_known_y ==0)
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
        val=f(x(:)');
        fes = feas_check(x(:),Aineq,bineq,g);
        itest=itest+1;
        Xtest(itest,:)=x';
        Ftest(itest)=val;
        Festest(itest)=fes;
        return
    end

    function isfeas = feas_check(x,Aineq,bineq,g)
    % check if the decision variable is feasible or not (for known
    % constraints, which is needed to help express preferences
        isfeas=true;
        if ~isempty(Aineq)
            isfeas=isfeas && all(Aineq*x<=bineq);
        end
        if ~isempty(g)
            isfeas=isfeas && all(g(x)<=0);
        end
    end
end