%%%%%%%%%%%%%%%%%%%%%%
function beta=get_rbf_weights_pref(M,n,I,Ieq,ibest)
% Fit RBF satisfying comparison constraints at sampled points

% optimization vector x=[beta;epsil] where:
%    beta  = rbf coefficients
%    epsil = vector of slack vars, one per constraint

global prob_setup

sepvalue=prob_setup.sepvalue;

normalize=0;

m=size(I,1);
meq=size(Ieq,1);
A=zeros(m+2*meq,n+m+meq);
b=zeros(m+2*meq,1);
for k=1:m
    i=I(k,1);
    j=I(k,2);
    % f(x(i))<f(x(j))
    % sum_h(beta(h)*phi(x(i,:),x(h,:))+a'*x(i,:)'+b'*(x(i,:)').^2
    % <= sum_h(beta(h)*phi(x(j,:),x(h,:))+a'*x(j,:)'+b'*(x(j,:)').^2
    %    +eps_k-sepvalue
    A(k,:)=[M(i,1:n)-M(j,1:n) zeros(1,k-1) -1 zeros(1,m+meq-k)];
    b(k)=-sepvalue;
end


% |f(x(i))-f(x(j))|<=comparetol
% --> f(x(i))<=f(x(j))+comparetol+epsil
% --> f(x(j))<=f(x(i))+comparetol+epsil
%
% sum_h(beta(h)*phi(x(i,:),x(h,:))+a'*x(i,:)'+b'*(x(i,:)').^2
% <= sum_h(beta(h)*phi(x(j,:),x(h,:))+a'*x(j,:)'+b'*(x(j,:)').^2+sepvalue+epsil
%
% sum_h(beta(h)*phi(x(j,:),x(h,:))+a'*x(i,:)'+b'*(x(i,:)').^2
% <= sum_h(beta(h)*phi(x(i,:),x(h,:))+a'*x(j,:)'+b'*(x(j,:)').^2+sepvalue+epsil

for k=1:meq
    i=Ieq(k,1);
    j=Ieq(k,2);
    A(m+2*(k-1)+1,:)=[M(i,1:n)-M(j,1:n) zeros(1,m+k-1) -1 zeros(1,meq-k)];
    A(m+2*k,:)=[M(j,1:n)-M(i,1:n) zeros(1,m+k-1) -1 zeros(1,meq-k)];
    b(m+2*(k-1)+1)=sepvalue;
    b(m+2*k)=sepvalue;
end

if normalize
    % Add constraints to avoid trivial solution surrogate=flat:
    %    sum_h(beta.*phi(x(ibest,:),x(h,:)))+a'*x(ibest,:)'+b'*(x(ibest,:)').^2  = 0
    %    sum_h(beta.*phi(x(ii,:),x(h,:)))+a'*x(ii,:)'+b'*(x(ii,:)').^2  = 1
    % Look for sample where function is worse
    ii=I(1,2);
    for k=1:m
        if I(k,1)==ii
            ii=I(k,2);
        end
    end
    Aeq=[M(ibest,1:n) zeros(1,m+meq);
        M(ii,1:n) zeros(1,m+meq)];
    beq=[0;
        1];
else
    Aeq=[];
    beq=[];
end
% Only impose surrogate=0 at x(ibest):
% Aeq=[M(ibest,1:n) zeros(1,m+meq)];
% beq=0;

e=ones(m+meq,1);
% penalize more violations involving zbest
if ~isempty(I)
    ii=(I(:,1)==ibest | I(:,2)==ibest);
    e(ii)=10;
end
if ~isempty(Ieq)
    ii=(Ieq(:,1)==ibest | Ieq(:,2)==ibest);
    e(m+ii)=10;
end

lb=[-inf(n,1);zeros(m+meq,1)]; % slacks >=0
c=[zeros(1,n) e'];

% solve QP with penalty on beta only, QUADPROG
opts=struct('Display','off');
xopt=quadprog(diag([1e-6*ones(n,1);zeros(m+meq,1)]),c,A,b,Aeq,beq,lb,[],[],opts);

beta=xopt(1:n);
end