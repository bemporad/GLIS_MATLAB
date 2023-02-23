function [f,fhat,dhat]=facquisition(x,X,F,N,alpha,delta_E,dF,W,rbf,isUnknownFeasibilityConstrained,isUnknownSatisfactionConstrained,Feasibility_unkn,SatConst_unkn,delta_G,delta_S,iw_ibest,maxevals)
% Acquisition function to minimize to get next sample in GLIS (surrogate + exploration)
% 
%     Note: in case samples that are infeasible wrt unknown constraints exist or if infeasible initial sampling is allowed
%         - here Xs only collects the K feasible ones, and W has dimension K.
%         - X_all and F_all collect all the samples (feasible and infeasible), while only the feasible ones are used to construct the surrogate


m=size(x,1); % number of points x to evaluate the acquisition function

f=zeros(m,1);

for i=1:m
    xx=x(i,:)';
    
    d=sum((X(1:N,:)-ones(N,1)*xx').^2,2);
    ii=find(d<1e-12);
    if ~isempty(ii)
        fhat=F(ii(1));
        dhat=0;
        if isUnknownFeasibilityConstrained
            Ghat=Feasibility_unkn(ii);
        else
            Ghat=1;
        end
        if isUnknownSatisfactionConstrained
            Shat=SatConst_unkn(ii);
        else
            Shat=1;
        end
    else
        w=exp(-d)./d;
        sw=sum(w);
        
        if ~isempty(rbf)
            v=zeros(N,1);
            for j=1:N
                v(j)=rbf(X(j,:),xx');
            end
            fhat=v'*W;
        else
            fhat=sum(F(1:N).*w)/sw;
        end
        if maxevals <= 30
            dhat=delta_E*atan(1/sum(1./d))*2/pi*dF+...
                alpha*sqrt(sum(w.*(F(1:N)-fhat).^2)/sw);  
        else
            dhat=delta_E*((1-N/maxevals)*atan((1/sum(1./d))/iw_ibest)+ N/maxevals *atan(1/sum(1./d)))*2/pi*dF+...
                alpha*sqrt(sum(w.*(F(1:N)-fhat).^2)/sw); 
        end

        if isUnknownFeasibilityConstrained
            Ghat=sum(Feasibility_unkn(1:N)'*w)/sw;
        else
            Ghat = 1;
        end  

        if isUnknownSatisfactionConstrained
            Shat=sum(SatConst_unkn(1:N)'*w)/sw;
        else
            Shat = 1;
        end
    end
    
    f(i)=fhat-dhat+(delta_G*(1-Ghat)+delta_S*(1-Shat))*dF;
end

end

