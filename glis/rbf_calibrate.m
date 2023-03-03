function rbf_calibrate(Xs)
% calibrate scaling of epsil parameter in RBF by cross-validation

global prob_setup

N = prob_setup.iter;
ibest = prob_setup.ibest;
itheta = prob_setup.itheta;
thetas = prob_setup.thetas;
% sepvalue = prob_setup.sepvalue;
comparetol = prob_setup.comparetol;
I = prob_setup.I;
Ieq = prob_setup.Ieq;
MM = prob_setup.MM;
epsil = prob_setup.rbf_epsil;
iM = prob_setup.iM;
rbf = prob_setup.rbf;

if prob_setup.display
    fprintf('Recalibrating RBF: ');
end

nth=numel(thetas);
success=zeros(nth,1);

for k=1:nth
    
    epsilth=epsil*thetas(k);

    % Update matrix containing RBF values for all thetas
    if ~(k==itheta)
        for j=iM+1:N
            for h=1:N
                MM(j,h,k)=rbf(Xs(j,:),Xs(h,:),epsilth);
                MM(h,j,k)=MM(j,h,k);
            end
        end
    end

    Ncomparisons=0;
    for i=1:N
        if i~=ibest
            Xi=Xs(1:N,:);
            Xi(i,:)=[];
            if ibest>i
                newibest=ibest-1;
            else
                newibest=ibest;
            end
            
            Ii=I;
            isi=[];
            if ~isempty(I)
                isi=(I(:,1)==i | I(:,2)==i);
                Ii(isi,:)=[];
                Ii(Ii(:,1)>i,1)=Ii(Ii(:,1)>i,1)-1;
                Ii(Ii(:,2)>i,2)=Ii(Ii(:,2)>i,2)-1;
            end
            Ieqi=Ieq;
            iseqi=[];
            if ~isempty(Ieq)
                iseqi=(Ieq(:,1)==i | Ieq(:,2)==i);
                Ieqi(iseqi,:)=[];
                Ieqi(Ieqi(:,1)>i,1)=Ieqi(Ieqi(:,1)>i,1)-1;
                Ieqi(Ieqi(:,2)>i,2)=Ieqi(Ieqi(:,2)>i,2)-1;
            end
            
            Mi=MM(1:N,1:N,k);
            Mi(i,:)=[];
            Mi(:,i)=[];
            
            Wi=get_rbf_weights_pref(Mi,N-1,Ii,Ieqi,newibest);
            
            % Compute RBF @Xs(i,:)'
            FH=zeros(N,1);
            FH([1:i-1,i+1:N])=Mi*Wi; % rbf at samples
            xx=Xs(i,:)';
            v=zeros(N-1,1);
            for j=1:N-1
                v(j)=rbf(Xi(j,:),xx',epsilth);
            end
            FH(i)=v'*Wi;
            
            % Cross validation
            jj=find(isi);
            Ncomparisons=Ncomparisons+numel(jj);
            for h=1:numel(jj)
                j=jj(h);
                i1=I(j,1);
                i2=I(j,2);
                if FH(i1)<=FH(i2)-comparetol
                    success(k)=success(k)+1;
                end
            end
            jj=find(iseqi);
            Ncomparisons=Ncomparisons+numel(jj);
            for h=1:numel(jj)
                j=jj(h);
                i1=Ieq(j,1);
                i2=Ieq(j,2);
                if abs(FH(i1)-FH(i2))<=comparetol
                    success(k)=success(k)+1;
                end
            end
        end
    end
    if prob_setup.display
        fprintf('.');
    end
    success(k)=success(k)/Ncomparisons*100; % NOTE: normalization is only for visualization purposes
end

% Find theta such that success is max, and closest to 1 among maximizers
[~,imax]=max(success);
theta_max=thetas(imax);
[~,ii]=min((theta_max-1).^2); % get the theta closest to 1 among maxima
prob_setup.theta=theta_max(ii);
prob_setup.itheta=imax(ii);
prob_setup.iM = N;
prob_setup.MM = MM;

if prob_setup.display
    fprintf(' done.\n');
end

