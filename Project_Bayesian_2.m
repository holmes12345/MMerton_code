%% Multivariate Merton Model
%% Loading the data and setup some few things
GSPC_data = readtable('GSPC.csv');
GSPC_adjusted = table2array(GSPC_data(:,7));
NDX_data = readtable('NDX.csv');
NDX_adjusted = table2array(NDX_data(:,7));
y1 = 100*log(GSPC_adjusted(2:end)./GSPC_adjusted(1:end-1));
y2 = 100*log(NDX_adjusted(2:end)./NDX_adjusted(1:end-1));
y= [y1;y2];
y_p = [y1 y2];
T = size(y,1);
X = ones(T,2);

%% Setting the draw
nsim = 5000; burnin = 1250; 
mu_draw = zeros(nsim,2);
sigma2_draw = zeros(nsim,2,2);
muz_draw = zeros(nsim,2);
sigma2z_draw = zeros(nsim,2,2);  
lambda_draw = zeros(nsim,1);

%% Setting the prior 
mu0 = zeros(2,1); iVmu0 = eye(2)/100;
nu0 = 5; S0 = eye(2);
muz0 = zeros(2,1); iVmuz0 =eye(2)/100;
nuz0 = 5; Sz0 = eye(2);
alpha0 = 0.5; beta0 = 0.5;

%% Setting up something (Intialize the Markov Chain)
muz = [0 0]; iSigz = [0.7051 -0.1645; -0.1645 0.7051];
Z_r = mvnrnd(muz,iSigz\speye(2),T/2);
Z = reshape(Z_r,[T,1]);
lambda = 0.5;
J_1=(rand(T/2,1)<lambda);
J=repmat(J_1,2,1);
y_tilde = y-Z.*J;
mu = (X'*X+0.01*eye(size(X, 2)))\(X'*y_tilde);  
e = reshape(y_tilde - X*mu,[T/2,2]);
Sig = e'*e/T;    
iSig = Sig\speye(2);
H = zeros(nsim,T);
%%
for isim = 1:nsim + burnin
        %% sample mu
    XiSig = X'*kron(speye(T/2),iSig);
    XiSigX = XiSig*X;
    Kmu = iVmu0 + XiSigX;
    mu_hat = Kmu\(iVmu0*mu0 + XiSig*y_tilde); 
    mu =  mu_hat + chol(Kmu,'lower')'\randn(2,1);
    %% sample Sig
    e = reshape(y_tilde - X*mu,[T/2,2]);    
    Sig = iwishrnd(S0 + e'*e,nu0 + T/2);
    diag1 = diag([sqrt(Sig(1,1)),sqrt(Sig(2,2))]);
    scale1 = sqrt(Sig(1, 1) * Sig(2, 2));
    corr1 = Sig(1,2)/scale1;
    corr1 = max(min(corr1, 1), -1);
    corr1_matrix = [1 corr1;corr1 1];
    Sig = diag1*corr1_matrix*diag1;
    iSig = Sig\speye(2);
    %% sample muz 
    ZSig = X' * kron(speye(T/2),iSigz);
    ZSigZ = ZSig * X;
    Kmuz = iVmuz0 + ZSigZ;
    muz_hat = Kmuz\(iVmuz0 * muz0 + ZSig * Z);
    muz =  muz_hat + chol(Kmuz,'lower')'\randn(2,1);
    %% sample Sigz
    e1 = reshape(Z - X*muz,[T/2,2]);    
    Sigz = iwishrnd(Sz0 + e1'*e1,nuz0 + T/2);
    diag2 = diag([sqrt(Sigz(1,1)),sqrt(Sigz(2,2))]);
    scale2 = sqrt(Sigz(1, 1) * Sigz(2, 2));
    corr2 = Sigz(1,2)/scale2;
    corr2 = max(min(corr2, 1), -1);
    corr2_matrix = [1 corr2;corr2 1];
    Sigz = diag2*corr2_matrix*diag2;
    iSigz = Sigz\speye(2);

    %% sample lambda 
    lambda = betarnd(alpha0 + sum(J_1),T-sum(J_1)+beta0);
    %% sample Zt
    V =zeros(T/2,2,2);
    m = zeros(T/2,2);
    Z_r = zeros(T/2,2);
    for i =1:T/2
        V(i,:,:) = reshape((((repmat(J_1(i),2,1)' .* iSig +iSigz))\speye(2)),[1 2 2]);
    end

    for i=1:T/2
        m(i,:) = (iSigz *((repmat(J_1(i),2,1)' .* iSig * (y_p(i,:)' - mu) + iSigz * muz)))';
        Z_r(i,:) =mvnrnd(m(i,:),squeeze(V(i,:,:)),1);
    end
    Z = reshape(Z_r,[T,1]);

    %% sample Jt
    A= zeros(T/2,1);
    B= zeros(T/2,1);
    q = zeros(T/2,1);
    J_1 = zeros(T/2,1);
    for i=1:T/2
        A(i) = lambda * exp(-1/2 * (y_p(i)-mu'-Z_r(i,:)) * iSig * (y_p(i)-mu'-Z_r(i,:))');
        B(i) = (1-lambda) * exp(-1/2 * (y_p(i)-mu') * iSig * (y_p(i)-mu')');
        q(i) =A(i)/(A(i)+B(i));
        J_1=(rand(T/2,1)<q(i));
        J=repmat(J_1,2,1);
    end

if isim > burnin
        isave = isim - burnin;        
        mu_draw(isave,:) = mu'; 
        muz_draw(isave,:) = muz';
        sigma2_draw(isave,:,:) = Sig;
        sigma2z_draw(isave,:,:) = Sigz;
        lambda_draw(isave,:) = lambda;
        H(isave,:)=J;
end
waitbar(isim/(nsim+burnin));
end

%% Create a table and draw a graph 
%% Extracting the parameters
mu1 = mu_draw(:,1);
mu2 = mu_draw(:,2);
muz1 = muz_draw(:,1);
muz2 = muz_draw(:,2);
sigma2_1 = zeros(nsim,1);
sigma2_2 = zeros(nsim,1);
covr = zeros(nsim,1);
corr = zeros(nsim,1);
for i = 1:nsim
sigma2_1(i) = sigma2_draw(i,1,1);
sigma2_2(i) = sigma2_draw(i,2,2);
covr(i) = sigma2_draw(i,1,2);
corr(i) = covr(i)/sqrt(sigma2_1(i)*sigma2_2(i));
end
sigmaz2_1 = zeros(nsim,1);
sigmaz2_2 = zeros(nsim,1);
covr_z = zeros(nsim,1);
corr_z = zeros(nsim,1);

for i = 1:nsim
sigmaz2_1(i) = sigma2z_draw(i,1,1)/100;
sigmaz2_2(i) = sigma2z_draw(i,2,2)/100;
covr_z(i) = sigma2z_draw(i,1,2)/100;
corr_z(i) = covr_z(i)/sqrt(sigmaz2_1(i)*sigmaz2_2(i));
end

%% Posteior mean, variance and HPD
% Mean
mu1_h = mean(mu1);mu2_h = mean(mu2);
muz1_h = mean(muz1);muz2_h = mean(muz2);
sigma2_1h = mean(sigma2_1); sigma2_2h = mean(sigma2_2);
sigmaz2_1h = mean(sigmaz2_1);sigmaz2_2h = mean(sigmaz2_2);
corr_h = mean(corr);corr_zh = mean(corr_z);
lambda_h = mean(lambda_draw);

%% Standard Deviation

mu1_sd = std(mu1);mu2_sd = std(mu2);
muz1_sd = std(muz1);muz2_sd = std(muz2);
sigma2_1sd = std(sigma2_1); sigma2_2sd = std(sigma2_2);
sigmaz2_1sd = std(sigmaz2_1);sigmaz2_2sd = std(sigmaz2_2);
corr_sd = std(corr);corr_zsd = std(corr_z);
lambda_sd = std(lambda_draw);

%% HPD region
q_mu1 = quantile(mu1,[0.025,0.975]);q_mu2 = quantile(mu2,[0.025,0.975]);
q_muz1 = quantile(muz1,[0.025,0.975]);q_muz2 = quantile(muz2,[0.025,0.975]);
q_sigma21 = quantile(sigma2_1,[0.025,0.975]);q_sigma22 = quantile(sigma2_2,[0.025,0.975]);
q_sigmaz21 = quantile(sigmaz2_1,[0.025,0.975]);q_sigmaz22 = quantile(sigmaz2_2,[0.025,0.975]);
q_corr = quantile(corr,[0.025,0.975]);q_corrz = quantile(corr_z,[0.025,0.975]);
q_lambda = quantile(lambda_draw,[0.025,0.975]);

%% Create a table
V = [mu1_h mu2_h sigma2_1h sigma2_2h corr_h...
    muz1_h muz2_h sigmaz2_1h sigmaz2_2h corr_zh lambda_h...
    mu1_sd mu2_sd sigma2_1sd sigma2_2sd corr_sd...
    muz1_sd muz2_sd sigmaz2_1sd sigmaz2_2sd corr_zsd lambda_sd...
    {q_mu1} {q_mu2} {q_sigma21} {q_sigma22} {q_corr}...
    {q_muz1} {q_muz2} {q_sigmaz21} {q_sigmaz22} {q_corrz} {q_lambda}];
V = reshape(V, 11, 3);
col_names = {'Mean', 'Std', 'Credible Interval'};
row_names = {'mu1', 'mu2','sigma21','sigma22','corr',...
        'muz1','muz2','sigmaz21','sigmaz22','corrz','lambda'};
V = array2table(V,'RowNames', row_names, 'VariableNames', col_names);
writetable(V, "Multi Merton Parameters.csv")

%%  %%%% Posterior Progressive MCMC trace plots
% S&P500
figure(1)
subplot(4,1,1)
plot(mu1);%ylim([-0.5 0.5]);
ylabel('$\mu_{1}$','interpreter','latex')
set(gca,'fontsize',12);
subplot(4,1,2)
plot(sigma2_1);%ylim([-0.5 0.5]);
ylabel('$\sigma_{1}^{2}$','interpreter','latex')
set(gca,'fontsize',12);
subplot(4,1,3)
plot(muz1)
ylabel('$\mu_{z1}$','interpreter','latex')
set(gca,'fontsize',12);
subplot(4,1,4)
plot(sigmaz2_1)
ylabel('$\sigma_{z1}^{2}$','interpreter','latex')
set(gca,'fontsize',12);
%% Nasdaq
figure(2)
subplot(4,1,1)
plot(mu2);%ylim([-0.5 0.5]);
ylabel('$\mu_{2}$','interpreter','latex')
set(gca,'fontsize',12);
subplot(4,1,2)
plot(sigma2_2);%ylim([-0.5 0.5]);
ylabel('$\sigma_{2}^{2}$','interpreter','latex')
set(gca,'fontsize',12);
subplot(4,1,3)
plot(muz2)
ylabel('$\mu_{z2}$','interpreter','latex')
set(gca,'fontsize',12);
subplot(4,1,4)
plot(sigmaz2_2)
ylabel('$\sigma_{z2}^{2}$','interpreter','latex')
set(gca,'fontsize',12);

%% Intensity and Correlation
figure(3)
subplot(3,1,1)
plot(lambda_draw)
ylabel('$\lambda$','interpreter','latex')
subplot(3,1,2)
plot(corr)
ylabel('$\rho_{12}$','interpreter','latex')
subplot(3,1,3)
plot(corr_z)
ylabel('$\rho_{z12}$','interpreter','latex')
%%  %%%% Posterior Progressive MCMC averages
%% S&P500
figure(4)
subplot(4,1,1)
plot(cumsum(mu1)./(1:nsim)');%ylim([-0.5 0.5]);
ylabel('$\mu_{1}$','interpreter','latex')
set(gca,'fontsize',12);
subplot(4,1,2)
plot(cumsum(sigma2_1)./(1:nsim)');%ylim([-0.5 0.5]);
ylabel('$\sigma^{2}_{1}$','interpreter','latex')
set(gca,'fontsize',12);
subplot(4,1,3)
plot(cumsum(muz1)./(1:nsim)')
ylabel('$\mu_{z1}$','interpreter','latex')
set(gca,'fontsize',12);
subplot(4,1,4)
plot(cumsum(sigmaz2_1)./(1:nsim)')
ylabel('$\sigma^{2}_{z1}$','interpreter','latex')
set(gca,'fontsize',12);

%% Nasdaq
figure(5)
subplot(4,1,1)
plot(cumsum(mu2)./(1:nsim)');%ylim([-0.5 0.5]);
ylabel('$\mu_{2}$','interpreter','latex')
set(gca,'fontsize',12);
subplot(4,1,2)
plot(cumsum(sigma2_2)./(1:nsim)');%ylim([-0.5 0.5]);
ylabel('$\sigma^{2}_{2}$','interpreter','latex')
set(gca,'fontsize',12);
subplot(4,1,3)
plot(cumsum(muz2)./(1:nsim)')
ylabel('$\mu_{z2}$','interpreter','latex')
set(gca,'fontsize',12);
subplot(4,1,4)
plot(cumsum(sigmaz2_2)./(1:nsim)')
ylabel('$\sigma^{2}_{z2}$','interpreter','latex')
set(gca,'fontsize',12);
%% Intensity and Correlation
figure(6)
subplot(3,1,1)
plot(cumsum(lambda_draw)./(1:nsim)')
ylabel('$\lambda$','interpreter','latex')
set(gca,'fontsize',12);
subplot(3,1,2)
plot(cumsum(corr)./(1:nsim)')
ylabel('$\rho_{12}$','interpreter','latex')
set(gca,'fontsize',12);
print(gcf,'MCMCav.eps','-depsc')
subplot(3,1,3)
plot(cumsum(corr_z)./(1:nsim)')
ylabel('$\rho_{z12}$','interpreter','latex')
set(gca,'fontsize',12);
%%  %%%% Posterior MCMC histogram
%% S&P500
figure(7)
subplot(2,2,1)
histogram(mu1,100);%ylim([-0.5 0.5]);
hold on
fitdist(mu1,'Normal')
hold off
ylabel('$\mu_{1}$','interpreter','latex')
set(gca,'fontsize',12);
subplot(2,2,2)
histogram(sigma2_1,100);%ylim([-0.5 0.5]);
ylabel('$\sigma^{2}_{1}$','interpreter','latex')
set(gca,'fontsize',12);
subplot(2,2,3)
histogram(muz1,100);
ylabel('$\mu_{z1}$','interpreter','latex')
set(gca,'fontsize',12);
subplot(2,2,4)
histogram(sigmaz2_1,100);
ylabel('$\sigma^{2}_{z1}$','interpreter','latex')
set(gca,'fontsize',12);

%% Nasdaq
figure(8)
subplot(2,2,1)
histogram(mu2,100);%ylim([-0.5 0.5]);
hold on
hold off
ylabel('$\mu_{2}$','interpreter','latex')
set(gca,'fontsize',12);
subplot(2,2,2)
histogram(sigma2_2,100);%ylim([-0.5 0.5]);
ylabel('$\sigma^{2}_{2}$','interpreter','latex')
set(gca,'fontsize',12);
subplot(2,2,3)
histogram(muz2,100);
ylabel('$\mu_{z2}$','interpreter','latex')
set(gca,'fontsize',12);
subplot(2,2,4)
histogram(sigmaz2_2,100);
ylabel('$\sigma^{2}_{z2}$','interpreter','latex')
set(gca,'fontsize',12);
%% Intensity and Correlation
figure(9)
subplot(3,1,1)
histogram(lambda_draw,100)
ylabel('$\lambda$','interpreter','latex')
set(gca,'fontsize',12);
subplot(3,1,2)
histogram(corr,100)
ylabel('$\rho_{12}$','interpreter','latex')
set(gca,'fontsize',12);
subplot(3,1,3)
histogram(corr_z,100)
ylabel('$\rho_{z12}$','interpreter','latex')
set(gca,'fontsize',12);
%% Jump times
startDate = datenum('03-01-2007');
endDate=datenum('08-12-2022');
yData = linspace(startDate,endDate,T);
figure(10)
yyaxis left;
tt=floor(T);
sl=(mean(H(1:nsim,1:tt))>0.284);
area(yData(1:tt),mean(H(1:nsim,1:tt)),'facecolor',[0.85,0.8,0.8],'linestyle','none')
hold on;
stairs(yData(1:tt),sl,'r-');
ylim([-0.1,2.1]);
%
yyaxis right;
rgT=(1:tt);
plot(yData(rgT)/100,abs(yData(rgT))/100,'color',[0 0 0]);
hold on;
ylim([-2.01 3]);
hold off;
ax = gca;
ax.XTick = yData;
xticks((yData(1):400:yData(tt)));
datetick('x','dd/mm/yy','keepticks')
legend('Not Jump $\left(J_{t}=0\right)$','Jump $\left(J_t=1\right)$','interpreter','latex')
set(gca,'fontsize',10);
print(gcf,'VolHat2a.eps','-depsc');