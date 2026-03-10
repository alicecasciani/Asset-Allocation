clear all
close all
clc

addpath('./Functions')
addpath('./Data')

%% ====================== 1. Data Import and Preprocessing ======================

% Read input data
table_prices           = readtable('asset_prices.csv');
mapping_table          = readtable('mapping_table.csv');
capitalization_weights = readtable('capitalization_weights.csv'); 

% Transform prices from table to timetable
dt     = table_prices{:,1};
values = table_prices{:,2:end};
nm     = table_prices.Properties.VariableNames(2:end);  % asset names

myPrice_dt = array2timetable(values, 'RowTimes', dt, 'VariableNames', nm);

% Select in-sample period: 2018–2022
[prices_val, ~] = selectPriceRange(myPrice_dt, '01/01/2018', '31/12/2022'); 

%% ====================== 2. Returns and Moments ======================

daysPerYear = 252;

% Daily log-returns: (T-1) x N
LogRet = log(prices_val(2:end,:) ./ prices_val(1:end-1,:));

% Daily moments
ExpRet_daily    = mean(LogRet);       % 1 x N
CovMatrix_daily = cov(LogRet);        % N x N
Std_daily       = std(LogRet);        % 1 x N
numAssets       = size(LogRet, 2);

% Annualized moments
ExpRet_ann    = ExpRet_daily * daysPerYear;         % 1 x N
CovMatrix_ann = CovMatrix_daily * daysPerYear;      % N x N
Std_ann       = Std_daily * sqrt(daysPerYear);      % 1 x N

% Individual asset volatilities (annualized)
vol_i = sqrt(diag(CovMatrix_ann));                  % N x 1

%% ====================== 3. Map Assets to Macro Groups ======================

[tf, loc] = ismember(nm', mapping_table.Asset);
if ~all(tf)
    error('Some assets in asset_prices.csv are not present in mapping_table.csv');
end

groups = mapping_table.MacroGroup(loc);   % e.g. 'Cyclical','Neutral','Defensive'

isCyc = strcmp(groups,'Cyclical');
isDef = strcmp(groups,'Defensive');
isNeu = strcmp(groups,'Neutral'); 

%% ====================== 4. Random Portfolios (for visualization) ======================

N = 100000;  % number of random portfolios

RetPtfs    = zeros(1, N);
VolaPtfs   = zeros(1, N);
SharpePtfs = zeros(1, N);

for n = 1:N
    w = rand(1, numAssets);
    w = w ./ sum(w);  % normalize weights to sum to 1

    RetPtfs(n)    = w * ExpRet_ann';
    VolaPtfs(n)   = sqrt(w * CovMatrix_ann * w');
    SharpePtfs(n) = RetPtfs(n) / VolaPtfs(n);
end

% Plot: random portfolios in mean–variance space
figure;
scatter(VolaPtfs, RetPtfs, [], SharpePtfs, 'filled')
colorbar
title('Random Portfolios: Expected Return vs Volatility')
xlabel('Volatility (annualized)')
ylabel('Expected return (annualized)')

%% ====================== 5. Constrained Efficient Frontier ======================

% Portfolio object with constraints
p    = Portfolio('AssetList', nm);
nPort = 100;

% Long-only with upper bound 25% per asset
lb = zeros(numAssets, 1);
ub = 0.25 * ones(numAssets, 1);
p  = setBounds(p, lb, ub);

% Logical masks for groups (row vectors)
isCyc = isCyc(:)';   
isDef = isDef(:)';

maskDef    = double(isDef);            % sum of Defensive weights
maskCycDef = double(isCyc | isDef);    % sum of Cyclical + Defensive weights

% Linear constraints A * w <= b
A = [
       maskDef;            % sum_Def      <= 0.40
       maskCycDef;         % sum_CycDef   <= 0.75
      -maskCycDef          % -sum_CycDef  <= -0.45  -> sum_CycDef >= 0.45
    ];
b = [0.40; 0.75; -0.45];

p = setInequality(p, A, b);

% Full investment constraint
p = setBudget(p, 1, 1);

% Set asset moments
P = setAssetMoments(p, ExpRet_ann, CovMatrix_ann);

% Compute constrained efficient frontier
W_OPT_estimate = estimateFrontier(P, nPort);    % numAssets x nPort

% Portfolio moments on the classical frontier
[portRisk, portExpRet] = estimatePortMoments(P, W_OPT_estimate);
% portRisk   : [1 x nPort] standard deviations
% portExpRet : [1 x nPort] expected returns

%% ====================== 6. Classical MVP and Maximum-Sharpe ======================

% Minimum-Variance Portfolio (classical)
[~, idxMinVar] = min(portRisk);
wMinVar    = W_OPT_estimate(:, idxMinVar);
retMinVar  = portExpRet(idxMinVar);
riskMinVar = portRisk(idxMinVar);

% Maximum-Sharpe Portfolio (classical)
rf       = 0.00;                       % risk-free (set to zero here)
excessRet = portExpRet - rf;
Sharpe    = excessRet ./ portRisk;

[~, idxMaxSharpe] = max(Sharpe);
wMaxSharpe    = W_OPT_estimate(:, idxMaxSharpe);
retMaxSharpe  = portExpRet(idxMaxSharpe);
riskMaxSharpe = portRisk(idxMaxSharpe);
maxSharpeVal  = Sharpe(idxMaxSharpe);

%% ====================== 7. Robust Frontier via Resampling ======================

tic
fprintf('\n=== Robust Resampling Procedure ===\n');

p = Portfolio('AssetList', nm);

% Same constraints as classical frontier
nPort = 100;
lb    = zeros(numAssets,1);
ub    = 0.25 * ones(numAssets,1);
p     = setBounds(p, lb, ub);

% Rebuild masks (row vectors)
isCyc = isCyc(:)';   
isDef = isDef(:)';

maskDef    = double(isDef);            % sum of Defensive weights
maskCycDef = double(isCyc | isDef);    % sum of Cyclical + Defensive weights

% Linear constraints A * w <= b
A = [
       maskDef;            % sum_Def      <= 0.40
       maskCycDef;         % sum_CycDef   <= 0.75
      -maskCycDef          % -sum_CycDef  <= -0.45  -> sum_CycDef >= 0.45
    ];
b = [0.40; 0.75; -0.45];

p = setInequality(p, A, b);
p = setBudget(p, 1, 1);   % full investment

% Resampling parameters
nResampling = 100;
T           = size(LogRet,1);   % number of historical observations
rf          = 0;                % risk-free for Sharpe (kept at 0)

% Preallocation for frontier weights across resamples
Weights = zeros(numAssets, nPort, nResampling);

rng(42)  % fixed seed for reproducibility
for i = 1:nResampling
    
    % 1) Simulate a daily return series of length T from N(mu_daily, Sigma_daily)
    R_sim = mvnrnd(ExpRet_daily, CovMatrix_daily, T);   % T x numAssets
    
    % 2) Estimate annualized moments from the simulated sample
    ExpRet_i = mean(R_sim) * daysPerYear;      % 1 x numAssets
    Cov_i    = cov(R_sim) * daysPerYear;       % numAssets x numAssets
    
    % 3) Set simulated moments in the portfolio object
    P_sim = setAssetMoments(p, ExpRet_i, Cov_i);
    
    % 4) Compute the constrained frontier for this resample
    W_OPT_i = estimateFrontier(P_sim, nPort);  % numAssets x nPort
    
    % Store weights
    Weights(:,:,i) = W_OPT_i;
end

% Robust frontier: average the weights across resamples
W_OPT_robust = mean(Weights, 3);   % numAssets x nPort

% Compute risk/return on the robust frontier using original (in-sample) moments
Ret_robust  = (ExpRet_ann(:)' * W_OPT_robust);                
Risk_robust = sqrt(diag(W_OPT_robust' * CovMatrix_ann * W_OPT_robust))';  

% Sharpe ratio for each robust portfolio
Sharpe_robust = (Ret_robust - rf) ./ Risk_robust;

%% ====================== 8. Robust MVP (C) and Robust MSRP (D) ======================

% Portfolio C: Robust Minimum-Variance Portfolio
[~, idxMinVar_rob] = min(Risk_robust);
w_C      = W_OPT_robust(:, idxMinVar_rob);
retC    = Ret_robust(idxMinVar_rob);
riskC   = Risk_robust(idxMinVar_rob);
sharpeC = Sharpe_robust(idxMinVar_rob);

% Portfolio D: Robust Maximum-Sharpe Portfolio
[~, idxMaxSharpe_rob] = max(Sharpe_robust);
w_D      = W_OPT_robust(:, idxMaxSharpe_rob);
retD    = Ret_robust(idxMaxSharpe_rob);
riskD   = Risk_robust(idxMaxSharpe_rob);
sharpeD = Sharpe_robust(idxMaxSharpe_rob);

elapsedTime = toc;
fprintf('Robust resampling completed in %.4f seconds.\n\n', elapsedTime);

%% ====================== 9. Print Results ======================
fprintf('================ EXERCISE 1: Constrained and Robust Efficient Frontier =================\n');
fprintf('================ Classical Constrained Frontier =================\n');
fprintf('Minimum-Variance Portfolio (Unrobust)\n');
fprintf('  Return    : %.4f\n', retMinVar);
fprintf('  Volatility: %.4f\n', riskMinVar);
fprintf('  Weights   :\n');
disp(wMinVar);

fprintf('Maximum-Sharpe Portfolio (Unrobust)\n');
fprintf('  Return    : %.4f\n', retMaxSharpe);
fprintf('  Volatility: %.4f\n', riskMaxSharpe);
fprintf('  Sharpe    : %.4f\n', maxSharpeVal);
fprintf('  Weights   :\n');
disp(wMaxSharpe);

fprintf('\n================ Robust Frontier (Resampling) ====================\n');
fprintf('Portfolio C – Robust Minimum-Variance Portfolio\n');
fprintf('  Return    : %.4f\n', retC);
fprintf('  Volatility: %.4f\n', riskC);
fprintf('  Sharpe    : %.4f\n', sharpeC);
fprintf('  Weights   :\n');
disp(w_C);

fprintf('Portfolio D – Robust Maximum-Sharpe Portfolio\n');
fprintf('  Return    : %.4f\n', retD);
fprintf('  Volatility: %.4f\n', riskD);
fprintf('  Sharpe    : %.4f\n', sharpeD);
fprintf('  Weights   :\n');
disp(w_D);

%% === Exercise 2 – Black–Litterman Model ===
fprintf('================ EXERCISE 2: Black-Litterman Model =================\n');
%% Calculate simple returns and Covariance Matrix
Ret         = tick2ret(prices_val);  % simple returns 
numAssets   = size(Ret, 2);
CovMatrix   = cov(Ret);
CovMatrix_simple_ann = CovMatrix * daysPerYear;  

ExpRet_simple      = mean(Ret);                     
ExpRet_simple_ann  = ExpRet_simple * daysPerYear;         

% Annualized expected returns
fprintf('\n=== Annualized expected returns (sample) ===\n');
Tab_ExpRet = table(nm', ExpRet_simple_ann', ...
    'VariableNames', {'Asset','ExpRet_ann'});
disp(Tab_ExpRet);

%% a) Compute equilibrium returns 

assetNames  = string(capitalization_weights.Asset);
MacroGroup  = string(capitalization_weights.MacroGroup);
wMKT        = capitalization_weights.MarketWeight;

tau = 1/length(Ret);  
% rf = 0; previously initialized
Rm_simple = Ret * wMKT; 

% Alternative computation of lambda. But then we chose to set lambda at 1.2, as in
% the lectures 

% mu_m_ann = (1 + mean(Rm_simple))^daysPerYear - 1;   
% var_m_ann = var(Rm_simple) * daysPerYear;   
% lambda = (mu_m_ann - rf) / var_m_ann;

lambda = 1.2;
mu_mkt = lambda * CovMatrix_simple_ann * wMKT;  
C      = tau * CovMatrix_simple_ann;    

fprintf('\n(a) Equilibrium returns annual:\n');
Tab_Equil = table(assetNames, mu_mkt, ...
    'VariableNames', {'Asset','EquilibriumRet_ann'});
disp(Tab_Equil);

% Plot prior distribution
X_prior = mvnrnd(mu_mkt, C, 200);  
figure;
histogram(X_prior);  
title('Prior Distribution of Returns (Equilibrium)');
xlabel('Annual Return')
ylabel('Frequency')

%% b) Introduce the views 
v   = 3;                           % number of views
P      = zeros(v, numAssets); 
q      = zeros(v, 1);        
Omega  = zeros(v);

% View 1: Cyclical assets expected to outperform Defensive ones by 2% annualized.  
Cyclical_idx  = find(MacroGroup == "Cyclical");
Defensive_idx = find(MacroGroup == "Defensive");

P(1, Cyclical_idx)  =  1 / length(Cyclical_idx);
P(1, Defensive_idx) = -1 / length(Defensive_idx);
q(1) = 0.02; 

% View 2: Asset 3 is expected to outperform Asset 11 by +1% annualized
P(2,3)  =  1;
P(2,11) = -1;
q(2)    = 0.01; 

% View 3: Asset 7 is expected to outperform the average Defensive group by +0.5% annualized
P(3,7)               =  1; 
P(3, Defensive_idx)  = -1 / length(Defensive_idx);
q(3)                 = 0.005; 

% Compute Omega as tau*P*Cov*P' (diagonal approximation)
for i = 1:v
    Omega(i,i) = tau * P(i,:) * CovMatrix_simple_ann * P(i,:)';
end

% Plot views distribution
X_views = mvnrnd(q, Omega, 200);  
figure;
hold on
for i = 1:v
    histogram(X_views(:,i), 'DisplayName', ['View ' num2str(i)], 'FaceAlpha',0.5);
end
legend
title('Distribution of Views')
hold off

%% c) Obtain posterior expected returns and compute the efficient frontier under standard
%     constraints (full investment & no short-selling ) using the in-sample data.

invC = C \ eye(numAssets);      

muBL = (invC + P'/Omega*P) \ (invC*mu_mkt + P'/Omega*q);
 
covBL = inv(invC + P'/Omega*P);
covBL = (covBL + covBL')/2;  % to assure simmetry 
Cov_total = CovMatrix_simple_ann + covBL;
Cov_total = (Cov_total + Cov_total')/2; % to assure simmetry 

% Compare prior vs BL posterior returns 
TBL = table(assetNames, mu_mkt, muBL, ...
    'VariableNames', ["Asset","PriorReturnAnnual","BLReturnAnnual"]);
disp(TBL) 

% Black-Litterman PTF
portBL = Portfolio('NumAssets', numAssets, 'Name', 'MV with BL');
portBL = setDefaultConstraints(portBL);
portBL = setAssetMoments(portBL, muBL, CovMatrix_simple_ann+covBL);

% Efficient frontier 
frontBL = estimateFrontier(portBL, 100);
[vol_BL, ret_BL] = estimatePortMoments(portBL,frontBL);

figure;
plot(vol_BL, ret_BL, 'LineWidth', 2);
xlabel('Volatility'); ylabel('Expected Return');
title('Efficient Frontier with Black–Litterman Views');
grid on;

%% d) Minimum Variance and Maximum Sharpe Ratio ptfs 

% Minimum Variance ptf 

% Portfolio E: Minimum Variance
w_minVarBL = estimateFrontierLimits(portBL, 'min');
[vol_minVarBL, ret_minVarBL] = estimatePortMoments(portBL, w_minVarBL);

PortfolioE_BL.weights = w_minVarBL;
PortfolioE_BL.vol     = vol_minVarBL;
PortfolioE_BL.ret     = ret_minVarBL;

% Portfolio F: Maximum Sharpe
w_maxSharpeBL = estimateMaxSharpeRatio(portBL);
[vol_maxSharpeBL, ret_maxSharpeBL] = estimatePortMoments(portBL, w_maxSharpeBL);

PortfolioF_BL.weights = w_maxSharpeBL;
PortfolioF_BL.vol     = vol_maxSharpeBL;
PortfolioF_BL.ret     = ret_maxSharpeBL;

fprintf('\nPortfolio E (BL Min Var):  vol = %.4f, ret = %.4f\n', vol_minVarBL, ret_minVarBL);
fprintf('Portfolio F (BL Max Sharpe): vol = %.4f, ret = %.4f\n', vol_maxSharpeBL, ret_maxSharpeBL);

fprintf('\nPortfolio E (BL Min Var):\n');
disp(w_minVarBL)

fprintf('Portfolio F (BL Max Sharpe):\n');
disp(w_maxSharpeBL)

% Check if Sharpe Ratio F> Sharpe Ratio E
Sharpe_E =(ret_minVarBL-rf)/vol_minVarBL; 
Sharpe_F =(ret_maxSharpeBL-rf)/vol_maxSharpeBL; 


%% Classical ptfs 

port = Portfolio('NumAssets', numAssets, 'Name', 'Mean-Variance');
port = setDefaultConstraints(port);
port = setAssetMoments(port, mean(Ret), CovMatrix_simple_ann);

w_minVar = estimateFrontierLimits(port, 'min');
[vol_minVar, ret_minVar] = estimatePortMoments(port, w_minVar);

w_maxSharpe = estimateMaxSharpeRatio(port);
[vol_maxSharpe, ret_maxSharpe] = estimatePortMoments(port, w_maxSharpe);

% Compare classical vs BL portfolio weights
Tweights = table(assetNames, w_minVarBL, w_minVar, w_maxSharpeBL, w_maxSharpe, ...
    'VariableNames', ["Asset","BL-MinVariance","Classical MinVariance","BL-MaxSharpe","Classical MaxSharpe"]);
disp(Tweights)

% Plot
figure;
subplot(2,2,4)
idxMS = w_maxSharpe > 0.001;
pie(w_maxSharpe(idxMS), assetNames(idxMS))
title('Classical Maximum Sharpe Portfolio')

subplot(2,2,3)
idxBLMS = w_maxSharpeBL > 0.001;
pie(w_maxSharpeBL(idxBLMS), assetNames(idxBLMS))
title('Black-Litterman Maximum Sharpe Portfolio')

subplot(2,2,2)
idxMV = w_minVar> 0.001;
pie(w_minVar(idxMV), assetNames(idxMV))
title('Classical Minimum Variance Portfolio')

subplot(2,2,1)
idxBLMV = w_minVarBL > 0.001;
pie(w_minVarBL(idxBLMV), assetNames(idxBLMV))
title('Black-Litterman Minimum Variance Portfolio')

%% Impact of views on portfolio (Delta weights) 

% BL MaxSharpe vs Classical MaxSharpe
delta_weights = w_maxSharpeBL - w_maxSharpe;
figure;
bar(delta_weights);
xlabel('Asset Index'); 
ylabel('Change in Weight');
title('Impact of Views on Max Sharpe Portfolio Allocation (BL - Classical)');

% Impact of views on Min Variance portfolio
delta_weights_minVar = w_minVarBL - w_minVar;

figure;
bar(delta_weights_minVar);
xlabel('Asset Index');
ylabel('Change in Weight');
title('Impact of Views on Min Var Portfolio Allocation (BL - Classical)');

%% Contribution of each view to expected returns
contrib = zeros(numAssets, v);

for i = 1:v
    P_i     = P(i,:)';
    Omega_i = Omega(i,i);

    contrib(:,i) = CovMatrix_simple_ann * P_i / ...
        (P_i' * CovMatrix_simple_ann * P_i + Omega_i) ...
        * (q(i) - P_i' * mu_mkt);
end

muBL_contrib = mu_mkt + sum(contrib,2);   

figure;
bar(contrib);
xlabel('Asset Index');
ylabel('Annualized Contribution');
title('Contribution of Each View to BL Expected Returns');
legend("View1","View2","View3");

%% Bar plot: classical vs BL weights 
figure;
bar([w_maxSharpe w_maxSharpeBL]);
legend('Classical','BL');
title('Comparison of Portfolio Weights: Classical vs BL (Max Sharpe)');
xlabel('Asset Index');
ylabel('Weight');

%% Exercise 3 – Diversification-Based Optimization ===

%% Common contstraints (G, H, EW)
[prices_val, dates] = selectPriceRange(myPrice_dt, '01/01/2018', '31/12/2022');

% 0 ≤ wi ≤ 0.2
lb = zeros(numAssets,1);
ub = 0.2 * ones(numAssets,1);

% Sum of weights = 1
Aeq = ones(1,numAssets);
beq = 1;

% Group constraints: each macro-group ≥ 15%
% sum_{i in group} w_i ≥ 0.15   ↔  -sum_{i in group} w_i ≤ -0.15
A = [];
b = [];

if any(isCyc)
    rowC = zeros(1,numAssets);
    rowC(isCyc) = -1;
    A = [A; rowC];
    b = [b; -0.15];
end

if any(isNeu)
    rowN = zeros(1,numAssets);
    rowN(isNeu) = -1;
    A = [A; rowN];
    b = [b; -0.15];
end

if any(isDef)
    rowD = zeros(1,numAssets);
    rowD(isDef) = -1;
    A = [A; rowD];
    b = [b; -0.15];
end

% Initial guess
w0 = ones(numAssets,1) / numAssets;

opts = optimoptions('fmincon','Display','off', ...
                    'Algorithm','sqp','MaxIterations',1e8);

%% Portfolio G: Max Diversification Ratio
% Diversification ratio:
% DR(w) = [(w' * vol_i) / sqrt(w' * Sigma * w)]
fun_DR = @(w) - (w' * vol_i) / sqrt(w' * CovMatrix_ann * w);   
[w_G, fval_G] = fmincon(fun_DR, w0, A, b, Aeq, beq, lb, ub, [], opts);

DR = -fval_G;   % maximum DR value

%% Portfolio H: Equal Risk Contribution
% Objective: equal risk contribution
% Minimize the dispersion of risk contributions
fun_RPC = @(w) riskParityObjective(w, CovMatrix_ann);

opts_minmax = optimoptions('fminimax', ...
    'Display', 'off', ...
    'MaxIterations', 1e4, ...
    'ConstraintTolerance', 1e-8, ...
    'OptimalityTolerance', 1e-8, ...
    'StepTolerance', 1e-10, ...
    'UseParallel', false);

[w_H, fval_H] = fminimax(fun_RPC, w0, A, b, Aeq, beq, lb, ub, [], opts_minmax);

%% Benchmark Equally-Weighted
w_EW = ones(numAssets,1) / numAssets;

%% Metrics: DR, VOL, Sharpe, N_eff
% Diversification Ratio
DR_G = getDiversificationRatio(w_G, LogRet);
DR_H = getDiversificationRatio(w_H, LogRet);
DR_EW = getDiversificationRatio(w_EW, LogRet);

% Annualized volatility
vol_G  = sqrt(w_G' * CovMatrix_ann * w_G);
vol_H  = sqrt(w_H' * CovMatrix_ann * w_H);
vol_EW = sqrt(w_EW' * CovMatrix_ann * w_EW);

% Annualized expected return
ret_G  = ExpRet_ann * w_G;
ret_H  = ExpRet_ann * w_H;
ret_EW = ExpRet_ann * w_EW;

% Sharpe ratio
Sharpe_G  = (ret_G  - rf) / vol_G;
Sharpe_H  = (ret_H  - rf) / vol_H;
Sharpe_EW = (ret_EW - rf) / vol_EW;

% Effective number of assets (Herfindahl index)
Neff_G  = 1 / sum(w_G.^2);
Neff_H  = 1 / sum(w_H.^2);
Neff_EW = 1 / sum(w_EW.^2);

% Summary table
Results = table(...
    [DR_G; DR_H; DR_EW], ...
    [vol_G;  vol_H;  vol_EW], ...
    [Sharpe_G; Sharpe_H; Sharpe_EW], ...
    [Neff_G; Neff_H; Neff_EW], ...
    'VariableNames', {'DivRatio','Vol_Ann','Sharpe','N_eff'}, ...
    'RowNames', {'G_MaxDR','H_RiskParity','EW_Benchmark'});

disp('================ EXERCISE 3: Diversification–Based Optimization ================')
disp(Results)

%% Plot of Weights
figure;
bar([w_G, w_H, w_EW])
legend({'G: Max DR','H: Risk Parity','EW'},'Location','bestoutside')
xlabel('Asset')
ylabel('Weight')
title('Comparison of G, H and Benchmark Portfolio Weights')
grid on


%% === Exercise 4 – PCA and Conditional Value-at-Risk ===
fprintf('================ EXERCISE 4: PCA and Conditional Value-at-Risk =================\n');
%% 4(a) PCA on covariance matrix – find k explaining at least 85% variance

% Standardize returns
RetStd = (LogRet - ExpRet_daily) ./ Std_daily; 
CovStd = cov(RetStd);

% full PCA
[factorLoading_full, factorRetn_full, latent_full, ~, explained_full] = pca(RetStd); 

% explained var
figure();
bar(explained_full);
title('Variance explained by each Principal Component');
xlabel('Principal Component');
ylabel('Explained Variance (%)');

% cumulative variance
n_list = linspace(1, 15, numAssets);
CumExplVar = cumsum(explained_full);
figure();
title('Total Percentage of Explained Variances for the first n-components')
plot(n_list,CumExplVar, 'm')
hold on
scatter(n_list,CumExplVar,'m', 'filled')
grid on
xlabel('Total number of Principal Components')
ylabel('Percentage of Explained Variances')

% find k
k = find(CumExplVar >= 85, 1, 'first');
fprintf('\n=== 4a) PCA ===\n');
fprintf('Smallest k with at least 85%% of variance: k = %d\n', k);

factorLoading = factorLoading_full(:,1:k);
factorRetn = factorRetn_full (:,1:k);
covarFactor = cov(factorRetn); 

% Rescale back to original return units
Lambda = diag(Std_daily); % diag matrix of std dev of original assets
D_std = diag(var(LogRet - (factorRetn*factorLoading' .* Std_daily+ ExpRet_daily))); % Idyiosyncratic risk
CovarPCA = Lambda * (factorLoading * covarFactor * factorLoading' + D_std) * Lambda; % PCA-based covariance matrix

% Reconstructed returns in the original units 
reconReturn = factorRetn * factorLoading' .* Std_daily+ ExpRet_daily;
unexplainedRetn = LogRet - reconReturn; % epsilon

%% 4(b) Portfolio I – Max Sharpe with PCA covariance & v1-neutrality

[V_eig, D_eig] = eig(CovMatrix_daily);
[~, idxMax] = max(diag(D_eig));
v1 = V_eig(:, idxMax);       
x0 = ones(numAssets,1)/numAssets;  % starting point: equal-weight portfolio

% Constraints: 0 <= w_i <= 0.25, w'v1 <= 0.5                              
lb = zeros(numAssets,1); 
ub = 0.25*ones(numAssets,1); 

% Full investment (sum of weights must be 1)
Aeq = ones(1,numAssets);
beq = 1;

% Inequality constraint: w' v1 <= 0.5  ->  (v1') * w <= 0.5
A  = v1';
b  = 0.5;

% Objective: maximize Sharpe = (mu'w - rf) / sqrt(w'Cov_PCA w)
% minimize -Sharpe
fun_sharpe_I = @(w) - ((ExpRet_daily * w - rf) / sqrt(w' * CovarPCA * w));

options = optimoptions('fmincon','Display','final','Algorithm','sqp');
[w_I, fval_I, exitflag_I] = fmincon(fun_sharpe_I, x0, A, b, Aeq, beq, lb, ub, [], options);

Sharpe_I_daily  = -fval_I;       
Sharpe_I_annual = sqrt(252) * Sharpe_I_daily;

% Check results
ExpRet_I_daily  = ExpRet_daily * w_I;
vol_I_daily  = sqrt(w_I' * CovarPCA * w_I);
vol_I_annual = vol_I_daily * sqrt(252);

% Recompute Sharpe ratios to validate consistency
Sharpe_I_daily_check  = ExpRet_I_daily / vol_I_daily;
Sharpe_I_annual_check = sqrt(252) * Sharpe_I_daily_check;

fprintf("\n=== Portfolio I (Max Sharpe, w'v1 <= 0.5) ===\n");
Tab_I = table(nm', w_I, 'VariableNames', {'Asset','Weight'});
disp(Tab_I);

fprintf('\n=== Portfolio I – Expected Return and Risk ===\n');
fprintf('Daily volatility          : %.6f\n', vol_I_daily);
fprintf('Annual volatility         : %.4f%%\n', 100 * vol_I_annual);
fprintf('Daily Sharpe (opt value)  : %.4f\n', Sharpe_I_daily);
fprintf('Daily Sharpe (recomputed) : %.4f\n', Sharpe_I_daily_check);
fprintf('Annual Sharpe (recomputed): %.4f\n', Sharpe_I_annual_check);
fprintf('PC1 exposure w''v1         : %.4f\n', w_I' * v1);


%% 4(b) Portfolio J – Min CVaR_5% with vol cap 15%% and 0<=w<=0.25

alpha = 0.95;            % CVaR at 5% tail
vol_cap = 0.15;          % 15% annualized volatility

% Objective: minimize CVaR_5% (historical, daily returns LogRet)
fun_CVaR = @(w) cvar_obj(w, LogRet, alpha);

% Standard constraints                            
% lb = zeros(numAssets,1); 
% ub = 0.25*ones(numAssets,1); 
% Aeq = ones(1,numAssets);
% beq = 1;

% Nonlinear constraint: vol(w) <= 15% annualized
nonlin_volcap = @(w) deal( sqrt(w'*CovMatrix_daily*w)*sqrt(daysPerYear) - vol_cap , [] );

optionsJ = optimoptions('fmincon','Display','iter','Algorithm','sqp','ConstraintTolerance',1e-3, 'OptimalityTolerance',1e-6, 'StepTolerance',1e-10);
[w_J, ~, ~] = fmincon(fun_CVaR, x0, [],[], Aeq, beq, lb, ub, nonlin_volcap, optionsJ);

% Performance of J (in-sample, daily)
pRet_J = LogRet * w_J;
VaR_J  = quantile(pRet_J, 1-alpha);
CVaR_J = -mean(pRet_J(pRet_J <= VaR_J));    % losses positive

ret_J  = ExpRet_ann * w_J;
vol_J  = sqrt(w_J' * CovMatrix_ann * w_J);
Sharpe_J = (ret_J - rf) / vol_J;

fprintf('\n=== Portfolio J (Min CVaR_5%%, vol<=15%%, w_i<=0.25) ===\n');


Tab_J = table(nm', w_J, 'VariableNames', {'Asset','Weight'});
disp(Tab_J);
fprintf('Portfolio J: ret = %.4f, vol = %.4f, Sharpe = %.4f, daily CVaR_5 = %.4f\n', ...
        ret_J, vol_J, Sharpe_J, CVaR_J);

%% Check the MVP under these constraints 
port_min = Portfolio('NumAssets', numAssets);
port_min = setDefaultConstraints(port_min);    % sum(w)=1, w>=0
port_min = setBounds(port_min, zeros(1,numAssets), 0.25*ones(1,numAssets));
port_min = setAssetMoments(port_min, ExpRet_ann, CovMatrix_ann);

w_minVarJ = estimateFrontierLimits(port_min, 'min');
sum(w_J)
g = sqrt(w_J' * CovMatrix_ann * w_J) - vol_cap;   
%vol_minVarJ = sqrt(w_minVarJ' * CovMatrix_ann * w_minVarJ);

% Given the in-sample covariance structure and the constraints (full investment, 0 ≤ wᵢ ≤ 0.25), the minimum 
% variance portfolio already exhibits an annual volatility above 15%. Therefore, a strict volatility cap at 15% 
% is infeasible in this universe. In practice, we keep the non-linear volatility constraint in the optimization 
% problem, but the solver converges to a solution with volatility 0.1799, which is the minimum level compatible
% with the given constraints

% In order to obtain a correct portfolio, we set the volatiliy cap equal to
% the minimum volatility: 17.99

vol_cap = 0.1799;  

% Nonlinear constraint: vol(w) <= 15% annualized
nonlin_volcap = @(w) deal( sqrt(w'*CovMatrix_daily*w)*sqrt(daysPerYear) - vol_cap , [] );

optionsJ = optimoptions('fmincon','Display','iter','Algorithm','sqp','ConstraintTolerance',1e-3, 'OptimalityTolerance',1e-6, 'StepTolerance',1e-10);

[w_J, fval_J, exitflag_J] = fmincon(fun_CVaR, x0, [],[], Aeq, beq, lb, ub, nonlin_volcap, optionsJ);

% Performance of J (in-sample, daily)
pRet_J = LogRet * w_J;
VaR_J  = quantile(pRet_J, 1-alpha);
CVaR_J = -mean(pRet_J(pRet_J <= VaR_J));   

ret_J  = ExpRet_ann * w_J;
vol_J  = sqrt(w_J' * CovMatrix_ann * w_J);
Sharpe_J = (ret_J - rf) / vol_J;

fprintf('\n=== Portfolio J (Min CVaR_5%%, vol<=17.99%%, w_i<=0.25) ===\n');

tol = 1e-6;      
w_J_round = round(w_J, 4);
w_J_round(abs(w_J_round) < tol) = 0;  

Tab_J = table(nm', w_J_round, 'VariableNames', {'Asset','Weight'});
disp(Tab_J);
fprintf('Portfolio J: ret = %.4f, vol = %.4f, Sharpe = %.4f, daily CVaR_5 = %.4f\n', ...
        ret_J, vol_J, Sharpe_J, CVaR_J);

%% 4(c) Tail risk (CVaR), Volatility, Max Drawdown – I vs J
fprintf('\n=== Portfolio I ===\n');
Tab_I_round = table(nm', round(w_I,4),'VariableNames', {'Asset','Weight'});
disp(Tab_I_round);

fprintf('\n=== Portfolio J ===\n');
Tab_J_round = table(nm', round(w_J,4), 'VariableNames', {'Asset','Weight'});
disp(Tab_J_round);

% Daily returns of portfolios (in-sample)
pRet_I = LogRet * w_I;
pRet_J = LogRet * w_J;

% CVaR 5% (daily)
VaR_I  = quantile(pRet_I, 1-alpha);
VaR_J  = quantile(pRet_J, 1-alpha);

CVaR_I = -mean(pRet_I(pRet_I <= VaR_I))*100; % (percentage)
CVaR_J = -mean(pRet_J(pRet_J <= VaR_J))*100;

equity_I = [100; 100 * exp(cumsum(pRet_I))];
equity_J = [100; 100 * exp(cumsum(pRet_J))];

% Performance metrics 
[annRet_I, annVol_I, Sharpe_I, MaxDD_I, Calmar_I] = getPerformanceMetrics(equity_I);
[annRet_J, annVol_J, Sharpe_J, MaxDD_J, Calmar_J] = getPerformanceMetrics(equity_J);

% I vs J table
perfTable_IJ = table( ...
    [CVaR_I; CVaR_J], ...
    [annVol_I; annVol_J], ...
    [MaxDD_I; MaxDD_J], ...
    'VariableNames', {'CVaR_5_daily','AnnVol','MaxDD'}, ...
    'RowNames', {'Portfolio I','Portfolio J'});

disp('=== Comparison Portfolio I vs Portfolio J ===');
disp(perfTable_IJ);

figure;
plot(dates, equity_I, 'LineWidth', 1.5); hold on;
plot(dates, equity_J, 'LineWidth', 1.5);

legend('Portfolio I (PCA Max Sharpe)', 'Portfolio J (Min CVaR)', 'Location','best');
xlabel('Date'); 
ylabel('Equity (base = 100)');
title('Equity curves – Portfolio I vs Portfolio J');
grid on;

vol_minVarJ = sqrt(w_minVarJ' * CovMatrix_ann * w_minVarJ);

%% Exercise 5 – Personal Strategy ===

[Vecs,Vals] = eig(CovMatrix_ann);
lambda      = diag(Vals);                         % eigenvalues (unsorted)

% Sort eigenvalues/eigenvectors in descending order
[lambda_sorted, idxEV] = sort(lambda, 'descend');
V_sorted               = Vecs(:, idxEV);

F = V_sorted(:, 1:3);
sig_f = diag(lambda_sorted(1:3));
mu = ExpRet_ann';

% specific variance vector
vec = zeros(numAssets, 1);
for i = 1 : numAssets
    var_explained = 0;
    for j = 1 : k
        var_explained = var_explained + lambda_sorted(j) * F(i, j)^2;
    end
    vec(i) = CovMatrix_ann(i, i) - var_explained;
end
D = diag(vec);

sig = @(w) w' * (F * sig_f * F' + D) * w;

gamma = 1;
f = @(w) mu'*w - gamma * sig(w);

% equality constraint: sum(w) = 1
Aeq = ones(1, numAssets);
beq = 1;

% bounds: 0 <= w_i <= 0.2
lb = zeros(numAssets, 1);
ub = 0.2 * ones(numAssets, 1);

% initial guess: equally-weighted portfolio
w0 = ones(numAssets, 1) / numAssets;

% fmincon MINIMIZES: we use the negative of f
obj = @(w) -f(w);

[w_opt_strategy, minus_f_opt, exitflag, output] = fmincon(obj, w0, ...
    [], [], Aeq, beq, lb, ub, [], opts);

% MAX value of the objective function
f_opt = -minus_f_opt;

% optimal weights
fprintf('\n\n==================== EXERCISE 5: Personal Allocation Strategy ====================\n\n');
disp('=== Optimal portfolio weights (%) ===');
for i = 1:length(w_opt_strategy)
    fprintf('%-10s : %6.2f %%\n', nm{i}, w_opt_strategy(i)*100);
end

%% Selection of a subset of Dates
[prices_val_validation, dates_validation] = selectPriceRange(myPrice_dt, '01/01/2023', '30/11/2024');

%% Portfolio performance with semi-annual rebalancing
V0 = 100;
rebalanceMonths = 6;

OutOfSamplePerformance(w_opt_strategy, prices_val_validation, dates_validation, ...
    V0, rebalanceMonths, 'Personal Strategy with rebalance', 1, 1, 1);

%% Portfolio performance without rebalancing
no_rebalance = Inf;
OutOfSamplePerformance(w_opt_strategy, prices_val_validation, dates_validation, ...
    V0, no_rebalance, 'Personal Strategy without rebalance', 1, 0, 1);

%% === Final Discussion ===

%% Rename of weights
w_A = wMinVar;
w_B = wMaxSharpe;
w_E = w_minVarBL;
w_F = w_maxSharpeBL;

%%w_
ptfLabels = 'A':'J';
nPortfolios = length(ptfLabels);

W_all = [w_A, w_B, w_C, w_D, w_E, w_F, w_G, w_H, w_I, w_J];

% Flags
flag_performance_table = 1;
flag_single_assets     = 0;
flag_ptf_vs_assets     = 0;

%%

fprintf('\n\n==================== IN SAMPLE SUMMARY ====================\n\n');
labels = 'A':'J';          
nPtf   = size(W_all, 2);         

AnnRet  = zeros(nPtf,1);
AnnVol  = zeros(nPtf,1);
SharpeR = zeros(nPtf,1);
MaxDDv  = zeros(nPtf,1);
CalmarR = zeros(nPtf,1);

for k = 1:nPtf

    L = labels(k);      
    w = W_all(:, k);    

    % Daily returns
    pRet = LogRet * w;
    ptf_value = [100; 100 * exp(cumsum(pRet))];

    % Performance metrics
    [annRet, annVol, sharpe, maxdd, calmar] = getPerformanceMetrics(ptf_value);

    fprintf('\n--- Portfolio performance (Portfolio %s) ---\n', L);
    fprintf('Annualized return:      %.4f\n', annRet);
    fprintf('Annualized volatility:  %.4f\n', annVol);
    fprintf('Sharpe ratio:           %.4f\n', sharpe);
    fprintf('Maximum drawdown:       %.4f\n', maxdd);
    fprintf('Calmar ratio:           %.4f\n', calmar);

    % Save for the table
    AnnRet(k)  = annRet;
    AnnVol(k)  = annVol;
    SharpeR(k) = sharpe;
    MaxDDv(k)  = maxdd;
    CalmarR(k) = calmar;
end

% Summary table
Portfolio = cellstr(labels'); 
resultsTable = table(Portfolio, AnnRet, AnnVol, SharpeR, MaxDDv, CalmarR, ...
    'VariableNames', {'Portfolio','AnnRet','AnnVol','Sharpe','MaxDD','Calmar'});

disp(resultsTable)

%% 
fprintf('\n\n==================== FINAL DISCUSSION ====================\n\n');
for k = 1:nPortfolios
    w_k = W_all(:, k);
    name_ptf_k = sprintf('Portfolio %c', ptfLabels(k));
    
    OutOfSamplePerformance( ...
        w_k, ...
        prices_val_validation, ...
        dates_validation, ...
        V0, ...
        no_rebalance, ...
        name_ptf_k, ...
        flag_performance_table, ...
        flag_ptf_vs_assets, ...
        flag_single_assets);
end

%% Plot asset prices evolution
norm_prices_validation = 100 * prices_val_validation ./ prices_val_validation(1, :);

figure('Position', [100 100 1200 500]);
plot(dates_validation, norm_prices_validation, 'LineWidth', 1.2);
grid on;

xlabel('Date');
ylabel('Normalized price (base = 100)');
title('All assets in validation period (normalized to 100 at start)');

% If you have asset names in variable "nm", use them in the legend
if exist('nm', 'var') && numel(nm) == size(prices_val_validation, 2)
    legend(nm, 'Location', 'bestoutside');
else
    % Fallback generic legend
    nAssets = size(prices_val_validation, 2);
    legNames = arrayfun(@(i) sprintf('Asset %d', i), 1:nAssets, 'UniformOutput', false);
    legend(legNames, 'Location', 'bestoutside');
end

%% Portfolio performance 2025
clc

prices_2025  = readtable('asset_prices_out_of_sample.csv');

% Transform prices from table to timetable
dt     = prices_2025{:,1};
values = prices_2025{:,2:end};
nm     = prices_2025.Properties.VariableNames(2:end);  % asset names

myPrice_2025 = array2timetable(values, 'RowTimes', dt, 'VariableNames', nm);

% Select out_of-sample period: 2025
[prices_val_2025, dates_2025] = selectPriceRange(myPrice_2025, '01/01/2025', '31/12/2025');

% Performance metrics
OutOfSamplePerformance(w_opt_strategy, prices_val_2025, dates_2025, ...
   V0, rebalanceMonths, 'Personal Strategy - 2025 OOS', 1, 1, 0);

% Comparison with equally weighted ptf
w_EW = ones(numAssets,1) / numAssets;
OutOfSamplePerformance(w_EW, prices_val_2025, dates_2025, ...
    V0, rebalanceMonths, 'EW - 2025 OOS', 1, 1, 0);

% Comparison with Min Var ptf
OutOfSamplePerformance(w_A, prices_val_2025, dates_2025, ...
    V0, rebalanceMonths, 'MVP - 2025 OOS', 1, 1, 0);