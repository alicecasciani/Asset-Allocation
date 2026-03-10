function f = riskParityObjective(w, Sigma)
    % Objective function for Equal Risk Contribution (Risk Parity)
    % w     : N x 1 vector of portfolio weights
    % Sigma : N x N covariance matrix of asset returns
    %
    % This function returns a VECTOR of deviations of each asset's
    % risk contribution from the common target. When used together
    % with FMINIMAX, MATLAB minimizes the infinity norm of this vector,
    % i.e. it minimizes the maximum absolute deviation:
    %
    %   min_w  || RC(w) - target ||_inf
    %
    % so that all assets' risk contributions become as equal as possible.

    % Total portfolio volatility σ_p
    sigma_p = sqrt(w' * Sigma * w);

    % Marginal risk contributions (MRC_i = ∂σ_p/∂w_i)
    mrc = Sigma * w ./ sigma_p;      % marginal risk contributions

    % Absolute Risk Contributions: RC_i = w_i * MRC_i
    RC  = w .* mrc;                  % absolute risk contributions

    % Target risk contribution: equal share of total risk σ_p / N
    target = sigma_p / length(w);

    % Vector of deviations from the target (RC_i - target)
    % FMINIMAX will minimize the maximum absolute component of this vector
    f = RC - target;
end
