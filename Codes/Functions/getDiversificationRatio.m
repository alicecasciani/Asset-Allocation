function DR = getDiversificationRatio(x, Ret)
    % Computes the Diversification Ratio (DR) of a portfolio.
    %
    % x   : N x 1 vector of portfolio weights
    % Ret : T x N matrix of asset returns
    %
    % The Diversification Ratio is defined as:
    %
    %          sum_i ( w_i * σ_i )
    % DR =  ----------------------------
    %            σ_p   = sqrt(w' Σ w)
    %
    % where σ_i is the volatility of asset i, and σ_p is the total
    % portfolio volatility. A higher DR indicates greater risk diversification.

    vola = std(Ret);          % individual asset volatilities (σ_i)
    V = cov(Ret);             % covariance matrix (Σ)
    volaPtf = sqrt(x' * V * x);  % portfolio volatility (σ_p)

    DR = (x' * vola') / volaPtf; % diversification ratio
end
