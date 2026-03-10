function cvar_val = cvar_obj(w, LogRet, alpha)
% CVAR_OBJ computes the Conditional Value-at-Risk (CVaR) of a portfolio.
%
% INPUTS:
%   w      : (N x 1) vector of portfolio weights
%   LogRet : (T x N) matrix of asset log-returns
%   alpha  : confidence level (e.g. 0.95 for 95% CVaR)
%
% OUTPUT:
%   cvar_val : positive scalar equal to the average loss in the worst
%              (1 - alpha) fraction of scenarios
%

    % 1) Portfolio returns across all scenarios
    pRet = LogRet * w;             % scenario returns

    % 2) Compute Value-at-Risk (VaR) at confidence level alpha
    VaR  = quantile(pRet, 1-alpha);

    % 3) Extract the tail of the distribution (worst (1 - alpha) scenarios)
    tail = pRet(pRet <= VaR);

    % 4) CVaR is the expected loss in the tail.
    cvar_val = -mean(tail);        % minimize losses in the 5% worst cases
end
