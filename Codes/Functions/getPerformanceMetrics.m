function [annRet, annVol, Sharpe, MaxDD, Calmar] = getPerformanceMetrics(equity)
% GETPERFORMANCEMETRICS computes standard performance indicators
% for a given equity curve (portfolio value over time).
%
% INPUT:
%   equity : (T x 1) vector of portfolio values (e.g. starting from 100)
%
% OUTPUTS:
%   annRet  : annualized return (log-return based)
%   annVol  : annualized volatility (standard deviation of log-returns)
%   Sharpe  : Sharpe ratio (excess return over volatility, rf = 0)
%   MaxDD   : maximum drawdown
%   Calmar  : Calmar ratio (annualized return divided by |MaxDD|)


    % 1) Daily log returns
    LogRet = log(equity(2:end,:) ./ equity(1:end-1,:));
    daysPerYear = 252;

    % 2) Annualized returns 
    annRet = mean(LogRet)*daysPerYear;

    % 3) Annualized volatility 
    annVol = std(LogRet) * sqrt(daysPerYear);

    % 4) Sharpe ratio (assuming rf = 0)
    rf = 0;
    Sharpe = (annRet - rf) / annVol;

    % 5) Max Drawdown
    runningMax = cummax(equity);
    drawdowns  = equity ./ runningMax - 1; 
    MaxDD      = min(drawdowns);             

    % 6) Calmar ratio (annRet / |MaxDD|)
    Calmar = annRet / abs(MaxDD);

end