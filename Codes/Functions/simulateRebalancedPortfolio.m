function [ptf_value, weights_time] = simulateRebalancedPortfolio(w_opt, prices_val, dates, V0, rebalanceMonths)
% rebalancePortfolio
% Computes portfolio value and weights over time with periodic rebalancing.
%
% INPUTS:
%   w_opt          : N x 1 vector of target portfolio weights
%   prices_val     : T x N matrix of asset prices over time
%   dates          : T x 1 datetime vector corresponding to prices_val
%   V0             : scalar, initial portfolio value
%   rebalanceMonths: scalar, number of months between rebalancing (e.g. 6)
%
% OUTPUTS:
%   ptf_value      : T x 1 vector, portfolio value over time
%   weights_time   : T x N matrix, portfolio weights over time

    T       = length(dates);
    nAssets = length(w_opt);

    ptf_value   = zeros(T, 1);
    weights_time = zeros(T, nAssets);   % each row: w(t)'

    % initial number of units of each asset
    n_asset = (w_opt * V0) ./ prices_val(1, :)';   % N x 1

    % initial portfolio value and weights
    ptf_value(1)    = prices_val(1, :) * n_asset;  
    weights_time(1,:) = (prices_val(1, :)' .* n_asset ./ ptf_value(1))';

    % last rebalancing date (initially: first date)
    last_rebal_date = dates(1);

    for t = 2:T

        % current portfolio value with existing holdings (before rebalancing)
        V_curr = prices_val(t, :) * n_asset;

        % check if at least rebalanceMonths months have passed
        if dates(t) >= last_rebal_date + calmonths(rebalanceMonths)
            % rebalance: reset weights to w_opt keeping portfolio value V_curr
            n_asset = (w_opt * V_curr) ./ prices_val(t, :)';
            last_rebal_date = dates(t);
        end

        % update portfolio value
        ptf_value(t) = prices_val(t, :) * n_asset;

        % current weights
        weights_time(t,:) = (prices_val(t, :)' .* n_asset ./ ptf_value(t))';
    end

end
