function [ptf_value, weights_time] = OutOfSamplePerformance( ...
    w, prices_val, dates_val, V0, rebalanceMonths, name_ptf, ...
    flag_performance_table, flag_ptf_vs_assets, flag_single_assets)
% OUTOFSAMPLEPERFORMANCE simulates and analyzes the out-of-sample behavior
% of a given portfolio under periodic rebalancing.
%
% INPUTS:
%   w                     - Column or row vector of portfolio weights (Nx1 or 1xN).
%                           Weights must sum to 1 (within numerical tolerance).
%   prices_val            - Matrix of asset prices over the evaluation period
%                           (T x N), where T is the number of dates and N is
%                           the number of assets.
%   dates_val             - Column vector (T x 1) of datetime objects
%                           corresponding to the rows of prices_val.
%   V0                    - Initial portfolio notional (starting value).
%   rebalanceMonths       - Rebalancing frequency in months (e.g., 6 for
%                           semi-annual rebalancing).
%   name_ptf              - String or char with the portfolio name to be used
%                           in the figure super-title ("PORTFOLIO <name_ptf>").
%   flag_single_assets    - Boolean flag:
%                           true  -> plot evolution of selected weights over time;
%                           false -> do not plot weights.
%   flag_performance_table- Boolean flag:
%                           true  -> print performance statistics to the console;
%                           false -> do not print them.
%   flag_ptf_vs_assets    - Boolean flag:
%                           true  -> plot portfolio vs normalized asset prices;
%                           false -> do not plot this comparison.
%
% OUTPUTS:
%   ptf_value             - Column vector (T x 1) of portfolio values over time.
%   weights_time          - Matrix (T x N) of portfolio weights over time
%                           after each rebalancing step and daily price move.
%
% DESCRIPTION:
%   The function:
%       1) Checks that the weights sum to 1 (within a small numerical tolerance).
%       2) Calls simulateRebalancedPortfolio to generate the out-of-sample
%          portfolio value and time-varying weights.
%       3) Selects the assets to display in plots as those with initial
%          weight strictly greater than 5% (w_i > 0.05).
%       4) Optionally prints performance metrics via getPerformanceMetrics
%          depending on flag_performance_table.
%       5) Optionally creates a figure (or combined figure) with:
%            - evolution of selected weights over time (flag_single_assets),
%            - portfolio vs normalized asset prices (flag_ptf_vs_assets),
%          always using a global title "PORTFOLIO <name_ptf>".
%

    % ---------------------------------------------------------------------
    % Basic sanity check on portfolio weights (they should sum to 1)
    % ---------------------------------------------------------------------
    tol   = 1e-8;              % numerical tolerance
    w_sum = sum(w);
    if abs(w_sum - 1) > tol
        error('OutOfSamplePerformance:InvalidWeights', ...
              'Portfolio weights must sum to 1 (current sum = %.10f).', w_sum);
    end

    % ---------------------------------------------------------------------
    % Simulate portfolio value and weights over time with periodic rebalancing
    % ---------------------------------------------------------------------
    [ptf_value, weights_time] = simulateRebalancedPortfolio( ...
        w, prices_val, dates_val, V0, rebalanceMonths);

    % ---------------------------------------------------------------------
    % Select assets to display: weights strictly greater than 5%
    % ---------------------------------------------------------------------
    idxAssets = find(w > 0.05);

    % ---------------------------------------------------------------------
    % Compute performance metrics (printed only if flag_performance_table)
    % ---------------------------------------------------------------------
    [annRet_sim, annVol_sim, Sharpe_sim, MaxDD_sim, Calmar_sim] = ...
        getPerformanceMetrics(ptf_value);

    if flag_performance_table
        fprintf('--- Portfolio performance (%s) ---\n', name_ptf);
        fprintf('Annualized return:         %.4f\n', annRet_sim);
        fprintf('Annualized volatility:     %.4f\n', annVol_sim);
        fprintf('Sharpe ratio:              %.4f\n', Sharpe_sim);
        fprintf('Maximum drawdown:          %.4f\n', MaxDD_sim);
        fprintf('Calmar ratio:              %.4f\n', Calmar_sim);
    end
    fprintf('\n');


    % ---------------------------------------------------------------------
    % Plots (controlled by flags)
    %  - flag_single_assets   -> evolution of selected weights
    %  - flag_ptf_vs_assets   -> portfolio vs normalized asset prices
    % If both are true, they are combined in a single figure with 2 subplots
    % SIDE BY SIDE. In any case, the figure has a global title
    % "PORTFOLIO <name_ptf>".
    % ---------------------------------------------------------------------
    doWeightsPlot = logical(flag_single_assets);
    doComparePlot = logical(flag_ptf_vs_assets);
    nSubplots     = doWeightsPlot + doComparePlot;

    if nSubplots > 0
        % Wider figure to avoid squashed plots
        figure('Position', [100 100 1200 500]);

        % Global title with portfolio name
        sgtitle(sprintf('PORTFOLIO %s', name_ptf), ...
                'FontWeight', 'bold');

        subplotIndex = 1;

        % -------------------------------------------------------------
        % (1) Evolution of selected weights over time
        % -------------------------------------------------------------
        if doWeightsPlot
            % Subplots arranged side by side
            subplot(1, nSubplots, subplotIndex);
            subplotIndex = subplotIndex + 1;

            if ~isempty(idxAssets)
                plot(dates_val, weights_time(:, idxAssets), 'LineWidth', 1.2);
                grid on;
                hold on;

                % Dynamic legend based on selected assets
                legNames = arrayfun(@(i) sprintf('Asset %d', i), ...
                                    idxAssets, 'UniformOutput', false);
                legNames = legNames(:)';   % force row cell array
                legend(legNames, 'Location', 'best');
            else
                % No asset above 5%: plot all weights with a generic legend
                plot(dates_val, weights_time, 'LineWidth', 0.5);
                grid on;
                legend('All assets (no w_i > 5%)', 'Location', 'best');
            end

            xlabel('Date');
            ylabel('Weight');
            title('Evolution of selected weights w_i(t)');
        end

        % -------------------------------------------------------------
        % (2) Portfolio vs individual assets (normalized prices)
        % -------------------------------------------------------------
        if doComparePlot
            subplot(1, nSubplots, subplotIndex);

            % Portfolio equity curve (already in currency units)
            plot(dates_val, ptf_value, 'LineWidth', 1.5);
            hold on;
            grid on;

            if ~isempty(idxAssets)
                % Prices normalized to V0 at the first date
                norm_prices = V0 * prices_val(:, idxAssets) ./ ...
                                    prices_val(1, idxAssets);

                plot(dates_val, norm_prices);

                % Build legend: first portfolio, then assets (as cell row)
                legNamesAssets = arrayfun(@(i) sprintf('Asset %d', i), ...
                                          idxAssets, 'UniformOutput', false);
                legNamesAssets = legNamesAssets(:)';   % force row
                legendEntries  = [{'Portfolio'}, legNamesAssets];
                legend(legendEntries, 'Location', 'best');
            else
                % Only portfolio (no asset with weight > 5%)
                legend('Portfolio', 'Location', 'best');
            end

            xlabel('Date');
            ylabel('Value');
            title('Portfolio vs normalized asset prices');
        end

    end

end
