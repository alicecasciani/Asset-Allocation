function [prices_val, dates] = selectPriceRange(myPrice_dt, startDate, endDate)
% selectPriceRange  Extracts prices and dates between two calendar dates.
%
% INPUTS:
%   myPrice_dt : timetable with Time as row times and asset prices as variables
%   startDate  : start date (string 'dd/MM/yyyy' or datetime)
%   endDate    : end date   (string 'dd/MM/yyyy' or datetime)
%
% OUTPUTS:
%   prices_val : matrix of prices in the selected range
%   dates      : datetime vector of corresponding dates

    % Convert inputs to datetime if they are strings
    if ischar(startDate) || isstring(startDate)
        start_dt = datetime(startDate, 'InputFormat', 'dd/MM/yyyy');
    else
        start_dt = startDate;
    end

    if ischar(endDate) || isstring(endDate)
        end_dt = datetime(endDate, 'InputFormat', 'dd/MM/yyyy');
    else
        end_dt = endDate;
    end

    % Select range
    range = timerange(start_dt, end_dt, 'closed'); 
    subsample = myPrice_dt(range, :);

    % Outputs
    prices_val = subsample.Variables;
    dates      = subsample.Time;
end
