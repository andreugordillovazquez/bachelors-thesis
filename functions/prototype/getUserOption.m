function selectedOption = getUserOption()
    % getUserOption creates a floating UI window that allows the user to select
    % one of three options: 'Optimize', 'Plot', or 'Show'.
    % The selected option is returned as a string.

    % Initialize the output variable.
    selectedOption = '';

    % Create a UI figure (a floating window)
    fig = uifigure('Name', 'Select Option', ...
                   'Position', [500 500 300 200], ...
                   'Resize', 'off', ...
                   'Color', [0.95 0.95 0.97]);  % light background

    % Create a label prompting the user to select an option
    uilabel(fig, 'Text', 'Please select an option:', ...
            'Position', [50 150 200 22], ...
            'HorizontalAlignment', 'center', ...
            'FontSize', 14, 'FontWeight', 'bold');

    % Create three buttons for the options

    % Button for "Optimize"
    btnOptimize = uibutton(fig, 'push', 'Text', 'Optimize', ...
                           'Position', [50 110 200 30], ...
                           'FontSize', 12, 'ButtonPushedFcn', @(btn,event) selectOption('Optimize'));

    % Button for "Plot"
    btnPlot = uibutton(fig, 'push', 'Text', 'Plot', ...
                       'Position', [50 70 200 30], ...
                       'FontSize', 12, 'ButtonPushedFcn', @(btn,event) selectOption('Plot'));

    % Button for "Show"
    btnShow = uibutton(fig, 'push', 'Text', 'Show', ...
                       'Position', [50 30 200 30], ...
                       'FontSize', 12, 'ButtonPushedFcn', @(btn,event) selectOption('Show'));

    % Block execution until the user makes a selection.
    uiwait(fig);

    % --- Callback Function ---
    function selectOption(option)
         % When an option is selected, store it and close the window.
         selectedOption = option;
         uiresume(fig);
         delete(fig);
    end
end
