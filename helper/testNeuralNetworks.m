function [testAccuracy, out] = testNeuralNetworks(net, input, target)

    out = sim(net, input);

    valid = true;
    r = 0;   % Contador de classificações corretas

    for i = 1:size(input, 2)
        try
            [~, c] = max(out(:, i));       % Classe predita (índice da ativação máxima)
            [~, e] = max(target(:, i));    % Classe real (índice da ativação one-hot)
    
            if isempty(c) || isempty(e)
                valid = false;
                break;
            end
    
            if c == e
                r = r + 1;     % Incrementa acertos se previsão for correta
            end
        catch
            valid = false;
            break;
        end
    end
    
    if valid
        testAccuracy = r / size(out, 2) * 100;
        fprintf('Precisão test: %.2f%%\n', testAccuracy);
    else
        fprintf('Erro: A saída da rede não foi válida para avaliar a precisão.\n');
        testAccuracy = NaN;
    end
end