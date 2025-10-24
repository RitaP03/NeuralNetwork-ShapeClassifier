function [trainAccuracy, net, tr, out] = trainNeuralNetworks(net, input, target)

    % Train the neural network
    % Treina a rede com os dados fornecidos
    [net, tr] = train(net, input, target);

    % Simulate the network
    % Simula a rede após o treino
    out = sim(net, input);
    % visualizar a rede
    % view(net);
    % disp(tr);
        
    % Calcule the global precision
    valid = true;
    r = 0;   % Contador de classificações corretas

    for i = 1:size(out, 2)
        try
            [~, c] = max(out(:, i));      % Classe predita pela rede
            [~, e] = max(target(:, i));   % Classe real (target)
            if isempty(c) || isempty(e)
                valid = false;
                break;
            end
            if c == e
                r = r + 1;
            end
        catch
            valid = false;
            break;
        end
    end
    
    if valid
        trainAccuracy = r / size(out, 2) * 100;     % Percentagem de acerto
        fprintf('Precisão Global: %.2f%%\n', trainAccuracy);
    else
        fprintf('Erro: A saída da rede não foi válida para avaliar a precisão.\n');
        trainAccuracy = NaN;
    end
end