function exci()
    % Limpa o terminal e Fecha janelas gráficas abertas
    clc;
    close all;

    % Garante acesso às funções auxiliares na pasta 'helper'
    addpath('helper\');

    % Define que os dados a testar serão os da pasta 'test'
    file = 'test';

    % Carrega os dados de teste já binarizados e vetorizados
    % [binaries, target] = tratarImagens(file);
    load(strcat('testData\binaryImages', file), 'binaries');
    load(strcat('testData\target', file), 'target');

    % Loop para aplicar as 3 melhores redes guardadas no ficheiro 'best_net_i.mat'
    % for all networks
    for j = 1:3
        % load networks
        % Carrega a i-ésima rede guardada em 'best_net_i.mat'
        filename = sprintf("saved_nets/best_net_%d.mat", j); 
        load(filename, 'net');  % A variável 'net' contém a estrutura {net, tr}
        network = net.net;      % Extrai a rede neuronal propriamente dita

        % Aplica a rede aos dados de teste e calcula a precisão
        [testAccuracy, out] = testNeuralNetworks(network, binaries, target);  % Treino = nan porque não houve
        writeNetToExcel(network, nan, testAccuracy, file);

        % Gera e guarda o gráfico da matriz de confusão
        savePlots(network, j, out, target);
    end
end