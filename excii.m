function excii()   

    % Limpa o terminal e Fecha janelas gráficas abertas
    clc;
    close all;

    % Garante acesso às funções auxiliares
    addpath('helper\');

    files = {'start', 'train', 'test'}; % Nomes dos conjuntos de dados disponíveis
    file = files{3}; % Neste caso, 'test'. A rede será treinada apenas com este conjunto

    % [binaries, target] = tratarImagens(file);
    % Carrega os dados da pasta 'test' (inputs binarizados e respetivos targets)
    load(strcat('testData\binaryImages', file), 'binaries');
    load(strcat('testData\target', file), 'target');

    numTests = 1; % Número de repetições do treino (poderia ser maior para médias)

    % for all networks
    % Loop pelas 3 melhores redes treinadas anteriormente
    for j = 1:3
        % load networks
        filename = sprintf("saved_nets/best_net_%d.mat", j);  % Nome do ficheiro da rede
        load(filename, 'net');   % Carrega a estrutura contendo a rede e info de treino anterior
        network = net.net;       % Extrai apenas a rede

        medianTrainAccuracy = 0;
        medianTestAccuracy = 0;

        % Treino com os dados da pasta test
        for i = 1:numTests
            % Train
            [trainAccuracy, network, tr, ~] = trainNeuralNetworks(network, binaries, target);
            medianTrainAccuracy = medianTrainAccuracy + trainAccuracy;

            % Test/Simulate
            [testAccuracy, ~] = testNeuralNetworks(network, binaries, target);
            medianTestAccuracy = medianTestAccuracy + testAccuracy;
            
            % Save data to excel
            % Guarda resultados da iteração no Excel
            writeNetToExcel(network, trainAccuracy, testAccuracy, files);
        end

        % Save medians of train/tests to excel
        % Guarda médias no Excel
        medianTestAccuracy = medianTestAccuracy / numTests;
        medianTrainAccuracy = medianTrainAccuracy / numTests;
        writeMediaAccToExcel(medianTrainAccuracy, medianTestAccuracy, file);

        % Classify all images
        % Testa a rede treinada nos 3 conjuntos: start, train, test
        for i = 1:3
            load(strcat('testData\binaryImages', files{i}), 'binaries');
            load(strcat('testData\target', files{i}), 'target');
            
            [testAccuracy, out] = testNeuralNetworks(network, binaries, target);
            writeNetToExcel(network, nan, testAccuracy, files{i});   % Não se regista o treino
            
            % Gera e guarda a matriz de confusão como imagem .jpg
            filepath = strcat(files{i} + "_best_net_" + j + '.jpg');
            saveas(plotconfusion(target, out), fullfile("plots", "\", filepath));
        end

        % Guarda a rede treinada nesta nova fase (com o conjunto test)
        net = struct('net', network, 'tr', tr);
        % save trained network
        filename = sprintf("saved_nets/best_net_%d.mat", j);
        save(filename, 'net');
    end
end