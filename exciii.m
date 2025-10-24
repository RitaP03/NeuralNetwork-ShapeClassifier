function exciii()   

    % Limpa o terminal e Fecha todas as janelas gráficas
    clc;
    close all;

    % Adiciona a pasta com funções auxiliares
    addpath('helper\');

    files = {'start', 'train', 'test'};  % Conjuntos de dados
    numTests = 10;                       % Nº de repetições para cálculo de médias

    % for all networks
    % Loop pelas 3 melhores redes gravadas anteriormente
    for k = 1:3
        % load networks
        filename = sprintf("saved_nets/best_net_%d.mat", k);
        load(filename, 'net');                  % Carrega rede e info de treino anterior
        network = net.net;                      % Extrai a rede neuronal
        network.trainParam.showWindow = false;  % Oculta janela gráfica do MATLAB

        % for all files
        % Para cada um dos 3 conjuntos (start, train, test)
        for j = 1:3
            medianTrainAccuracy = 0;
            medianTestAccuracy = 0;
            
            % Load inputs and targets
            % [binaries, target] = tratarImagens(file);
            % Carrega imagens binarizadas e targets
            load(strcat('testData\binaryImages', files{j}), 'binaries');
            load(strcat('testData\target', files{j}), 'target');

            % Train 10 times for a file
            % Treina a rede 10 vezes com o mesmo conjunto
            for i = 1:numTests
                % Train
                [trainAccuracy, network, ~, ~] = trainNeuralNetworks(network, binaries, target);
                medianTrainAccuracy = medianTrainAccuracy + trainAccuracy;

                % Test/Simulate
                [testAccuracy, network] = testNeuralNetworks(network, binaries, target);
                medianTestAccuracy = medianTestAccuracy + testAccuracy;
                
                % Save data to excel
                % Guarda resultados da execução no Excel
                writeNetToExcel(network, trainAccuracy, testAccuracy, files{j});
            end

            % Save medians to excel
            % Guarda as médias no Excel
            medianTrainAccuracy = medianTrainAccuracy / numTests;
            medianTestAccuracy = medianTestAccuracy / numTests;
            writeMediaAccToExcel(medianTrainAccuracy, medianTestAccuracy, files{j});
        end

        % Classify all images
        % Classifica novamente todos os datasets com a última rede treinada
        for i = 1:3
            load(strcat('testData\binaryImages', files{i}), 'binaries');
            load(strcat('testData\target', files{i}), 'target');

            [testAccuracy, network, out] = testNeuralNetworks(network, binaries, target);
            writeNetToExcel(network, nan, testAccuracy, files{i});    % Treino = nan (já feito)

            filepath = strcat(files{i} + "_best_net_" + k + '.jpg');
            saveas(plotconfusion(target, out), fullfile("plots", "\", filepath));
        end
        
        % Guarda a rede final após treino com todos os dados
        net = struct('net', network, 'tr', tr);
        % save trained network
        filename = sprintf("saved_nets/best_net_%d.mat", k);
        save(filename, 'net');
    end
end