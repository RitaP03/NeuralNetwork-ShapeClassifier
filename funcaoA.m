function funcaoA()
   % Limpa o ambiente e fecha janelas anteriores 
    clc;
    close all;

    % Adiciona a pasta "helper" ao caminho, onde estão as funções auxiliares
    addpath('helper\');

    % Define os nomes das pastas de dados
    files = {'start', 'train', 'test'};


    % Lista de configurações a testar: 
    % {dataset, neurónios, funções de ativação, função de treino, estratégia de divisão, rácios}

    configs = {%  A
    % {files{1}, 10,        {'tansig', 'purelin'}, 'trainlm',   'dividetrain',  [0.7 0.15 0.15]}, ...
    % {files{1}, 5,         {'tansig', 'purelin'}, 'trainlm',   'dividetrain',  [0.7 0.15 0.15]}, ...
    % {files{1}, 20,        {'tansig', 'purelin'}, 'trainlm',   'dividetrain',  [0.7 0.15 0.15]}, ...
    % {files{1}, [10 10],   {'tansig', 'purelin'}, 'trainlm',   'dividetrain',  [0.7 0.15 0.15]}, ... 
    % {files{1}, [20 20],   {'tansig', 'purelin'}, 'trainlm',   'dividetrain',  [0.7 0.15 0.15]}, ...
    % {files{1}, [10 10 10],{'tansig', 'purelin'}, 'trainlm',   'dividetrain',  [0.7 0.15 0.15]}, ...
    {files{2},  10,       {'tansig', 'purelin'}, 'trainlm',   'dividerand',   [0.7 0.15 0.15]}, ... 
    % {files{2},  5,        {'tansig', 'purelin'}, 'trainlm',   'dividerand',   [0.7 0.15 0.15]}, ...
    % {files{2},  20,       {'tansig', 'purelin'}, 'trainlm',   'dividerand',   [0.7 0.15 0.15]}, ...
    % {files{2},  [5 5],    {'tansig', 'purelin'}, 'trainlm',   'dividerand',   [0.7 0.15 0.15]}, ...
    % {files{2},  [10 10],  {'tansig', 'purelin'}, 'trainlm',   'dividerand',   [0.7 0.15 0.15]}, ... % Training function
    % {files{2},  10,       {'tansig', 'purelin'}, 'traingd',   'dividerand',   [0.7 0.15 0.15]}, ...
    % {files{2},  10,       {'tansig', 'purelin'}, 'trainbfg',  'dividerand',   [0.7 0.15 0.15]}, ...
    % {files{2},  10,       {'tansig', 'purelin'}, 'trainbr',   'dividerand',   [0.7 0.15 0.15]}, ...
    % {files{2},  10,       {'tansig', 'purelin'}, 'traincgf',  'dividerand',   [0.7 0.15 0.15]}, ... % Transfer function
    % {files{2},  10,       {'tansig', 'logsig'},  'trainlm',   'dividerand',   [0.7 0.15 0.15]}, ...
    % {files{2},  10,       {'purelin', 'logsig'}, 'trainlm',   'dividerand',   [0.7 0.15 0.15]}, ...
    % {files{2},  10,       {'logsig', 'tansig'},  'trainlm',   'dividerand',   [0.7 0.15 0.15]}, ...
    % {files{2},  10,       {'radbasn', 'tansig'}, 'trainlm',   'dividerand',   [0.7 0.15 0.15]}, ... % Ratios
    % {files{2},  10,       {'tansig', 'purelin'}, 'trainlm',   'dividerand',   [0.33 0.33 0.33]}, ...
    % {files{2},  10,       {'tansig', 'purelin'}, 'trainlm',   'dividerand',   [0.9 0.05 0.05]}, ...
    % {files{2},  10,       {'tansig', 'purelin'}, 'trainlm',   'dividerand',   [0.8 0.1 0.1]}, ...
    % {files{2},  10,       {'tansig', 'purelin'}, 'trainlm',   'dividerand',   [0.7 0.15 0.15]}, ...
    };

    % Número de execuções por configuração
    numTests = 10;

    % Inicializa vetores para guardar as melhores redes
    currentAccuracies = zeros(3); % Vetor auxiliar para guardar as 3 melhores precisões

    % Top 3 melhores redes
    % Estrutura para guardar as melhores redes de todas as configurações
    bestNets = struct('net', {}, ...
        'trainAccuracy', 0, ...
        'testAccuracy', 0, ...
        'tr', {}, ...
        'target', {}, ...
        'out', {});
    % Estrutura para guardar a melhor rede *dentro* de cada configuração testada
    bestNetInConfig = struct('net', {}, ...
        'trainAccuracy', 0, ...
        'testAccuracy', 0, ...
        'tr', {}, ...
        'target', {}, ...
        'out', {});
    
    % For each configuration
    % Loop principal sobre todas as configurações especificadas
    for j = 1:numel(configs)
        medianTrainAccuracy = 0;
        medianTestAccuracy = 0;
    
        % Get the configuration
        % Configuração atual
        runConfig = configs{j};
    
        % Load the images and target for the file
        % [binaries, target] = tratarImagens(file);
        % Carregamento dos dados binarizados e dos targets para a pasta correspondente
        load(strcat('testData\binaryImages', runConfig{1}), 'binaries');
        load(strcat('testData\target', runConfig{1}), 'target');

        % % % % % % % SETUP THE CONFIGURATION % % % % % % %
        % % % % % % % % CONFIGURAÇÃO DE REDE % % % % % % % %
        neurons = runConfig{2};                  % Número de neurónios/camadas ocultas
        trainFcn = runConfig{4};                 % Função de treino (ex: 'trainlm')
        net = feedforwardnet(neurons, trainFcn); % Cria a rede com a topologia e função de treino

        net.trainParam.showWindow = false; % Não mostrar a janela gráfica de treino do MATLAB

        % Lista com funções de ativação por camada
        transferFcn = runConfig{3};

        % For all but the end transferFunction
        % Atribuição das funções de ativação às camadas ocultas e de saída
        if size(neurons) == (numel(transferFcn) - 1)
            for k = 1:numel(neurons)
                % Set the transfer function
                net.layers{k}.transferFcn = transferFcn{k};
            end
        else
            % Set the ONLY transfer function
            net.layers{1:end-1}.transferFcn = transferFcn{1}; 
        end
        % Sets the output function
        net.layers{end}.transferFcn=transferFcn{end};

        % Num epochs
        net.trainParam.epochs=100; % Número de épocas de treino

        % Sets the ratios
        % Definir método e proporções de divisão dos dados (train/val/test)
        if ~isempty(runConfig{5})         % Função de divisão (ex: 'dividerand')
            net.divideFcn = runConfig{5}; % Vetor com as percentagens
            ratioTrain = runConfig{6};
            net.divideParam.trainratio = ratioTrain(1);
            net.divideParam.valRatio = ratioTrain(2);
            net.divideParam.testRatio = ratioTrain(3);
        end

        % % % % % % % % % % % % % % % % % % % % % % % % % % 

        savedNet = false;
        % 10 tests for each configuraion
        % Repetição dos testes para média estatística (10 execuções)
        for i = 1:numTests
            fprintf("Config: %d, test: %d\n", j, i);

            % Train the network and simulate to get the global accuracy
            % Treina a rede e devolve a precisão no treino
            [trainAccuracy, net, tr, ~] = trainNeuralNetworks(net, binaries, target);
            medianTrainAccuracy = medianTrainAccuracy + trainAccuracy;

            testAccuracy = nan;

            % I only want to test when the file is 'train'
            % Só realiza teste se estiver a usar o dataset 'train'
            if strcmp(runConfig{1}, files{2}) == 1
                % Get the inputs and targets for testing
                TInput = binaries(:, tr.testInd);    % Dados de teste (inputs)
                TTargets = target(:, tr.testInd);    % Dados de teste (targets)

                % Tests the network with the test parameters
                [testAccuracy, out] = testNeuralNetworks(net, TInput, TTargets);
                medianTestAccuracy = medianTestAccuracy + testAccuracy;

                % Save the best net in the config
                % Estrutura auxiliar para guardar rede atual, se for uma das melhores
                auxNet = struct( ...
                        'net', net, ...
                        'trainAccuracy', trainAccuracy, ...
                        'testAccuracy', testAccuracy, ...
                        'tr', tr, ...
                        'target', TTargets, ...
                        'out', out);

                % Verifica se é a melhor dentro da configuração atual
                if savedNet == false
                    if isempty(bestNetInConfig)
                        bestNetInConfig = auxNet;
                    else
                        acc1 = median([auxNet.testAccuracy auxNet.trainAccuracy]);
                        acc2 = median([bestNetInConfig.testAccuracy bestNetInConfig.trainAccuracy]);
         
                        if acc1 > acc2
                           bestNetInConfig = auxNet;     
                        end
                    end
                    savedNet = true;
                end
            end

            % Guarda os resultados dessa execução no Excel
            writeNetToExcel(net, trainAccuracy, testAccuracy, runConfig{1});
        end

        % Adiciona a melhor rede da configuração atual ao top 3 geral
        if strcmp(runConfig{1}, files{2}) == 1
            % Case 1: fewer than 3 saved just add
            if length(bestNets) < 3
                bestNets(end+1) = bestNetInConfig;    
            else
                % Case 2: find worst accuracy in current top 3
                for k = 1:3
                    currentAccuracies(k) = (bestNets(k).testAccuracy + bestNets(k).trainAccuracy) / 2;
                end
                [worstAcc, worstIdx] = min(currentAccuracies);
                            
                % If new net is better than the worst in top 3 → replace it
                if bestNetInConfig.testAccuracy > worstAcc
                    bestNets(worstIdx) = bestNetInConfig;
                end
            end
        end

        % Guarda as médias finais de precisão de treino e teste no Excel
        medianTestAccuracy = medianTestAccuracy / numTests;
        medianTrainAccuracy = medianTrainAccuracy / numTests;
        writeMediaAccToExcel(medianTrainAccuracy, medianTestAccuracy, runConfig{1});
    end

    % Salva as 3 melhores redes em ficheiros .mat e gera gráficos de confusão
    for i = 1:length(bestNets)
        filename = sprintf("saved_nets/best_net_%d.mat", i);
        net = struct('net', bestNets(i).net, 'tr', bestNets(i).tr);

         savePlots(bestNets(i).net, i, bestNets(i).out, bestNets(i).target); % Salva o modelo treinado

        save(filename, 'net');
    end
end