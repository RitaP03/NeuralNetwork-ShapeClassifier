function savePlots(net, idx, out, target)

    % Esta função guarda o gráfico da matriz de confusão gerado pela rede
    % O nome do ficheiro inclui: índice da rede, topologia, funções de ativação e divisão

    neurons = "";   % String para construir a descrição da topologia (nº de neurónios)
    trainFcn = "";  % String para guardar as funções de ativação usadas

     % Se a rede tiver mais de 2 camadas (entrada + >=2 camadas ocultas)
    if net.numLayers > 2 
        for i = 1:net.numLayers-2
             % Concatena o número de neurónios e função de ativação de cada camada oculta
            neurons = neurons + net.layers{i}.dimensions + ", ";
            trainFcn = trainFcn + net.layers{i}.transferFcn + ", ";
        end   
    end  

    % Adiciona a penúltima camada (última oculta) à string
    neurons = neurons + net.layers{end-1}.dimensions;
    trainFcn = trainFcn + net.layers{end-1}.transferFcn;

    % Gera o nome completo do ficheiro .jpg
    filepath = strcat("confusion_", ...
        string(idx), "_", ...                         % Índice da rede (ex: 1, 2, 3)
        neurons, "_", ...                             % Topologia (ex: 10,10)
        net.trainFcn, "_", ...                        % Função de treino usada (ex: trainlm)
        trainFcn, "_", ...                            % Funções de ativação das camadas ocultas
        net.layers{end}.transferFcn, "_", ...         % Função de ativação da camada de saída
        net.divideFcn, '.jpg');                       % Função de divisão dos dados (ex: dividerand)

    % Mostra no terminal o caminho onde o gráfico será guardado
    fprintf("Saving plot confusion at:\n (plots\\%s)\n", filepath);

    % Gera o gráfico da matriz de confusão e guarda-o como imagem
    saveas(plotconfusion(target, out), fullfile("plots", "\", filepath));

    % filepath = strcat("performance_", ...
    %     string(bestIdx), "_", ...
    %     neurons, "_", ...
    %     bestNet.net.trainFcn, "_", ...
    %     trainFcn, "_", ...
    %     bestNet.net.layers{end}.transferFcn, "_", ...
    %     bestNet.net.divideFcn, '.jpg');
    % saveas(plotperform(bestNet.tr), fullfile("plots", "\", filepath));
end