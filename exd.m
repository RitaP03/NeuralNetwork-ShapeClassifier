function exd()
    
    % Limpa o terminal e Fecha todas as janelas gráficas
    clc;
    close all;

    % Garante acesso às funções auxiliares
    addpath('helper\')

    % Tamanho padrão das imagens de entrada
    imageSize = [25 25];
    % Nomes das classes
    classes = {'circle', 'kite', 'parallelogram', 'square', 'trapezoid', 'triangle'};

    % [imgName, folderPath] = uigetfile('desenhadas\*.png', 'Seleciona uma imagem PNG');
    % if isequal(imgName, 0)
    %     return;
    % end
    % 
    % currentImage = imread(fullfile(folderPath, imgName));
    % currentImage = rgb2gray(currentImage);
    % currentImage = imresize(currentImage, imageSize);
    % binaryImage = imbinarize(currentImage);
    % 
    % binaryImage = binaryImage(:);

    % Carrega todas as imagens binarizadas da pasta 'desenhadas'
    % A função tratarImagens devolve a matriz binaries com imagens vetorizadas
    [binaries, ~] = tratarImagens('desenhadas');
    
    % Loop pelas 3 redes previamente treinadas
    for j = 1:3
        filename = sprintf("saved_nets/best_net_%d.mat", j);
        load(filename, 'net');       % Carrega a rede
        network = net.net;

        fprintf("Testing the %d net:\n", j);

        % Para cada classe (1 a 6) e para cada das 5 imagens desenhadas
        for file = 1:6
            for image = 1:5
                % Aplica a rede à imagem específica
                out = sim(network, binaries(:, image));

                [~, real] = max(out); % Obtém o índice da classe predita (one-hot)

                % Mostra o resultado esperado vs predito
                fprintf("Imagem esperada: %s\n", classes{file});
                fprintf("A rede indica: %s\n", classes{real});
                disp("----------------------------------------")
            end
        end
    end
end