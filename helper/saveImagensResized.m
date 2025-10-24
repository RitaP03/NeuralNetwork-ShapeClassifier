function [binaries, target] = saveImagensResized(folderName)

    % Função que:
    % - Lê imagens de cada subpasta (classe),
    % - Converte para binário (0/1),
    % - Codifica a classe em one-hot,
    % - Guarda os dados em ficheiros .mat


    % Sets the file names to get the target and training data
    % Lista das classes a serem lidas
    classes = {'circle', 'kite', 'parallelogram', ...
        'square', 'trapezoid', 'triangle'};
    numberOfClasses = numel(classes);
    
    % Used to pre alocating memory and resizing odds
    % Making the image smaller helps reduce testing times
    % imageSize = [224 224];
    % Define o tamanho final das imagens (reduz carga de treino)
    imageSize = [25 25];
    inputSize = prod(imageSize);
    
    % Uses the first directory in the training folder
    % to know how many images are going to be used
    % Lê a primeira pasta de classe para saber quantas imagens existem por classe
    folderPath = fullfile(folderName,"\", classes(1), "\");
    imageFiles = dir(fullfile(folderPath, "*.png"));
    numFilesPerClass = length(imageFiles);
    
    totalImages = numFilesPerClass * numberOfClasses;
    
    % Pre alocates memory for the data and target
    % Pré-alocação das matrizes
    binaries = zeros(inputSize, totalImages);
    target = zeros(numberOfClasses, numFilesPerClass);
    
    
    imageIndex = 1;
    % For each folder/class
    for classIndex = 1:numberOfClasses
        % Caminho da subpasta atual
        folderPath = fullfile(folderName,"\", classes(classIndex), "\");
        imageFiles = dir(fullfile(folderPath, "*.png"));
    
        % For each image in folder/class
        for i = 1:length(imageFiles)
            % Get the image as a Matrix
            % Leitura da imagem
            currentImage = imread(fullfile(folderPath + imageFiles(i).name));

            % Remove any RGB into 0 and 255 values
            % Converte para grayscale se for RGB
            if size(currentImage, 3) == 3
                currentImage = rgb2gray(currentImage);
            end

            % Resize to imageSize to fit binaries matrix
            % Redimensiona e binariza a imagem (0 e 1)
            currentImage = imresize(currentImage, imageSize);
            % Convert grayed picture to logical values
            % 0 and 1
            binaryImage = imbinarize(currentImage);
               
            % Flatten to a single column
            % Vetoriza e armazena a imagem processada
            binaries(:, imageIndex) = binaryImage(:);
        
            % Set the correct target value for training
            % Criação do vetor one-hot para a classe atual
            % Will create a matrix of:
            % [1 0 0 0 0 0; 0 1 0 0 0 0; ... ; 0 0 0 0 0 1]
            % Each represents a class
            target(classIndex, imageIndex) = 1;
            % Próxima posição nas matrizes
            imageIndex = imageIndex + 1;
        end
    end

    % Guarda os dados processados:
    save(strcat('testData\target', folderName, '.mat'), 'target');
    save(strcat('testData\binaryImages', folderName, '.mat'), "binaries");
end