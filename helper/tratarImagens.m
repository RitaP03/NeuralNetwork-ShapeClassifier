function [binaries, target] = tratarImagens(folderName)
    % Sets the file names to get the target and training data
    classes = {'circle', 'kite', 'parallelogram', ...
        'square', 'trapezoid', 'triangle'};
    numberOfClasses = numel(classes);
    
    % Used to pre alocating memory and resizing odds
    % Making the image smaller helps reduce testing times
    % imageSize = [224 224];
    imageSize = [25 25];
    inputSize = prod(imageSize);
    
    % Uses the first directory in the training folder
    % to know how many images are going to be used
    folderPath = fullfile(folderName,"\", classes(1), "\");
    imageFiles = dir(fullfile(folderPath, "*.png"));
    numFilesPerClass = length(imageFiles);
    
    totalImages = numFilesPerClass * numberOfClasses;
    
    % Pre alocates memory for the data and target
    binaries = zeros(inputSize, totalImages);
    target = zeros(numberOfClasses, numFilesPerClass);
   
    imageIndex = 1;
    % For each folder/class
    for classIndex = 1:numberOfClasses
        folderPath = fullfile(folderName,"\", classes(classIndex), "\");
        imageFiles = dir(fullfile(folderPath, "*.png"));
    
        % For each image in folder/class
        for i = 1:length(imageFiles)
            % Get the image as a Matrix
            currentImage = imread(fullfile(folderPath + imageFiles(i).name));

            % Remove any RGB into 0 and 255 values
            if size(currentImage, 3) == 3
                currentImage = rgb2gray(currentImage);     % Converte RGB → grayscale
            end

            % Resize to imageSize to fit binaries matrix
            % Redimensiona para 25x25
            currentImage = imresize(currentImage, imageSize);   

            % Convert grayed picture to logical values
            % 0 and 1
            % Binariza a imagem (0/1)
            binaryImage = imbinarize(currentImage);
            
            % Flatten to a single column
            % Vetoriza e guarda na matriz
            binaries(:, imageIndex) = binaryImage(:);
        
            % Set the correct target value for training
            % Will create a matrix of:
            % [1 0 0 0 0 0; 
            % 0 0 0 0 0 0 0, 0 1 0 0 0 0; 
            % ... ;
            % 0 0 0 0 0 1]
            % Each represents a class
            target(classIndex, imageIndex) = 1;
            imageIndex = imageIndex + 1;
        end
    end
end