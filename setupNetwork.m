function [net] = setupNetwork()

    % TESTES 
    % Alterar o número de neuronios por camada
    %       Manter 10 por default
    %       Mudar para 20, 30, 5 para outros testes
    % neurons = [20 10];
    neurons = 10;
    % Alterar o número de layers escondidas
    %       Manter 1 por default
    %       Mudar para [10 8 3] para outros testes 

    % Alterar a função de treino para 
    %       'trainlm', 'trainbr', 'trainbfg', 'traincgf'
    % Função de treino 
    trainFcn = "trainlm";

    % Funções de ativação
    % help nntransfer para mais fcns
    %       Da camada escondida
    %       default - "tansig"
    % nota: se houver mais que 1 layer, a transferFcnInput
    % pode ter mais que 1 funcao, colocar dentro de []
    % com tamanho igual a neurons
    % transferFcnInput = ["tansig", "satlins"];
    transferFcnInput = "logsig";
    %       Da camada de saída 
    %       default - "purelin"
    transferFcnOutput = "purelin"; 

    % Numero de epocas de treino
    numEpochs = 100;

    % Função e rácios de divisão em treino/validação/teste
    % default - 'dividerand'
    divideFcn = '';
    % se divideFcn for ''
    % todos os exemplos de inpus sao usados no treino
    % divide
    if ~isempty(divideFcn)
        trainRatio = 0.7;
        valRatio = 0.15;
        testRatio = 0.15;
    end
    % valores default
    % 70 % treino
    % 15 % validação
    % 15 % teste

% % % % % % % % % % % % % % % % % % % % % % % % % % % % 

    % Create the Neural Network
    net = feedforwardnet(neurons, trainFcn);
    
    % Check and update transfer function for each layer
    if size(neurons) == size(transferFcnInput)
        for i = 1:numel(neurons)
            net.layers{i}.transferFcn=transferFcnInput(i);
        end
    else
        net.layers{1:end-1}.transferFcn=transferFcnInput(1); 
    end
    net.layers{end}.transferFcn=transferFcnOutput;
    % Set the epochs of training
    net.trainParam.epochs=numEpochs;

    % Check and update the functions that divide the input data
    if ~isempty(divideFcn)
        net.divideFcn = divideFcn;
        net.divideParam.trainratio = trainRatio;
        net.divideParam.valRatio = valRatio;
        net.divideParam.testRatio = testRatio;
    end
end