%TRAINAGENT trains an agent (a deep Q-model) to behave optimally in
%the Travelling Salesman task
%
%   Parameters
%   ==========
%   N            - number (of cities in the task)
%   NUM_TARGETS  - number (of cities to be visited)
%   BASE_REWARD  - double (reward for visiting a target city)
%   GAMMA        - double (temporal discount factor)
%   NUM_TASKS    - number (of training iterations)
%   LRATE        - function (learning rate schedule)
%   BETA         - double (inverse temperature for softmax decision)
%   SZ           - number (of units in each hidden layer)
%
%   Author
%   ======
%   Ngoc Tran,      2018-2019. ntran@cshl.edu
%   Sergey Shuvaev, 2019-2021. sshuvaev@cshl.edu

close all
clear
clc

addpath(genpath('Scripts'));
rng('shuffle')

N = 10;
NUM_TARGETS = 3;
BASE_REWARD = 5;
SZ = 200;
NUM_TASKS = 2e5;
BETA = 1;
GAMMA = 0.9;
LRATE = @(i) 1e-2 * 1e-2 ^ (i / NUM_TASKS);

%Define the roads
x = randn(N, 1);
y = randn(N, 1);
DISTANCES = squareform(pdist([x, y]));

triangles = delaunay(x, y);
ROADS = zeros(N);
for i = 1 : N
    ROADS(i, triangles(sum(triangles == i, 2) > 0, :)) = 1;
end
ROADS = logical(ROADS);

%Define the Agent's neural network
%    #                 TYPE      INDIM     OUTIM     WDIM     NLTYPE      OS
Anet(1)=DLNetworkLayer('input',	 [2*N 1 1],[2*N 1 1],[],      [],         []);
Anet(2)=DLNetworkLayer('full',	 [2*N 1 1],[SZ 1 1], [SZ 2*N],'leakyrelu',[]);
Anet(3)=DLNetworkLayer('full',	 [SZ 1 1], [N 1 1],	 [N SZ],  'linear',   []);
Anet(4)=DLNetworkLayer('target', [N 1 1],  [N 1 1],	 [],      [],         []);

len = length(Anet);

%Train the model
TD_ERROR_LOG = zeros(NUM_TASKS, 1) * NaN; %Delta (Reward presiction error)
NUM_STEPS = zeros(NUM_TASKS, 1) * NaN; %Number of steps to complete a task
tic

for i = 1 : NUM_TASKS
    allNodeSets = randperm(N, NUM_TARGETS + 1);
    S = zeros(1, N);
    S(allNodeSets(1)) = 1;
    M = zeros(1, N);
    M(allNodeSets(2 : end)) = 1;
    
    TD_ERROR_LOG(i) = 0; NUM_STEPS(i) = 0;
    
    while sum(M) > 0
        %Agent's action
        [Q0, A, Snew] = actionAgent(S, M, Anet, ROADS, BETA);
        R = -DISTANCES(find(S), find(Snew));
        if Snew * M' > 0
            R = R + BASE_REWARD;
        end
        Mnew = updateMotivation(Snew, M);
        
        if sum(Mnew) ~= 0
            Q = actionAgent(Snew, Mnew, Anet, ROADS, BETA);
            actionAgent(S, M, Anet, ROADS, BETA);
        else
            Q = 0;
        end
        M = Mnew; S = Snew;
        
        %Perform the TD update
        TD_ERROR = R + GAMMA * max(squeeze(Q)) - Q0(A);
        delta_vector = zeros(N, 1);
        delta_vector(A) = -TD_ERROR;
        Anet(len - 1).delta = delta_vector;
        for j = len - 1 : - 1 : 2
            stepBackward(Anet, j, LRATE(i));
        end
        
        %Log the variables
        TD_ERROR_LOG(i) = TD_ERROR_LOG(i) + abs(TD_ERROR);
        NUM_STEPS(i) = NUM_STEPS(i) + 1;
    end
    
    %Plot the progress
    if ~mod(i,1000)
        subplot(2, 1, 1);
        loglog(conv(abs(TD_ERROR_LOG ./ NUM_STEPS), ones(1000, 1)) / 1000), grid
        title('TD error')
        subplot(2, 1, 2);
        semilogx(conv(abs(NUM_STEPS), ones(1000, 1)) / 1000), grid
        title('Number of steps')
        xlabel('Task number')
        drawnow
    end
end

t = toc;
fprintf('\nTraining time: %d min %d sec.\n', ...
    floor(t / 60), round(t - floor(t / 60) * 60));

%Remove training-related variables and save the model
for i = 2 : len - 1
    Anet(i).strip();
end
save(fullfile('Models', 'agent.mat'), 'Anet', 'ROADS', 'DISTANCES', ...
    'N', 'NUM_TARGETS')
