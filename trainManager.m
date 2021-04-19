%TRAINMANAGER trains a manager (a deep Q-model) to infer motivation in
%the Travelling Salesman task
%
%   Parameters
%   ==========
%   MANAGER_TYPE    - string ('unsupervised' | 'supervised')
%   NUM_TASKS       - number (of training iterations)
%   NUM_STEPS_MAX   - number (of steps after which a task is terminated)
%   MODEL_FILE      - string (for the trained agent, e.g. 'agent')
%   SZ              - number (of units in each hidden layer)
%   BETA            - double (inverse temperature for agent's decision)
%   GAMMA           - double (temporal discount factor)
%   EPSILON         - double (fraction of manager's random actions)
%   LRATE           - function (learning rate schedule)
%   MOMENTUM        - double (momentum)
%   WDECAY          - double (weight decay, as a fraction of learning rate)
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

MANAGER_TYPE = 'unsupervised';
NUM_TASKS = 2e5;
NUM_STEPS_MAX = 50;
MODEL_FILE = 'agent';
SZ = 200;
BETA = 10;
GAMMA = 0.9;
EPSILON = 0.1;
LRATE = @(i) 1e-5 * 1e-3 ^ (i / NUM_TASKS);
MOMENTUM = 0.9;
WDECAY = 1e-3;

load(fullfile('Models', MODEL_FILE));

%Define the Manager's neural network
%    #                 TYPE      INDIM     OUTIM     WDIM     NLTYPE      OS
Mnet(1)=DLNetworkLayer('input',	 [2*N 1 1],[2*N 1 1],[],      [],         []);
Mnet(2)=DLNetworkLayer('full',	 [2*N 1 1],[SZ 1 1], [SZ 2*N],'leakyrelu',[]);
Mnet(3)=DLNetworkLayer('full',	 [SZ 1 1], [SZ 1 1], [SZ SZ], 'leakyrelu',[]);
Mnet(4)=DLNetworkLayer('full',	 [SZ 1 1], [SZ 1 1], [SZ SZ], 'leakyrelu',[]);
Mnet(5)=DLNetworkLayer('full',	 [SZ 1 1], [N+1 1 1],[N+1 SZ],'leakyrelu',[]);
Mnet(6)=DLNetworkLayer('target', [N+1 1 1],[N+1 1 1],[],      [],         []);

len = length(Mnet);

%Train the model
TD_ERROR_LOG = zeros(NUM_TASKS, 1) * NaN; %Delta (Reward presiction error)
NUM_STEPS = zeros(NUM_TASKS, 1) * NaN; %Number of steps to complete a task
tic

for i = 1 : NUM_TASKS
    allNodeSets = randperm(N, NUM_TARGETS + 1);
    S = zeros(1, N);
    S(allNodeSets(1)) = 1;
    Mprev = zeros(1, N);
    Mprev(allNodeSets(2 : end)) = 1;
    M = Mprev; Mprev_man = Mprev;
    
    TD_ERROR_LOG(i) = 0; NUM_STEPS(i) = 0;
    
    while sum(M) > 0
        NUM_STEPS(i) = NUM_STEPS(i) + 1;
        
        %In supervised mode, provide manager with agent's true motivation
        if strcmp(MANAGER_TYPE, 'supervised')
            Mprev_man = Mprev;
        end
        
        %Manager's action to update motivation for agent's existing state
        [Q0_man, A_man, M_man] = actionManager(S, Mprev_man, Mnet, EPSILON);
        
        %Agent's action to update state using *manager-provided* motivation
        M = updateMotivation(S, Mprev);
        [~, ~, Snew] = actionAgent(S, M_man, Anet, ROADS, BETA);
        
        %If the *agent's actual* motivation is still not zero
        if sum(M) ~= 0
            Q_man = actionManager(Snew, M_man, Mnet, EPSILON);
            actionManager(S, Mprev_man, Mnet, EPSILON);
        else
            Q_man = 0;
        end
        Mprev_man = M_man; Mprev = M; S = Snew;
        
        %Assign a reward to the manager
        if strcmp(MANAGER_TYPE, 'supervised')
            if isequal(M_man, M); R = 2; ...
            elseif (A_man == N + 1); R = -1;
            else; R = -2;
            end
        else %unsupervised
            if Snew * M' > 0; R = 5; ...
            elseif (A_man == N + 1); R = -1;
            else; R = -2;
            end
        end
        
        %Perform the TD update
        TD_ERROR = R + GAMMA * max(squeeze(Q_man)) - Q0_man(A_man);
        delta_vector = zeros(N + 1, 1);
        delta_vector(A_man) = -TD_ERROR;
        Mnet(len - 1).delta = delta_vector;
        for j = len - 1 : - 1 : 2
            stepBackward(Mnet, j, LRATE(i), MOMENTUM, WDECAY, 1);
        end
        
        %Log the variables
        TD_ERROR_LOG(i) = TD_ERROR_LOG(i) + abs(TD_ERROR);
        
        if NUM_STEPS(i) == NUM_STEPS_MAX
            break
        end
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
    Mnet(i).strip();
end
save(fullfile('Models', 'manager.mat'), 'Mnet')
