%TESTMODEL tests a model on a Travelling Salesman task
%
%   Parameters
%   ==========
%   TEST_TYPE       - string ('agent' | 'manager')
%   AGENT_FILE      - string (e.g. 'agent')
%   MANAGER_FILE    - string (e.g. 'managerUnsupervised')
%   NUM_TASKS_TEST  - number (of testing iterations)
%   NUM_STEPS_MAX   - number (of steps after which a task is terminated)
%   BETA            - double (inverse temperature for softmax decision)
%   EPSILON         - double (fraction of manager's random actions)
%   RANDOM_SEED     - number (to avoid saving the testing set)
%
%   Author
%   ======
%   Ngoc Tran,      2018-2019. ntran@cshl.edu
%   Sergey Shuvaev, 2019-2021. sshuvaev@cshl.edu

close all
clear
clc

addpath(genpath('Scripts'));

TEST_TYPE = 'manager';
AGENT_FILE = 'agent';
MANAGER_FILE = 'managerUnsupervised';
NUM_TASKS_TEST = 100;
NUM_STEPS_MAX = 50;
BETA = 10;
EPSILON = 0;
RANDOM_SEED = 1;

load(fullfile('Models', AGENT_FILE));
if strcmp(TEST_TYPE, 'manager')
    load(fullfile('Models', MANAGER_FILE));
end

%Generate a testing set
rng(RANDOM_SEED);
allNodeSets = zeros(NUM_TASKS_TEST, NUM_TARGETS + 1);
for i = 1 : NUM_TASKS_TEST
    allNodeSets(i, :) = randperm(N, NUM_TARGETS + 1);
end
shortestPaths = shortestPath(allNodeSets, ROADS, DISTANCES);
actualPaths = zeros(NUM_TASKS_TEST, 1);
%ROADS = ROADS - eye(size(ROADS));

%Evaluate the model
for i = 1 : NUM_TASKS_TEST
    S = zeros(1, N);
    S(allNodeSets(i, 1)) = 1;
    M = zeros(1, N);
    M(allNodeSets(i, 2 : end)) = 1;
    M_true = M;
    
    NUM_STEPS = 0;
    while sum(M_true) > 0
        NUM_STEPS = NUM_STEPS + 1;
        [Q0, A, Snew] = actionAgent(S, M, Anet, ROADS, BETA);
        Mnew_true = updateMotivation(Snew, M_true);
        if strcmp(TEST_TYPE, 'agent')
            Mnew = updateMotivation(Snew, M);
        else %manager
            [~, ~, Mnew] = actionManager(Snew, M, Mnet, EPSILON);
        end
        actualPaths(i) = actualPaths(i) + DISTANCES(find(S), find(Snew));
        M = Mnew; S = Snew; M_true = Mnew_true;
        
        if NUM_STEPS == NUM_STEPS_MAX
            actualPaths(i) = NaN;
            break
        end
    end
end

%Plot the results
figure, loglog(shortestPaths, actualPaths, '.', 'markersize', 15), hold on
line([1 100], [1 100], 'color', 'k')
axis([1 10 1 100])
pbaspect([1 2 1])
fprintf('Converged: %d%%,\nAverage path excess: %d%%\n', ...
        round(sum(~isnan(actualPaths)) / length(actualPaths) * 100), ...
        round((nanmean(actualPaths ./ shortestPaths) - 1) * 100));
