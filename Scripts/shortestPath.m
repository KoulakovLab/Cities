function shortestLength = shortestPath(allNodeSets, ROADS, DISTANCES)
%SHORTESTPATH computes the shortest path lengths for all provided sets of
%starting and target cities (a solution for the Travelling Salesman task)
%
%   Parameters
%   ==========
%   ALLNODESETS     - matrix (sets of starting and target cities)
%   S               - vector (agent's state, one-hot)
%   ROADS           - matrix (cities' adjacency matrix)
%   DISTANCES       - matrix (pairwise distances between the cities)
%   SHORTESTLENGTH  - vector (shortest path lengths for every input)
%
%   Author
%   ======
%   Ngoc Tran,      2018-2019. ntran@cshl.edu
%   Sergey Shuvaev, 2019-2021. sshuvaev@cshl.edu

NUM_TARGETS = size(allNodeSets, 2) - 1;
shortestLength = zeros(size(allNodeSets, 1), 1);
G = graph(ROADS .* DISTANCES);

for k = 1 : size(allNodeSets, 1)
    pathDist = zeros(NUM_TARGETS + 1);

    %Compute shortest pairwise distances between starting and target cities 
    for i = 1 : NUM_TARGETS + 1
        for j = i + 1 : NUM_TARGETS + 1
            [~, pathDist(i, j)] = shortestpath(G, ...
                allNodeSets(k, i), allNodeSets(k, j));
        end
    end
    pathDist = pathDist + pathDist';
    
    %Compute distances for visiting target cities in any order
    possiblePaths = [ones(factorial(NUM_TARGETS), 1), ...
        perms(1 : NUM_TARGETS) + 1];
    possiblePathLengths = zeros(size(possiblePaths, 1), 1);
    for i = 1 : size(possiblePaths, 1)
        for j = 1 : size(possiblePaths, 2) - 1
            possiblePathLengths(i) = possiblePathLengths(i) + ...
                pathDist(possiblePaths(i, j), possiblePaths(i, j + 1));
        end
    end

    %Pick the shortest distance
    shortestLength(k) = min(possiblePathLengths);
end
