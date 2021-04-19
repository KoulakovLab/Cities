function M = updateMotivation(Snew, M)
%UPDATEMOTIVATION Updates the motivation in the task
%
%   Parameters
%   ==========
%   Snew         - vector (agent's updated state, one-hot)
%   M            - vector (agent's motivation towards each city)
%
%   Author
%   ======
%   Ngoc Tran,      2018-2019. ntran@cshl.edu
%   Sergey Shuvaev, 2019-2021. sshuvaev@cshl.edu

if Snew * M' > 0
    M = M - Snew;
end
