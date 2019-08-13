clear;
clc;
times = 10;
cross = 5;
% rcv4 2^-6, 2, 0.1;

data_file = {'arts', 'birds', 'cal500', 'flags', 'genbase', 'medical' ,'rcvsubset1', 'rcvsubset2','rcvsubset3', 'rcvsubset4', 'rcvsubset5', 'slashdot'};
params = [2^-9,   2^-6,   2^-1,    100;        % arts
          2^-10,  2^-7,   2^-10,   1;        % birds
          2^2,    2^-1,   2^-2,    1000;     % cal500
          2^-10,  2^-5,   2^-10,   1;           % flags
          2^-7,   2^-8,   2^-8,    1;        % genbase
          2^-8,   2^-5,   2^-3,    1;         % medical
          2^-10,  2^-8,   2^2,     10000;       % rcv1
          2^-9,   2^-7,   2^2,     10000;       % rcv2
          2^-9,   2^-8,   2^1,     10000;       % rcv3
          2^-9,   2^-9,   2^1,     10000;       % rcv4
          2^-10,  2^-8,   2^2,     10000;       % rcv5
          2^-4,   2^-10,  2^1,     10;];        % slashdot
for exp = 7:12
    dataset = data_file{exp};
    cd('data');
        eval(['load ', dataset]);
    cd('..');
    if exp == 4
        features = zscore(features);
    end
    num_instance = size(features, 1);
    opt_params.lambda1 = params(exp,1); % label correlation
    opt_params.lambda2 = params(exp,2); % sample similary
    opt_params.lambda3 = params(exp,3); % sparsity
    opt_params.neg = 10;
    opt_params.gamma = params(exp,4);
    opt_params.maxIter = 100;
    opt_params.minimumLossMargin = 0.0001;
    
    % Experiment results
    exp_pre_labels = cell(times, cross);
    exp_pre_distributions = cell(times, cross);
    exp_true_labels = cell(times, cross);
    % Evaluation result
    Result = zeros(times, 7);
    fprintf('=============== %s ============== \n', dataset);
    for itrator = 1:times
    %     indices = tenfold{iter};
        indices = crossvalind('Kfold', num_instance, cross);
        temp_result = zeros(cross, 7);
        for rep=1:cross
        fprintf('=============== %d %d  %s ============== \n', itrator, rep, datestr(now));
           testIdx = find(indices == rep);
           trainIdx = setdiff(find(indices),testIdx);
           test_data = features(testIdx,:);
           test_target = labels(testIdx,:);
           train_data = features(trainIdx,:);
           train_target = labels(trainIdx,:);
           % Train model
           [W]  = LSF_CI(train_data, train_target, opt_params);
           % Prediction
           [pre_labels, pre_dis , res_once] = LSF_CI_predict(W, test_data, test_target);
           % Save experiment results
           exp_pre_labels{itrator, rep} = pre_labels;
           exp_pre_distributions{itrator, rep} = pre_dis;
           exp_true_labels{itrator, rep} = test_target;
           % evaluation result
           temp_result(rep, :) = res_once;
        end
        Result(itrator, :) = mean(temp_result, 1);
    end
    meanres = mean(Result, 1)
    stdres = std(Result, 1)
    cd('expres');
        eval(['save ', dataset, '_res.mat meanres stdres']);
    cd('..');
    cd('experimentres');
        eval(['save ', dataset, '_expres.mat exp_pre_labels exp_pre_distributions exp_true_labels']);
    cd('..');
end