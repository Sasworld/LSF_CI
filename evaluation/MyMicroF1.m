function microF1 = MyMicroF1(pre_target,test_target)
%     test_target: N * L
    label_P = 1;
    label_N = 0;
    [num_instance, num_class] = size(test_target);
    TP_i = zeros(num_class,1);
    TN_i = zeros(num_class,1);
    FP_i = zeros(num_class,1);
    FN_i = zeros(num_class,1);
    for i = 1:num_class
        num_P_pre = sum(pre_target(:,i) == label_P); % positive prediction number
        num_N_pre = num_instance - num_P_pre;
        P_index = find(test_target(:,i) == label_P); % positive instance index
        N_index = find(test_target(:,i) == label_N);
        pre_eq_test_index = find(pre_target(:,i) == test_target(:,i)); % the index of prediction and test target are same 
        TP_i(i, 1) = size(intersect(pre_eq_test_index,P_index), 1);
        TN_i(i, 1) = size(intersect(pre_eq_test_index,N_index), 1);
        FP_i(i, 1) = num_P_pre - TP_i(i, 1);
        FN_i(i, 1) = num_N_pre - TN_i(i, 1);
    end
    micro_P = sum(TP_i) / (sum(TP_i) + sum(FP_i) + 0.000001);
    micro_R = sum(TP_i) / (sum(TP_i) + sum(FN_i) + 0.000001);
    microF1 = (2 * micro_P * micro_R) / (micro_P + micro_R + 0.000001);
end