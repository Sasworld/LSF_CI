function macroF1 = MyMacroF1(pre_target,test_target)
%     test_target: N * L
    label_P = 1;
    label_N = 0;
    [num_instance, num_class] = size(test_target);
    precision = zeros(num_class,1);
    recall = zeros(num_class,1);
    for i = 1:num_class
        num_P_pre = sum(pre_target(:,i) == label_P); % positive prediction number
        num_N_pre = num_instance - num_P_pre;
        P_index = find(test_target(:,i) == label_P); % positive instance index
        N_index = find(test_target(:,i) == label_N);
        pre_eq_test_index = find(pre_target(:,i) == test_target(:,i)); % the index of prediction and test target are same 
        TP_i = size(intersect(pre_eq_test_index,P_index), 1);
        TN_i = size(intersect(pre_eq_test_index,N_index), 1);
        FP_i = num_P_pre - TP_i;
        FN_i = num_N_pre - TN_i;
        if TP_i + FP_i + FN_i == 0
            precision(i,1) = 1;
            recall(i,1) = 1;
        else
            precision(i,1) = TP_i / (TP_i + FP_i + 0.0000001);
            recall(i,1) = TP_i / (TP_i + FN_i + 0.0000001);            
        end
    end
    macro_P = sum(precision) / num_class;
    macro_R = sum(recall) / num_class;
    macroF1 = (2 * macro_P * macro_R) / (macro_P + macro_R + 0.000001);
end