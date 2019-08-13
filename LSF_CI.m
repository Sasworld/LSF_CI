function [model_LLSF] = LSF_CI( X, Y, optmParameter)
    
   %% optimization parameters
    lambda1          = optmParameter.lambda1;
    lambda2          = optmParameter.lambda2;
    lambda3          = optmParameter.lambda3;
    neg              = optmParameter.neg;
    gamma            = optmParameter.gamma;
    maxIter          = optmParameter.maxIter;
    miniLossMargin   = optmParameter.minimumLossMargin;

   %% initializtion
    num_dim = size(X,2);
    XTX = X'*X;
    XTY = X'*Y;
    W_s   = (XTX + gamma*eye(num_dim)) \ (XTY);
    W_s_1 = W_s;
    % 标记相关性
    R     = pdist2( Y'+eps, Y'+eps, 'cosine' );
    % 样本相关性
    S = sample_similary(X, neg);
    A = diag(sum(S, 2));
    L = A - S;
    
    iter    = 1;
    oldloss = 0;
    
    Lip = sqrt(3 * (norm(XTX)^2 + norm(lambda1 * R)^2 + norm(lambda2 * X'*L*X)^2));

    bk = 1;
    bk_1 = 1; 
    
   %% proximal gradient
    while iter <= maxIter

       W_s_k  = W_s + (bk_1 - 1)/bk * (W_s - W_s_1);
       Gw_s_k = W_s_k - 1/Lip * ((XTX*W_s_k - XTY) + lambda1 * W_s_k * R + lambda2 * X'*L*X*W_s_k);
       bk_1   = bk;
       bk     = (1 + sqrt(4*bk^2 + 1))/2;
       W_s_1  = W_s;
       W_s    = softthres(Gw_s_k,lambda3/Lip);
       
       predictionLoss = trace((X*W_s - Y)'*(X*W_s - Y));
       correlation     = trace(R*W_s'*W_s);
       sparsity = sum(sum(abs(W_s)), 2);
       sample = trace(W_s'*X'*L*X*W_s);
%        sparsity    = sum(sum(W_s~=0));
       totalloss = predictionLoss / 2 + lambda1 * correlation / 2 + lambda2 * sample / 2 + lambda3 * sparsity;
%        fprintf('=============== predictionLoss: %f ================ \n', predictionLoss);
%        fprintf('=============== correlation: %f ================ \n', correlation);
%        fprintf('=============== sparsity: %f ================ \n', sparsity);
%        fprintf('=============== %d %f ================ \n', iter, totalloss);
       if abs(oldloss - totalloss) <= miniLossMargin
           break;
       elseif totalloss <=0
           break;
       else
           oldloss = totalloss;
       end
       
       iter=iter+1;
    end
    model_LLSF = W_s;
end


%% soft thresholding operator
function W = softthres(W_t,lambda)
    W = max(W_t-lambda,0) - max(-W_t-lambda,0); 
end
