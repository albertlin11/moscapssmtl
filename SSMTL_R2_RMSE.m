%% Train_SSMTL
clc;
file = readtable('./SSMTL/csv/SSMTL_train_50.csv'); %%change name
file.Properties.VariableNames;
true = reshape(file.Cm_pF,[],1);
predict = reshape(file.Cm_pred_pF,[],1);
mse = mean((true - predict).^2);
rmse = sqrt(mse);
r2 = 1 - sum((true - predict).^2)/sum((true - mean(true)).^2);
fprintf(1,'==================Train==================');
fprintf(1,'\nmse  = %6.4f',mse);
fprintf(1,'\nrmse = %6.4f',rmse);
fprintf(1,'\nr2   = %6.4f',r2);
%% Validation_SSMTL
file = readtable('./SSMTL/csv/SSMTL_val_50.csv'); %%change name
file.Properties.VariableNames;
true = reshape(file.Cm_pF,[],1);
predict = reshape(file.Cm_pred_pF,[],1);
mse = mean((true - predict).^2);
rmse = sqrt(mse);
r2 = 1 - sum((true - predict).^2)/sum((true - mean(true)).^2);
fprintf(1,'\n==================Valid==================');
fprintf(1,'\nmse  = %6.4f',mse);
fprintf(1,'\nrmse = %6.4f',rmse);
fprintf(1,'\nr2   = %6.4f',r2);
%% Test_SSMTL
file = readtable('./SSMTL/csv/SSMTL_test_50.csv'); %%change name
file.Properties.VariableNames;

%%%%%%%%%%%%%%%%%% test 3k %%%%%%%%%%%%%%%%%%

test_3k = file(file.frequency_kHz == 3 ,:);
true_3k = reshape(test_3k.Cm_pF,[],1);
predict_3k = reshape(test_3k.Cm_pred_pF,[],1);
mse_3k = mean((true_3k - predict_3k).^2);
rmse_3k = sqrt(mse_3k);
r2_3k = 1 - sum((true_3k - predict_3k).^2)/sum((true_3k - mean(true_3k)).^2);

%%%%%%%%%%%%%%%%%% test 5k %%%%%%%%%%%%%%%%%%

test_5k = file(file.frequency_kHz == 5 ,:);
true_5k = reshape(test_5k.Cm_pF,[],1);
predict_5k = reshape(test_5k.Cm_pred_pF,[],1);
mse_5k = mean((true_5k - predict_5k).^2);
rmse_5k = sqrt(mse_5k);
r2_5k = 1 - sum((true_5k - predict_5k).^2)/sum((true_5k - mean(true_5k)).^2);

%%%%%%%%%%%%%%%%%% test 10k %%%%%%%%%%%%%%%%%

test_10k = file(file.frequency_kHz == 10 ,:);
true_10k = reshape(test_10k.Cm_pF,[],1);
predict_10k = reshape(test_10k.Cm_pred_pF,[],1);
mse_10k = mean((true_10k - predict_10k).^2);
rmse_10k = sqrt(mse_10k);
r2_10k = 1 - sum((true_10k - predict_10k).^2)/sum((true_10k - mean(true_10k)).^2);

%%%%%%%%%%%%%%%%%% test 50k %%%%%%%%%%%%%%%%%

test_50k = file(file.frequency_kHz == 50 ,:);
true_50k = reshape(test_50k.Cm_pF,[],1);
predict_50k = reshape(test_50k.Cm_pred_pF,[],1);
mse_50k = mean((true_50k - predict_50k).^2);
rmse_50k = sqrt(mse_50k);
r2_50k = 1 - sum((true_50k - predict_50k).^2)/sum((true_50k - mean(true_50k)).^2);

%%%%%%%%%%%%%%%%%%%% all %%%%%%%%%%%%%%%%%%%%

true = reshape(file.Cm_pF,[],1);
predict = reshape(file.Cm_pred_pF,[],1);
mse = mean((true - predict).^2);
rmse = sqrt(mse);
r2 = 1 - sum((true - predict).^2)/sum((true - mean(true)).^2);

fprintf(1,'\n===================Test==================');
fprintf(1,'\n----------------for 3k Hz----------------');
fprintf(1,'\nmse  = %6.4f',mse_3k);
fprintf(1,'\nrmse = %6.4f',rmse_3k);
fprintf(1,'\nr2   = %6.4f',r2_3k);
fprintf(1,'\n----------------for 5k Hz----------------');
fprintf(1,'\nmse  = %6.4f',mse_5k);
fprintf(1,'\nrmse = %6.4f',rmse_5k);
fprintf(1,'\nr2   = %6.4f',r2_5k);
fprintf(1,'\n----------------for 10k Hz---------------');
fprintf(1,'\nmse  = %6.4f',mse_10k);
fprintf(1,'\nrmse = %6.4f',rmse_10k);
fprintf(1,'\nr2   = %6.4f',r2_10k);
fprintf(1,'\n----------------for 50k Hz---------------');
fprintf(1,'\nmse  = %6.4f',mse_50k);
fprintf(1,'\nrmse = %6.4f',rmse_50k);
fprintf(1,'\nr2   = %6.4f',r2_50k);
fprintf(1,'\n-----------------for all-----------------');
fprintf(1,'\nmse  = %6.4f',mse);
fprintf(1,'\nrmse = %6.4f',rmse);
fprintf(1,'\nr2   = %6.4f',r2);