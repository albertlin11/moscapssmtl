function R2 = R2(train,predict)

error = train-predict;
norm_error = norm(error);
SSE = norm_error.^2;
SST = norm(train-mean(train))^2;
R2 = 1 - SSE/SST;
