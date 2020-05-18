rm (list=ls())
cat("\014")

library(dplyr)
library(MASS)
library(glmnet)
library(randomForest)
library(ggplot2)
library(reshape)
library(gridExtra)

#################################################################################################
# read csv file
Twitter = read.csv("twitter.csv")

# standardize the dataset
for (i in 1:70) {
  Twitter[, i] = Twitter[,i]/sd(Twitter[,i])
}

a = Twitter
dim(a)
colnames(a)[dim(a)[2]] = c("Target")
colMeans(a)
t.data = a

colMeans(t.data)
apply(t.data, 2, sd)
mean(t.data$Target)
attach(t.data)

# data structure and rates
p = dim(t.data)[2]-1
n = dim(t.data)[1]

# Modelling factors
iterations = 100
Dlearn_rate = 0.8
sampling.rate = 1

# train and test error rate matrix
train_error = matrix(0, nrow = iterations, ncol = 4)
colnames(train_error) = c("LASSO","Elastic", "Ridge", "RF") #, "OOB", "OOBsd")

cv_error = matrix(0, nrow = iterations, ncol = 3)
colnames(cv_error) = c("LASSO","Elastic", "Ridge")

test_error = matrix(0, nrow = iterations, ncol = 4)
colnames(test_error) = c("LASSO","Elastic", "Ridge", "RF")

lasso.coef = matrix(0, ncol = iterations, nrow = p+1)
el.coef = matrix(0, ncol = iterations, nrow = p+1)
ridge.coef = matrix(0, ncol = iterations, nrow = p+1)

# convert to data frame
train_error = data.frame(train_error)
test_error = data.frame(test_error)

# time of cv and fit
time.cv = matrix(0, nrow = iterations, ncol = 3)
colnames(time.cv) = c("LASSO", "Elastic", "Ridge")

time.fit = matrix(0, nrow = iterations, ncol = 4)
colnames(time.fit) = c("LASSO", "Elastic", "Ridge", "RF")

# rf importance
rf.importace = matrix(0, nrow = iterations, ncol = p)

# sampling from t.data
sampling = sample(n,n*sampling.rate)
sampling.data = data.frame(t.data[sampling,])
sampling.n = dim(sampling.data)[1]

# preparation for lasso and ridge
X = model.matrix(Target ~., sampling.data)[,-1]
y = sampling.data$Target


# 100 iteration for error rates, time, and coefficients
for(m in 1:iterations){
  
  cat("iteration = ", m, "\n")
  
  # create a training data vector for dividing the data set.
  train = sample(sampling.n, sampling.n*Dlearn_rate)
  
  cat("lasso iteration = ", m, "\n")
  # lasso cross validation and tune lambda
  # record lasso cv time
  ptm = proc.time()
  cv.lasso = cv.glmnet(X[train,], y[train], alpha = 1, family = "gaussian", 
                       intercept = T)
  ptm = proc.time() - ptm
  time.cv[m,1]  = ptm["elapsed"]
  
  cv_error[m,1] = min(cv.lasso$cvm)
  bestlam = cv.lasso$lambda.min
  
  # record lasso fit time
  ptm = proc.time()
  lasso.mod = glmnet(X[train,], y[train], alpha = 1, family = "gaussian",
                     intercept = T, lambda = bestlam,
                     standardize = F)
  ptm = proc.time() - ptm
  time.fit[m,1]  = ptm["elapsed"]
  
  lasso.coef[,m] = coef(lasso.mod)[,1]
  lasso.pred = as.vector(predict(lasso.mod, s = bestlam, newx = X[train,], type ="response"))
  lasso.train.residual = y[train] - lasso.pred
  train_error[m,1] = 1- (mean((y[train] - lasso.pred)^2)/mean((y-mean(y))^2))
  
  lasso.pred = as.vector(predict(lasso.mod, s = bestlam, newx = X[-train,], type="response"))
  lasso.test.residual = y[-train] - lasso.pred
  test_error[m,1] = 1- (mean((y[-train] - lasso.pred)^2)/mean((y-mean(y))^2))

  cat("elastic net iteration = ", m, "\n")
  # elastic net cross validation and tune lambda
  # record elastic net cv time
  ptm = proc.time()
  cv.el = cv.glmnet(X[train,], y[train], alpha = 0.5, family = "gaussian", 
                       intercept = T)
  ptm = proc.time() - ptm
  time.cv[m,2]  = ptm["elapsed"]
  
  cv_error[m,2] = min(cv.el$cvm)
  bestlam = cv.el$lambda.min
  
  # record elastic net fit time
  ptm = proc.time()
  el.mod = glmnet(X[train,], y[train], alpha = 0.5, family = "gaussian",
                     intercept = T, lambda = bestlam,
                     standardize = F)
  ptm = proc.time() - ptm
  time.fit[m,2]  = ptm["elapsed"]
  
  el.coef[,m] = coef(el.mod)[,1]
  el.pred = as.vector(predict(el.mod, s = bestlam, newx = X[train,], type ="response"))
  el.train.residual = y[train] - el.pred
  train_error[m,2] = 1- (mean((y[train] - el.pred)^2)/mean((y-mean(y))^2))
  el.pred = as.vector(predict(el.mod, s = bestlam, newx = X[-train,], type="response"))
  el.test.residual = y[-train] - el.pred
  test_error[m,2] = 1- (mean((y[-train] - el.pred)^2)/mean((y-mean(y))^2))
    
  cat("ridge iteration = ", m, "\n")
  # ridge cross validation and tune lambda
  # record ridge cv time
  ptm = proc.time()
  cv.ridge = cv.glmnet(X[train,], y[train], alpha = 0, family = "gaussian", intercept = T)
  ptm = proc.time() - ptm
  time.cv[m,3]  = ptm["elapsed"]
  
  cv_error[m,3] = min(cv.ridge$cvm)
  bestlam = cv.ridge$lambda.min
  
  # record ridge fit time
  ptm = proc.time()
  ridge.mod = glmnet(X[train,], y[train], alpha = 0, family = "gaussian", 
                     intercept = T, lambda = bestlam,
                     standardize = F)
  ptm = proc.time() - ptm
  time.fit[m,3]  = ptm["elapsed"]
  
  ridge.coef[,m] = as.matrix(coef(ridge.mod))
  ridge.pred = as.vector(predict(ridge.mod, s = bestlam, newx = X[train,], type = "response"))
  ridge.train.residual = y[train] - ridge.pred
  train_error[m,3] = 1- (mean((y[train] - ridge.pred)^2)/mean((y-mean(y))^2))
  ridge.pred = as.vector(predict(ridge.mod, s = bestlam, newx = X[-train,], type = "response"))
  ridge.test.residual = y[-train] - ridge.pred
  test_error[m,3] = 1- (mean((y[-train] - ridge.pred)^2)/mean((y-mean(y))^2))

  cat("random forest iteration = ", m, "\n")
  #random forest with 500 bootstrapped trees
  ptm = proc.time()
  rf = randomForest(X[train,], y[train], mtry = sqrt(p), importance = T)
  ptm = proc.time() - ptm
  time.fit[m,4]  = ptm["elapsed"]
  
  rf.pred = predict(rf, newdata = X[train,])
  rf.train.residual = y[train] - rf.pred
  train_error[m,4] = 1- (mean(y[train] - rf.pred)^2)/mean((y-mean(y))^2)
  rf.pred = predict(rf, newdata = X[-train,])
  rf.test.residual = y[-train] - rf.pred
  test_error[m,4] = 1- (mean(y[-train] - rf.pred)^2)/mean((y-mean(y))^2)

}

apply(train_error, 2, mean)
apply(train_error, 2, sd)

apply(test_error, 2, mean)
apply(test_error, 2, sd)

apply(time.cv, 2, mean)
apply(time.cv, 2, sd)

apply(time.fit, 2, mean)
apply(time.fit, 2, sd)

time.total = time.fit

for(i in 1:3){
  time.total[ , i] = time.cv[ , i] + time.fit[ , i]
}

apply(time.total, 2, mean)
apply(time.total, 2, sd)

############################################
############################################

# store error rate and coef
write.csv(ridge.coef, file ="ridge_coef.csv")
write.csv(lasso.coef, file = "lasso_coef.csv")
write.csv(el.coef, file = "el_coef.csv")

write.csv(cv_error, file = "cv_error.csv")
write.csv(test_error, file = "test_error.csv")
write.csv(train_error, file = "train_error.csv")

write.csv(time.cv, file = "time_cv.csv")
write.csv(time.fit, file = "time_fit.csv")

cv_error = 1-cv_error/mean((y-mean(y))^2)

# boxplot of r-square
f1_1 = ggplot(melt(train_error[,1:4]), aes(x = variable, y = value, color = variable)) + 
  geom_boxplot() + ylim(0.9,1.1) + theme(legend.position="none") + scale_color_brewer(palette="Dark2") +
  labs(x = element_blank(), y = "R-Squred", title = expression(Train~R2))
f1_2 = ggplot(melt(test_error), aes(x = variable, y = value, color = variable)) + 
  geom_boxplot() + ylim(0.9,1.1) + theme(legend.position="none") + scale_color_brewer(palette="Dark2") +
  labs(x = element_blank(), y = "R-Squred", title = expression(Test~R2))
# f1_3 = ggplot(melt(cv_error), aes(x = variable, y = value, color = variable), inherit.aes = FALSE ) + 
#   geom_boxplot() + ylim(0.9,1.1) + theme(legend.position="none") + scale_color_brewer(palette="Dark2") +
#   labs(x = element_blank(), y = "R-Squred", title = expression(CV~R2))
f1 = grid.arrange(f1_1, f1_2, nrow = 1, widths = c(1.5,1.5))


# cv curves
par(mfrow=c(1,3))
f2_1 = plot(cv.lasso, ylim=c(0,5e+05), main="LASSO")
f2_2 = plot(cv.el, ylim=c(0,5e+05), main="Elastic Net")
f2_3 = plot(cv.ridge, ylim=c(0,5e+05), main="Ridge")
f2 = grid.arrange(f2_1, f2_2, f2_3, nrow = 1)

# residual boxplots
par(mfrow=c(1,1))
train_residual = data.frame(cbind(lasso.train.residual, el.train.residual, ridge.train.residual, rf.train.residual))
colnames(train_residual) = c("LASSO", "Elastic", "Ridge", "RF")
test_residual = data.frame(cbind(lasso.test.residual, el.test.residual, ridge.test.residual, rf.test.residual))
colnames(test_residual) = c("LASSO", "Elastic", "Ridge", "RF")

f3_1 = ggplot(melt(train_residual), aes(x = variable, y = value, color = variable)) + 
  geom_boxplot() + ylim(-1000,11000) + theme(legend.position="none") + scale_color_brewer(palette="Dark2") +
  labs(x = element_blank(), y = "Residuals", title = expression(Train~Residual))
f3_2 = ggplot(melt(test_residual), aes(x = variable, y = value, color = variable)) + 
  geom_boxplot() + ylim(-1000,11000) + theme(legend.position="none") + scale_color_brewer(palette="Dark2") +
  labs(x = element_blank(), y = "Residuals", title = expression(Test~Residual))
f3 = grid.arrange(f3_1, f3_2, nrow = 1, widths = c(1.5,1.5))

############################################
############################################

bootstrapSamples =     100
beta.rf.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.ls.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)         
beta.en.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)         
beta.rd.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)         

for (m in 1:bootstrapSamples){
  bs_indexes       =     sample(n, replace=T)
  X.bs             =     X[bs_indexes, ]
  y.bs             =     y[bs_indexes]
  
  cat(sprintf("Bootstrap Sample %3.f \n", m))
  
  cat(sprintf("Bootstrap Sample - lasso %3.f \n", m))
  # fit bs lasso
  a                =     1 # lasso
  cv.fit           =     cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, alpha = a, lambda = cv.fit$lambda.min)  
  beta.ls.bs[,m]   =     as.vector(fit$beta)
  
  cat(sprintf("Bootstrap Sample - elastic %3.f \n", m))
  # fit bs en
  a                =     0.5 # elastic-net
  cv.fit           =     cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, alpha = a, lambda = cv.fit$lambda.min)  
  beta.en.bs[,m]   =     as.vector(fit$beta)

  cat(sprintf("Bootstrap Sample - ridge %3.f \n", m))
  # fit bs ridge
  a                =     0 # ridge
  cv.fit           =     cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, alpha = a, lambda = cv.fit$lambda.min)  
  beta.rd.bs[,m]   =     as.vector(fit$beta)

    cat(sprintf("Bootstrap Sample - random forest %3.f \n", m))
  # fit bs rf
  rf               =     randomForest(X.bs, y.bs, mtry = sqrt(p), importance = TRUE)
  beta.rf.bs[,m]   =     as.vector(rf$importance[,1])

}

# calculate bootstrapped standard errors / alternatively you could use qunatiles to find upper and lower bounds
rf.bs.sd    = apply(beta.rf.bs, 1, "sd")
ls.bs.sd    = apply(beta.ls.bs, 1, "sd")
en.bs.sd    = apply(beta.en.bs, 1, "sd")
rd.bs.sd    = apply(beta.rd.bs, 1, "sd")

# fit lasso to the whole data
a=1 # lasso
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit              =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)
betaS.ls               =     data.frame(names(X[1,]), as.vector(fit$beta), 2*ls.bs.sd)
colnames(betaS.ls)     =     c( "feature", "value", "err")

# fit en to the whole data
a=0.5 # elastic-net
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit              =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)
betaS.en               =     data.frame(names(X[1,]), as.vector(fit$beta), 2*en.bs.sd)
colnames(betaS.en)     =     c( "feature", "value", "err")

# fit ridge to the whole data
a=0 # ridge
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit              =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)
betaS.rd               =     data.frame(names(X[1,]), as.vector(fit$beta), 2*rd.bs.sd)
colnames(betaS.rd)     =     c( "feature", "value", "err")

# fit rf to the whole data
rf               =     randomForest(X, y, mtry = sqrt(p), importance = TRUE)
betaS.rf               =     data.frame(names(X[1,]), as.vector(rf$importance[,1]), 2*rf.bs.sd)
colnames(betaS.rf)     =     c( "feature", "value", "err")

# we need to change the order of factor levels by specifying the order explicitly.
betaS.rf$feature     =  factor(betaS.rf$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.ls$feature     =  factor(betaS.ls$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.en$feature     =  factor(betaS.en$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.rd$feature     =  factor(betaS.rd$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])


lsPlot =  ggplot(betaS.ls, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  labs(x = element_blank(), y = "Coefficients", title = expression(LASSO))

enPlot =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  labs(x = element_blank(), y = "Coefficients", title = expression(Elastic))

rdPlot =  ggplot(betaS.rd, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  labs(x = element_blank(), y = "Coefficients", title = expression(Ridge))

rfPlot =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  labs(x = element_blank(), y = "Importance", title = expression(Randon~Forest))

f4 = grid.arrange(lsPlot, enPlot, rdPlot, rfPlot, nrow = 4)
