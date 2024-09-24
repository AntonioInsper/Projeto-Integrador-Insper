library(tidyverse)
library(rsample)    
library(rpart)     
library(rpart.plot) 
library(yardstick) 
library(skimr)
library(modeldata)
library(glmnet)
library(plotmo)
library(pROC)
library(caret)
library(tidymodels)
library(readr)
library(partykit)
library(MASS)
library(ranger)
library(dplyr)
library(ggthemes)
library(vip)
library(fastDummies)
library(GGally)
library(gbm)
library(readr)
library(xgboost)
library(forcats)
library(survival)
library(survminer)




bisnode_tratado <- read_csv("//wsl.localhost/Ubuntu/home/antonio/Insper/Computação Para A Ciência de Dados/bisnode_tratado")
View(bisnode_tratado)

bis <- bisnode_tratado


# 1. Defina as métricas de Desempenho -------------------------------------

#Pela variavel resposta ser uma variavél qualitativa binomial, em um momento inicial escolhemos usar com mé-
#tricas de avaliação de desempenho Acurácia e Curva ROC.

# Active =1
# Inactive = 0

# 2. Modelos --------------------------------------------------------------



resultados <- tibble(modelo = c("Logística", "Ridge", "Lasso", "Floresta Aleatoria","XGB"), 
                     ACURACIA = NA, AUC = NA)
bis <-subset(bis, select = -exit_year) # == retirar no Phyton
bis <-subset(bis, select = -exit_date)
bis <- subset(bis, select = -`Log10 Sales`)
bis <- subset(bis, select = -balsheet_flag)                  
bis <- subset(bis, select = -balsheet_length)
bis <- subset(bis, select = -birth_year) 
bis <- subset(bis, select = -ind)  
bis <- subset(bis, select = -ind2)
bis <- subset(bis, select = -begin)
bis <- subset(bis, select = -end)
#bis <- subset(bis, select = -company_age)




bis %>% 
  group_by(status) %>% 
  skim()

bis <- bis %>% 
  mutate(status = factor(status, levels = c("Inactive", "Active")))

corte <- 0.50

set.seed(7777)

splits <- initial_split(bis, prop = .75, strata  = "status")

treinamento <- training(splits)
teste <- testing(splits)


treinamento$status <- as.factor(treinamento$status)
teste$status <- as.factor(teste$status)

# 2.1 GLM -----------------------------------------------------------------

#Company_age e Founded_year parecem só apresentar problemas na regrassão logística. 

fit <- glm(status ~ . -founded_year -company_age, data = treinamento, family = "binomial")

summary(fit)

prob_logistica <- predict(fit, teste, type = "response")

Acuracia_Log <- mean(teste$status == ifelse(prob_logistica >= corte, "Active", "Inactive"))

resultados$ACURACIA[resultados$modelo == "Logística"] <- Acuracia_Log

 roc_curva <- roc(teste$status, prob_logistica)

 
resultados$AUC[resultados$modelo == "Logística"] <-roc_curva$auc


# 2.2 RIDGE -------------------------------------------------------------------
idx <- sample(nrow(bis), size = .75 * nrow(bis), replace = FALSE)

#X_tr <- model.matrix(status ~ .-founded_year -company_age, bis)[idx, -1]
X_tr <- model.matrix(status ~ ., bis)[idx, -1]
y_tr <- bis$status[idx]

#X_test <- model.matrix(status ~ .-founded_year -company_age, bis)[-idx,-1]
X_test <- model.matrix(status ~ ., bis)[-idx,-1]
y_test <- bis$status[-idx]


ridge <- glmnet(X_tr, y_tr, alpha = 0, family = "binomial")

plot_glmnet(ridge)

cv_ridge <- cv.glmnet(X_tr, y_tr, alpha = 0, family = "binomial")

plot(cv_ridge)

lambda_ridge <- cv_ridge$lambda.1se

predict(ridge, newx = X_test, type = "response", s = lambda_ridge)

predict(ridge, newx = X_test, type = "class", s = lambda_ridge)

prob_ridge <- as.numeric(predict(ridge, newx = X_test, type = "response", s = lambda_ridge))

Acuracia_Ridge <- mean(y_test == ifelse(prob_ridge >= corte, "Active", "Inactive"))


resultados$ACURACIA[resultados$modelo == "Ridge"] <- Acuracia_Ridge

roc_curva_ridge <- roc(y_test, prob_ridge)
resultados$AUC[resultados$modelo == "Ridge"] <-roc_curva_ridge$auc

# 2.3 LASSO -------------------------------------------------------------------
lasso <- glmnet(X_tr, y_tr, alpha = 1, family = "binomial")

plot_glmnet(lasso)

cv_lasso <- cv.glmnet(X_tr, y_tr, alpha = 1, family = "binomial")

plot(cv_lasso)

lambda_lasso <- cv_lasso$lambda.1se

predict(lasso, newx = X_test, type = "response", s = lambda_lasso)

predict(lasso, newx = X_test, type = "class", s = lambda_lasso)

prob_lasso <- as.numeric(predict(lasso, newx = X_test, type = "response", s = lambda_lasso))

Acuracia_Lasso <- mean(y_test == ifelse(prob_lasso >= corte, "Active", "Inactive"))


resultados$ACURACIA[resultados$modelo == "Lasso"] <- Acuracia_Lasso

roc_curva_lasso <- roc(y_test, prob_lasso)
resultados$AUC[resultados$modelo == "Lasso"] <-roc_curva_lasso$auc
# 2.4 Floresta Aleatoria --------------------------------------------------


colnames(treinamento) <- make.names(colnames(treinamento))
colnames(teste) <- make.names(colnames(teste))

#rf <- ranger(status ~ . -founded_year -company_age, data = treinamento) 

rf <- ranger(status ~ . , data = treinamento, probability = T) 

summary(rf)
rf
pred_rf <- predict(rf, data = teste, type= "response")
probabilidade_rf <- pred_rf$predictions[, "Active"]


table(observado = teste$status,
      predito = ifelse(probabilidade_rf >= corte,"Active", "Inactive"))



acuracia_rf <- mean(teste$status == ifelse(probabilidade_rf >= corte, "Active", "Inactive"))

resultados$ACURACIA[resultados$modelo == "Floresta Aleatoria"] <- acuracia_rf
resultados

floresta <- tibble(n_arvores = 1:500, mse = NA)
for (i in 1:nrow(floresta)) {
  rf <- ranger(status ~ ., num.trees = floresta$n_arvores[i], data = treinamento)
  floresta$mse[i] <- rf$prediction.error
}
floresta %>% 
  ggplot(aes(n_arvores, mse)) + 
  geom_line(color = "#5B5FFF", size = 1.2) + 
  labs(x = "Numero de Arvores", y = "MSE (OOB)") + 
  theme_bw()

floresta %>%
  dplyr::select(n_arvores, mse) %>%
  slice_min(mse)

#rf <- ranger(status ~ . -founded_year -company_age, num.trees = 106, data = treinamento)
rf <- ranger(status ~ ., num.trees = 106, data = treinamento, probability = T)
rf 

probabilidade_rf <- predict(rf, data = teste)$predictions[, "Active"]

acuracia_rf <- mean(teste$status == ifelse(probabilidade_rf >= corte, "Active", "Inactive"))

resultados$ACURACIA[resultados$modelo == "Floresta Aleatoria"] <- acuracia_rf
resultados

roc_curva_floresta <- roc(teste$status, probabilidade_rf)
resultados$AUC[resultados$modelo == "Floresta Aleatoria"] <-roc_curva_floresta$auc
resultados





# 2.5 XGB -----------------------------------------------------------------

X <-model.matrix(status ~ ., -1, data = bis)

#bis_x <- data.frame(X) %>% 
#  mutate(status = bis$status)

treinamento$status <- ifelse(treinamento$status == "Active", 1, 0)
teste$status <- ifelse(teste$status == "Active", 1, 0)


b_treino <- xgb.DMatrix(label = treinamento$status, 
                        data = model.matrix(~ . + 0, data = treinamento %>% dplyr::select(-status)))

b_teste <- xgb.DMatrix(label = teste$status, 
                       data = model.matrix(~ . + 0, data = teste %>% dplyr::select(-status)))
(fit_xgb <- xgb.train(data = b_treino, nrounds = 100, max_depth = 1, eta = 0.3,
                      nthread = 3, verbose = FALSE, objective = "binary:logistic"))
summary(fit_xgb)

prob_xgb <- predict(fit_xgb, b_teste)
acuracia_xgb <- mean(teste$status == ifelse(prob_xgb >= corte, 1, 0))
resultados$ACURACIA[resultados$modelo == "XGB"] <- acuracia_xgb
resultados


roc_curva_XGB <- roc(teste$status, prob_xgb)
resultados$AUC[resultados$modelo == "XGB"] <-roc_curva_XGB$auc
resultados


#Vamos tentar ajustar?


ajusta_bst <- function(splits, eta, nrounds, max_depth) {
  tr_ajs <- training(splits)
  teste_ajs <- testing(splits)
  b_treino <- xgb.DMatrix(label = treinamento$status, 
                                    data = model.matrix(~ . + 0, data = treinamento %>% dplyr::select(-status)))
  b_teste <- xgb.DMatrix(label = teste$status, 
                         data = model.matrix(~ . + 0, data = teste %>% dplyr::select(-status)))
  fit <- xgb.train(data = b_treino, nrounds = nrounds, max_depth = max_depth, eta = eta,
                   nthread = 3, verbose = FALSE, objective = "binary:logistic")
  acc <- mean(teste$status == ifelse(predict(fit_xgb, b_teste) >= corte, 1, 0))
  return(acc)
 }


hiperparametros <- crossing(eta = c(.01, .1),
                            nrounds = c(250, 750),
                            max_depth = c(1, 4))

resultados_ajs <- rsample::vfold_cv(treinamento, 5) %>%
  crossing(hiperparametros) %>%
  mutate(acc = pmap_dbl(list(splits, eta, nrounds, max_depth), ajusta_bst))

resultados_ajs %>%
  group_by(eta, nrounds, max_depth) %>%
  summarise(acc = mean(acc)) %>%
  arrange(acc)

fit_xgb <- xgb.train(data = b_treino, nrounds = 250, max_depth = 1, eta = 0.1,
                     nthread = 3, verbose = FALSE, objective = "binary:logistic")

prob_xgb <- predict(fit_xgb, b_teste)
acuracia_xgb <- mean(teste$status == ifelse(prob_xgb >= corte, 1, 0))
resultados$ACURACIA[resultados$modelo == "XGB"] <- acuracia_xgb


roc_curva_XGB <- roc(teste$status, prob_xgb)
resultados$AUC[resultados$modelo == "XGB"] <-roc_curva_XGB$auc
resultados

#Com essa seed (7777) náo mudou nada -.-



# 3 Comparações -------------------------------------------------------------

print(resultados)

# 4 Importância das Preditoras ----------------------------------------------

rf <- ranger(status ~ ., num.trees = 106, importance='impurity', data = treinamento, probability = T)
vip::vip(rf, aesthetics = list(fill = "blue"))
rf <- ranger(status ~ ., num.trees = 106, importance='permutation', data = treinamento, probability = T)
vip::vip(rf, aesthetics = list(fill = "red"))


fit_glm <- glm(status ~ ., data = treinamento, family = "binomial")
vip::vip(fit_glm, aesthetics = list(fill = "yellow"))

vip::vip(lasso, aesthetics = list(fill = "orange"))

vip::vip(ridge, aesthetics = list(fill = "purple"))

importancia <- xgb.importance(model = fit_xgb)
xgb.plot.importance(importancia, rel_to_first = TRUE, top_n = 10, xlab = "Relative Import")




