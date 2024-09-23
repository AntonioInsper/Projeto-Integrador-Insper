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

bis <- bisnode_tratado


# 1. Defina as métricas de Desempenho -------------------------------------

#Pela variavel resposta ser uma variavél qualitativa binomial, em um momento inicial escolhemos usar com mé-
#tricas de avaliação de desempenho Acurácia e Curva ROC.

# Active =1
# Inactive = 0

# 2. Modelos --------------------------------------------------------------



resultados <- tibble(modelo = c("Logística", "Ridge", "Lasso", "Floresta Aleatoria"), 
                     ACURACIA = NA, AUC = NA)
#bis <-subset(bis, select = -exit_year)  == retirar no Phyton
#bis <-subset(bis, select = -exit_date)
#bis <- subset(bis, select = -`Log10 Sales`)
#bis <- subset(bis, select = -balsheet_flag)                  
#bis <- subset(bis, select = -balsheet_length)
#bis <- subset(bis, select = -birth_year) 
#bis <- subset(bis, select = -ind)  
#bis <- subset(bis, select = -ind2)  



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


fit <- glm(status ~ . -founded_year -company_age, data = treinamento, family = "binomial")

summary(fit)

prob_logistica <- predict(fit, teste, type = "response")

Acuracia_Log <- mean(teste$status == ifelse(prob_logistica >= corte, "Active", "Inactive"))

resultados$ACURACIA[resultados$modelo == "Logística"] <- Acuracia_Log

 roc_curva <- roc(teste$status, prob_logistica)

 
resultados$AUC[resultados$modelo == "Logística"] <-roc_curva$auc


# 2.2 RIDGE -------------------------------------------------------------------
idx <- sample(nrow(bis), size = .75 * nrow(bis), replace = FALSE)

X_tr <- model.matrix(status ~ .-founded_year -company_age, bis)[idx, -1]
y_tr <- bis$status[idx]

X_test <- model.matrix(status ~ .-founded_year -company_age, bis)[-idx,-1]
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



# Importância das Preditoras ----------------------------------------------

rf <- ranger(status ~ ., num.trees = 106, importance='impurity', data = treinamento, probability = T)
vip::vip(rf, aesthetics = list(fill = "blue"))
rf <- ranger(status ~ ., num.trees = 106, importance='permutation', data = treinamento, probability = T)
vip::vip(rf, aesthetics = list(fill = "red"))


fit_glm <- glm(status ~ ., data = treinamento, family = "binomial")
vip::vip(fit_glm, aesthetics = list(fill = "yellow"))

vip::vip(lasso, aesthetics = list(fill = "orange"))

vip::vip(ridge, aesthetics = list(fill = "purple"))






