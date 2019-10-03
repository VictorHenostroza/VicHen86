# Utilizaré XGBoost para el modelo de clasificación

# Activar librerías necesarias para el algoritmo
library(mice)
library(VIM)
library(caret)

# Importar el dataset y analizar las variables
training_set = read.csv('train.csv', stringsAsFactors = TRUE)
testing_set = read.csv('test.csv', stringsAsFactors = TRUE)
testing_set$id=NULL
str(training_set)        # Hay variables de tipo numérica y de tipo categórica
head(training_set)       # Ver primeros valores de las variables
summary(training_set)    # Ver proncipales estadísticos
md.pattern(training_set) # Existen 329 valores perdidos (NAs), hay que rellenarlos con alguna técnica.

# Llenar valores perdidos con imputaciones
# a) Si la variable es numéricá con predictive mean matching
# b) Si la variable es categórica con multinomial logistic regression
p <- function(x) {sum(is.na(x))/length(x)*100}
apply(training_set, 2, p)

impute <- mice(training_set[,2:14], m=5, seed = 1234)
print(impute)

# Completar los datos perdidos 
# Haré 5 training sets para ver con cual se clasifica mejor
training_new1 <- complete(impute, 1)
training_new2 <- complete(impute, 2)
training_new3 <- complete(impute, 3)
training_new4 <- complete(impute, 4)
training_new5 <- complete(impute, 5)

# Distribución de valores observados e imputados 
# Los rojos son los imputados y los azules los observados
stripplot(impute, pch = 20, cex = 1.2)

# Codificar los factores para tener imput correcto del modelo
str(training_new1)

training_new1$Term = as.numeric(factor(training_new1$Term, labels = c(1:3)))
training_new1$State = as.numeric(factor(training_new1$State, labels = c(1:49)))
training_new1$Income.Verification.Status = as.numeric(factor(training_new1$Income.Verification.Status, labels = c(1:4)))
training_new1$Home.Ownership = as.numeric(factor(training_new1$Home.Ownership, labels = c(1:5)))
training_new1$Loan.Purpose = as.numeric(factor(training_new1$Loan.Purpose, labels = c(1:11)))
training_new1$Due.Settlement = as.numeric(factor(training_new1$Due.Settlement, labels = c(1:3)))
training_new1$Payment.Plan = as.numeric(factor(training_new1$Payment.Plan, labels = c(1:2)))

testing_set$Term = as.numeric(factor(testing_set$Term, labels = c(1:2)))
testing_set$State = as.numeric(factor(testing_set$State, labels = c(1:49)))
testing_set$Income.Verification.Status = as.numeric(factor(testing_set$Income.Verification.Status, labels = c(1:3)))
testing_set$Home.Ownership = as.numeric(factor(testing_set$Home.Ownership, labels = c(1:4)))
testing_set$Loan.Purpose = as.numeric(factor(testing_set$Loan.Purpose, labels = c(1:11)))
testing_set$Due.Settlement = as.numeric(factor(testing_set$Due.Settlement, labels = c(1:2)))
testing_set$Payment.Plan = as.numeric(factor(testing_set$Payment.Plan, labels = c(1)))


str(training_new1)
str(testing_set)
md.pattern(training_new1)

# Aplicando el algoritmoo XGBoost
classifier = xgboost(data = as.matrix(training_new1[, -13]),
                     label = training_new1$Approve.Loan, 
                     nrounds = 100,
                     objective = "binary:logistic",
                     max.depth = 5,
                     nthread = 2,
                     verbose = 1)

y_pred = predict(classifier, newdata = as.matrix(testing_set[,]))
testing_set$Approve.Loan = ifelse((y_pred > 0.5),1,0)
head(testing_set)
