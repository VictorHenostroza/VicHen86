# Utilizaré XGBoost para el modelo de clasificación

# Activar librerías necesarias para el algoritmo
library(mice)
library(VIM)
library(caret)
library(ggplot2)
library(GGally)

# Importar el dataset y analizar las variables
training_set = read.csv('train.csv', stringsAsFactors = TRUE)

# Ploteo Multivariante 
ggpairs(training_set[,5:14], 
        aes(colour = Approve.Loan, 
            alpha = 0.4),
        title = "Análisis multivariante de las variables independientes vs Approve.Loan",
        upper = list(continuous = "density"),
        lower = list(combo = "denstrip"))+
  theme(plot.title = element_text(hjust = 0.5))  

data_nueva = read.csv('test.csv', stringsAsFactors = TRUE)
data_nueva$id=NULL
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

data_nueva$Term = as.numeric(factor(data_nueva$Term, labels = c(1:2)))
data_nueva$State = as.numeric(factor(data_nueva$State, labels = c(1:49)))
data_nueva$Income.Verification.Status = as.numeric(factor(data_nueva$Income.Verification.Status, labels = c(1:3)))
data_nueva$Home.Ownership = as.numeric(factor(data_nueva$Home.Ownership, labels = c(1:4)))
data_nueva$Loan.Purpose = as.numeric(factor(data_nueva$Loan.Purpose, labels = c(1:11)))
data_nueva$Due.Settlement = as.numeric(factor(data_nueva$Due.Settlement, labels = c(1:2)))
data_nueva$Payment.Plan = as.numeric(factor(data_nueva$Payment.Plan, labels = c(1)))

str(training_new1)
str(data_nueva)
md.pattern(training_new1)

# Dividir los datos en conjunto de entrenamiento y conjunto de test
# install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(training_new1$Approve.Loan, SplitRatio = 0.8)
entrenamiento = subset(training_new1, split == TRUE)
testeo = subset(training_new1, split == FALSE)

# Aplicar el algoritmoo XGBoost
classifier = xgboost(data = as.matrix(entrenamiento[, -13]),
                     label = entrenamiento$Approve.Loan, 
                     nrounds = 100,
                     objective = "binary:logistic",
                     max.depth = 5,
                     nthread = 2,
                     verbose = 1)

# Predecir con el clasificador
y_pred = predict(classifier, newdata = as.matrix(testeo[,-13]))
y_pred = (y_pred > 0.5)

# Hallar matríz de confusión y porcentaje de aciertos
cm = table(testeo[, 13], y_pred)
accuracy=sum(diag(cm))/sum(cm)
