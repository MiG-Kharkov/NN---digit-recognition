install.packages("readr")
library(readr)
train <- read_csv("train.csv")
test <- read_csv("test.csv")
head(train[1:10])


# Create a 28*28 matrix with pixel color values
displayDigit <- function(X){
  m <- matrix(unlist(X), nrow=28, byrow = T)
  m <- t(apply(m, 2, rev))
  image(m,col=grey.colors(255))
}


install.packages("h2o")
library(h2o)

## start a local h2o cluster
localH2O = h2o.init(max_mem_size = '6g', # use 6GB of RAM of *GB available
                    nthreads = -1) # use all CPUs (8 on my personal computer :3)

## MNIST data as H2O
train$label = as.factor(train$label) # convert digit labels to factor for classification
train_h2o = as.h2o(train)
test_h2o = as.h2o(test)

## set timer
s <- proc.time()

## train model
model =
  h2o.deeplearning(x = 2:785,  # column numbers for predictors
                   y = 1,   # column number for label
                   training_frame = train_h2o, # data in H2O format
                   activation = "RectifierWithDropout", # algorithm
                   input_dropout_ratio = 0.2, # % of inputs dropout
                   hidden_dropout_ratios = c(0.5,0.5), # % for nodes dropout
                   balance_classes = TRUE, 
                   hidden = c(100,100), # two layers of 100 nodes
                   momentum_stable = 0.99,
                   nesterov_accelerated_gradient = T, # use it for speed
                   epochs = 15) # no. of epochs

## print time elapsed
s - proc.time()

## print confusion matrix
h2o.confusionMatrix(model)

## classify test set
h2o_y_test <- h2o.predict(model, test_h2o)

## convert H2O format into data frame and  save as csv
df_y_test = as.data.frame(h2o_y_test)
df_y_test = data.frame(ImageId = seq(1,length(df_y_test$predict)), Label = df_y_test$predict)
write.csv(df_y_test, file = "submission-r-h2o.csv", row.names=F)

## shut down virutal H2O cluster
h2o.shutdown(prompt = F)

