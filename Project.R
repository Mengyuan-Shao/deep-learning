library(ggplot2)
library(dplyr)
library(tidytext)
library(stringr)
library(stopwords)
library(tm)
library(caTools)
library(caret)
library(keras)

path = "/Users/shaomengyuan/assignment/BDP/project/Womens\ Clothing\ Reviews.csv"
datas <- read.csv(path, stringsAsFactors = FALSE)[-1]

###################Perpare Data####################

# only need column 1, 2, 4 and 5.
review_rating <- cbind(datas[1:2], datas[4:5])
# drop the rows out which review is empty.
review_rating <- review_rating[!(is.na(review_rating$Review.Text) | review_rating$Review.Text==""), ]
# rename column name.
names(review_rating) <- c("Clother_ID","Age","Review", "Rating")

# randomly split data to train, validation and 
# test with 0.6, 0.2 and 0.2.
set.seed(1)
# train <- createDataPartition(review_rating$Review, p=0.8,list=FALSE)
all_data <- sample.split(review_rating, SplitRatio = 0.6)
train_data <- subset(review_rating, all_data==TRUE)
test_data <- subset(review_rating, all_data==FALSE)
val_test <- sample.split(rest_data, SplitRatio = 0.5)
validation_data <- subset(rest_data, val_test==FALSE)
test_data <- subset(rest_data, val_test==TRUE)

data(stop_words)

split_review <- function(data) {
  # apply stop words and per word per row.
  split_review <- data %>% mutate(review_ID = row_number()) %>% 
    unnest_tokens(word, Review) %>% 
    anti_join(stop_words, by=c("word"="word")) 
  # adjust column order. 
  split_review <- split_review[, c(1,2,4,5,3)]
  
  # give each word scores based on "afinn" dictionary.
  afinn_score <- split_review %>%
    inner_join(get_sentiments("afinn"), by="word") #%>%
  # summarise(sentiment = mean(afinn_score_train$score))
  afinn_score <-afinn_score[-4]
  the_dataset <- aggregate(score~Clother_ID+Age+review_ID+Rating, afinn_score, mean)
  the_dataset <- the_dataset[, c(1,2,5,4)]
  return(the_dataset)
}

# prepare the dataset for training
train_dataset <- split_review(train_data)
# translate the input train_data to matrix
train_x <- data.matrix(train_dataset[,1:3])  
train_y <- train_dataset[4]
# translate the output tain_data to matrix
train_y <- to_categorical(as.matrix(train_y))[,c(-1)]

# # prepare the dataset for validating
validate_dataset <- split_review(validation_data)
validate_x <- data.matrix(validate_dataset[,1:3])
validate_y <- validate_dataset[4]
validate_y <- to_categorical(as.matrix(validate_y))[,c(-1)] 

# prepare the dataset for testing
test_dataset <- split_review(test_data)
test_x <- data.matrix(test_dataset[,1:3])  
test_y <- test_dataset[4]
test_y <- to_categorical(as.matrix(test_y))[,c(-1)] 

# label each word with "positive" or "negative".
# bing_score <- split_review %>% inner_join(get_sentiments("bing"), by="word") %>%
#               group_by(index=review_ID)

####################Deep Learning#####################

model <- keras_model_sequential()
# set the layer, units and activation function.
model %>%
  layer_dense(units = 3, activation = "relu", input_shape = ncol(train_x)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(0.1) %>%
  layer_dense(units = 5, activation = "softmax")

summary(model)

# set the hyperparameter
model %>% compile (
                  loss = "categorical_crossentropy",
                  # optimizer = "SGD",
                  optimizer = "rmsprop",
                  metric = c("accuracy"))

history <- model %>% fit(
  train_x,
  train_y,
  epochs = 100,
  batch_size = 32,
  # validation_split = 0.25
  validation_data = list(validate_x, validate_y)
)

###################Visualization####################
# plot the model
plot(history)

# Test the accurarcy of the model
model %>% evaluate(test_x, test_y)


