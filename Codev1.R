library(tidyverse)
library(tidytext)
library(tm)
library(textstem)
library(e1071)
library(class)
library(caret)
library(MLmetrics)
library(rpart)
library(nnet)

# import data
data = read.csv('cleaned_data.csv', stringsAsFactors = FALSE)
data$ID = 1:nrow(data)
data = data %>% mutate(label = case_when(
  label == "F" ~ "Favor",
  label == "O" ~ "Opposed",
  label == "N" ~ "Neutral"
)) %>% mutate(label = as.factor(label)) %>% 
  na.omit()
data = data %>% select(ID, comment, label) 
data$ID = as.character(data$ID)

# tokenize, remove stop words, and remove infrequent words
# tokenize and remove stop words. show most frequent words
data %>% 
  unnest_tokens(word, 'comment') %>%  # tokenize
  anti_join(stop_words)%>% # Remove stop words
  count(word, sort = TRUE) %>% # count by word
  arrange(desc(n)) %>% # Everything from this point on is just to graph
  head(20) %>% 
  ggplot(aes(x = reorder(word, n), y = n)) +
  geom_bar(stat= 'identity') + 
  coord_flip() +
  theme_bw()

# remove infrequent words. show distribution
data %>% 
  unnest_tokens(word, 'comment') %>%  # tokenize
  anti_join(stop_words)%>% # Remove stop words
  count(word, sort = TRUE) %>% # count by word
  filter(n >= 2) %>%  # Remove words that occur less than 2 times
  ggplot(aes(n)) +
  geom_histogram() +
  scale_x_log10()

# text preprocessing
comment_corpus = Corpus(VectorSource(data$comment))
# Lower text
comment_corpus = tm_map(comment_corpus, content_transformer(tolower))
comment_corpus = tm_map(comment_corpus, removeNumbers)
comment_corpus = tm_map(comment_corpus, removePunctuation)
comment_corpus = tm_map(comment_corpus, removeWords, c("the", "and", stopwords("english")))
# Lemmatization
comment_corpus = tm_map(comment_corpus, lemmatize_words)
comment_dtm = DocumentTermMatrix(comment_corpus)
# Take a look at tweet_dtm
comment_dtm
inspect(comment_dtm)

comment_dtm = removeSparseTerms(comment_dtm, 0.99)
# Inspect the matrix again
inspect(comment_dtm)

# Convert the matrix to a data frame
comment_dtm_df = as.data.frame(as.matrix(comment_dtm))

# Add label variable to the data frame
comment_dtm_df$ID = data$ID
comment_dtm_df$label = data$label
# train and test split
features = subset(comment_dtm_df, select = -ID)
set.seed(123)
trainIndex = createDataPartition(features$label, p = 0.8, list = FALSE)
train = features[trainIndex,]
test = features[-trainIndex,]

# KNN method
control = trainControl(method = "cv", number = 10,
                        savePredictions = "final", 
                        classProbs = TRUE, 
                        summaryFunction = multiClassSummary)
k_values = c(3,4,5,6,7,8,9)
results = data.frame(K = integer(), Accuracy = numeric(),
                      Precision = numeric(), Recall = numeric())
confusion1 = list()
for(k in k_values){
  set.seed(123)
  knn_fit = train(label ~ ., data = train, method = "knn", trControl = control,
                   preProcess = c("center", "scale"),
                   tuneGrid = expand.grid(k = k),
                   metric = "Accuracy")
  
  accuracy = knn_fit$results$Accuracy
  precision = Precision(knn_fit$pred$pred, knn_fit$pred$obs, positive = "Favor")
  recall = Recall(knn_fit$pred$pred, knn_fit$pred$obs, positive = "Favor")
  results = rbind(results, data.frame(K = k, Accuracy = accuracy,
                                       Precision = precision, Recall = recall))
  
  predictions = predict(knn_fit, test)
  c = confusionMatrix(predictions, test$label)
  confusion1[[as.character(k)]] = c
}

print(results)
print(confusion1)

# neural network method
control2 = trainControl(method = "cv", number = 10,
                        savePredictions = "final", 
                        classProbs = TRUE, 
                        summaryFunction = multiClassSummary)
results2 = data.frame()
confusion2 = list()
sizes = 1
decays = c(0.1, 0.01, 0.001) 
for (decays in decays) {
  set.seed(123)
  nnet_fit = train(label ~ ., data = train, method = "nnet",
                   trControl = control2, 
                   tuneGrid = expand.grid(size = 1, decay = decays),
                   metric = "Accuracy")
  results2 = rbind(results2, nnet_fit$results)
  predictions = predict(nnet_fit, test)
  c2 = confusionMatrix(predictions, test$label)
  confusion2[[as.character(decays)]] = c2
}

print(results2)
print(confusion2)

## Svm method
# control3 = trainControl(method = "cv", number = 10,
#                         savePredictions = "final", 
#                         classProbs = TRUE, 
#                         summaryFunction = multiClassSummary)
# tuneGrid = expand.grid(C = c(0.01, 0.1, 1, 10),
#                         sigma = c(0.01, 0.1, 1, 10))
# set.seed(123)
# svm_fit = train(label ~ ., data = train, method = "svmRadial",
#                  trControl = control3, tuneLength = 5,
#                  tuneGrid = tuneGrid, preProcess = c("center", "scale"))
# results3 = svm_fit$results
# predictions = predict(svm_fit, test)
# confusion3 = confusionMatrix(predictions, test$label)

save(confusion1, confusion2, file = "confusion.RData")
