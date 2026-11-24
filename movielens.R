##########################################################
# MovieLens Capstone: movie rating prediction
#
# Goal: predict the rating a user gives a movie, scored by RMSE
# on the final_holdout_test set.
#
# Approach (each step is compared on a validation set cut from edx):
#   1. Baseline: the overall mean rating
#   2. Movie effect, then movie + user effects
#   3. Regularized movie + user effects (penalty lambda tuned on edx only)
#   4. Matrix factorization with the recosystem package (final model)
#
# final_holdout_test is used exactly once, at the very end (section 7),
# to report the RMSE of the final model. It is never used for training,
# tuning or choosing between models.
#
##########################################################


##########################################################
# 1. Packages
##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "https://cloud.r-project.org")
if(!require(caret)) install.packages("caret", repos = "https://cloud.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "https://cloud.r-project.org")

library(tidyverse)
library(caret)
library(recosystem)


##########################################################
# 2. Create the edx and final_holdout_test sets
##########################################################

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 600)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl, mode = "wb")

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

# ratings.dat and movies.dat use "::" as the separator, so we split each line by hand
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


##########################################################
# 3. Split edx into a training set and a validation set
##########################################################

set.seed(1, sample.kind = "Rounding")
valid_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-valid_index,]
temp <- edx[valid_index,]

# Keep only validation rows whose user and movie also appear in the training set,
# otherwise those predictions would be impossible, and move the rest back to training
valid_set <- temp %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")
train_set <- rbind(train_set, anti_join(temp, valid_set))

rm(valid_index, temp)


##########################################################
# 4. Loss function and baseline model
##########################################################

# Root Mean Squared Error: typical size of the prediction error, in rating stars
rmse <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Baseline: predict the average training rating for every movie and user
mu <- mean(train_set$rating)
rmse_results <- tibble(method = "Overall mean rating",
                       RMSE = rmse(valid_set$rating, mu))


##########################################################
# 5. Movie and user effects, with and without regularization
##########################################################

# Prediction = mu + b_i (movie effect) + b_u (user effect).
# Each effect is the average residual, shrunk toward 0 by lambda:
#   b = sum(residuals) / (n + lambda)
# lambda = 0 means no regularization. A larger lambda pulls effects
# estimated from few ratings toward 0 (this is regularization).
effects_rmse <- function(lambda, use_user = TRUE) {
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + lambda))

  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i) / (n() + lambda))

  predictions <- valid_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(b_u = if (use_user) b_u else 0,
           pred = mu + b_i + b_u) %>%
    pull(pred)

  rmse(valid_set$rating, predictions)
}

rmse_results <- bind_rows(
  rmse_results,
  tibble(method = "Movie effect", RMSE = effects_rmse(0, use_user = FALSE)),
  tibble(method = "Movie + user effects", RMSE = effects_rmse(0))
)

# Tune lambda on the validation set 
lambdas <- seq(2, 8, 0.5)
lambda_rmses <- sapply(lambdas, effects_rmse)
best_lambda <- lambdas[which.min(lambda_rmses)]

rmse_results <- bind_rows(
  rmse_results,
  tibble(method = "Regularized movie + user effects", RMSE = min(lambda_rmses))
)


##########################################################
# 6. Matrix factorization (recosystem)
##########################################################

# Idea: arrange the ratings as a large user x movie matrix with mostly missing
# cells. Matrix factorization approximates it as P x Q, where each user and each
# movie gets a short vector of latent factors (for example "likes action" or
# "prefers old classics"). The predicted rating is the dot product of the
# user's and the movie's vectors, which captures user-movie interactions
# that simple additive effects cannot.

# Convert a data frame into the format recosystem expects
to_reco <- function(df) {
  data_memory(user_index = df$userId,
              item_index = df$movieId,
              rating = df$rating,
              index1 = TRUE)
}

n_threads <- max(1, parallel::detectCores() - 1)

# Tune the factorization parameters on the training set (cross-validation inside tune())
set.seed(1, sample.kind = "Rounding")
reco <- Reco()
tuning <- reco$tune(to_reco(train_set),
                    opts = list(dim = c(30, 50),
                                costp_l2 = c(0.01, 0.1),
                                costq_l2 = c(0.01, 0.1),
                                lrate = 0.1,
                                nthread = n_threads,
                                niter = 10,
                                nfold = 3))
best_params <- tuning$min

# Train on the training set and evaluate on the validation set
set.seed(1, sample.kind = "Rounding")
reco$train(to_reco(train_set), opts = c(best_params, list(niter = 30, nthread = n_threads)))
mf_predictions <- reco$predict(to_reco(valid_set), out_memory())

# Ratings can only be between 0.5 and 5, so clip predictions to that range
mf_predictions <- pmin(pmax(mf_predictions, 0.5), 5)

rmse_results <- bind_rows(
  rmse_results,
  tibble(method = "Matrix factorization", RMSE = rmse(valid_set$rating, mf_predictions))
)

print(rmse_results)


##########################################################
# 7. Final model: train on the full edx set, test once on final_holdout_test
##########################################################

# The best model on the validation set is matrix factorization. We retrain it with
# the tuned parameters on all of edx (more data), then use final_holdout_test for
# the first and only time.
set.seed(1, sample.kind = "Rounding")
reco_final <- Reco()
reco_final$train(to_reco(edx), opts = c(best_params, list(niter = 30, nthread = n_threads)))

final_predictions <- reco_final$predict(to_reco(final_holdout_test), out_memory())
final_predictions <- pmin(pmax(final_predictions, 0.5), 5)

final_rmse <- rmse(final_holdout_test$rating, final_predictions)
cat("Final RMSE on final_holdout_test:", final_rmse, "\n")
