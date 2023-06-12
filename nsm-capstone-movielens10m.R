# CAPSTONE PROJECT: ALL LEARNERS
# NOTE: running time is 7 minutes


# Project Overview: MovieLens


# we need to create a movie recommendation system using the MovieLens 10M Dataset (https://grouplens.org/datasets/movielens/10m/) to predict the rating of a particular user for a particular movie with a root mean squared error (RMSE) of <0.86490


# Create Train and Final Hold-out Test Sets


# we download the dataset and generate an edx set and a final_holdout_test set using the provided code

##########################################################
# Create edx and final_holdout_test sets
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org") # require() checks if the package exists
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120) # timeout in seconds for some Internet operations

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

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
# 'data.frame':	10000054 obs. of  6 variables:
# $ userId   : int  1 1 1 1 1 1 1 1 1 1 ...
# $ movieId  : int  122 185 231 292 316 329 355 356 362 364 ...
# $ rating   : num  5 5 5 5 5 5 5 5 5 5 ...
# $ timestamp: int  838985046 838983525 838983392 838983421 838983392 838983392 838984474 838983653 838984885 838983707 ...
# $ title    : chr  "Boomerang (1992)" "Net, The (1995)" "Dumb & Dumber (1994)" "Outbreak (1995)" ...
# $ genres   : chr  "Comedy|Romance" "Action|Crime|Thriller" "Comedy" "Action|Drama|Sci-Fi|Thriller" ...

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
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


# MovieLens Instructions


# NOTE: we will use the edx set to train and test algorithms, whereas we will reserve the final_holdout_test set only to test the final algorithm


# edx Set


# we use the edx set to train and test algorithms


# we examine the edx set

head(edx)
str(edx)
# 'data.frame':	9000055 obs. of  6 variables:
# $ userId   : int  1 1 1 1 1 1 1 1 1 1 ...
# $ movieId  : int  122 185 292 316 329 355 356 362 364 370 ...
# $ rating   : num  5 5 5 5 5 5 5 5 5 5 ...
# $ timestamp: int  838985046 838983525 838983421 838983392 838983392 838984474 838983653 838984885 838983707 838984596 ...
# $ title    : chr  "Boomerang (1992)" "Net, The (1995)" "Outbreak (1995)" "Stargate (1994)" ...
# $ genres   : chr  "Comedy|Romance" "Action|Crime|Thriller" "Action|Drama|Sci-Fi|Thriller" "Action|Adventure|Sci-Fi" ...


# we examine user preferences

sample_index <- sample(1:nrow(edx), 1000)
edx_figure1 <- edx[sample_index,] |>
  ggplot(aes(movieId, userId)) +
  geom_point(aes(color = rating)) +
  xlab("Movie Identification Number") +
  ylab("User Identification Number") +
  scale_color_gradient(name = "Rating", low = "skyblue", high = "darkblue") + # reverses the color gradient
  labs(caption = "Figure 1. Plot of user ratings for movies of a random sample of the dataset.") +
  theme(plot.caption = element_text(color = "darkblue", size = 11, hjust = 0)) # moves the caption to the left
edx_figure1
# users rate certain movies more than others

edx_figure2 <- edx[sample_index,] |>
  ggplot(aes(genres, userId)) +
  geom_point(aes(color = rating)) +
  theme(axis.text.x = element_text(size = 3, hjust = 1, vjust = 0.5, angle = 90)) +
  xlab("Genre") +
  ylab("User Identification Number") +
  scale_color_gradient(name = "Rating", low = "skyblue", high = "darkblue") +
  labs(caption = "Figure 2. Plot of user ratings for genres of a random sample of the dataset.") +
  theme(plot.caption = element_text(color = "darkblue", size = 11, hjust = 0))
edx_figure2
# users rate certain genres more than others

edx_figure3 <- edx[sample_index,] |>
  mutate(year_rel = str_extract(title, "\\(\\d{4}\\)$")) |> # simplifies extraction of the year of release because of the presence of multiple () in the titles
  mutate(year_rel = str_replace_all(year_rel, "[:punct:]", "")) |>
  mutate(year_rel = as.integer(year_rel)) |>
  ggplot(aes(year_rel, userId)) +
  geom_point(aes(color = rating)) +
  xlab("Year of Release") +
  ylab("User Identification Number") +
  scale_color_gradient(name = "Rating", low = "skyblue", high = "darkblue") +
  labs(caption = "Figure 3. Plot of user ratings per year of release of a random sample of the dataset.") +
  theme(plot.caption = element_text(color = "darkblue", size = 11, hjust = 0))
edx_figure3
# users rate movies from certain years of release more than others

edx_figure4 <- edx |>
  ggplot(aes(userId)) +
  geom_histogram(color = "darkblue", bins = 69878) +
  xlab("User Identification Number") +
  ylab("Number of Ratings") +
  labs(caption = "Figure 4. Histogram of user ratings of a large sample of the dataset.") +
  theme(plot.caption = element_text(color = "darkblue", size = 11, hjust = 0))
edx_figure4
# certain users rate more than others

if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
library(lubridate)
edx_figure5 <- edx[sample_index,] |>
  mutate(time = as_datetime(timestamp)) |>
  mutate(hour = hour(time)) |>
  ggplot(aes(hour, userId)) +
  geom_point(aes(color = rating)) +
  xlab("Hour") +
  ylab("User Identification Number") +
  scale_color_gradient(name = "Rating", low = "skyblue", high = "darkblue") +
  labs(caption = "Figure 5. Plot of user ratings per hour of the day of a random sample of the dataset.") +
  theme(plot.caption = element_text(color = "darkblue", size = 11, hjust = 0))
edx_figure5

edx_figure6 <- edx[sample_index,] |>
  mutate(time = as_datetime(timestamp)) |>
  mutate(day_w = wday(time, label = TRUE, abbr = FALSE)) |>
  ggplot(aes(day_w, userId)) +
  geom_point(aes(color = rating)) +
  xlab("Day of the Week") +
  ylab("User Identification Number") +
  scale_color_gradient(name = "Rating", low = "skyblue", high = "darkblue") +
  labs(caption = "Figure 6. Plot of user ratings per day of the week of a random sample of the dataset.") +
  theme(plot.caption = element_text(color = "darkblue", size = 11, hjust = 0))
edx_figure6

edx_figure7 <- edx[sample_index,] |>
  mutate(time = as_datetime(timestamp)) |>
  mutate(day_m = day(time)) |>
  ggplot(aes(day_m, userId)) +
  geom_point(aes(color = rating)) +
  xlab("Day of the Month") +
  ylab("User Identification Number") +
  scale_color_gradient(name = "Rating", low = "skyblue", high = "darkblue") +
  labs(caption = "Figure 7. Plot of user ratings per day of the month of a random sample of the dataset.") +
  theme(plot.caption = element_text(color = "darkblue", size = 11, hjust = 0))
edx_figure7

edx_figure8 <- edx[sample_index,] |>
  mutate(time = as_datetime(timestamp)) |>
  mutate(month = month(time, label = TRUE, abbr = FALSE)) |>
  ggplot(aes(month, userId)) +
  geom_point(aes(color = rating)) +
  theme(axis.text.x = element_text(hjust = 1, vjust = 0.5, angle = 90)) +
  xlab("Month") +
  ylab("User Identification Number") +
  scale_color_gradient(name = "Rating", low = "skyblue", high = "darkblue") +
  labs(caption = "Figure 8. Plot of user ratings per month of the year of a random sample of the dataset.") +
  theme(plot.caption = element_text(color = "darkblue", size = 11, hjust = 0))
edx_figure8

edx_figure9 <- edx[sample_index,] |>
  mutate(year_rel = str_extract(title, "\\(\\d{4}\\)$")) |>
  mutate(year_rel = str_replace_all(year_rel, "[:punct:]", "")) |>
  mutate(year_rel = as.integer(year_rel)) |>
  mutate(time = as_datetime(timestamp)) |>
  mutate(year_rat = year(time)) |>
  mutate(year_s = year_rat - year_rel) |>
  ggplot(aes(year_s, userId)) +
  geom_point(aes(color = rating)) +
  xlab("Year Since Release") +
  ylab("User Identification Number") +
  scale_color_gradient(name = "Rating", low = "skyblue", high = "darkblue") +
  labs(caption = "Figure 9. Plot of user ratings per year since release of a random sample of the dataset.") +
  theme(plot.caption = element_text(color = "darkblue", size = 11, hjust = 0))
edx_figure9
# users rate movies within certain years since release more than others


# we separate the edx set into a train set and a test set

set.seed(10, sample.kind = "Rounding") # if using R 3.6 or later # for reproducibility during peer assessment
# set.seed(10) # if using R 3.5 or earlier
test_index <- createDataPartition(edx$rating, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temporary_set <- edx[test_index,]
test_set <- temporary_set |>
  semi_join(train_set, by = "movieId") |>
  semi_join(train_set, by = "userId")
train_set <- rbind(train_set, anti_join(temporary_set, test_set))


# we define a function of the rmse by which we will measure the discrepancy between the actual ratings in the test set and the ratings predicted by each trained algorithm

rmse <- function(actual_rating, predicted_rating){
  sqrt(mean((actual_rating - predicted_rating)^2))
}


# NOTE: we can use train() to conveniently train algorithms with if we are not constrained by time and/or computer capability


# we define and test the baseline algorithm: average rating mu
# this algorithm simply predicts that the average rating mu in the train set will be the rating of users for movies in the test set as a basis of comparison for the succeeding algorithms

options(digits = 5)
mu <- mean(train_set$rating)
mu
# [1] 3.5124
average_rmse <- rmse(test_set$rating, mu)
average_rmse
# [1] 1.0593

# we tabulate the rmses
rmse_tibble <- tibble(Algorithm = "Baseline: Average Rating", RMSE = average_rmse)


# we train and test algorithm 1: average rating mu + movie bias bi
# this algorithm predicts that the average rating mu plus a movie bias bi derived from the average rating per movie will be the rating of users for a particular movie
# this is based on our previous observation that users rate certain movies more than others

bi_tibble <- train_set |>
  group_by(movieId) |>
  summarize(bi = mean(rating - mu))
# A tibble: 10,677 × 2
#   movieId         bi
#     <int>        <dbl>
# 1       1       0.419
# 2       2      -0.309
# ...
algorithm1_rating <- test_set |>
  left_join(bi_tibble, by = "movieId") |> # adds the bi column
  mutate(algorithm1_rating = mu + bi) |>
  pull(algorithm1_rating)
algorithm1_rmse <- rmse(test_set$rating, algorithm1_rating)
algorithm1_rmse
# [1] 0.94292 # lower than average_rmse
rmse_tibble <- rbind(rmse_tibble, tibble(Algorithm = "1: Average Rating + Movie Bias", RMSE = algorithm1_rmse))
# we add genre as another possible predictor


# we train and test algorithm 2: average rating mu + movie bias bi + genre bias bg
# this algorithm predicts that the average rating mu plus the movie bias bi plus the genre bias bg derived from the average rating per genre will be the rating of users for a particular movie
# this is based on our previous observation that users rate certain genres more than others

bg_tibble <- train_set |>
  left_join(bi_tibble, by = "movieId") |>
  group_by(genres) |>
  summarize(bg = mean(rating - mu - bi))
# A tibble: 797 × 2
# genres                                                    bg
# <chr>                                                  <dbl>
# 1 (no genres listed)                                  0
# 2 Action                                             -8.97e-17
# 3 Action|Adventure                                   -1.91e-15
# ...
algorithm2_rating <- test_set |>
  left_join(bi_tibble, by = "movieId") |>
  left_join(bg_tibble, by = "genres") |>
  mutate(algorithm2_rating = mu + bi + bg) |>
  pull(algorithm2_rating)
algorithm2_rmse <- rmse(test_set$rating, algorithm2_rating)
algorithm2_rmse
# [1] 0.94292 # same as the algorithm1_rmse

# we retrain and retest using only the highest-rated genre of each genre combination
genre_vector <- c("Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",  "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western") # from https://files.grouplens.org/datasets/movielens/ml-10m-README.html
genre_rating <- sapply(genre_vector, function(g){
  train_set_a <- train_set |> filter(str_detect(genres, g) == TRUE)
  mean(train_set_a$rating)
})
genre_tibble <- tibble(genre = genre_vector, rating = genre_rating) |> arrange(-rating) # arranges the genres according to decreasing rating
# A tibble: 18 × 2
#   genre       rating
#   <chr>        <dbl>
# 1 Film-Noir     4.01
# 2 Documentary   3.78
# 3 War           3.78
# 4 Mystery       3.68
# 5 Drama         3.67
# 6 Crime         3.67
# 7 Animation     3.60
# 8 Musical       3.56
# 9 Western       3.55
# 10 Romance       3.55
# 11 Thriller      3.51
# 12 Fantasy       3.50
# 13 Adventure     3.49
# 14 Comedy        3.44
# 15 Action        3.42
# 16 Children      3.42
# 17 Sci-Fi        3.40
# 18 Horror        3.27
bg_tibble <- train_set |>
  left_join(bi_tibble, by = "movieId") |>
  mutate(h_genre = case_when(str_detect(genres, "Film-Noir") ~ "Film-Noir",
                             str_detect(genres, "Documentary") ~ "Documentary",
                             str_detect(genres, "War") ~ "War",
                             str_detect(genres, "Mystery") ~ "Mystery",
                             str_detect(genres, "Drama") ~ "Drama",
                             str_detect(genres, "Crime") ~ "Crime",
                             str_detect(genres, "Animation") ~ "Animation",
                             str_detect(genres, "Musical") ~ "Musical",
                             str_detect(genres, "Western") ~ "Western",
                             str_detect(genres, "Romance") ~ "Romance",
                             str_detect(genres, "Thriller") ~ "Thriller",
                             str_detect(genres, "Fantasy") ~ "Fantasy",
                             str_detect(genres, "Adventure") ~ "Adventure",
                             str_detect(genres, "Comedy") ~ "Comedy",
                             str_detect(genres, "Action") ~ "Action",
                             str_detect(genres, "Children") ~ "Children",
                             str_detect(genres, "Sci-Fi") ~ "Sci-Fi",
                             str_detect(genres, "Horror") ~ "Horror",
                             str_detect(genres, "IMAX") ~ "(no genres listed)", # categorizes  movies with only IMAX as genre as (no genres listed)
                             str_detect(genres, "(no genres listed)") ~ "(no genres listed)")) |> # extracts the highest-rated genre of each genre combination
  group_by(h_genre) |>
  summarize(bg = mean(rating - mu - bi))
# A tibble: 19 × 2
#   h_genre                   bg
#   <chr>                  <dbl>
# 1 (no genres listed)  5.84e-17
# 2 Action              2.34e-16
# ...
algorithm2_rating <- test_set |>
  left_join(bi_tibble, by = "movieId") |>
  mutate(h_genre = case_when(str_detect(genres, "Film-Noir") ~ "Film-Noir",
                             str_detect(genres, "Documentary") ~ "Documentary",
                             str_detect(genres, "War") ~ "War",
                             str_detect(genres, "Mystery") ~ "Mystery",
                             str_detect(genres, "Drama") ~ "Drama",
                             str_detect(genres, "Crime") ~ "Crime",
                             str_detect(genres, "Animation") ~ "Animation",
                             str_detect(genres, "Musical") ~ "Musical",
                             str_detect(genres, "Western") ~ "Western",
                             str_detect(genres, "Romance") ~ "Romance",
                             str_detect(genres, "Thriller") ~ "Thriller",
                             str_detect(genres, "Fantasy") ~ "Fantasy",
                             str_detect(genres, "Adventure") ~ "Adventure",
                             str_detect(genres, "Comedy") ~ "Comedy",
                             str_detect(genres, "Action") ~ "Action",
                             str_detect(genres, "Children") ~ "Children",
                             str_detect(genres, "Sci-Fi") ~ "Sci-Fi",
                             str_detect(genres, "Horror") ~ "Horror",
                             str_detect(genres, "IMAX") ~ "(no genres listed)",
                             str_detect(genres, "(no genres listed)") ~ "(no genres listed)")) |>
  left_join(bg_tibble, by = "h_genre") |>
  mutate(algorithm2_rating = mu + bi + bg) |>
  pull(algorithm2_rating)
algorithm2_rmse <- rmse(test_set$rating, algorithm2_rating)
algorithm2_rmse
# [1] 0.94292 # still the same as the algorithm1_rmse
rmse_tibble <- rbind(rmse_tibble, tibble(Algorithm = "2: Average Rating + Movie Bias + Genre Bias", RMSE = algorithm2_rmse))
# we replace genre with year of release as another possible predictor


# we train and test algorithm 3: average rating mu + movie bias bi + year-of-release bias byr
# this algorithm predicts that the average rating mu plus the movie bias bi plus the year-of-release bias byr derived from the average rating per year of release will be the rating of users for a particular movie
# this is based on our previous observation that users rate movies from certain years of release more than others

byr_tibble <- train_set |>
  left_join(bi_tibble, by = "movieId") |>
  mutate(year_rel = str_extract(title, "\\(\\d{4}\\)$")) |>
  mutate(year_rel = str_replace_all(year_rel, "[:punct:]", "")) |>
  mutate(year_rel = as.integer(year_rel)) |>
  group_by(year_rel) |>
  summarize(byr = mean(rating - mu - bi))
# A tibble: 94 × 2
#     year_rel     byr
#     <int>     <dbl>
# 1     1915 -5.71e-17
# 2     1916 -5.48e-17
# ...
algorithm3_rating <- test_set |>
  left_join(bi_tibble, by = "movieId") |>
  mutate(year_rel = str_extract(title, "\\(\\d{4}\\)$")) |>
  mutate(year_rel = str_replace_all(year_rel, "[:punct:]", "")) |>
  mutate(year_rel = as.integer(year_rel)) |>
  left_join(byr_tibble, by = "year_rel") |>
  mutate(algorithm3_rating = mu + bi + byr) |>
  pull(algorithm3_rating)
algorithm3_rmse <- rmse(test_set$rating, algorithm3_rating)
algorithm3_rmse
# [1] 0.94292 # same as the algorithm1_rmse
rmse_tibble <- rbind(rmse_tibble, tibble(Algorithm = "3: Average Rating + Movie Bias + Year-of-Release Bias", RMSE = algorithm3_rmse))
# we replace year of release with user as another possible predictor


# we train and test algorithm 4: average rating mu + movie bias bi + user bias bu
# this algorithm predicts that the average rating mu plus a movie bias bi plus a user bias bu derived from the average rating per user will be the rating of a particular user for a particular movie
# this is based on our previous observation that certain users rate more than others

bu_tibble <- train_set |>
  left_join(bi_tibble, by = "movieId") |>
  group_by(userId) |>
  summarize(bu = mean(rating - mu - bi))
# A tibble: 69,878 × 2
#   userId      bu
#   <int>       <dbl>
# 1      1     1.66
# 2      2    -0.130
# ...
algorithm4_rating <- test_set |>
  left_join(bi_tibble, by = "movieId") |>
  left_join(bu_tibble, by = "userId") |>
  mutate(algorithm4_rating = mu + bi + bu) |>
  pull(algorithm4_rating)
algorithm4_rmse <- rmse(test_set$rating, algorithm4_rating)
algorithm4_rmse
# [1] 0.86458 # lower than the algorithm1_rmse and the required rmse
rmse_tibble <- rbind(rmse_tibble, tibble(Algorithm = "Required RMSE", RMSE = 0.86490))
rmse_tibble <- rbind(rmse_tibble, tibble(Algorithm = "4: Average Rating + Movie Bias + User Bias", RMSE = algorithm4_rmse))
# we add year since release as another possible predictor


# we train and test algorithm 5: average rating mu + movie bias bi + user bias bu + year-since-release bias bys
# this algorithm predicts that the average rating mu plus a movie bias bi plus a user bias bu plus a year-since-release bias bys derived from the average rating per year since release will be the rating of a particular user for a particular movie
# this is based on our previous observation that users rate movies within certain years since release more than others

bys_tibble <- train_set |>
  left_join(bi_tibble, by = "movieId") |>
  left_join(bu_tibble, by = "userId") |>
  mutate(year_rel = str_extract(title, "\\(\\d{4}\\)$")) |>  mutate(year_rel = str_replace_all(year_rel, "[:punct:]", "")) |>
  mutate(year_rel = as.integer(year_rel)) |>
  mutate(time = as_datetime(timestamp)) |>
  mutate(year_rat = year(time)) |>
  mutate(year_s = year_rat - year_rel) |>
  group_by(year_s) |>
  summarize(bys = mean(rating - mu - bi - bu))
# A tibble: 96 × 2
#   year_s     bys
#     <dbl>   <dbl>
# 1     -2  0.0228
# 2     -1  0.147
# ...
algorithm5_rating <- test_set |>
  left_join(bi_tibble, by = "movieId") |>
  left_join(bu_tibble, by = "userId") |>
  mutate(year_rel = str_extract(title, "\\(\\d{4}\\)$")) |>
  mutate(year_rel = str_replace_all(year_rel, "[:punct:]", "")) |>
  mutate(year_rel = as.integer(year_rel)) |>
  mutate(time = as_datetime(timestamp)) |>
  mutate(year_rat = year(time)) |>
  mutate(year_s = year_rat - year_rel) |>
  left_join(bys_tibble, by = "year_s") |>
  mutate(algorithm5_rating = mu + bi + bu + bys) |>
  pull(algorithm5_rating)
algorithm5_rmse <- rmse(test_set$rating, algorithm5_rating)
algorithm5_rmse
# [1] 0.86414 # lower than algorithm4_rmse and the required rmse
rmse_tibble <- rbind(rmse_tibble, tibble(Algorithm = "5: Average Rating + Movie Bias + User Bias + Year-Since-Release Bias", RMSE = algorithm5_rmse))


# we evaluate algorithm 5 based on the the "best" movies predicted by the movie bias bi

title_tibble <- edx |>
  as_tibble() |>
  select(movieId, title) |>
  distinct()
# A tibble: 10,677 × 2
# movieId title
# <int> <chr>
# 1     122 Boomerang (1992)
# 2     185 Net, The (1995)
# ...
train_set |>
  count(movieId) |>
  left_join(title_tibble, by = "movieId") |>
  left_join(bi_tibble, by = "movieId") |>
  select(title, bi, n) |>
  arrange(desc(bi)) |>
  head(n = 10)
#                                                                               title       bi n
# 1                                                     Hellhounds on My Trail (1999) 1.487613 1
# 2                                                 Satan's Tango (Sátántangó) (1994) 1.487613 2
# 3                                             Shadows of Forgotten Ancestors (1964) 1.487613 1
# 4                                              Fighting Elegy (Kenka erejii) (1966) 1.487613 1
# 5                                                    Sun Alley (Sonnenallee) (1999) 1.487613 1
# 6                                                      Maradona by Kusturica (2008) 1.487613 1
# 7                                          Blue Light, The (Das Blaue Licht) (1932) 1.487613 1
# 8                               Human Condition II, The (Ningen no joken II) (1959) 1.320946 3
# 9  Who's Singin' Over There? (a.k.a. Who Sings Over There) (Ko to tamo peva) (1980) 1.237613 4
# 10                            Human Condition III, The (Ningen no joken III) (1961) 1.237613 4
# the "best" movies are unheard of and come with very low numbers of ratings
# we adjust the movie bias bi and other predictors in algorithm 5 for the number of ratings using regularization


###############
# ALGORITHM 6 #
###############
# we train and test algorithm 6: average rating mu + regularized movie bias r_bi + regularized user bias r_bu + regularized year-since-release bias r_bys
# this algorithm predicts that the average rating mu plus the movie bias adjusted or regularized for the number of ratings per movie with a factor lambda r_bi plus the user bias also regularized for the number of ratings per user with a factor lambda r_bu plus the year-since-release bias also regularized for the number of ratings per year since release with a factor lambda r_bys will be the rating of users for a particular movie
# it improves on algorithm 5

# we optimize the lambda such that it minimizes the rmse
lambda_vector <- seq(4.5, 6.5, 0.1)
lambda_rmse <- sapply(lambda_vector, function(l){
  r_bi_tibble <- train_set |>
    group_by(movieId) |>
    summarize(r_bi = sum(rating - mu)/(l + n()))
  r_bu_tibble <- train_set |>
    left_join(r_bi_tibble, by = "movieId") |>
    group_by(userId) |>
    summarize(r_bu = sum(rating - mu - r_bi)/(l + n()))
  r_bys_tibble <- train_set |>
    left_join(r_bi_tibble, by = "movieId") |>
    left_join(r_bu_tibble, by = "userId") |>
    mutate(year_rel = str_extract(title, "\\(\\d{4}\\)$")) |>
    mutate(year_rel = str_replace_all(year_rel, "[:punct:]", "")) |>
    mutate(year_rel = as.integer(year_rel)) |>
    mutate(time = as_datetime(timestamp)) |>
    mutate(year_rat = year(time)) |>
    mutate(year_s = year_rat - year_rel) |>
    group_by(year_s) |>
    summarize(r_bys = sum(rating - mu - r_bi - r_bu)/(l + n()))
  algorithm6_rating <- test_set |>
    left_join(r_bi_tibble, by = "movieId") |>
    left_join(r_bu_tibble, by = "userId") |>
    mutate(year_rel = str_extract(title, "\\(\\d{4}\\)$")) |>
    mutate(year_rel = str_replace_all(year_rel, "[:punct:]", "")) |>
    mutate(year_rel = as.integer(year_rel)) |>
    mutate(time = as_datetime(timestamp)) |>
    mutate(year_rat = year(time)) |>
    mutate(year_s = year_rat - year_rel) |>
    left_join(r_bys_tibble, by = "year_s") |>
    mutate(algorithm6_rating = mu + r_bi + r_bu + r_bys) |>
    pull(algorithm6_rating)
  return(rmse(test_set$rating, algorithm6_rating))
})
lambda <- lambda_vector[which.min(lambda_rmse)]
# [1] 5.3

# we use the final lambda
r_bi_tibble <- train_set |>
  group_by(movieId) |>
  summarize(r_bi = sum(rating - mu)/(lambda + n()))
r_bu_tibble <- train_set |>
  left_join(r_bi_tibble, by = "movieId") |>
  group_by(userId) |>
  summarize(r_bu = sum(rating - mu - r_bi)/(lambda + n()))
r_bys_tibble <- train_set |>
  left_join(r_bi_tibble, by = "movieId") |>
  left_join(r_bu_tibble, by = "userId") |>
  mutate(year_rel = str_extract(title, "\\(\\d{4}\\)$")) |>
  mutate(year_rel = str_replace_all(year_rel, "[:punct:]", "")) |>
  mutate(year_rel = as.integer(year_rel)) |>
  mutate(time = as_datetime(timestamp)) |>
  mutate(year_rat = year(time)) |>
  mutate(year_s = year_rat - year_rel) |>
  group_by(year_s) |>
  summarize(r_bys = sum(rating - mu - r_bi - r_bu)/(lambda + n()))
algorithm6_rating <- test_set |>
  left_join(r_bi_tibble, by = "movieId") |>
  left_join(r_bu_tibble, by = "userId") |>
  mutate(year_rel = str_extract(title, "\\(\\d{4}\\)$")) |>
  mutate(year_rel = str_replace_all(year_rel, "[:punct:]", "")) |>
  mutate(year_rel = as.integer(year_rel)) |>
  mutate(time = as_datetime(timestamp)) |>
  mutate(year_rat = year(time)) |>
  mutate(year_s = year_rat - year_rel) |>
  left_join(r_bys_tibble, by = "year_s") |>
  mutate(algorithm6_rating = mu + r_bi + r_bu + r_bys) |>
  pull(algorithm6_rating)
algorithm6_rmse <- rmse(test_set$rating, algorithm6_rating)
algorithm6_rmse
# [1] 0.86353 # lower than the algorithm5_rmse and the required rmse
rmse_tibble <- rbind(rmse_tibble, tibble(Algorithm = "6: Average Rating + Regularized Movie Bias + Regularized User Bias + Regularized Year-Since-Release Bias", RMSE = algorithm6_rmse))
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
library(kableExtra)
kbl(rmse_tibble) |> # Table 1. Root Mean Squared Errors (RMSEs) of the Algorithms
  kable_styling() |>
  row_spec(0, color = "darkblue") |>
  row_spec(5, color = "lightgray")
###############
# ALGORITHM 6 #
###############


# we evaluate algorithm 6 based on the the best movies predicted by the regularized movie bias r_bi

train_set |>
  count(movieId) |>
  left_join(title_tibble, by = "movieId") |>
  left_join(r_bi_tibble, by = "movieId") |>
  select(title, r_bi, n) |>
  arrange(desc(r_bi)) |>
  head(n = 10)
#                                           title      r_bi     n
# 1               Shawshank Redemption, The (1994) 0.9419646 25232
# 2                          Godfather, The (1972) 0.8998882 16017
# 3                        Schindler's List (1993) 0.8531383 20895
# 4                     Usual Suspects, The (1995) 0.8530202 19491
# 5                             Rear Window (1954) 0.8080552  7146
# 6                              Casablanca (1942) 0.8054104 10091
# 7  Sunset Blvd. (a.k.a. Sunset Boulevard) (1950) 0.8009191  2609
# 8                          Third Man, The (1949) 0.7988919  2671
# 9                        Double Indemnity (1944) 0.7969676  1928
# 10   Seven Samurai (Shichinin no samurai) (1954) 0.7968523  4658
# the best movies are well known and come with high numbers of ratings compared to those predicted by the movie bias bi of algorithm 5


# we evaluate the robustness of algorithm 6 on new data using bootstrapping

cv_algorithm6_rmses <- c()
for(i in 1:5){
  cv_test_index <- sample(1:nrow(edx), size = 1000000, replace = TRUE)
  cv_test_set <- edx[cv_test_index,]
  cv_algorithm6_rating <- cv_test_set |>
    left_join(r_bi_tibble, by = "movieId") |>
    left_join(r_bu_tibble, by = "userId") |>
    mutate(year_rel = str_extract(title, "\\(\\d{4}\\)$")) |>
    mutate(year_rel = str_replace_all(year_rel, "[:punct:]", "")) |>
    mutate(year_rel = as.integer(year_rel)) |>
    mutate(time = as_datetime(timestamp)) |>
    mutate(year_rat = year(time)) |>
    mutate(year_s = year_rat - year_rel) |>
    left_join(r_bys_tibble, by = "year_s") |>
    mutate(cv_algorithm6_rating = mu + r_bi + r_bu + r_bys) |>
    pull(cv_algorithm6_rating)
  cv_algorithm6_rmse <- rmse(cv_test_set$rating, cv_algorithm6_rating)
  cv_algorithm6_rmses <- c(cv_algorithm6_rmses, cv_algorithm6_rmse)
}
cv_algorithm6_rmses
mean(cv_algorithm6_rmses)
# around 0.857 # still lower than the the required rmse
# we use algorithm 6 as the final algorithm


# final_holdout_test Set


# we reserve the final_holdout_test set only to test the final algorithm


# we predict the ratings in the the final_holdout_test set using the final algorithm 6: average rating mu + regularized movie bias r_bi + regularized user bias r_bu + regularized year-since-release bias r_bys

predicted_rating <- final_holdout_test |>
  left_join(r_bi_tibble, by = "movieId") |>
  left_join(r_bu_tibble, by = "userId") |>
  mutate(year_rel = str_extract(title, "\\(\\d{4}\\)$")) |>
  mutate(year_rel = str_replace_all(year_rel, "[:punct:]", "")) |>
  mutate(year_rel = as.integer(year_rel)) |>
  mutate(time = as_datetime(timestamp)) |>
  mutate(year_rat = year(time)) |>
  mutate(year_s = year_rat - year_rel) |>
  left_join(r_bys_tibble, by = "year_s") |>
  mutate(predicted_rating = mu + r_bi + r_bu + r_bys) |>
  pull(predicted_rating)
final_holdout_test_rmse <- rmse(final_holdout_test$rating, predicted_rating)
final_holdout_test_rmse
# [1] 0.86469 # lower than the required rmse of 0.86490

