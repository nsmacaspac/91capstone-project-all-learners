# CONTENT-BASED RECOMMENDATION SYSTEM DEVELOPED FROM THE MOVIELENS 10M DATASET
# NICELLE SERNADILLA MACASPAC
# RUNNING TIME: 7 minutes



# Introduction



# we need to develop a modest recommendation system that is based on the content information from the baseline predictors in a similar dataset of the movie recommender MovieLens and that predicts future movie ratings of users with a root mean squared error (RMSE) rate of <0.86490



# MovieLens 10M Dataset



# we download, wrangle and partition the dataset into the edx set and the final_holdout_test set using a code provided by Harvard Online

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



# we use the edx set to examine user preferences and to train and test content-based algorithms to develop the recommendation system

head(edx, n = 5) # fig1 in the rmd file
str(edx)
# 'data.frame':	9000055 obs. of  6 variables:
# $ userId   : int  1 1 1 1 1 1 1 1 1 1 ...
# $ movieId  : int  122 185 292 316 329 355 356 362 364 370 ...
# $ rating   : num  5 5 5 5 5 5 5 5 5 5 ...
# $ timestamp: int  838985046 838983525 838983421 838983392 838983392 838984474 838983653 838984885 838983707 838984596 ...
# $ title    : chr  "Boomerang (1992)" "Net, The (1995)" "Outbreak (1995)" "Stargate (1994)" ...
# $ genres   : chr  "Comedy|Romance" "Action|Crime|Thriller" "Action|Drama|Sci-Fi|Thriller" "Action|Adventure|Sci-Fi" ...
n_distinct(edx$userId)
# [1] 69878
n_distinct(edx$movieId)
# [1] 10677
unique(edx$rating)
# [1] 5.0 3.0 2.0 4.0 4.5 3.5 1.0 1.5 2.5 0.5



# we reserve the final_holdout_test set to evaluate the recommendation system

str(final_holdout_test)
# 'data.frame':	999999 obs. of  6 variables:
# $ userId   : int  1 1 1 2 2 2 3 3 4 4 ...
# $ movieId  : int  231 480 586 151 858 1544 590 4995 34 432 ...
# $ rating   : num  5 5 5 3 2 3 3.5 4.5 5 3 ...
# $ timestamp: int  838983392 838983653 838984068 868246450 868245645 868245920 1136075494 1133571200 844416936 844417070 ...
# $ title    : chr  "Dumb & Dumber (1994)" "Jurassic Park (1993)" "Home Alone (1990)" "Rob Roy (1995)" ...
# $ genres   : chr  "Comedy" "Action|Adventure|Sci-Fi|Thriller" "Children|Comedy" "Action|Drama|Romance|War" ...



# User Preference



# we use the edx set to examine user preferences

set.seed(2, sample.kind = "Rounding") # if using R 3.6 or later # for reproducibility during peer assessment
# set.seed(2) # if using R 3.5 or earlier
sample_index <- sample(1:nrow(edx), 1000)
edx_figure2 <- edx[sample_index,] |>
  ggplot(aes(movieId, userId)) +
  geom_point(aes(color = rating)) +
  xlab("Movie Identification Number") +
  ylab("User Identification Number") +
  scale_color_gradient(name = "Rating", low = "skyblue", high = "darkblue") # reverses the color gradient
edx_figure2
# users tend to rate certain movies more than others

edx_figure3 <- edx[sample_index,] |>
  ggplot(aes(genres, userId)) +
  geom_point(aes(color = rating)) +
  theme(axis.text.x = element_text(size = 3, hjust = 1, vjust = 0.5, angle = 90)) +
  xlab("Genre") +
  ylab("User Identification Number") +
  scale_color_gradient(name = "Rating", low = "skyblue", high = "darkblue")
edx_figure3
# users prefer certain genres such as Comedy and Drama

edx_figure4 <- edx[sample_index,] |>
  mutate(year_rel = str_extract(title, "\\(\\d{4}\\)$")) |> # simplifies extraction of the year of release because of the presence of multiple () in the titles
  mutate(year_rel = str_replace_all(year_rel, "[:punct:]", "")) |>
  mutate(year_rel = as.integer(year_rel)) |>
  ggplot(aes(year_rel, userId)) +
  geom_point(aes(color = rating)) +
  xlab("Year of Release") +
  ylab("User Identification Number") +
  scale_color_gradient(name = "Rating", low = "skyblue", high = "darkblue")
edx_figure4
# users are inclined toward movies from the 1990s to 2000s

set.seed(6, sample.kind = "Rounding") # if using R 3.6 or later # for reproducibility during peer assessment
# set.seed(6) # if using R 3.5 or earlier
sample_index <- sample(1:nrow(edx), 25)
edx_figure5 <- edx[sample_index,] |>
  group_by(userId) |>
  summarize(ave_rating = mean(rating)) |>
  ggplot(aes(userId, ave_rating)) +
  geom_col(color = "darkblue") +
  xlab("User Identification Number") +
  ylab("Average Rating")
edx_figure5
# certain users tend to give a generous average rating of 5, whereas some users tend to give a stingy average rating of 1

if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
library(lubridate)
set.seed(7, sample.kind = "Rounding") # if using R 3.6 or later # for reproducibility during peer assessment
# set.seed(7) # if using R 3.5 or earlier
sample_index <- sample(1:nrow(edx), 1000)
edx_figure6 <- edx[sample_index,] |>
  mutate(time = as_datetime(timestamp)) |>
  mutate(hour = hour(time)) |>
  ggplot(aes(hour, userId)) +
  geom_point(aes(color = rating)) +
  xlab("Hour") +
  ylab("User Identification Number") +
  scale_color_gradient(name = "Rating", low = "skyblue", high = "darkblue")
edx_figure6

edx_figure7 <- edx[sample_index,] |>
  mutate(time = as_datetime(timestamp)) |>
  mutate(day_w = wday(time, label = TRUE, abbr = FALSE)) |>
  ggplot(aes(day_w, userId)) +
  geom_point(aes(color = rating)) +
  xlab("Day of the Week") +
  ylab("User Identification Number") +
  scale_color_gradient(name = "Rating", low = "skyblue", high = "darkblue")
edx_figure7

edx_figure8 <- edx[sample_index,] |>
  mutate(time = as_datetime(timestamp)) |>
  mutate(day_m = day(time)) |>
  ggplot(aes(day_m, userId)) +
  geom_point(aes(color = rating)) +
  xlab("Day of the Month") +
  ylab("User Identification Number") +
  scale_color_gradient(name = "Rating", low = "skyblue", high = "darkblue")
edx_figure8

edx_figure9 <- edx[sample_index,] |>
  mutate(time = as_datetime(timestamp)) |>
  mutate(month = month(time, label = TRUE, abbr = FALSE)) |>
  ggplot(aes(month, userId)) +
  geom_point(aes(color = rating)) +
  theme(axis.text.x = element_text(hjust = 1, vjust = 0.5, angle = 90)) +
  xlab("Month") +
  ylab("User Identification Number") +
  scale_color_gradient(name = "Rating", low = "skyblue", high = "darkblue")
edx_figure9

edx_figure10 <- edx[sample_index,] |>
  mutate(year_rel = str_extract(title, "\\(\\d{4}\\)$")) |>
  mutate(year_rel = str_replace_all(year_rel, "[:punct:]", "")) |>
  mutate(year_rel = as.integer(year_rel)) |>
  mutate(time = as_datetime(timestamp)) |>
  mutate(year_rat = year(time)) |>
  mutate(rel_age = year_rat - year_rel) |>
  ggplot(aes(rel_age, userId)) +
  geom_point(aes(color = rating)) +
  xlab("Relative Age of the Movie") +
  ylab("User Identification Number") +
  scale_color_gradient(name = "Rating", low = "skyblue", high = "darkblue")
edx_figure10
# users prefer to rate movies within 10 years of their release



# Content-Based Algorithm



# we use the edx set to train and test content-based algorithms

options(digits = 5)
set.seed(10, sample.kind = "Rounding") # if using R 3.6 or later # for reproducibility during peer assessment
# set.seed(10) # if using R 3.5 or earlier
test_index <- createDataPartition(edx$rating, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temporary_set <- edx[test_index,]
test_set <- temporary_set |>
  semi_join(train_set, by = "movieId") |>
  semi_join(train_set, by = "userId")
train_set <- rbind(train_set, anti_join(temporary_set, test_set))



# we define a function that calculates the rmse

rmse <- function(actual_rating, predicted_rating){
  sqrt(mean((actual_rating - predicted_rating)^2))
}



# NOTE: we use the penalized least squares equation and not train() to train algorithms with as we are constrained by time and/or computer capability



# we define and test the baseline algorithm: average rating mu

mu <- mean(train_set$rating)
mu
# [1] 3.5124
baseline_rmse <- rmse(test_set$rating, mu)
baseline_rmse
# [1] 1.0593

# we tabulate the rmses
rmse_tibble <- tibble(Algorithm = "Baseline: Average Rating", RMSE = baseline_rmse)



# we train and test algorithm 1: average rating mu + movie bias bi

bi_tibble <- train_set |>
  group_by(movieId) |>
  summarize(bi = mean(rating - mu))
head(bi_tibble, n = 5)
# # A tibble: 5 × 2
#   movieId     bi
#     <int>  <dbl>
# 1       1  0.419
# 2       2 -0.309
# 3       3 -0.366
# 4       4 -0.645
# 5       5 -0.442
algorithm1_rating <- test_set |>
  left_join(bi_tibble, by = "movieId") |> # adds the bi column
  mutate(algorithm1_rating = mu + bi) |>
  pull(algorithm1_rating)
algorithm1_rmse <- rmse(test_set$rating, algorithm1_rating)
algorithm1_rmse
# [1] 0.94292 # lower than baseline_rmse



# we train and test algorithm 2: average rating mu + movie bias bi + genre bias bg

bg_tibble <- train_set |>
  left_join(bi_tibble, by = "movieId") |>
  group_by(genres) |>
  summarize(bg = mean(rating - mu - bi))
head(bg_tibble, n = 5)
# # A tibble: 5 × 2
#   genres                                                    bg
#   <chr>                                                  <dbl>
# 1 (no genres listed)                                  0
# 2 Action                                             -8.97e-17
# 3 Action|Adventure                                   -1.91e-15
# 4 Action|Adventure|Animation|Children|Comedy         -4.81e-16
# 5 Action|Adventure|Animation|Children|Comedy|Fantasy  4.47e-17
algorithm2_rating <- test_set |>
  left_join(bi_tibble, by = "movieId") |>
  left_join(bg_tibble, by = "genres") |>
  mutate(algorithm2_rating = mu + bi + bg) |>
  pull(algorithm2_rating)
algorithm2_rmse <- rmse(test_set$rating, algorithm2_rating)
algorithm2_rmse
# [1] 0.94292 # same as the algorithm1_rmse

# we retrain and retest using only the top-rated genre of each genre combination
genre_vector <- c("Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",  "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western") # from https://files.grouplens.org/datasets/movielens/ml-10m-README.html
genre_rating <- sapply(genre_vector, function(g){
  train_set_a <- train_set |> filter(str_detect(genres, g) == TRUE)
  mean(train_set_a$rating)
})
genre_tibble <- tibble(genre = genre_vector, rating = genre_rating) |> arrange(-rating) # arranges the genres according to decreasing rating
genre_tibble
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
  mutate(t_genre = case_when(str_detect(genres, "Film-Noir") ~ "Film-Noir",
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
                             str_detect(genres, "Horror") ~ "Horror")) |>
  group_by(t_genre) |>
  summarize(bg = mean(rating - mu - bi))
head(bg_tibble, n = 5)
# # A tibble: 5 × 2
#   t_genre          bg
#   <chr>         <dbl>
# 1 Action     2.34e-16
# 2 Adventure  2.60e-15
# 3 Animation  3.45e-15
# 4 Children  -1.71e-16
# 5 Comedy     1.10e-15
algorithm2_rating <- test_set |>
  left_join(bi_tibble, by = "movieId") |>
  mutate(t_genre = case_when(str_detect(genres, "Film-Noir") ~ "Film-Noir",
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
                             str_detect(genres, "Horror") ~ "Horror")) |>
  left_join(bg_tibble, by = "t_genre") |>
  mutate(algorithm2_rating = mu + bi + bg) |>
  pull(algorithm2_rating)
algorithm2_rmse <- rmse(test_set$rating, algorithm2_rating)
algorithm2_rmse
# [1] 0.94292 # still the same as the algorithm1_rmse
rmse_tibble <- rbind(rmse_tibble, tibble(Algorithm = "2: Average Rating + Movie Bias + Genre Bias", RMSE = algorithm2_rmse))
# we replace genre with year of release as another possible predictor



# we train and test algorithm 3: average rating mu + movie bias bi + release bias br

br_tibble <- train_set |>
  left_join(bi_tibble, by = "movieId") |>
  mutate(year_rel = str_extract(title, "\\(\\d{4}\\)$")) |>
  mutate(year_rel = str_replace_all(year_rel, "[:punct:]", "")) |>
  mutate(year_rel = as.integer(year_rel)) |>
  group_by(year_rel) |>
  summarize(br = mean(rating - mu - bi))
head(br_tibble, n = 5)
# # A tibble: 5 × 2
#   year_rel        br
#       <int>     <dbl>
# 1     1915 -5.71e-17
# 2     1916 -5.48e-17
# 3     1917 -1.53e-17
# 4     1918 -3.22e-18
# 5     1919  2.25e-17
algorithm3_rating <- test_set |>
  left_join(bi_tibble, by = "movieId") |>
  mutate(year_rel = str_extract(title, "\\(\\d{4}\\)$")) |>
  mutate(year_rel = str_replace_all(year_rel, "[:punct:]", "")) |>
  mutate(year_rel = as.integer(year_rel)) |>
  left_join(br_tibble, by = "year_rel") |>
  mutate(algorithm3_rating = mu + bi + br) |>
  pull(algorithm3_rating)
algorithm3_rmse <- rmse(test_set$rating, algorithm3_rating)
algorithm3_rmse
# [1] 0.94292 # same as the algorithm1_rmse
rmse_tibble <- rbind(rmse_tibble, tibble(Algorithm = "3: Average Rating + Movie Bias + Release Bias", RMSE = algorithm3_rmse))
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
# we add relative age of the movie as another possible predictor


# we train and test algorithm 5: average rating mu + movie bias bi + user bias bu + age bias ba
# this algorithm predicts that the average rating mu plus a movie bias bi plus a user bias bu plus a age bias ba derived from the average rating per relative age of the movie will be the rating of a particular user for a particular movie
# this is based on our previous observation that users rate movies within certain relative ages more than others

ba_tibble <- train_set |>
  left_join(bi_tibble, by = "movieId") |>
  left_join(bu_tibble, by = "userId") |>
  mutate(year_rel = str_extract(title, "\\(\\d{4}\\)$")) |>  mutate(year_rel = str_replace_all(year_rel, "[:punct:]", "")) |>
  mutate(year_rel = as.integer(year_rel)) |>
  mutate(time = as_datetime(timestamp)) |>
  mutate(year_rat = year(time)) |>
  mutate(rel_age = year_rat - year_rel) |>
  group_by(rel_age) |>
  summarize(ba = mean(rating - mu - bi - bu))
# A tibble: 96 × 2
#   rel_age     ba
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
  mutate(rel_age = year_rat - year_rel) |>
  left_join(ba_tibble, by = "rel_age") |>
  mutate(algorithm5_rating = mu + bi + bu + ba) |>
  pull(algorithm5_rating)
algorithm5_rmse <- rmse(test_set$rating, algorithm5_rating)
algorithm5_rmse
# [1] 0.86414 # lower than algorithm4_rmse and the required rmse
rmse_tibble <- rbind(rmse_tibble, tibble(Algorithm = "5: Average Rating + Movie Bias + User Bias + Age Bias", RMSE = algorithm5_rmse))


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
# we train and test algorithm 6: average rating mu + regularized movie bias r_bi + regularized user bias r_bu + regularized age bias r_ba
# this algorithm predicts that the average rating mu plus the movie bias adjusted or regularized for the number of ratings per movie with a factor lambda r_bi plus the user bias also regularized for the number of ratings per user with a factor lambda r_bu plus the age bias also regularized for the number of ratings per relative age of the movie with a factor lambda r_ba will be the rating of users for a particular movie
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
  r_ba_tibble <- train_set |>
    left_join(r_bi_tibble, by = "movieId") |>
    left_join(r_bu_tibble, by = "userId") |>
    mutate(year_rel = str_extract(title, "\\(\\d{4}\\)$")) |>
    mutate(year_rel = str_replace_all(year_rel, "[:punct:]", "")) |>
    mutate(year_rel = as.integer(year_rel)) |>
    mutate(time = as_datetime(timestamp)) |>
    mutate(year_rat = year(time)) |>
    mutate(rel_age = year_rat - year_rel) |>
    group_by(rel_age) |>
    summarize(r_ba = sum(rating - mu - r_bi - r_bu)/(l + n()))
  algorithm6_rating <- test_set |>
    left_join(r_bi_tibble, by = "movieId") |>
    left_join(r_bu_tibble, by = "userId") |>
    mutate(year_rel = str_extract(title, "\\(\\d{4}\\)$")) |>
    mutate(year_rel = str_replace_all(year_rel, "[:punct:]", "")) |>
    mutate(year_rel = as.integer(year_rel)) |>
    mutate(time = as_datetime(timestamp)) |>
    mutate(year_rat = year(time)) |>
    mutate(rel_age = year_rat - year_rel) |>
    left_join(r_ba_tibble, by = "rel_age") |>
    mutate(algorithm6_rating = mu + r_bi + r_bu + r_ba) |>
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
r_ba_tibble <- train_set |>
  left_join(r_bi_tibble, by = "movieId") |>
  left_join(r_bu_tibble, by = "userId") |>
  mutate(year_rel = str_extract(title, "\\(\\d{4}\\)$")) |>
  mutate(year_rel = str_replace_all(year_rel, "[:punct:]", "")) |>
  mutate(year_rel = as.integer(year_rel)) |>
  mutate(time = as_datetime(timestamp)) |>
  mutate(year_rat = year(time)) |>
  mutate(rel_age = year_rat - year_rel) |>
  group_by(rel_age) |>
  summarize(r_ba = sum(rating - mu - r_bi - r_bu)/(lambda + n()))
algorithm6_rating <- test_set |>
  left_join(r_bi_tibble, by = "movieId") |>
  left_join(r_bu_tibble, by = "userId") |>
  mutate(year_rel = str_extract(title, "\\(\\d{4}\\)$")) |>
  mutate(year_rel = str_replace_all(year_rel, "[:punct:]", "")) |>
  mutate(year_rel = as.integer(year_rel)) |>
  mutate(time = as_datetime(timestamp)) |>
  mutate(year_rat = year(time)) |>
  mutate(rel_age = year_rat - year_rel) |>
  left_join(r_ba_tibble, by = "rel_age") |>
  mutate(algorithm6_rating = mu + r_bi + r_bu + r_ba) |>
  pull(algorithm6_rating)
algorithm6_rmse <- rmse(test_set$rating, algorithm6_rating)
algorithm6_rmse
# [1] 0.86353 # lower than the algorithm5_rmse and the required rmse
rmse_tibble <- rbind(rmse_tibble, tibble(Algorithm = "6: Average Rating + Regularized Movie Bias + Regularized User Bias + Regularized Age Bias", RMSE = algorithm6_rmse))
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
    mutate(rel_age = year_rat - year_rel) |>
    left_join(r_ba_tibble, by = "rel_age") |>
    mutate(cv_algorithm6_rating = mu + r_bi + r_bu + r_ba) |>
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


# we predict the ratings in the the final_holdout_test set using the final algorithm 6: average rating mu + regularized movie bias r_bi + regularized user bias r_bu + regularized age bias r_ba

holdout_rating <- final_holdout_test |>
  left_join(r_bi_tibble, by = "movieId") |>
  left_join(r_bu_tibble, by = "userId") |>
  mutate(year_rel = str_extract(title, "\\(\\d{4}\\)$")) |>
  mutate(year_rel = str_replace_all(year_rel, "[:punct:]", "")) |>
  mutate(year_rel = as.integer(year_rel)) |>
  mutate(time = as_datetime(timestamp)) |>
  mutate(year_rat = year(time)) |>
  mutate(rel_age = year_rat - year_rel) |>
  left_join(r_ba_tibble, by = "rel_age") |>
  mutate(holdout_rating = mu + r_bi + r_bu + r_ba) |>
  pull(holdout_rating)
holdout_rmse <- rmse(final_holdout_test$rating, holdout_rating)
holdout_rmse
# [1] 0.86469 # lower than the required rmse of 0.86490

