"""
Module containing custom scorers and regressors for movie rating predictions.

Includes functions for calculating custom scoring metrics and classes for
different regression models based on genres, clusters, and movie-specific data.
Also includes preprocessing transformers for preparing the movie dataset.

#### Functions:
- custom_mse_scorer: Calculates the mean squared error between true and predicted values.
- custom_mae_scorer: Calculates the mean absolute error between true and predicted values.
- custom_accuracy_scorer: Calculates the accuracy based on a tolerance value.
- get_performance_stats: Returns a dictionary of performance statistics (mse, mae, accuracy).
- print_stats: Prints a dictionary of statistics in a formatted way.

#### Classes:
- GenreBasedRegressor:
    Regressor to predict movie ratings based on user average rating for movie genres.
- ClusterBasedRegressor:
    Regressor to predict movie ratings based on clustering movies and user ratings.
- MovieBasedRegressor:
    Regressor to predict movie ratings based on average ratings of individual movies.
- HybridRegressor:
    Regressor that combines predictions from genre-based, cluster-based, and movie-based regressors.
- PreProcessingBase:
    Transformer to preprocess movie data, including splitting genres.
- PreProcessingAggregated:
    Transformer to preprocess movie data and aggregate additional information like rating means and release years.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from sklearn.cluster import KMeans


def custom_mse_scorer(y_true, y_pred):
    """
    Calculates the mean squared error between true and predicted values.

    #### Parameters:
    - y_true: array-like
        True values.
    - y_pred: array-like
        Predicted values.

    #### Returns:
    Mean squared error.
    """
    return np.mean((y_true - y_pred) ** 2)


def custom_mae_scorer(y_true, y_pred):
    """
    Calculates the mean absolute error between true and predicted values.

    #### Parameters:
    - y_true: array-like
        True values.
    - y_pred: array-like
        Predicted values.

    #### Returns:
    Mean absolute error.
    """
    return np.mean(abs(y_true - y_pred))


def custom_accuracy_scorer(y_true, y_pred, tol=(1.0 + 1e-9)):
    """
    Calculates the accuracy based on a tolerance value.

    #### Parameters:
    - y_true: array-like
        True values.
    - y_pred: array-like
        Predicted values.
    - tol: float
        Tolerance value for considering predictions as accurate.

    #### Returns:
    Accuracy score.
    """
    accuracy = np.isclose(y_pred, y_true, atol=tol).mean()
    return accuracy


def get_performance_stats(y_true, y_pred):
    """
    Returns a dictionary of performance statistics (mse, mae, accuracy).

    #### Parameters:
    - y_true: array-like
        True values.
    - y_pred: array-like
        Predicted values.

    #### Returns:
    Dictionary containing mse, mae, and accuracy.
    """
    mse = custom_mse_scorer(np.array(y_pred), np.array(y_true))
    mae = custom_mae_scorer(np.array(y_pred), np.array(y_true))
    acc = custom_accuracy_scorer(np.array(y_pred), np.array(y_true))
    return {"mse": mse, "mae": mae, "accuracy": acc}


def print_stats(stats):
    """
    Prints a dictionary of statistics in a formatted way.

    #### Parameters:
    - stats: dict
        dictionary of statistics to print.
    """
    for key, value in stats.items():
        print(key.upper() + ": " + str(round(value, 3)))


class GenreBasedRegressor(BaseEstimator, RegressorMixin):
    """
    Regressor to predict movie ratings based on user preferences for movie genres.

    #### Parameters:
    - movies_hot_df: DataFrame
        Preprocessed movie dataset with genres.
    """

    def __init__(self, movies_hot_df):
        self.user_genre_df = None
        self.movies_hot_df = movies_hot_df

    def fit(self, X, y=None):
        """
        Fits the regressor on the training data.

        #### Parameters:
        - X: DataFrame
            Feature set.
        - y: Series or DataFrame
            Target variable.

        #### Returns:
        Fitted regressor.
        """
        ratings_train_df = pd.concat([X, y], axis=1)
        self.user_genre_df = pd.DataFrame({"userId": ratings_train_df["userId"].unique()})
        genres = self.movies_hot_df["Genres_Split"].explode().unique()

        for genre in genres:
            self.user_genre_df[genre] = pd.Series(np.zeros(self.user_genre_df.shape[0]))
            self.user_genre_df["count_" + genre] = pd.Series(np.zeros(self.user_genre_df.shape[0]))

        self.user_genre_df.set_index("userId", inplace=True)

        movies_exploded_df = self.movies_hot_df[["Genres_Split"]].explode('Genres_Split').rename(
            columns={'Genres_Split': 'genre'})
        merged_df = ratings_train_df.merge(movies_exploded_df, on='movieId')

        agg_df = merged_df.groupby(['userId', 'genre']).agg(
            total_rating=('rating', 'sum'),
            count=('rating', 'count')
        ).reset_index()

        agg_df['average_rating'] = agg_df['total_rating'] / agg_df['count']

        self.user_genre_df = agg_df.pivot(index='userId', columns='genre', values=['average_rating', 'count'])
        self.user_genre_df.columns = [f'{stat}_{genre}' for stat, genre in self.user_genre_df.columns]
        self.user_genre_df = self.user_genre_df.reset_index()
        self.user_genre_df.fillna(3.5, inplace=True)
        self.user_genre_df.drop(["count_" + col for col in genres], axis=1, inplace=True)
        self.user_genre_df.set_index("userId", inplace=True)

        return self

    def predict(self, X, rounded=True, default=3.5):
        """
        Predicts movie ratings for the given data.

        #### Parameters:
        - X: DataFrame
            Feature set.
        - rounded: bool
            Whether to round predictions to the nearest half.
        - default: float
            Default rating to use if user or movie is not found.

        #### Returns:
        Predicted ratings.
        """
        y_pred = []
        for i, row in X.iterrows():
            y_pred.append(self._predict(row["userId"], row["movieId"], rounded, default))
        return y_pred

    def _predict(self, user_id, movie_id, rounded=True, default=3.5):
        if user_id not in self.user_genre_df.index or movie_id not in self.movies_hot_df.index:
            return default

        movie_genres = self.movies_hot_df.loc[movie_id]["Genres_Split"]
        movie_genres = ["average_rating_" + x for x in movie_genres]
        res = self.user_genre_df.loc[user_id][movie_genres].mean()
        return round(res * 2) / 2 if rounded else res


class ClusterBasedRegressor(BaseEstimator, RegressorMixin):
    """
    Regressor to predict movie ratings based on clustering movies and user ratings.

    #### Parameters:
    - movies_hot_df: DataFrame
        Preprocessed movie dataset.
    - n_movie_clusters: int
        Number of clusters for KMeans.
    - rating_multiplier: float
        Multiplier for rating values.
    - year_multiplier: float
        Multiplier for year values.
    """

    def __init__(self, movies_hot_df, n_movie_clusters=5, rating_multiplier=5, year_multiplier=0.05):
        self.users_df = None
        self.movies_hot_df = movies_hot_df.copy()
        self.n_movie_clusters = n_movie_clusters
        self.rating_multiplier = rating_multiplier
        self.year_multiplier = year_multiplier

    def fit(self, X, y=None):
        """
        Fits the regressor on the training data.

        #### Parameters:
        - X: DataFrame
            Feature set.
        - y: Series or DataFrame
            Target variable.

        #### Returns:
        Fitted regressor.
        """
        ratings_train_df = pd.concat([X, y], axis=1)

        self.movies_hot_df["rating_mean"] *= self.rating_multiplier
        self.movies_hot_df['year'] *= self.year_multiplier

        kmeans = KMeans(n_clusters=self.n_movie_clusters, random_state=42, n_init="auto")
        self.movies_hot_df['Cluster'] = kmeans.fit_predict(self.movies_hot_df.drop(["Genres_Split"], axis=1))

        self.movies_hot_df["rating_mean"] /= self.rating_multiplier

        ratings_with_clusters = ratings_train_df.merge(self.movies_hot_df, left_on='movieId', right_index=True)

        user_cluster_stats = ratings_with_clusters.groupby(['userId', 'Cluster'])['rating'].agg(
            ['count', 'sum']).reset_index()
        user_cluster_pivot = user_cluster_stats.pivot(index='userId', columns='Cluster',
                                                      values=['count', 'sum']).fillna(0)
        user_cluster_pivot.columns = [f'Cluster_{stat}_{cluster}' for stat, cluster in user_cluster_pivot.columns]

        for cluster in range(self.n_movie_clusters):
            count_col = f'Cluster_count_{cluster}'
            sum_col = f'Cluster_sum_{cluster}'
            mean_col = f'Cluster_mean_{cluster}'
            if count_col in user_cluster_pivot.columns and sum_col in user_cluster_pivot.columns:
                user_cluster_pivot[mean_col] = user_cluster_pivot[sum_col] / user_cluster_pivot[count_col]
            else:
                user_cluster_pivot[count_col] = 0
                user_cluster_pivot[sum_col] = 0
                user_cluster_pivot[mean_col] = 0

        user_cluster_pivot = user_cluster_pivot.fillna(0)
        self.users_df = user_cluster_pivot.reset_index()
        self.users_df.set_index("userId", inplace=True)

        sum_sums = self.users_df[["Cluster_sum_" + str(x) for x in range(self.n_movie_clusters)]].sum(axis=1)
        count_sums = self.users_df[["Cluster_count_" + str(x) for x in range(self.n_movie_clusters)]].sum(axis=1)
        self.users_df["rating_mean"] = sum_sums / count_sums
        return self

    def predict(self, X, rounded=True):
        """
        Predicts movie ratings for the given data.

        #### Parameters:
        - X: DataFrame
            Feature set.
        - rounded: bool
            Whether to round predictions to the nearest half.

        #### Returns:
        Predicted ratings.
        """
        y_pred = []
        for i, row in X.iterrows():
            y_pred.append(self._predict(row["userId"], row["movieId"], rounded))
        return y_pred

    def _predict(self, user_id, movie_id, rounded=True):
        if user_id not in self.users_df.index or movie_id not in self.movies_hot_df.index:
            return 3.5

        cluster = int(self.movies_hot_df.loc[movie_id]["Cluster"])
        user_pred = self.users_df.loc[user_id][f"Cluster_mean_{cluster}"]
        return round(user_pred * 2) / 2 if rounded else user_pred


class MovieBasedRegressor(BaseEstimator, RegressorMixin):
    """
    Regressor to predict movie ratings based on average ratings of individual movies.
    """

    def __init__(self):
        self.movie_ratings = None

    def fit(self, X, y=None):
        """
        Fits the regressor on the training data.

        #### Parameters:
        - X: DataFrame
            Feature set.
        - y: Series or DataFrame
            Target variable.

        #### Returns:
        Fitted regressor.
        """
        self.movie_ratings = X["rating_mean"]
        return self

    def predict(self, X, rounded=True):
        """
        Predicts movie ratings for the given data.

        #### Parameters:
        - X: DataFrame
            Feature set.
        - rounded: bool
            Whether to round predictions to the nearest half.

        #### Returns:
        Predicted ratings.
        """
        y_pred = []
        for i, row in X.iterrows():
            y_pred.append(self._predict(row["userId"], row["movieId"], rounded))
        return y_pred

    def _predict(self, user_id, movie_id, rounded=True):
        if movie_id not in self.movie_ratings.index:
            return 3.5

        film_pred = self.movie_ratings.loc[movie_id]
        return round(film_pred * 2) / 2 if rounded else film_pred


class HybridRegressor(BaseEstimator, RegressorMixin):
    """
    Regressor that combines predictions from genre-based, cluster-based, and movie-based regressors.

    #### Parameters:
    - movies_hot_df: DataFrame
        Preprocessed movie dataset.
    - weights: tuple of float
        Weights for combining predictions from different regressors.
    - genre_based_regressor: GenreBasedRegressor
        Genre-based regressor instance.
    - cluster_based_regressor: ClusterBasedRegressor
        Cluster-based regressor instance.
    - movie_based_regressor: MovieBasedRegressor
        Movie-based regressor instance.
    """

    def __init__(
            self,
            movies_hot_df,
            weights=(0.35, 0.45, 0.2),
            genre_based_regressor: GenreBasedRegressor = None,
            cluster_based_regressor: ClusterBasedRegressor = None,
            movie_based_regressor: MovieBasedRegressor = None):
        self.movies_hot_df = movies_hot_df
        self.weights = weights
        self.genre_based_regressor = (
            genre_based_regressor
            if genre_based_regressor is not None
            else GenreBasedRegressor(movies_hot_df)
        )
        self.cluster_based_regressor = (
            cluster_based_regressor
            if cluster_based_regressor is not None
            else ClusterBasedRegressor(movies_hot_df)
        )
        self.movie_based_regressor = (
            movie_based_regressor
            if movie_based_regressor is not None
            else MovieBasedRegressor()
        )

    def fit(self, X, y=None):
        """
        Fits the regressor on the training data.

        #### Parameters:
        - X: DataFrame
            Feature set.
        - y: Series or DataFrame
            Target variable.

        #### Returns:
        Fitted regressor.
        """
        self.genre_based_regressor = self.genre_based_regressor.fit(X, y)
        self.cluster_based_regressor = self.cluster_based_regressor.fit(X, y)
        self.movie_based_regressor = self.movie_based_regressor.fit(self.movies_hot_df)
        return self

    def predict(self, X, rounded=True):
        """
        Predicts movie ratings for the given data.

        #### Parameters:
        - X: DataFrame
            Feature set.
        - rounded: bool
            Whether to round predictions to the nearest half.

        #### Returns:
        Predicted ratings.
        """
        genre_predictions = self.genre_based_regressor.predict(X, False)
        cluster_predictions = self.cluster_based_regressor.predict(X, False)
        movie_predictions = self.movie_based_regressor.predict(X, False)

        y_pred = np.array([
            np.array([genre_pred, cluster_pred, movie_pred]).dot(np.array(self.weights))
            for genre_pred, cluster_pred, movie_pred
            in zip(genre_predictions, cluster_predictions, movie_predictions)
        ])
        return np.round(y_pred * 2) / 2 if rounded else y_pred


class PreProcessingBase(BaseEstimator, TransformerMixin):
    """
    Transformer to preprocess movie data, including splitting genres.
    """

    def fit(self, X, y=None):
        """
        Fits the transformer on the data.

        #### Parameters:
        - X: DataFrame
            Feature set.
        - y: Series or DataFrame
            Target variable.

        #### Returns:
        Fitted transformer.
        """
        return self

    def transform(self, X):
        """
        Transforms the movie data by splitting genres.

        #### Parameters:
        - X: DataFrame
            Feature set.

        #### Returns:
        Transformed DataFrame with genres split into separate columns.
        """
        X = X.copy()
        X['Genres_Split'] = X['genres'].apply(lambda x: x.split('|'))
        mlb = MultiLabelBinarizer()
        binary_matrix = mlb.fit_transform(X['Genres_Split'])
        binary_df = pd.DataFrame(binary_matrix, columns=mlb.classes_)
        movies_hot_df = pd.concat([X.reset_index(), binary_df], axis=1).set_index("movieId")
        return movies_hot_df.drop(["title", "genres"], axis=1)


class PreProcessingAggregated(PreProcessingBase):
    """
    Transformer to preprocess movie data and aggregate additional information like rating means and release years.
    """

    def fit(self, X, y=None):
        """
        Fits the transformer on the data.

        #### Parameters:
        - X: DataFrame
            Feature set.
        - y: Series or DataFrame
            Target variable.

        #### Returns:
        Fitted transformer.
        """
        return self

    def transform(self, X):
        """
        Transforms the movie data by aggregating additional information like rating means and release years.

        #### Parameters:
        - X: tuple of DataFrames
            Tuple containing movies DataFrame and ratings DataFrame.

        #### Returns:
        Transformed DataFrame with aggregated information.
        """
        movies_df, ratings_train_df = X
        movies_hot_df = super().transform(movies_df)

        movies_df = movies_df.copy()
        movies_hot_df = movies_hot_df.merge(ratings_train_df.groupby("movieId")["rating"].mean().reset_index(),
                                            on="movieId")
        movies_hot_df.rename(columns={"rating": "rating_mean"}, inplace=True)
        movies_hot_df = movies_hot_df.set_index("movieId")

        movies_hot_year_df = movies_hot_df.copy()
        movies_df.reset_index(inplace=True)
        years = movies_df[movies_df['movieId'].isin(movies_hot_df.index)]['title'].str.extract(r'\((\d{4})\)')
        years.index = movies_hot_year_df.index
        years[0] = pd.to_numeric(years[0], errors='coerce')
        movies_hot_df['year'] = years.fillna(years.median())
        return movies_hot_df
