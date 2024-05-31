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
    mse = custom_mse_scorer(np.array(y_true), np.array(y_pred))
    mae = custom_mae_scorer(np.array(y_true), np.array(y_pred))
    acc = custom_accuracy_scorer(np.array(y_true), np.array(y_pred))
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
        self.genres_rating_columns = None
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
        def normalize(df):
            df["total"] /= df["count"]
            return df
        
        def column_labels(df):
            df.columns = [f'{stat}_{genre}' for stat, genre in df.columns]
            return df

        self.user_genre_df = (
            self.movies_hot_df[["Genres_Split"]]
                .explode('Genres_Split')
                .merge(pd.concat([X, y], axis=1).drop(['timestamp'], axis=1),on='movieId')
                .drop(["movieId"], axis=1)
                .groupby(['userId', 'Genres_Split'])
                .agg(total=('rating', 'sum'), count=('rating', 'count'))
                .pipe(normalize)
                .drop(["count"], axis=1)
                .reset_index()
                .rename(columns={"total": "mean"})
                .pivot(
                    index='userId',
                    columns='Genres_Split',
                    values=['mean'])
                .pipe(column_labels)
                .fillna(3.5)
        )

        self.genres_rating_columns = (self.movies_hot_df["Genres_Split"]
                                      .apply(lambda x: self.user_genre_df.columns.get_indexer(["mean_"+y for y in x])))
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
        y_pred = np.zeros(X.shape[0]) + default
        is_present = (X["userId"].isin(self.user_genre_df.index) & X["movieId"].isin(self.movies_hot_df.index))
        present_df = X.loc[is_present]

        ravelled = self.user_genre_df.loc[present_df["userId"]].to_numpy().ravel()

        choices_relative = self.genres_rating_columns.loc[present_df["movieId"]]
        choices_absolute = choices_relative + np.arange(choices_relative.shape[0]) * self.user_genre_df.shape[1]

        y_pred[is_present] = choices_absolute.apply(lambda x: ravelled[x].mean())
        return np.round(y_pred * 2) / 2 if rounded else y_pred


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

    def __init__(self, movies_hot_df, n_movie_clusters=5, rating_multiplier=5, year_multiplier=0.05, random_state=None):
        self.users_df = None
        self.movies_hot_df = movies_hot_df.copy()
        self.n_movie_clusters = n_movie_clusters
        self.rating_multiplier = rating_multiplier
        self.year_multiplier = year_multiplier
        self.random_state = random_state
        self.cluster_columns = np.array([f"Cluster_mean_{i}" for i in range(n_movie_clusters)])

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
        self.movies_hot_df["rating_mean"] *= self.rating_multiplier
        self.movies_hot_df['year'] *= self.year_multiplier

        kmeans = KMeans(n_clusters=self.n_movie_clusters, random_state=self.random_state, n_init="auto")
        self.movies_hot_df['Cluster'] = kmeans.fit_predict(self.movies_hot_df.drop(["Genres_Split"], axis=1))

        self.movies_hot_df["rating_mean"] /= self.rating_multiplier

        def column_labels(df):
            df.columns = [f'Cluster_{stat}_{cluster}' for stat, cluster in df.columns]
            return df

        def calculate_mean(df):
            for cluster in range(self.n_movie_clusters):
                count_col = f'Cluster_count_{cluster}'
                sum_col = f'Cluster_sum_{cluster}'
                mean_col = f'Cluster_mean_{cluster}'

                df[mean_col] = df[sum_col] / df[count_col]

            return df

        self.users_df = (
            pd.concat([X.drop(['timestamp'], axis=1), y], axis=1)
                .merge(self.movies_hot_df['Cluster'], left_on='movieId', right_index=True)
                .drop(['movieId'], axis=1)
                .groupby(['userId', 'Cluster'])['rating']
                .agg(['count', 'sum'])
                .reset_index()
                .pivot(
                    index='userId',
                    columns='Cluster',
                    values=['count', 'sum'])
                .fillna(0)
                .pipe(column_labels)
                .pipe(calculate_mean)
                .fillna(0)
        )
        # sum_sums = self.users_df[["Cluster_sum_" + str(x) for x in range(self.n_movie_clusters)]].sum(axis=1)
        # count_sums = self.users_df[["Cluster_count_" + str(x) for x in range(self.n_movie_clusters)]].sum(axis=1)
        # self.users_df["rating_mean"] = sum_sums / count_sums
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
        y_pred = np.zeros(X.shape[0]) + 3.5
        is_present = (X["userId"].isin(self.users_df.index) & X["movieId"].isin(self.movies_hot_df.index))
        present_df = X.loc[is_present]

        clusters = self.movies_hot_df.loc[present_df["movieId"]]["Cluster"].astype(int)
        choices = clusters + np.arange(clusters.shape[0]) * self.cluster_columns.shape[0]

        y_pred[is_present] = self.users_df.loc[present_df["userId"]][self.cluster_columns].to_numpy().ravel()[choices]
        return np.round(y_pred * 2) / 2 if rounded else y_pred


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
        y_pred = np.zeros(X.shape[0]) + 3.5
        is_present = X["movieId"].isin(self.movie_ratings.index)

        film_pred = self.movie_ratings.loc[X.loc[is_present]["movieId"]]

        y_pred[is_present] = film_pred
        return np.round(y_pred * 2) / 2 if rounded else y_pred


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
