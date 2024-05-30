# Movie recommendation system

## About the project
This repository is created for educational purposes. Famous [Movie Lens](https://grouplens.org/datasets/movielens/) was considered. The version we tuned the parameters on is `ml-latest.zip`. The description of the dataset among with the problem definition can be found on the website.

It is also important to understand that this was also an attempt to present a simple solution without that intensive computations, which is wise considering the size of the data. Of course, in some situation the implementation is strongly coupled with the dataset, however, in this way we employ our knowledge about dataset.

Additionally, the code follows the schema of [Transformer Mixin](https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html), [BaseEstimator](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html), [RegressorMixin](https://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html) class from `scikit-learn` library to be support API.


## Files
Each file in the repository has it's own purpose.
| File name | Meaning |
| -- | -- |
| [.dvc](.dvc) | Folder for DVC
| [data](data) | Directory with dataset metadata
| [.dvcignore](.dvcignore) | File to ensure proper data version control
| [analysis.ipynb](analysis.ipynb)  | Exploratory analysis of the features
| [movie_lens_libs.py](movie_lens_lib.py) | The library itself
| [sample.ipynb](samples.ipynb) | A set of use cases from the library
| [test.ipynb](tests.ipynb) | The comparison of the model with the baselines using train-test split and MSE, MAE and accuracy as metrics 

## Getting started

### Prerequisites

The library alone uses only [Pandas](https://pandas.pydata.org/) and [scikit-learn](https://scikit-learn.org/stable/index.html).

As for the date of writing this description, the following command is used to download the dependencies:

- if you are using [conda](https://conda.io/en/latest/):
```sh
conda install pandas scikit-learn
```

- if you are using [pip](https://pypi.org/project/pip/):
```sh
pip install pandas scikit-learn
```

However, you might also want to play around with the analysis. Here are additional packages for data visualization and Jupyter Notebook support:
```sh
# conda
conda install ipykernel matplotlib seaborn
```

```sh
# pip
pip install ipykernel matplotlib seaborn
```

## Installation
If you have all those dependencies described above, the usage is simply "plug and play".

You just have to clone the repository:
```sh
git clone https://github.com/vasia-korz/recommendation-system
```

Then, you can use the files within the repository. As for the usage of the library, you just have to copy [movie_lens_lib.py](movie_lens_lib.py) to whatever directory you are interested in and then import in the project using Python language, for example:
```py
import movie_lens_lib
```
or any other appropriate way the language allows.