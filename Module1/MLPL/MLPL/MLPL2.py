#Sarah Wilson 

import pandas as pd


def main(
    data_file: str,
    result_file: str,
    tune: bool,
    categorical: List,
    discretize: List,
    missing: List,
    standarize: List
):
    raw_df: pd.DataFrame = read_data(data_file)
    filled_df = fill_missing_values(raw_df, missing)

    # For purpose of the demo, run these back to back. However, we could probably optimize this a little bit more
    # Leaving that for future work. Created story and added to backlog for that.

    categorical_filled_df = map_categorical_data(filled_df, categorical)
    discretize_filled_df = map_categorical_data(filled_df, discretize)

    tune_df, test_df = create_tune_test(categorical_filled_df)
    list_tune_folds: List[pd.DataFrame] = create_stratified_folds(tune_df)
    list_test_folds: List[pd.DataFrame] = create_stratified_folds(test_df)

    for index, pd.DataFrame in enumerate(list_test_folds):
        train_fold, test_fold = generate_train_test(list_test_folds, index)
        mu_sigma: List[Tuple[float, float]] = calculate_mean_std(train_fold, standarize)
        std_train_fold = standardize_data(train_fold, standarize, mu_sigma)
        std_test_fold = standardize_data(test_fold, standarize, mu_sigma)

        simple_ml = MLExampleAlgorithm()
        simple_ml.train(std_train_fold)
        accuracy = simple_ml.classify(std_test_fold)
        logging.info(f"Accuracy for fold {index} was {accuracy}")

    pass


def read_data(data_file: str) -> pd.DataFrame:
    """
    :param data_file: File name to read into dataframe
    :return: Pandas dataframe for the data file read in
    This method doesn't do much, but allows me to quickly add params when reading in data (if needed)
    """
    dataset: pd.DataFrame = pd.read_csv(data_file)
    logging.debug(f"Read in data {data_file} with shape {dataset.shape}")
    return dataset


def fill_missing_values(raw_df: pd.DataFrame, apply: List[int] = None) -> pd.DataFrame:
    """
    Fill in missing values for a dataframe
    :param raw_df: A raw dataframe with missing values
    :param apply: A list of column indexes to replace values on.
    :return: Dataframe with all missing values filled
    """
    # TODO: Write this method
    return raw_df


def map_categorical_data(
    raw_df: pd.DataFrame,
    apply_ordinal: List[int] = None,
    apply_one_hot: List[int] = None
) -> pd.DataFrame:
    """
    Fill in missing values for a dataframe
    :param raw_df: A raw dataframe to apply categorical mapping on
    :param apply_ordinal: A list of column indexes to map categorical values using name -> index
    :param apply_one_hot: A list of column indexes to map categorical values using one hot encoding
    :return: Dataframe with all categorical values mapped to numeric values
    """
    # TODO: Write this method
    return raw_df


def discretize_data(raw_df: pd.DataFrame, apply: List[int] = None) -> pd.DataFrame:
    """
    Run discretization on the dataset
    :param raw_df: Dataframe to apply discretization on
    :param apply: Columns to discretize
    :return:
    """
    return raw_df


def create_tune_test(raw_data_file: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into a tune (20%) and test data set (80%). This does NOT create folds
    :param raw_data_file: Dataframe for all data points
    :return: A tuple of two dataframes. First is the tune set. Second is the test set
    """
    tune = raw_data_file.sample(frac=0.2)
    test = raw_data_file.drop(tune.index)
    logging.debug(f"Split data into tune and test with shapes {tune.shape}, {test.shape}")

    return tune, test


def create_stratified_folds(tune_df: pd.DataFrame, num_folds: int = 5) -> List[pd.DataFrame]:
    """
    Split data into stratified folds
    :param tune_df: Dataframe to split up into k folds
    :param num_folds: Number of folds to create
    :return: A list of k disjoint folds, each a dataframe
    """
    # TODO: Implement stratification
    folds = [pd.DataFrame() for i in range(0, num_folds)]
    return folds


def generate_train_test(list_test_folds: List[pd.DataFrame], index: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # TODO: Actually create the test/train set here
    # Will just hard code sum numbers here to get mypy to not complain
    return list_test_folds[0], list_test_folds[index]


def calculate_mean_std(train_folds: pd.DataFrame, apply: List[int] = None) -> List[Tuple[float, float]]:
    """
    Determine mean and std deviation for standardization
    :param train_folds: Dataframe to calculate the mean and std from
    :param apply: Columns to calculate mean, std
    :return: Tuple with mean and std deviation
    """
    # TODO: Build method
    return [(0.5, 0.25)]


def standardize_data(
    train_fold: pd.DataFrame,
    apply: List[int] = None,
    mu_std: List[Tuple[float, float]] = None
) -> pd.DataFrame:
    # TODO: Build method
    return train_fold


def main_batcher():
    main(data_set="colAvg.data", 
        result_file="colAvg_result.xlsx",
        tune=False,
        categorical=[1, 2, 3],
        discretize=[4, 5, 6],
        missing=[2, 5],
        standarize=[1, 2, 3, 4, 5, 6])

if __name__ == "__main__":
    main_batcher()
