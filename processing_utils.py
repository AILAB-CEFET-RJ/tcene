import pandas as pd
from rapidfuzz import process

def transformDataInCategory(column):
    """
    Transforms a given column into one-hot encoding.

    Args:
        column (pd.Series): A pandas Series containing categorical data.

    Returns:
        pd.DataFrame: A DataFrame with one-hot encoded categories.
    """
    # Detect unique categories and transform into one-hot encoding
    one_hot_encoded = pd.get_dummies(column, sparse=True)
    return one_hot_encoded

def group_similar_categories(values, threshold=100):
    """
    Groups similar values in a column based on a similarity threshold.

    Args:
        values (pd.Series): The column containing categorical data.
        threshold (int): Similarity threshold (0-100) for grouping.

    Returns:
        pd.Series: A column with similar values grouped together.
        dict: A dictionary showing the mapping of original values to grouped values.
    """
    unique_values = values.unique()
    grouped = {}

    for value in unique_values:
        # Find the closest match from already grouped categories
        match = process.extractOne(value, grouped.keys(), score_cutoff=threshold)
        if match:
            grouped[value] = match[0]
        else:
            grouped[value] = value

    # Map the original values to their grouped counterparts
    grouped_series = values.map(grouped)
    return grouped_series, grouped