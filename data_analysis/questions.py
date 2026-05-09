import seaborn as sns


def most_common_embarkation(data):
    """
    Finds the embarkation port with the highest number of passengers.
    Args:
        data: pandas DataFrame containing the Titanic dataset.
    Returns:
        Most common embarkation port ('C','Q','S').
    """
    # write your code here

    most_frequent_value = titanic_data["embarked"].mode()[0]

    titanic_data["embarked"] = titanic_data["embarked"].fillna(most_frequent_value)

    print(titanic_data["embarked"].isnull().sum())

    return most_frequent_value


titanic_data = sns.load_dataset("titanic")

common_port = most_common_embarkation(titanic_data)
print("Most common embarkation port:", common_port)
