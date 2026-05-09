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
    embark_include = ["S", "C", "Q"]
    biggest_port = "h"
    passenger_num = -1

    for i in embark_include:
        port_passenger = data[data["embarked"] == i]["embarked"].count()

        if port_passenger > passenger_num:
            passenger_num = port_passenger
            biggest_port = i
    return biggest_port


titanic_data = sns.load_dataset("titanic")

common_port = most_common_embarkation(titanic_data)
print("Most common embarkation port:", common_port)
