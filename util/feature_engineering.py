
def feature_engineer(data):
    import pandas as pd

    #* New feature `FamilySize` derived from columns `SibSpi` and `Parch`
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1    # +1 for the passenger themselves


    #* New feature `IsAlone` derived from column `FamilySize`
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)


    #* New feature `AgeGroup` derived from column `Age`
    age_bins = [0, 18, 30, 50, 100]
    age_labels = ["Child", "Young Adult", "Adult", "Senior"]

    data["Age_Group"] = pd.cut(data["Age"], bins=age_bins, labels=age_labels, right=False)


    #* New feature `Cabins_booked` derived from column `Cabin`
    # >> Assuming Rich people ( = people who can afford multiple cabins) are more likely to survive
    if "Cabin" in data.columns:
        data["Cabins_booked"] = data["Cabin"].str.split(" ").apply(lambda x: len(x) if isinstance(x, list) else 0)
        data = data.drop(columns=["Cabin"], axis=1)


    #* New feature `FareGroup` derived from column `Fare`
    fare_bins = [0, 20, 50, 100, 1000]
    fare_labels = ["Low", "Mediun", "High", "Very High"]

    data['Fare_Group'] = pd.cut(data['Fare'], bins=fare_bins, labels=fare_labels, right=False)


    #* One-hot-encode column `Embarked`
    if "Embarked" in data.columns:
        data["Embarked"] = data["Embarked"].map({"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown", "U": "Unknown"})
        data = pd.get_dummies(data, columns=["Embarked"], prefix="Embarked_in", dtype="int64")


    #* One-hot-encode column `Sex`
    if "Sex" in data.columns:
        data = pd.get_dummies(data, columns=["Sex"], prefix="Sex", dtype="int64")


    #* One-hot-encode column `Age_Group`
    if "Age_Group" in data.columns:
        data = pd.get_dummies(data, columns=["Age_Group"], prefix="Age_Group", dtype="int64")


    #* One-hot-encode column `Fare_Group`
    if "Fare_Group" in data.columns:
        data = pd.get_dummies(data, columns=["Fare_Group"], prefix="Fare_Group", dtype="int64")


    #* New feature `Age_Class` derived from columns `Age` and `Pclass`
    data["Age_Class"] = data["Age"] * data["Pclass"]


    #* New feature `Fare_per_person` derived from columns `Fare` and `FamilySize`
    # People who spend more per person, are rich, and probably more likely to survive
    data["Fare_per_person"] = data["Fare"] / data["FamilySize"]
    

    return data