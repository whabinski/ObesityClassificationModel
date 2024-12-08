# create new feature BMI - make sure to add to preprocessing
# correlation analysis to determine feature selection (idk if we do before or after preprocessing)


# funciton to calculate bmi and add as new column to data
def add_bmi_column(data):
    data['BMI'] = (data['Weight'] / (data['Height'] ** 2)).round(2)     # BMI = weight (kg) / height (meters) squared
    return data

