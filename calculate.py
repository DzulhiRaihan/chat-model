def calculate_bmi (weight, height):
    return weight / (height/100)**2

def calculate_bmr (weight, height, age, gender):
    if gender == "Pria":
        return 66 + (13.7 * weight) + (5 * height) - (6.8 * age)
    else:
        return 665 + (9.6 * weight) + (1.8 * height) - (6.8 * age)
    
def calculate_tdee (activity, bmr):
    if activity == "Sangat Jarang":
        return bmr * 1.2
    elif activity == "Jarang":
        return bmr * 1.375
    elif activity == "Normal":
        return bmr * 1.55
    elif activity == "Sering":
        return bmr * 1.725
    else:
        return bmr * 1.9