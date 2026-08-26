import pandas as pd
import joblib


# ==========================================
# EMIPredict AI - Prediction System
# ==========================================

CLASSIFICATION_MODEL = "models/emi_classification_model.joblib"
REGRESSION_MODEL = "models/emi_regression_model.joblib"


# Load trained models
classification_model = joblib.load(CLASSIFICATION_MODEL)
regression_model = joblib.load(REGRESSION_MODEL)


print("=" * 60)
print("              EMIPREDICT AI")
print("          EMI ELIGIBILITY PREDICTION")
print("=" * 60)


# ==========================================
# GET USER INPUT
# ==========================================

age = float(input("Enter age: "))

gender = input("Enter gender: ")

marital_status = input("Enter marital status: ")

education = input("Enter education: ")

monthly_salary = float(
    input("Enter monthly salary: ")
)

employment_type = input(
    "Enter employment type: "
)

years_of_employment = float(
    input("Enter years of employment: ")
)

company_type = input(
    "Enter company type: "
)

house_type = input(
    "Enter house type: "
)

monthly_rent = float(
    input("Enter monthly rent: ")
)

family_size = int(
    input("Enter family size: ")
)

dependents = int(
    input("Enter number of dependents: ")
)

school_fees = float(
    input("Enter school fees: ")
)

college_fees = float(
    input("Enter college fees: ")
)

travel_expenses = float(
    input("Enter travel expenses: ")
)

groceries_utilities = float(
    input("Enter groceries/utilities expenses: ")
)

other_monthly_expenses = float(
    input("Enter other monthly expenses: ")
)

existing_loans = input(
    "Enter existing loans: "
)

current_emi_amount = float(
    input("Enter current EMI amount: ")
)

credit_score = float(
    input("Enter credit score: ")
)

bank_balance = float(
    input("Enter bank balance: ")
)

emergency_fund = float(
    input("Enter emergency fund: ")
)

emi_scenario = input(
    "Enter EMI scenario: "
)

requested_amount = float(
    input("Enter requested loan amount: ")
)

requested_tenure = int(
    input("Enter requested tenure: ")
)


# ==========================================
# CREATE INPUT DATAFRAME
# ==========================================

input_data = pd.DataFrame([{

    "age": age,
    "gender": gender,
    "marital_status": marital_status,
    "education": education,
    "monthly_salary": monthly_salary,
    "employment_type": employment_type,
    "years_of_employment": years_of_employment,
    "company_type": company_type,
    "house_type": house_type,
    "monthly_rent": monthly_rent,
    "family_size": family_size,
    "dependents": dependents,
    "school_fees": school_fees,
    "college_fees": college_fees,
    "travel_expenses": travel_expenses,
    "groceries_utilities": groceries_utilities,
    "other_monthly_expenses": other_monthly_expenses,
    "existing_loans": existing_loans,
    "current_emi_amount": current_emi_amount,
    "credit_score": credit_score,
    "bank_balance": bank_balance,
    "emergency_fund": emergency_fund,
    "emi_scenario": emi_scenario,
    "requested_amount": requested_amount,
    "requested_tenure": requested_tenure

}])


# ==========================================
# CLASSIFICATION
# ==========================================

eligibility_prediction = classification_model.predict(
    input_data
)[0]


# ==========================================
# REGRESSION
# ==========================================

emi_prediction = regression_model.predict(
    input_data
)[0]


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n")
print("=" * 60)
print("                 PREDICTION RESULT")
print("=" * 60)

print(
    "\nEMI Eligibility:",
    eligibility_prediction
)

print(
    "\nMaximum Monthly EMI:",
    round(emi_prediction, 2)
)

print("\n" + "=" * 60)
print("             PREDICTION COMPLETED")
print("=" * 60)