import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import joblib

try:
    df_mech = pd.read_csv('predictive_maintenance.csv')
    df_mech.columns = ['UDI', 'ID', 'Type', 'Air', 'Process', 'RPM', 'Torque', 'Wear', 'Target', 'FailType']
    
    X = df_mech[['Air', 'Process', 'RPM', 'Torque', 'Wear']]
    y = df_mech['Target']
    
    model_mech = RandomForestClassifier(n_estimators=100, random_state=42)
    model_mech.fit(X, y)
    
    joblib.dump(model_mech, 'vehicle_model.pkl')
    print("done 1")
except FileNotFoundError:
    print("predictive wali file missing hai")
try:
    df_bikes = pd.read_csv('all_bikez_curated.csv', low_memory=False)
    
    df_bikes['Brand'] = df_bikes['Brand'].astype(str).str.strip().str.title()
    
    target_brands = ['Yamaha', 'Honda', 'Royal Enfield', 'Kawasaki', 'Ktm', 'Suzuki', 'Harley-Davidson']
    
    final_bike_list = []
    for brand in target_brands:
        brand_df = df_bikes[df_bikes['Brand'] == brand]

        top_models = brand_df.sort_values('Year', ascending=False)['Model'].unique()[:10]
        
        for model in top_models:
            final_bike_list.append(f"{brand} - {model}")

    joblib.dump(final_bike_list, 'bike_list.pkl')
    print("done 2")
except Exception as e:
    print(f"bike load nai hui")
    joblib.dump(['Yamaha - Generic Sport', 'Honda - Generic Cruiser'], 'bike_list.pkl')
try:
    df_salary = pd.read_csv('Salary_dataset.csv')
    X_sal = df_salary[['YearsExperience']]
    y_sal = df_salary['Salary']
    
    model_sal = LinearRegression()
    model_sal.fit(X_sal, y_sal)
    
    joblib.dump(model_sal, 'salary_model.pkl')
    print("done 3")
except FileNotFoundError:
    print("'Salary_Data.csv' is missing!")

