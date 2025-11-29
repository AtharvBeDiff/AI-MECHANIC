from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# --- LOAD MODELS ---
try:
    # We use the vehicle_model from the real data training
    vehicle_model = joblib.load('vehicle_model.pkl')
    
    # We load the real bike list we extracted
    real_bike_list = joblib.load('bike_list.pkl')
    
    print("✅ Models & Real Data Loaded Successfully")
except Exception as e:
    print(f"⚠️ Error loading files: {e}")
    print("Did you run 'clean_and_train.py' yet?")
    vehicle_model = None
    real_bike_list = ["Error Loading Data"]

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html', bike_options=real_bike_list)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get Inputs
        bike_model = request.form['bike_model']
        
        # We assume standard shop ambient temp (300K / 27C)
        air_temp = 300 
        
        # User Inputs
        eng_temp_c = float(request.form['temp'])
        voltage = float(request.form['voltage'])
        chain = float(request.form['chain'])
        vibe = float(request.form['vibe'])
        rpm = float(request.form['rpm'])
        work_exp = float(request.form['work_exp'])

        # 2. Map Inputs to Model Features
        # The Kaggle dataset used: [Air(K), Process(K), RPM, Torque(Nm), Wear(min)]
        
        # Conversion Logic:
        process_k = eng_temp_c + 273.15 # Celsius to Kelvin
        
        # Loose chain (high slack) means poor torque transfer. 
        # We map slack to Torque: 60mm slack = 0 torque, 10mm slack = 50 torque
        torque_est = max(10, 60 - chain) 
        
        # High vibration usually means high tool/engine wear.
        wear_est = vibe * 15 
        
        features = [[air_temp, process_k, rpm, torque_est, wear_est]]
        
        # 3. Predict Failure
        if vehicle_model:
            fail_prob = vehicle_model.predict_proba(features)[0][1] # Probability
        else:
            fail_prob = 0.5 # Fallback
            
        health_score = int((1 - fail_prob) * 100)

        # 4. Generate Report
        report = {
            "score": health_score,
            "status": "OPERATIONAL",
            "color": "#00d26a",
            "advice": "Vehicle condition is good.",
            "issues": [],
            "est_cost": 0
        }

        # Logic: If health is low, identify WHY
        if health_score < 80:
            report["status"] = "SERVICE NEEDED"
            report["color"] = "#fcd53f"
            
            # Diagnostics
            if eng_temp_c > 105:
                report["issues"].append({"part": "Coolant System", "action": "Check Fan/Fluid", "cost": 40})
                report["est_cost"] += 40
            if voltage < 12.4:
                report["issues"].append({"part": "Battery", "action": "Replace", "cost": 50})
                report["est_cost"] += 50
            if vibe > 5:
                 report["issues"].append({"part": "Mountings", "action": "Tighten", "cost": 20})
                 report["est_cost"] += 20
            if chain > 35:
                 report["issues"].append({"part": "Chain", "action": "Tighten/Replace", "cost": 30})
                 report["est_cost"] += 30

        if health_score < 50:
             report["status"] = "CRITICAL RISK"
             report["color"] = "#f8312f"
             report["advice"] = "Unsafe to ride. Immediate repair required."

        # Financial Check (Simple estimates)
        est_income = work_exp * 500 + 1500 # Simple formula
        impact_pct = (report["est_cost"] / est_income) * 100

        return render_template('index.html', 
                             bike_options=real_bike_list,
                             report=report, 
                             bike=bike_model, 
                             impact=round(impact_pct, 1))

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)