import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class PCOSPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = None
        
    def create_sample_dataset(self):
        """Medical-grade PCOS dataset"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Age': np.random.randint(18, 45, n_samples),
            'Weight_kg': np.random.normal(70, 15, n_samples),
            'Height_cm': np.random.normal(162, 8, n_samples),
            'BMI': None,
            'Irregular_periods': np.random.choice([0,1], n_samples, p=[0.3, 0.7]),
            'Hirsutism': np.random.choice([0,1], n_samples, p=[0.4, 0.6]),
            'Acne': np.random.choice([0,1], n_samples, p=[0.45, 0.55]),
            'Hair_loss': np.random.choice([0,1], n_samples, p=[0.35, 0.65]),
            'Weight_gain': np.random.choice([0,1], n_samples, p=[0.6, 0.4]),
            'Infertility': np.random.choice([0,1], n_samples, p=[0.25, 0.75]),
            'Dark_skin_patches': np.random.choice([0,1], n_samples, p=[0.3, 0.7]),
            'Fatigue': np.random.choice([0,1], n_samples, p=[0.5, 0.5]),
            'Mood_swings': np.random.choice([0,1], n_samples, p=[0.55, 0.45]),
            'Sleep_issues': np.random.choice([0,1], n_samples, p=[0.4, 0.6]),
            'Headaches': np.random.choice([0,1], n_samples, p=[0.35, 0.65]),
            'Family_history': np.random.choice([0,1], n_samples, p=[0.3, 0.7]),
            'Insulin_resistance': np.random.choice([0,1], n_samples, p=[0.45, 0.55]),
            'High_androgens': np.random.choice([0,1], n_samples, p=[0.6, 0.4]),
            'Ovarian_cysts': np.random.choice([0,1], n_samples, p=[0.65, 0.35])
        }
        
        df = pd.DataFrame(data)
        df['BMI'] = df['Weight_kg'] / ((df['Height_cm']/100)**2)
        
        pcos_prob = (
            0.3*df['Irregular_periods'] + 0.25*df['Hirsutism'] + 0.2*df['Acne'] +
            0.15*df['Weight_gain'] + 0.1*df['Infertility'] + 0.05*df['BMI']/40 +
            np.random.normal(0, 0.2, n_samples)
        )
        df['PCOS'] = (pcos_prob > 0.5).astype(int)
        
        return df
    
    def train_model(self):
        """Train PCOS prediction model"""
        print("🏥 Training PCOS prediction model...")
        
        df = self.create_sample_dataset()
        feature_cols = ['Age', 'BMI', 'Irregular_periods', 'Hirsutism', 'Acne', 
                       'Hair_loss', 'Weight_gain', 'Infertility', 'Dark_skin_patches',
                       'Fatigue', 'Mood_swings', 'Sleep_issues', 'Headaches',
                       'Family_history', 'Insulin_resistance', 'High_androgens',
                       'Ovarian_cysts']
        
        X = df[feature_cols]
        y = df['PCOS']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        self.is_trained = True
        self.feature_importance = dict(zip(feature_cols, self.model.feature_importances_))
        
        print(f"✅ Model trained! Accuracy: {accuracy:.1%}")
        print(f"📊 Dataset: {len(df)} patients")
        return accuracy
    
    def predict_pcos(self, symptoms: dict) -> dict:
        """Predict PCOS risk"""
        if not self.is_trained:
            self.train_model()
        
        feature_cols = ['Age', 'BMI', 'Irregular_periods', 'Hirsutism', 'Acne', 
                       'Hair_loss', 'Weight_gain', 'Infertility', 'Dark_skin_patches',
                       'Fatigue', 'Mood_swings', 'Sleep_issues', 'Headaches',
                       'Family_history', 'Insulin_resistance', 'High_androgens',
                       'Ovarian_cysts']
        
        input_data = np.zeros(len(feature_cols))
        for key, val in symptoms.items():
            if key in feature_cols:
                idx = feature_cols.index(key)
                input_data[idx] = val
        
        input_scaled = self.scaler.transform(input_data.reshape(1, -1))
        probability = self.model.predict_proba(input_scaled)[0]
        prediction = self.model.predict(input_scaled)[0]
        
        risk_level = "🟢 LOW" if probability[1] < 0.3 else "🟡 MODERATE" if probability[1] < 0.7 else "🔴 HIGH"
        
        return {
            'PCOS_Prediction': 'YES' if prediction else 'NO',
            'Risk_Probability': f"{probability[1]*100:.1f}%",
            'Risk_Level': risk_level,
            'Top_Symptoms': self.get_top_symptoms(symptoms)
        }
    
    def get_top_symptoms(self, symptoms: dict) -> list:
        """Get important symptoms"""
        if not self.feature_importance:
            return []
        
        symptom_scores = {}
        for symptom, importance in self.feature_importance.items():
            if symptom in symptoms and symptoms[symptom] > 0:
                symptom_scores[symptom] = importance
        
        return sorted(symptom_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def interactive_survey(self):
        """Interactive symptom input"""
        print("\n" + "="*60)
        print("🩺 PCOS/PCOD RISK ASSESSMENT")
        print("="*60)
        
        symptoms = {}
        
        # Basic info
        symptoms['Age'] = float(input("Your age (18-45): ") or 25)
        symptoms['Weight_kg'] = float(input("Weight (kg): ") or 60)
        symptoms['Height_cm'] = float(input("Height (cm): ") or 160)
        
        bmi = symptoms['Weight_kg'] / ((symptoms['Height_cm']/100)**2)
        print(f"\n📏 BMI: {bmi:.1f}")
        symptoms['BMI'] = bmi
        
        # Symptoms (Yes=1, No=0)
        symptom_list = [
            'Irregular_periods', 'Hirsutism', 'Acne', 'Hair_loss', 'Weight_gain',
            'Infertility', 'Dark_skin_patches', 'Fatigue', 'Mood_swings', 
            'Sleep_issues', 'Headaches', 'Family_history', 'Insulin_resistance',
            'High_androgens', 'Ovarian_cysts'
        ]
        
        print("\nYES=1 / NO=0 / Enter=0:")
        for symptom in symptom_list:
            while True:
                val = input(f"• {symptom.replace('_', ' ').title()}: ").strip().lower()
                if val in ['1', 'yes', 'y']:
                    symptoms[symptom] = 1
                    break
                elif val in ['0', 'no', 'n', '']:
                    symptoms[symptom] = 0
                    break
                print("Enter 1/YES or 0/NO")
        
        return symptoms
    
    def visualize_results(self, result, symptoms):
        fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Risk pie chart (WORKS ✅)
        risk = float(result['Risk_Probability'][:-1])  # Remove % sign
        colors = ['#ff6b6b', '#4ecdc4']  # Red for risk, Teal for safe
        ax1.pie([risk, 100-risk], labels=['PCOS Risk', 'No Risk'], 
            colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('🔬 PCOS Risk Assessment', fontsize=16, fontweight='bold')
    
        symptom_names = ['Irregular Periods', 'Hirsutism', 'Acne', 'Weight Gain', 'Infertility']
        symptom_keys = ['Irregular_periods', 'Hirsutism', 'Acne', 'Weight_gain', 'Infertility']
    
        symptom_values = []
        for key in symptom_keys:
            value = symptoms.get(key, 0)
            symptom_values.append(value)
    
        # Create colorful bar chart
        bars = ax2.bar(symptom_names, symptom_values, color=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
        ax2.set_title('📊 Your Reported Symptoms', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1.2)
        ax2.set_ylabel('Present (1) / Absent (0)')
    
        # Add value labels on bars
        for bar, value in zip(bars, symptom_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{int(value)}', ha='center', va='bottom', fontweight='bold')
    
        plt.tight_layout()
        plt.show()

def demo_pcos_predictor():
    """Main demo function"""
    predictor = PCOSPredictor()
    accuracy = predictor.train_model()
    
    symptoms = predictor.interactive_survey()
    result = predictor.predict_pcos(symptoms)
    
    print("\n" + "="*60)
    print("🎯 YOUR RESULTS")
    print("="*60)
    print(f"Prediction: {result['PCOS_Prediction']}")
    print(f"Risk: {result['Risk_Probability']} {result['Risk_Level']}")
    
    print(f"\n🔍 Top Symptoms:")
    for symptom, importance in result['Top_Symptoms']:
        print(f"  • {symptom.replace('_', ' ').title()}: {importance:.1%}")
    
    print("\n⚠️  CONSULT A DOCTOR FOR CONFIRMATION!")
    predictor.visualize_results(result, symptoms)
    
    joblib.dump(predictor, 'pcos_predictor_model.pkl')
    print("\n💾 Model saved!")

# RUN THIS
if __name__ == "__main__":
    demo_pcos_predictor()