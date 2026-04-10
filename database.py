import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------------------------------------------------------
# 1. Database Schema Setup
# ---------------------------------------------------------
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    age = Column(Integer)
    foot_morphology = Column(String(50))
    medical_history = Column(Text)
    average_weekly_mileage = Column(Float)
    n1_model_path = Column(String(255), nullable=True)

class DailyLog(Base):
    __tablename__ = 'daily_logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    date = Column(DateTime)
    
    # Features
    previous_week_mileage = Column(Float, nullable=True)
    current_week_mileage = Column(Float)
    mileage_spike_pct = Column(Float, nullable=True)
    average_rpe = Column(Integer)
    max_pain_score = Column(Integer)
    
    # Target Variable (Moved directly into the log for XGBoost alignment)
    injury_risk_level = Column(String(20))

# Create engine and session
engine = create_engine('sqlite:///run_safe.db', connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# ---------------------------------------------------------
# 2. Synthetic Data Generation Logic
# ---------------------------------------------------------
def generate_synthetic_data(n_users=1000, max_days_per_user=52):
    print("Generating synthetic user profiles and daily logs...")
    
    users = []
    daily_logs = []

    age_range = (18, 65)
    foot_probs = {'normal': 0.4, 'flat_arched': 0.3, 'high_arched': 0.3}
    medical_conditions = [
        "No significant health issues", 
        "History of knee problems",
        "Hip issues",
        "Back pain",
        "Flat feet"
    ]
    
    for i in range(n_users):
        age = np.random.randint(*age_range)
        foot = np.random.choice(list(foot_probs.keys()), p=list(foot_probs.values()))
        
        if np.random.rand() < 0.7:
            medical_history = "No significant health issues"
        else:
            medical_history = np.random.choice(medical_conditions)
        
        avg_mileage = max(0, 30 - (age * 0.2))
        weekly_mileage = max(5, np.random.normal(avg_mileage, 5))
        
        users.append({
            'id': i+1,
            'age': int(age),
            'foot_morphology': foot,
            'medical_history': medical_history,
            'average_weekly_mileage': float(weekly_mileage),
            'n1_model_path': f"models/n1_weights/user_{i+1}.pt"
        })

    date_rng = pd.date_range(end=pd.Timestamp.now(), periods=365)
    
    for user in users:
        user_id = user['id']
        n_user_days = min(max_days_per_user, int(np.random.exponential(scale=20)))
        if n_user_days < 5: 
            continue
            
        sample_indices = sorted(np.random.choice(len(date_rng), n_user_days, replace=False))
        base_weekly_mileage = user['average_weekly_mileage']
        
        for sequence_idx, date_idx in enumerate(sample_indices):
            date = date_rng[date_idx]
            
            # FIX: Properly define and calculate previous week's mileage
            if sequence_idx == 0:
                prev_week = None
                spike_pct = None
            else:
                prev_week = max(0.0, base_weekly_mileage + np.random.normal(0, 3))
            
            current_weekly_mileage = max(0.0, base_weekly_mileage + np.random.normal(0, 5))
            
            # FIX: Calculate spike percentage safely
            if prev_week and prev_week > 0:
                spike_pct = ((current_weekly_mileage - prev_week) / prev_week) * 100
            else:
                spike_pct = 0.0 if prev_week is not None else None

            rpe = max(2, min(10, int(round(5 + (current_weekly_mileage/30 * 2) + user['age']/40))))
            pain = max(0, min(5, int(round(np.random.exponential(scale=1)/2))))

            # FIX: Calculate target variable directly inside the log
            risk_score = 0
            if spike_pct and spike_pct > 15: risk_score += 1.5
            if pain >= 3: risk_score += 2.0
            if rpe >= 8: risk_score += 1.0
            if user['age'] > 45: risk_score += 0.5
            
            if risk_score < 1.5:
                risk_level = 'Low'
            elif 1.5 <= risk_score < 3.5:
                risk_level = 'Medium'
            else:
                risk_level = 'High'

            daily_logs.append({
                'user_id': user_id,
                'date': date.to_pydatetime(),
                'previous_week_mileage': prev_week,
                'current_week_mileage': current_weekly_mileage,
                'mileage_spike_pct': spike_pct,
                'average_rpe': rpe,
                'max_pain_score': pain,
                'injury_risk_level': risk_level
            })

    # Write data to SQLite Database
    db = SessionLocal()
    try:
        db.query(User).delete()
        db.query(DailyLog).delete()

        for user in users:
            db.add(User(**user))
        
        # Insert logs in bulk for performance
        db.bulk_insert_mappings(DailyLog, daily_logs)
        
        db.commit()
        print(f"Successfully saved {len(users)} users and {len(daily_logs)} logs to run_safe.db")
    except Exception as e:
        print(f"Error saving data: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    generate_synthetic_data()
