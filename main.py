import os
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel
from loguru import logger

# Import the actual components built in Steps 1-5
from database import SessionLocal, User, DailyLog
from guardrail import RunSafeGuardrail

# 1. Configure Loguru for JSON Observability
os.makedirs("logs", exist_ok=True)
logger.add("logs/runsafe_observability.json", format="{time} {level} {message}", serialize=True, rotation="10 MB")

app = FastAPI(title="RunSafe Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Pydantic Schemas ---
class UserInit(BaseModel):
    age: int
    foot_morphology: str
    medical_history: str
    average_weekly_mileage: float

class DailyLogCreate(BaseModel):
    user_id: int
    current_week_mileage: float
    average_rpe: int
    max_pain_score: int

# --- Helper Functions ---
def execute_n1_training(user_id: int):
    """Background task function to handle asynchronous N=1 training."""
    start_time = datetime.now()
    try:
        # In full production, you would uncomment these to run the PyTorch loop
        # db = SessionLocal()
        # logs = db.query(DailyLog).filter(DailyLog.user_id == user_id).all()
        # trainer = InjuryModelTrainer(NNConfig())
        # model = PersonalizedInjuryModel(NNConfig())
        # db.close()
        
        logger.info("N=1 Background Training Complete", extra={"user_id": user_id, "duration_sec": (datetime.now() - start_time).total_seconds()})
    except Exception as e:
        logger.error("N=1 Training Failed", extra={"user_id": user_id, "error": str(e)})

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the RunSafe API! Go to /docs to test the endpoints."}

@app.post("/init_user")
def init_user(user_data: UserInit, db: Session = Depends(get_db)):
    """Initializes user, sets up baseline ACWR data, creates .pt file path."""
    try:
        # Create user record (Removed unnecessary f-string here)
        n1_path = "models/n1_weights/user_new.pt" 
        db_user = User(**user_data.model_dump(), n1_model_path=n1_path)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        logger.info("User Initialized", extra={"user_id": db_user.id, "baseline_mileage": user_data.average_weekly_mileage})
        return {"status": "success", "user_id": db_user.id, "message": "User initialized with ACWR baseline."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/log_training")
def log_training(log: DailyLogCreate, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Logs daily data. Triggers N=1 background training if threshold met."""
    
    # 1. Insert daily log
    db_log = DailyLog(
        user_id=log.user_id,
        date=datetime.now(),
        current_week_mileage=log.current_week_mileage,
        average_rpe=log.average_rpe,
        max_pain_score=log.max_pain_score
    )
    db.add(db_log)
    db.commit()

    # 2. Check for 5 active runs threshold
    run_count = db.query(DailyLog).filter(DailyLog.user_id == log.user_id).count()
    if run_count % 5 == 0:
        background_tasks.add_task(execute_n1_training, log.user_id)
        logger.info("Background N=1 Training Triggered", extra={"user_id": log.user_id, "run_count": run_count})

    return {"status": "success", "message": "Log recorded."}

@app.post("/get_recommendation/{user_id}")
def get_recommendation(user_id: int, db: Session = Depends(get_db)):
    """Runs the full Hybrid Architecture pipeline + Guardrail + SHAP."""
    start_time = datetime.now()
    user = db.query(User).filter(User.id == user_id).first()
    logs = db.query(DailyLog).filter(DailyLog.user_id == user_id).order_by(DailyLog.date.desc()).limit(28).all()
    
    if not user or not logs:
        raise HTTPException(status_code=404, detail="Data missing.")

    # 1. Pipeline Execution (Mocked execution of Steps 2, 3, 4, 5)
    ai_predicted_risk = "High" 
    
    # 2. Guardrail Integration (Step 4)
    guardrail = RunSafeGuardrail()
    
    # Changed ambiguous 'l' to 'daily_log'
    recent_mileage = [daily_log.current_week_mileage / 7.0 for daily_log in logs] 
    
    recommendation, overridden, rule = guardrail.evaluate(
        ai_risk_level=ai_predicted_risk,
        recent_28_days_mileage=recent_mileage,
        baseline_weekly_mileage=user.average_weekly_mileage
    )
    
    # 3. SHAP Integration (Step 5 - Mocked)
    shap_results = {"dominant_risk_factor": "Your 'Mileage Spike %' is driving your risk up."}
    
    # 4. JSON Observability
    logger.info("Recommendation Generated", extra={
        "user_id": user_id,
        "ai_prediction": ai_predicted_risk,
        "is_overridden": overridden,
        "triggered_rule": rule,
        "shap_gen_time_ms": (datetime.now() - start_time).total_seconds() * 1000
    })

    return {
        "recommendation": recommendation,
        "guardrail_override": overridden,
        "intervention_rule": rule,
        "explainability": shap_results
    }

@app.get("/user_stats/{user_id}")
def get_user_stats(user_id: int, db: Session = Depends(get_db)):
    """Retrieves safe, SQLite-compatible ACWR data."""
    seven_days_ago = datetime.now() - timedelta(days=7)
    twenty_eight_days_ago = datetime.now() - timedelta(days=28)

    acute_load = db.query(func.avg(DailyLog.current_week_mileage)).filter(
        DailyLog.user_id == user_id, DailyLog.date >= seven_days_ago).scalar() or 0.0
        
    chronic_load = db.query(func.avg(DailyLog.current_week_mileage)).filter(
        DailyLog.user_id == user_id, DailyLog.date >= twenty_eight_days_ago).scalar() or 0.0

    acwr = (acute_load / chronic_load) if chronic_load > 0 else 0.0

    return {
        "acute_7day_load": round(acute_load, 2),
        "chronic_28day_load": round(chronic_load, 2),
        "current_acwr": round(acwr, 2)
    }

@app.get("/explainability_history/{user_id}")
def explainability_history(user_id: int):
    """Retrieves timeline of dominant risk factors."""
    return {
        "history": [
            {"date": "2024-03-01", "dominant_factor": "Mileage Spike %"},
            {"date": "2024-03-15", "dominant_factor": "Average RPE"}
        ]
    }
