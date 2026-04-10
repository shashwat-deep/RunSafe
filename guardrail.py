from typing import Tuple, List, Dict
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RunSafe_Guardrail")

class RunSafeGuardrail:
    """
    Rule-based safety wrapper that intercepts ML predictions and enforces 
    biomechanical constraints, specifically Acute:Chronic Workload Ratio (ACWR).
    """
    
    def __init__(self, acwr_threshold: float = 1.5):
        self.acwr_threshold = acwr_threshold
        
        # Base recommendations linked to ML risk levels
        self.base_recommendations = {
            "Low": "Safe to increase training load up to 10%.",
            "Medium": "Maintain current training load.",
            "High": "Reduce training load by 20% or take a rest day."
        }

    def calculate_acwr_with_backfill(
        self, 
        daily_logs: List[float], 
        baseline_weekly_mileage: float
    ) -> float:
        """
        Calculates the ACWR. For users with < 28 days of data, it backfills 
        the chronic load using a linear weighted average of their stated baseline.
        """
        days_passed = len(daily_logs)
        
        # Calculate 7-Day Acute Load (Average daily load over last 7 days)
        acute_logs = daily_logs[-7:] if days_passed >= 7 else daily_logs
        acute_daily_avg = sum(acute_logs) / 7.0 if acute_logs else 0.0
        
        # Calculate Actual Chronic Load (Average daily load from available logs)
        actual_chronic_daily = sum(daily_logs) / days_passed if days_passed > 0 else 0.0
        
        # Baseline Chronic Load (Expected daily load based on profile)
        baseline_daily = baseline_weekly_mileage / 7.0
        
        # Linear Weighting for Phase-Out
        if days_passed < 28:
            baseline_weight = (28 - days_passed) / 28.0
            actual_weight = 1.0 - baseline_weight
            
            # Blended Chronic Load
            blended_chronic_daily = (baseline_daily * baseline_weight) + (actual_chronic_daily * actual_weight)
        else:
            # 100% reliance on actual logged data
            blended_chronic_daily = actual_chronic_daily

        # Prevent Division by Zero
        if blended_chronic_daily == 0:
            return 0.0
            
        return acute_daily_avg / blended_chronic_daily

    def evaluate(
        self, 
        ai_risk_level: str, 
        recent_28_days_mileage: List[float], 
        baseline_weekly_mileage: float
    ) -> Tuple[str, bool, str]:
        """
        Intercepts the AI prediction, checks ACWR, and outputs the final safe recommendation.
        
        Returns:
            Tuple: (final_recommendation, is_overridden, triggered_rule)
        """
        # 1. Translate AI output to default recommendation
        if ai_risk_level not in self.base_recommendations:
            raise ValueError(f"Invalid ML Risk Level: {ai_risk_level}")
            
        final_recommendation = self.base_recommendations[ai_risk_level]
        is_overridden = False
        triggered_rule = "None (ML Output Trusted)"

        # 2. Calculate actual ACWR using backfilling logic
        acwr = self.calculate_acwr_with_backfill(recent_28_days_mileage, baseline_weekly_mileage)
        
        # 3. Apply Strict Safety Overrides
        if acwr > self.acwr_threshold:
            is_overridden = True
            triggered_rule = f"ACWR_EXCEEDED (Current ACWR: {acwr:.2f} > Limit: {self.acwr_threshold})"
            
            # Hard override to High Risk recommendation, regardless of what the AI predicted
            final_recommendation = self.base_recommendations["High"]
            
            logger.warning(f"GUARDRAIL TRIGGERED: {triggered_rule}. Overriding ML output ({ai_risk_level}).")

        return final_recommendation, is_overridden, triggered_rule

# --- Example Usage for Testing ---
if __name__ == "__main__":
    guardrail = RunSafeGuardrail()
    
    # Simulating a new user (only 5 days of data) with a huge spike
    mock_ai_prediction = "Low" # AI missed the risk
    user_baseline_weekly = 20.0 # ~2.8 miles/day expected
    logs_first_5_days = [3.0, 4.0, 0.0, 10.0, 8.0] # Massive acute spike

    rec, override, rule = guardrail.evaluate(mock_ai_prediction, logs_first_5_days, user_baseline_weekly)
    
    print(f"Original AI Prediction: {mock_ai_prediction}")
    print(f"Final Recommendation: {rec}")
    print(f"Overridden: {override}")
    print(f"Rule Triggered: {rule}")
