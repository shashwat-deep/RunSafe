import pytest
from guardrail import RunSafeGuardrail  # Importing the actual class from Step 4

def test_safe_acwr_trusted():
    """Verify that a safe ACWR (< 1.5) trusts the AI's recommendation."""
    guardrail = RunSafeGuardrail(acwr_threshold=1.5)
    
    ai_prediction = "Low"
    baseline_weekly = 28.0 # ~4 miles/day
    recent_logs = [4.0] * 28 # Perfectly consistent 4 miles/day (ACWR = 1.0)
    
    rec, override, rule = guardrail.evaluate(ai_prediction, recent_logs, baseline_weekly)
    
    assert override is False
    assert rule == "None (ML Output Trusted)"
    assert rec == "Safe to increase training load up to 10%."

def test_extreme_acwr_override():
    """Verify that a dangerous ACWR (> 1.5) strictly overrides a 'Low Risk' AI prediction."""
    guardrail = RunSafeGuardrail(acwr_threshold=1.5)
    
    ai_prediction = "Low" # AI incorrectly thinks user is safe
    baseline_weekly = 14.0 # ~2 miles/day
    
    # 21 days of 2 miles, followed by a massive 7-day spike of 10 miles/day
    recent_logs = ([2.0] * 21) + ([10.0] * 7) 
    
    rec, override, rule = guardrail.evaluate(ai_prediction, recent_logs, baseline_weekly)
    
    assert override is True
    assert "ACWR_EXCEEDED" in rule
    # Must override the AI's "Low" and force the "High" recommendation
    assert rec == "Reduce training load by 20% or take a rest day."

def test_new_user_backfilling():
    """Verify the linear backfilling safely protects new users with little data."""
    guardrail = RunSafeGuardrail(acwr_threshold=1.5)
    
    ai_prediction = "Medium"
    baseline_weekly = 21.0 # 3 miles/day expected
    
    # User just started using the app: only 3 days logged, but ran 12 miles each day!
    logs_first_3_days = [12.0, 12.0, 12.0]
    
    rec, override, rule = guardrail.evaluate(ai_prediction, logs_first_3_days, baseline_weekly)
    
    # The backfill mechanism should catch this massive acute spike against the baseline
    assert override is True
    assert "ACWR_EXCEEDED" in rule