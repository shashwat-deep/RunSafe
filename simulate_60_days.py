import random
from typing import List

class RunSafeSimulator:
    """
    Simulates a 60-day user journey to demonstrate the ACWR backfill phase-out
    and the asynchronous N=1 PyTorch fine-tuning triggers.
    """
    def __init__(self, baseline_weekly_mileage: float):
        self.baseline_weekly = baseline_weekly_mileage
        self.daily_logs: List[float] = []
        self.active_run_count = 0
        
    def calculate_acwr_with_backfill(self) -> tuple[float, str]:
        """Matches the exact math from Step 4 Guardrails."""
        days_passed = len(self.daily_logs)
        if days_passed == 0:
            return 0.0, "No Data"
            
        # Acute Load (7 days)
        acute_logs = self.daily_logs[-7:]
        acute_daily_avg = sum(acute_logs) / 7.0 
        
        # Chronic Load Calculation
        actual_chronic_daily = sum(self.daily_logs) / days_passed
        baseline_daily = self.baseline_weekly / 7.0
        
        # Linear Weighting for Phase-Out
        if days_passed < 28:
            baseline_weight = (28 - days_passed) / 28.0
            actual_weight = 1.0 - baseline_weight
            blended_chronic_daily = (baseline_daily * baseline_weight) + (actual_chronic_daily * actual_weight)
            
            phase_status = f"Phase-Out Active: {baseline_weight*100:.0f}% Profile / {actual_weight*100:.0f}% Actual"
        else:
            blended_chronic_daily = actual_chronic_daily
            phase_status = "100% Actual Data Used"

        if blended_chronic_daily == 0:
            return 0.0, phase_status
            
        return (acute_daily_avg / blended_chronic_daily), phase_status

    def simulate_60_days(self):
        print("="*60)
        print(f"RUNSAFE 60-DAY SIMULATION")
        print(f"User Stated Baseline: {self.baseline_weekly} miles/week (~{self.baseline_weekly/7:.1f} mi/day)")
        print("="*60 + "\n")
        
        for day in range(1, 61):
            # 1. Simulate Runner Behavior (30% chance of a rest day)
            is_rest_day = random.random() < 0.3
            if is_rest_day:
                daily_mileage = 0.0
            else:
                # Simulate a run around their baseline, with occasional spikes
                daily_mileage = random.uniform(2.0, 6.0)
                self.active_run_count += 1
            
            self.daily_logs.append(daily_mileage)
            
            # 2. Calculate ACWR & Check Phase-Out Status
            acwr, phase_status = self.calculate_acwr_with_backfill()
            
            # 3. Print Logs (Print daily for first 7 days, then every 7 days to keep it clean)
            if day <= 7 or day % 7 == 0 or day == 28:
                print(f"Day {day:02d} | Logged: {daily_mileage:4.1f} mi | ACWR: {acwr:4.2f} | Context: {phase_status}")
                if day == 28:
                    print("-" * 60)
                    print(">>> [MILESTONE] Day 28 Reached. Backfill phase-out complete.")
                    print("-" * 60)
                
            # 4. Check N=1 PyTorch Trigger (Every 5 active runs)
            # We only trigger on days where they actually ran, and hit a multiple of 5
            if not is_rest_day and self.active_run_count % 5 == 0:
                print(f"  ⚡ [BACKGROUND TASK] 5 Active Runs Logged (Total: {self.active_run_count}). Triggering N=1 PyTorch Replay Buffer & Training...")

if __name__ == "__main__":
    # Simulate a runner who claims they run 21 miles a week
    sim = RunSafeSimulator(baseline_weekly_mileage=21.0)
    sim.simulate_60_days()
