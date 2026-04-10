import torch
import shap
import numpy as np

# Exact features aligning with Objective 1 & 3 inputs
FEATURE_NAMES = [
    "Age", "Average Weekly Mileage", 
    "Previous Week Mileage", "Current Week Mileage", "Mileage Spike %", 
    "Average RPE", "Max Pain Score", 
    "XGBoost Low Risk Prob", "XGBoost Medium Risk Prob", "XGBoost High Risk Prob"
]

def explain_n1_model(model: torch.nn.Module, user_input: torch.Tensor) -> dict:
    """
    Generates SHAP explanations for the N=1 personalized model, targeting 
    the 'High Risk' class to identify specific injury triggers.
    """
    
    # CRITICAL: Must be in eval() mode to disable Dropout layer randomness
    model.eval()
    
    # Ensure tensor is float32 and shape (1, num_features)
    if not isinstance(user_input, torch.Tensor):
        user_input = torch.tensor(user_input, dtype=torch.float32).unsqueeze(0)
    
    # Create a mathematically sound background tensor: 0-load rest day baseline
    # Assuming standard scaling, 0s represent mean/baseline, and we set XGBoost Low Risk to 1.0
    background_data = torch.zeros((1, 10), dtype=torch.float32)
    background_data[0, 7] = 1.0 # Set XGBoost 'Low Risk' to 100%
    
    try:
        # Initialize DeepExplainer
        explainer = shap.DeepExplainer(model, background_data)
        
        # Calculate SHAP values
        # shap_values is a list of length 3 (Low, Med, High). We want index 2 (High Risk)
        shap_values_all = explainer.shap_values(user_input)
        high_risk_shap = shap_values_all[2][0] # Get Class 2, first (and only) sample
        
        # Find the dominant factor driving the High Risk probability UP (max positive SHAP)
        max_idx = np.argmax(high_risk_shap)
        dominant_feature = FEATURE_NAMES[max_idx]
        max_shap_val = high_risk_shap[max_idx]
        
        # Generate Natural Language Translation
        if max_shap_val > 0.01:
            explanation_str = (
                f"The primary factor driving your injury risk up is your '{dominant_feature}'. "
                f"This metric is contributing the most heavily to your current high-risk profile."
            )
        else:
            # If no strong positive drivers, find the strongest protective factor (max negative)
            min_idx = np.argmin(high_risk_shap)
            protective_feature = FEATURE_NAMES[min_idx]
            explanation_str = (
                f"Your injury risk is currently stable. The strongest protective factor "
                f"keeping your risk down right now is your '{protective_feature}'."
            )
            
        return {
            "raw_shap_values": high_risk_shap.tolist(),
            "dominant_risk_factor": explanation_str,
            "feature_names": FEATURE_NAMES
        }
        
    except Exception as e:
        return {"error": f"SHAP calculation failed: {str(e)}"}
