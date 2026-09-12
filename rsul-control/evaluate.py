from pathlib import Path
import numpy as np
import pandas as pd
from src.models import train_models
from src.rsul import crossing_from_forecast

if __name__=="__main__":
 d=pd.read_csv("data/processed/amr_features.csv"); comp,cls,test=train_models(d); Path("results").mkdir(exist_ok=True); comp.to_csv("results/model_comparison.csv",index=False)
 # RSUL evaluation uses each chronological test row's actual threshold crossing in its remaining simulated trajectory.
 threshold=100.; errors=[]
 for i in range(len(test)-1):
   future=test.iloc[i+1:]; actual=crossing_from_forecast(future.runtime_minutes.to_numpy(),future.ai_latency_ms.to_numpy(),threshold)
   slope=max(float(test.iloc[i].get("latency_trend",0)),0); predicted=None if slope==0 else float(test.iloc[i].runtime_minutes)+(threshold-float(test.iloc[i].ai_latency_ms))/slope
   if actual is not None and predicted is not None: errors.append(abs(actual-predicted))
 rsul_mae=float(np.mean(errors)) if errors else float("nan")
 metrics=pd.DataFrame([{**cls,"RSUL_MAE_minutes":rsul_mae}]);metrics.to_csv("results/evaluation_metrics.csv",index=False)
 print("Chronological holdout regression comparison:\n",comp.round(3).to_string(index=False));print("\nClassification and RSUL metrics:\n",metrics.round(3).to_string(index=False))
