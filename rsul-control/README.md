# PulseOS RSUL Control (Person 3)

Person 3 predicts future runtime health, failure risk and remaining software useful life (RSUL), then produces a maintenance recommendation.

## Person 2 adapter

`src/rsul_control/adapter/person2_adapter.py` accepts two input contracts:

1. **Detailed contract:** the agreed 47-field Person 2 telemetry interface.
2. **Legacy PulseOS contract:** `timestamp`, `shi`, `health_state`, `degradation_rate`, `degradation_acceleration`, `root_cause`, `cause_confidence`.

The adapter does **not** invent missing detailed telemetry in legacy mode. Legacy mode is an explicit compatibility fallback and predicts time to the critical SHI band using only the fields actually supplied.

## Run locally

```powershell
cd rsul-control
pip install -r requirements.txt
$env:PYTHONPATH="src"
python predict.py --row 100
```

API:

```powershell
$env:PYTHONPATH="src"
uvicorn rsul_control.api.main:app --reload
```

Then POST a Person 2 JSON payload to `/predict`.

## Train

The synthetic training data is in:

`data/synthetic/person2_dummy_training_2000.csv`

Train the detailed models with:

```powershell
$env:PYTHONPATH="src"
python train.py --input data/synthetic/person2_dummy_training_2000.csv
```

The synthetic dataset is for prototype/demo training. It is not claimed to be measured ESP32-S3 telemetry.
