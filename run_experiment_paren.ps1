# Parameters
$model = "EleutherAI/pythia-70m"
$batch_size = 2
$ig_steps = 3
$device = "cuda" # change to "cpu" if you don't have CUDA support

# Print information
Write-Host "Running Circuit Stability experiment (with parentheses) with:" -ForegroundColor Cyan
Write-Host "- Model: $model" -ForegroundColor Cyan
Write-Host "- Batch size: $batch_size" -ForegroundColor Cyan
Write-Host "- IG steps: $ig_steps" -ForegroundColor Cyan
Write-Host "- Device: $device" -ForegroundColor Cyan
Write-Host ""

# Run the experiment
python -m src.experiments.circuit_discovery "$model" output_paren --batch_size $batch_size --dataset bool --data_params "allow_parentheses=True" --device $device --format zero-shot --ig_steps $ig_steps 