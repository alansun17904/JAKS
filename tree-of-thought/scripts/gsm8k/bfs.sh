python tree-of-thought/run.py \
      --backend meta-llama/llama-3.2-3B-Instruct \
      --task gsm8k  \
      --method_generate propose \
      --method_evaluate circuits \
      --method_select sample \
      --task_start_index 1 \
      --task_end_index 2 \
      --batch_size 1 \
      --ndevices 1 \
      --device "cuda" \
      --seed 42 \
      --dataset thought \
      --format zero-shot \
      --extraction tail
       "${@}"