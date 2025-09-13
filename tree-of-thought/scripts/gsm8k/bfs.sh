python tree-of-thought/run.py \
      --backend meta-llama/llama-3.2-3B-Instruct \
      --task gsm8k  \
      --method_generate propose \
      --method_evaluate value \
      --method_select sample \
      --task_start_index 1 \
      --task_end_index 2 \
       "${@}"