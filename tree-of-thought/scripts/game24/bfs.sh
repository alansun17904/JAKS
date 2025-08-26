python tree-of-thought/run.py \
      --backend google/gemma-2-9b-it \
      --task game24  \
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

# google/gemma-2-9b-it

: <<'COMMENT'
--backend
gpt2
--task
game24
--task_start_index
0
--task_end_index
1
--method_generate
propose
--method_evaluate
value6 m
--n_generate_sample
2
COMMENT