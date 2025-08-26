python tree-of-thought/run.py \
      --backend gpt2 \
      --task game24  \
      --method_generate propose \
      --method_evaluate value \
      --method_select sample \
      --task_start_index 1 \
      --task_end_index 2 \
       "${@}"



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