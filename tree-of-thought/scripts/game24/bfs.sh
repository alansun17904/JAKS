#python run.py \
#    --task game24 \
#    --task_start_index 900 \
#    --task_end_index 1000 \
#    --method_generate propose \
#    --method_evaluate value \
#    --method_select greedy \
#    --n_evaluate_sample 3 \
#    --n_select_sample 1 \
#    ${@}

export PYTHONPATH="$(pwd)/tree-of-thought:${PYTHONPATH:-}"


python tree-of-thought/run.py \
      --backend gpt2 \
      --task game24  \
      --method_generate propose \
      --method_evaluate value \
      --method_select sample \
      --task_start_index 1 \
      --task_end_index 2 \
       "${@}"

#python run.py \
#      --backend meta-llama/llama-3.2-3B-Instruct\
#      --task game24  \
#      --method_generate propose \
#      --method_evaluate value \
#      --task_start_index 1 \
#      --task_end_index 2 \
#       "${@}"

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
value
--n_generate_sample
2
COMMENT