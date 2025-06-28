CUDA_VISIBLE_DEVICES=3 python examples/libero/main.py \
    --args.lang_transform_type no_transform \
    --args.num_rephrase_candidates 1 \
    --args.port 10000 &

CUDA_VISIBLE_DEVICES=4 python examples/libero/main.py \
    --args.lang_transform_type rephrase \
    --args.num_rephrase_candidates 1 \
    --args.port 10000 &

CUDA_VISIBLE_DEVICES=5 python examples/libero/main.py \
    --args.lang_transform_type rephrase \
    --args.num_rephrase_candidates 5 \
    --args.port 10000 &

CUDA_VISIBLE_DEVICES=6 python examples/libero/main.py \
    --args.lang_transform_type rephrase \
    --args.num_rephrase_candidates 15 \
    --args.port 10000 &

CUDA_VISIBLE_DEVICES=7 python examples/libero/main.py \
    --args.lang_transform_type rephrase \
    --args.num_rephrase_candidates 25 \
    --args.port 10000 &
