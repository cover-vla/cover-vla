# CUDA_VISIBLE_DEVICES=0 python examples/libero/main.py \
#     --args.lang_transform_type no_transform \
#     --args.num_rephrase_candidates 1 \
#     --args.port 8000

# python examples/libero/main.py \
#     --args.lang_transform_type rephrase \
#     --args.num_rephrase_candidates 1 \
#     --args.port 8000

# python examples/libero/main.py \
#     --args.lang_transform_type rephrase \
#     --args.num_rephrase_candidates 5 \
#     --args.port 8000

# python examples/libero/main.py \
#     --args.lang_transform_type rephrase \
#     --args.num_rephrase_candidates 15 \
#     --args.port 8000

python examples/libero/main.py \
    --args.lang_transform_type rephrase \
    --args.num_rephrase_candidates 25 \
    --args.port 8000

