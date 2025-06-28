CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --env LIBERO --port 8000 &

CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py --env LIBERO --port 10000 &
