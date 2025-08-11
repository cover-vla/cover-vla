export PRISMATIC_DATA_ROOT=. && export PYTHONPATH=.

xvfb-run --auto-servernum -s "-screen 0 640x480x24" \
python experiments/robot/simpler/run_simpler_eval.py \
  --initial_samples 9 \
  --augmented_samples 32