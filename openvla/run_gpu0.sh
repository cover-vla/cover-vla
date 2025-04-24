# #!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
#     --model_family openvla \
#     --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
#     --task_suite_name libero_spatial \
#     --center_crop True \
#     --clip_model_path /home/xilun/vla-clip/clip_verifier/bash/model_checkpoints/spatial_clip_final.pt 



for beta in 0.05 0.1 0.5 1; do
    CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True \
    --clip_model_path /home/xilun/vla-clip/clip_verifier/bash/model_checkpoints/all_clip_action_vision_epoch_1.pt \
    --use_gradient_optimization True \
    --topk 5 \
    --beta $beta
done

for beta in 0.05 0.1 0.5 1 ; do
    for alignment_text in original transformed; do
        CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial \
        --center_crop True \
        --language_transformation_type rephrase \
        --clip_model_path /home/xilun/vla-clip/clip_verifier/bash/model_checkpoints/all_clip_action_vision_epoch_1.pt \
        --alignment_text $alignment_text \
        --use_gradient_optimization True \
        --topk 5 \
        --beta $beta
    done
done

# for alignment_text in transformed; do
#     CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
#     --model_family openvla \
#     --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
#     --task_suite_name libero_spatial \
#     --center_crop True \
#     --clip_action_iter 5 \
#     --language_transformation_type rephrase \
#     --clip_model_path /home/xilun/vla-clip/clip_verifier/bash/model_checkpoints/spatial_clip_final.pt \
#     --sampling_based_optimization True \
#     --beta 0.2 \
#     --alignment_text $alignment_text

# done

# # loop over beta values
# for beta in 0.5 1 3; do
#     for alignment_text in original transformed; do
#         CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
#         --model_family openvla \
#         --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
#         --task_suite_name libero_spatial \
#         --center_crop True \
#         --clip_action_iter 5 \
#         --language_transformation_type rephrase \
#         --clip_model_path /home/xilun/vla-clip/clip_verifier/bash/model_checkpoints/spatial_clip_final.pt \
#         --sampling_based_optimization True \
#         --beta $beta \
#         --alignment_text $alignment_text
#     done
# done



