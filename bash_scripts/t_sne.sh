python analysis/t_sne.py \
  --test_data GenImage \
  --dataset_keys Midjourney \
  --arch "CLIP:ViT-L/14" \
  --ckpt "./checkpoints/ptrain_3090_0_1/best_model.pth" \
  --result_folder "./results/t_sne/midjourney" \
  --select_k 5 \
  --max_sample 500 \
  --batch_size 128 \
  --p 0.1