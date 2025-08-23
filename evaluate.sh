cd evaluate

python test_spatbench.py

python test_flickr30k.py \
  --flickr-root /path/to/flickr30k/flickr30k-images \
  --token-file /path/to/results_20130124.token \
  --clip-repo-id demo911/spatialclip \
  --clip-filename iter_5000.pth \
  --depth-model-encoder vitl \
  --depth-ckpt-path /path/to/depth_anything_v2_vitl.pth \
  --device cuda:0 \
  --batch-size 64 \
  --num-workers 4


# python test_coco.py \
#    --data-root "/path/to/your/coco/val2017" \
#     --ann-file "/path/to/your/coco/annotations/captions_val2017.json" \
#     --clip-repo-id "demo911/spatialclip" \
#     --depth-model-encoder "vitl" \
#     --depth-ckpt-path "/path/to/your/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth" \
#     --device "cuda:0" \




