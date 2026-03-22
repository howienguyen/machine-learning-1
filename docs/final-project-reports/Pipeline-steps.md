Pipeline steps:

python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
    --mode stage1 \
    --stage1a-sources fma \
    --stage1b-sources fma


python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
    --mode stage1 \
    --stage1a-sources additional \
    --stage1b-sources additional

python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
	--mode stage2

python MelCNN-MGR/preprocessing/2_build_log_mel_dataset.py