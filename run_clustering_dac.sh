source venv/bin/activate
N_CLUSTERS=1000
TYPE=dac
CKPT_PATH=./model.pt
LAYER=0
MANIFEST=/mnt/hdd/datasets/libritts/manifest_train.txt
KM_MODEL_PATH=./kmeans_model_dac_1000.pkl
FEATURE_PATH=/mnt/hdd/GSLM/features_dac.npy

PYTHONPATH=. python examples/textless_nlp/gslm/speech2unit/clustering/cluster_kmeans.py \
	--num_clusters $N_CLUSTERS \
	--feature_type $TYPE \
	--checkpoint_path $CKPT_PATH \
	--layer $LAYER \
	--manifest_path $MANIFEST \
	--out_kmeans_model_path $KM_MODEL_PATH \
	--out_features_path $FEATURE_PATH \
	--sample_pct 0.05
