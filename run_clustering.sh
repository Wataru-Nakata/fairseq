source venv/bin/activate
N_CLUSTERS=1000
TYPE=hubert
CKPT_PATH=./hubert_base_ls960.pt
LAYER=6
MANIFEST=/mnt/hdd/datasets/libritts/manifest_train.txt
KM_MODEL_PATH=./kmeans_model_hubert_1000.pkl
FEATURE_PATH=/mnt/hdd/GSLM/features_hubert.npy

PYTHONPATH=. python examples/textless_nlp/gslm/speech2unit/clustering/cluster_kmeans.py \
    --num_clusters $N_CLUSTERS \
    --feature_type $TYPE \
    --checkpoint_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_kmeans_model_path $KM_MODEL_PATH \
    --out_features_path $FEATURE_PATH \
    --sample_pct 0.05 
