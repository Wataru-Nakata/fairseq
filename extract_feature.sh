source venv/bin/activate
TYPE=$1
CKPT_PATH=./hubert_base_ls960.pt
LAYER=$2
KM_MODEL_PATH=$3
CLUSTERS=$4
NUM_SAMPLES_PER_MANIFEST=100000
for subset in train dev test; do
	MANIFEST=/mnt/hdd/datasets/libritts/manifest_${subset}.txt
	OUT_QUANTIZED_FILE=/mnt/hdd/datasets/libritts/"$TYPE"_quantized_l"$LAYER"_km"$CLUSTERS"_"$subset".txt
	tail -n +2 $MANIFEST | split -l $NUM_SAMPLES_PER_MANIFEST - split_"$TYPE"_l"$LAYER"_km"$CLUSTERS"_"$subset"
	for file in split_"$TYPE"_l"$LAYER"_km"$CLUSTERS"_"$subset"*; do
		head -n 1 $MANIFEST >tmp_file
		cat "$file" >>tmp_file
		mv -f tmp_file "$file"
	done

	for file in split_"$TYPE"_l"$LAYER"_km"$CLUSTERS"_"$subset"*; do
		echo $file
		OUT_QUANTIZED_FILE_2=$OUT_QUANTIZED_FILE.$file
		PYTHONPATH=. python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
			--feature_type $TYPE \
			--acoustic_model_path $CKPT_PATH \
			--layer $LAYER \
			--manifest_path $file \
			--out_quantized_file_path $OUT_QUANTIZED_FILE_2 \
			--kmeans_model_path $KM_MODEL_PATH \
			--extension ".flac"
	done
	rm split_"$TYPE"_l"$LAYER"_km"$CLUSTERS"_"$subset"*
done
