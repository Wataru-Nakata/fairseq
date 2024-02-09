from pathlib import Path
from itertools import chain
import torchaudio
import random
from tqdm import tqdm
libritts_wavs_train = Path("/mnt/hdd/datasets/libritts/LibriTTS/").glob("train*/**/*.wav")
libritts_wavs_dev = Path("/mnt/hdd/datasets/libritts/LibriTTS/").glob("dev*/**/*.wav")
libritts_wavs_test = Path("/mnt/hdd/datasets/libritts/LibriTTS/").glob("test*/**/*.wav")
output_manifest_path = Path("/mnt/hdd/datasets/libritts")

# for data,subset in [(libritts_wavs_train,'train'), (libritts_wavs_dev,'dev'), (libritts_wavs_test,'test')]:
for data,subset in [(libritts_wavs_test,'test')]:
    datas = []
    for wav_path in tqdm(data):
        info:torchaudio.AudioMetaData = torchaudio.info(wav_path)
        datas.append( (str(wav_path.relative_to(output_manifest_path)), info.num_frames))
    with open(output_manifest_path / f"manifest_{subset}.txt", "w") as f:
        for data in datas:
            f.write(f"{data[0]}\t{data[1]}\n")