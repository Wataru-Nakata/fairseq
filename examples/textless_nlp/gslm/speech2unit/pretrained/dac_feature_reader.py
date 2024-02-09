import torch
import fairseq
import soundfile as sf
import torch.nn.functional as F
import torchaudio
import transformers


class DACFeatureReader:
    """
    Wrapper class to run inference on NecoBERT model.
    Helps extract features for a given audio file.
    """
    def __init__(self,checkpoint_path, layer, max_chunk=240000, use_cuda=True) -> None:
        super().__init__()
        self.model = transformers.AutoModel.from_pretrained('Wataru/necobert-base-ls', trust_remote_code=True).eval()
        self.sample_rate = 24000
        self.use_cuda = use_cuda
        self.max_chunk = max_chunk
        self.layer = layer
        if self.use_cuda and (torch.cuda.device_count() > 0):
            self.model.cuda()
    def read_audio(self, path, ref_len=None, channel_id=None):
        wav, sr = sf.read(path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(torch.tensor(wav), sr, self.sample_rate).numpy()
            sr = self.sample_rate

        if channel_id is not None:
            assert wav.ndim == 2, \
                f"Expected stereo input when channel_id is given ({path})"
            assert channel_id in [1, 2], \
                "channel_id is expected to be in [1, 2]"
            wav = wav[:, channel_id-1]
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        assert sr == self.sample_rate, sr
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            print(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav
    def get_feats(self, file_path, ref_len=None, channel_id=None):
        x = self.read_audio(file_path, ref_len, channel_id)
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            if self.use_cuda:
                x = x.cuda()
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                _, _, feat_chunk = self.model.preprocessor(x_chunk.view(1,1,-1), 24_000)
                feat_chunk = feat_chunk.transpose(1,2)
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)
