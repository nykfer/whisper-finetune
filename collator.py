from dataclasses import dataclass
from typing import Any, Dict, List

import soundfile as sf
import soxr
import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    sampling_rate: int = 16000

    def load_audio(self, path):
        audio, sr = sf.read(path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sampling_rate:
            audio = soxr.resample(audio, sr, self.sampling_rate, quality='LQ')
        return audio

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audios = [self.load_audio(f['filename']) for f in features]
        batch = self.processor.feature_extractor(
            audios, sampling_rate=self.sampling_rate, return_tensors='pt'
        )

        labels_batch = self.processor.tokenizer(
            [f['text'] for f in features], padding=True, return_tensors='pt'
        )
        labels = labels_batch['input_ids'].masked_fill(labels_batch['attention_mask'].ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch['labels'] = labels
        return batch
