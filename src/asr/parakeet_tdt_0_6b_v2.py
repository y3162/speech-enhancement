from types import SimpleNamespace
from typing import Sequence, Tuple

from omegaconf import open_dict
import torch
import torch.nn as nn
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.numba.rnnt_loss import TDTLossNumba
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures
from nemo.core.utils import numba_utils
from nemo.utils import logging as nemo_logging
nemo_logging.set_verbosity(nemo_logging.ERROR)
numba_utils.set_numba_compat_strictness(False)


class FrozenParakeetTDT06BV2(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name='nvidia/parakeet-tdt-0.6b-v2')
        self.model.preprocessor.featurizer.use_grads = True
        tdt_kwargs = self.model.cfg.loss.tdt_kwargs
        self.tdt_loss = TDTLossNumba(
            blank=int(self.model.loss._blank),
            durations=[int(duration) for duration in tdt_kwargs.durations],
            reduction='none',
            sigma=float(tdt_kwargs.sigma),
            omega=float(tdt_kwargs.get('omega', 0.0)),
        )
        decoding_cfg = self.model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = 'greedy_batch'
            decoding_cfg.compute_timestamps = False
            decoding_cfg.tdt_include_token_duration = True
        self.model.change_decoding_strategy(decoding_cfg)
        self.samples_per_encoder_frame = (
            int(self.model.preprocessor.featurizer.hop_length)
            * int(self.model.encoder.subsampling_factor)
        )
        self.eval()

    def train(
        self,
        mode: bool = True,
    ):
        super().train(mode)
        self.model.eval()
        self._set_decoder_backward_enabled(mode)
        for parameter in self.model.parameters():
            parameter.requires_grad_(self.training)
        return self

    def _set_decoder_backward_enabled(
        self,
        enabled: bool,
    ):
        if enabled:
            self.model.decoder.train()
            for module in self.model.decoder.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()
        else:
            self.model.decoder.eval()

    def _tokenize_batch(
        self,
        texts: Sequence[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token_ids = [self.model.tokenizer.text_to_ids(text) for text in texts]
        target_lengths = torch.tensor([len(ids) for ids in token_ids], device=device, dtype=torch.int64)
        max_length = int(target_lengths.max().item())
        padded = torch.zeros((len(token_ids), max_length), device=device, dtype=torch.int64)
        for i, ids in enumerate(token_ids):
            if ids:
                padded[i, :len(ids)] = torch.tensor(ids, device=device, dtype=torch.int64)
        return padded, target_lengths

    def _to_int_list(
        self,
        value,
    ) -> list:
        if value is None:
            return []
        if torch.is_tensor(value):
            if value.numel() == 0:
                return []
            return [int(v) for v in value.detach().cpu().tolist()]
        return [int(v) for v in value]

    def _timed_recognition(
        self,
        prediction,
        encoded_length: int,
        return_text: bool = True,
    ) -> SimpleNamespace:
        token_ids = self._to_int_list(prediction.y_sequence)
        starts = self._to_int_list(prediction.timestamp)
        durations = self._to_int_list(getattr(prediction, 'token_duration', None))
        if not (len(token_ids) == len(starts) == len(durations)):
            raise RuntimeError(
                'y_sequence/timestamp/token_duration lengths differ: '
                f'{len(token_ids)}, {len(starts)}, {len(durations)}'
            )
        blank_id = int(self.model.loss._blank)
        kept = []
        for token_id, start, duration in zip(token_ids, starts, durations):
            if token_id >= blank_id:
                continue
            start_offset = max(int(start), 0)
            if start_offset >= encoded_length:
                continue
            end_offset = min(start_offset + max(int(duration), 1), encoded_length)
            kept.append((token_id, start_offset, end_offset))
        if return_text:
            token_texts = self.model.decoding.decode_ids_to_tokens(
                [token_id for token_id, _, _ in kept]
            )
            text = prediction.text
        else:
            token_texts = [''] * len(kept)
            text = ''
        tokens = [
            SimpleNamespace(
                token_id=token_id,
                token=token_text,
                start_offset=start_offset,
                end_offset=end_offset,
            )
            for (token_id, start_offset, end_offset), token_text in zip(kept, token_texts)
        ]
        frame_token_index = [-1] * encoded_length
        for token_index, token in enumerate(tokens):
            for frame in range(token.start_offset, token.end_offset):
                frame_token_index[frame] = token_index
        return SimpleNamespace(
            text=text,
            encoded_length=encoded_length,
            samples_per_encoder_frame=self.samples_per_encoder_frame,
            tokens=tokens,
            frame_token_index=frame_token_index,
        )

    def _empty_recognition(
        self,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            text='',
            encoded_length=0,
            samples_per_encoder_frame=self.samples_per_encoder_frame,
            tokens=[],
            frame_token_index=[],
        )

    def loss(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        texts: Sequence[str],
    ) -> torch.Tensor:
        batch_size = waveforms.shape[0]
        token_id_lists = [self.model.tokenizer.text_to_ids(text) for text in texts]
        valid = (lengths > 0) & torch.tensor(
            [len(ids) > 0 for ids in token_id_lists],
            device=lengths.device,
            dtype=torch.bool,
        )
        if not bool(valid.any()):
            return waveforms.new_zeros(batch_size) + waveforms.sum() * 0
        idx = None
        if not bool(valid.all()):
            idx = valid.nonzero(as_tuple=False).squeeze(1)
            waveforms = waveforms.index_select(0, idx)
            lengths = lengths.index_select(0, idx)
            texts = [texts[i] for i in idx.tolist()]
        targets, target_lengths = self._tokenize_batch(texts, device=waveforms.device)
        features, feature_lengths = FilterbankFeatures.forward(
            self.model.preprocessor.featurizer,
            waveforms.to(dtype=torch.float32),
            lengths,
        )
        encoded, encoded_lengths = self.model.forward(
            processed_signal=features,
            processed_signal_length=feature_lengths,
        )
        decoder_outputs, _, _ = self.model.decoder(
            targets=targets,
            target_length=target_lengths,
        )
        logits = self.model.joint.joint(
            encoded.transpose(1, 2),
            decoder_outputs.transpose(1, 2),
        )
        nll_valid = self.tdt_loss(
            acts=logits,
            labels=targets.to(dtype=torch.int64),
            act_lens=encoded_lengths.to(dtype=torch.int64),
            label_lens=target_lengths.to(dtype=torch.int64),
        )
        if idx is None:
            return nll_valid
        return nll_valid.new_zeros(batch_size).scatter(0, idx, nll_valid)

    def decode(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
    ):
        decoder_training = self.model.decoder.training
        self.model.decoder.eval()
        with torch.no_grad():
            processed_signal, processed_length = self.model.preprocessor(
                input_signal=waveforms.to(dtype=torch.float32),
                length=lengths,
            )
            encoded, encoded_length = self.model.encoder(
                audio_signal=processed_signal,
                length=processed_length,
            )
            predictions = self.model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded,
                encoded_lengths=encoded_length,
                return_hypotheses=True,
            )
        self._set_decoder_backward_enabled(decoder_training)
        return encoded_length, predictions

    def align(
        self,
        encoded_length: torch.Tensor,
        predictions,
        return_text: bool = True,
    ) -> list:
        encoded_lengths = encoded_length.detach().cpu().tolist()
        return [
            self._timed_recognition(prediction, int(encoded_lengths[i]), return_text=return_text)
            for i, prediction in enumerate(predictions)
        ]

    def _encode(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features, feature_lengths = FilterbankFeatures.forward(
            self.model.preprocessor.featurizer,
            waveforms.to(dtype=torch.float32),
            lengths,
        )
        return self.model.forward(
            processed_signal=features,
            processed_signal_length=feature_lengths,
        )

    def _empty_encoded(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = waveforms.shape[0]
        d_model = int(self.model.encoder.d_model)
        encoded = waveforms.new_zeros(batch_size, d_model, 0) + waveforms.sum() * 0
        encoded_length = lengths.new_zeros(batch_size)
        return encoded, encoded_length

    def _scatter_encoded(
        self,
        encoded_valid: torch.Tensor,
        encoded_length_valid: torch.Tensor,
        idx,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx is None:
            return encoded_valid, encoded_length_valid
        d_model = encoded_valid.shape[1]
        time = encoded_valid.shape[2]
        index = idx.view(-1, 1, 1).expand_as(encoded_valid)
        encoded = encoded_valid.new_zeros(batch_size, d_model, time).scatter(0, index, encoded_valid)
        encoded_length = encoded_length_valid.new_zeros(batch_size).scatter(0, idx, encoded_length_valid)
        return encoded, encoded_length

    def _scatter_recognition(
        self,
        timed: list,
        idx,
        batch_size: int,
    ) -> list:
        if idx is None:
            return timed
        results = [self._empty_recognition() for _ in range(batch_size)]
        for source, destination in enumerate(idx.tolist()):
            results[destination] = timed[source]
        return results

    def _rnnt_predictions(
        self,
        encoded: torch.Tensor,
        encoded_length: torch.Tensor,
    ):
        decoder_training = self.model.decoder.training
        self.model.decoder.eval()
        with torch.no_grad():
            predictions = self.model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded.detach(),
                encoded_lengths=encoded_length,
                return_hypotheses=True,
            )
        self._set_decoder_backward_enabled(decoder_training)
        return predictions

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        return_text: bool = True,
        return_recognition: bool = True,
        return_encoded: bool = False,
    ) -> SimpleNamespace:
        """
        return_recognition=True, return_encoded=False
        SimpleNamespace(
            recognition=[
                SimpleNamespace(
                    text='We truthful ones, the nobility in ancient Greece called themselves,',
                    encoded_length=56,
                    samples_per_encoder_frame=1280,
                    tokens=[
                        SimpleNamespace(token_id=248, token='▁We', start_offset=4, end_offset=6),
                        ...,
                    ],
                    frame_token_index=[-1, -1, -1, -1, 0, 0, ..., 23, -1],
                ),
                ...,
            ],
        )
        return_recognition=True, return_encoded=True
        SimpleNamespace(
            recognition=[...],
            encoded=torch.Size([2, 1024, 56]),
            encoded_length=tensor([56, 56], dtype=torch.int64),
        )
        return_recognition=False, return_encoded=True
        SimpleNamespace(
            encoded=torch.Size([2, 1024, 56]),
            encoded_length=tensor([56, 56], dtype=torch.int64),
        )
        """
        if not return_recognition and not return_encoded:
            raise ValueError('at least one of return_recognition and return_encoded must be True')
        batch_size = waveforms.shape[0]
        valid = lengths > 0
        result = SimpleNamespace()
        if not bool(valid.any()):
            if return_recognition:
                result.recognition = [self._empty_recognition() for _ in range(batch_size)]
            if return_encoded:
                result.encoded, result.encoded_length = self._empty_encoded(waveforms, lengths)
            return result
        idx = None
        waveforms_valid = waveforms
        lengths_valid = lengths
        if not bool(valid.all()):
            idx = valid.nonzero(as_tuple=False).squeeze(1)
            waveforms_valid = waveforms.index_select(0, idx)
            lengths_valid = lengths.index_select(0, idx)
        encoded_valid = None
        encoded_length_valid = None
        if return_encoded:
            encoded_valid, encoded_length_valid = self._encode(waveforms_valid, lengths_valid)
            result.encoded, result.encoded_length = self._scatter_encoded(
                encoded_valid,
                encoded_length_valid,
                idx,
                batch_size,
            )
        if return_recognition:
            if return_encoded:
                timed = self.align(
                    encoded_length_valid,
                    self._rnnt_predictions(encoded_valid, encoded_length_valid),
                    return_text=return_text,
                )
            else:
                timed = self.align(*self.decode(waveforms_valid, lengths_valid), return_text=return_text)
            result.recognition = self._scatter_recognition(timed, idx, batch_size)
        return result
