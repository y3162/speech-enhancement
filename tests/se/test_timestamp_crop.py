import unittest
from types import SimpleNamespace

from src.asr.timestamp_crop import (
    SAMPLES_PER_ENCODER_FRAME,
    crop_transcript,
    join_token_texts,
    slice_tokens_for_crop,
    token_center_sample,
)


def _token(start: int, end: int, text: str = "x", token_id: int = 1) -> SimpleNamespace:
    return SimpleNamespace(start_offset=start, end_offset=end, token=text, token_id=token_id)


class TokenCenterSampleTest(unittest.TestCase):
    def test_half_open_frame_span(self) -> None:
        self.assertEqual(token_center_sample(0, 1), 640.0)
        self.assertEqual(token_center_sample(0, 2), 1280.0)
        self.assertEqual(token_center_sample(5, 8), 6.5 * SAMPLES_PER_ENCODER_FRAME)

    def test_rejects_empty_or_inverted_span(self) -> None:
        with self.assertRaises(ValueError):
            token_center_sample(3, 3)
        with self.assertRaises(ValueError):
            token_center_sample(4, 2)


class SliceTokensForCropTest(unittest.TestCase):
    def test_keeps_tokens_whose_center_is_inside_half_open_crop(self) -> None:
        tokens = [_token(0, 1, "a"), _token(1, 3, "b"), _token(3, 4, "c")]
        crop_start, crop_end = 1280, 3840
        selected = slice_tokens_for_crop(tokens, crop_start, crop_end)
        self.assertEqual([token.token for token in selected], ["b"])
        self.assertEqual(token_center_sample(1, 3), 2560.0)
        self.assertTrue(crop_start <= 2560.0 < crop_end)

    def test_includes_center_on_crop_start_excludes_crop_end(self) -> None:
        on_start = _token(0, 2, "start")
        on_end = _token(2, 4, "end")
        self.assertEqual(token_center_sample(0, 2), 1280.0)
        self.assertEqual(token_center_sample(2, 4), 3840.0)
        selected = slice_tokens_for_crop([on_start, on_end], 1280, 3840)
        self.assertEqual([token.token for token in selected], ["start"])

    def test_preserves_original_order(self) -> None:
        tokens = [_token(6, 7, "c"), _token(0, 1, "a"), _token(3, 4, "b")]
        selected = slice_tokens_for_crop(tokens, 0, 10_000)
        self.assertEqual([token.token for token in selected], ["c", "a", "b"])

    def test_empty_when_no_center_falls_in_crop(self) -> None:
        tokens = [_token(0, 2, "left"), _token(10, 12, "right")]
        self.assertEqual(slice_tokens_for_crop(tokens, 3000, 4000), [])

    def test_padding_region_is_excluded_when_crop_end_is_real_audio(self) -> None:
        tokens = [_token(0, 2, "in"), _token(4, 6, "pad")]
        self.assertEqual(token_center_sample(0, 2), 1280.0)
        self.assertEqual(token_center_sample(4, 6), 6400.0)
        selected = slice_tokens_for_crop(tokens, 0, 4000)
        self.assertEqual([token.token for token in selected], ["in"])


class JoinTokenTextsTest(unittest.TestCase):
    def test_sentencepiece_word_boundaries(self) -> None:
        self.assertEqual(join_token_texts(["\u2581came", "\u2581to", "\u2581a"]), "came to a")
        self.assertEqual(join_token_texts(["\u2581travel", "ing"]), "traveling")
        self.assertEqual(join_token_texts([]), "")

    def test_crop_transcript_joins_selected_tokens(self) -> None:
        tokens = [
            _token(0, 1, "\u2581out"),
            _token(2, 3, "\u2581in"),
            _token(3, 4, "side"),
            _token(8, 9, "\u2581after"),
        ]
        self.assertEqual(crop_transcript(tokens, 2560, 5120), "inside")
