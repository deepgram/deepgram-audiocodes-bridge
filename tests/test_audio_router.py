"""Tests for AudioRouter — bidirectional audio routing between handlers."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepgram_audiocodes_bridge.audio_router import AudioRouter
from deepgram_audiocodes_bridge.audiocodes_handler import AudioCodesHandler, PlayStream
from deepgram_audiocodes_bridge.deepgram_handler import DeepgramHandler
from tests.conftest import FakeSocket


def make_router() -> tuple[AudioRouter, MagicMock, MagicMock]:
    """Return (router, mock_ac_handler, mock_dg_handler)."""
    ac = MagicMock(spec=AudioCodesHandler)
    dg = MagicMock(spec=DeepgramHandler)

    fake_stream = MagicMock(spec=PlayStream)
    fake_stream.write_chunk = AsyncMock()
    fake_stream.end = AsyncMock()

    ac.start_play_stream = AsyncMock(return_value=fake_stream)
    ac.cancel_current_play_stream = AsyncMock()
    dg.send_audio = AsyncMock()

    router = AudioRouter(ac, dg)  # type: ignore[arg-type]
    return router, ac, dg


async def test_forward_to_deepgram_calls_send_audio() -> None:
    router, ac, dg = make_router()
    chunk = b"\x01\x02\x03"
    await router.forward_to_deepgram(chunk)
    dg.send_audio.assert_called_once_with(chunk)  # type: ignore[union-attr]


async def test_forward_to_audiocodes_opens_stream_on_first_chunk() -> None:
    router, ac, dg = make_router()
    await router.forward_to_audiocodes(b"\xaa\xbb")
    ac.start_play_stream.assert_called_once()  # type: ignore[union-attr]


async def test_forward_to_audiocodes_reuses_stream_for_subsequent_chunks() -> None:
    router, ac, dg = make_router()
    await router.forward_to_audiocodes(b"\x01")
    await router.forward_to_audiocodes(b"\x02")
    # start_play_stream should only be called once (stream is reused)
    ac.start_play_stream.assert_called_once()  # type: ignore[union-attr]


async def test_forward_to_audiocodes_writes_chunk_to_stream() -> None:
    router, ac, dg = make_router()
    chunk = b"\xde\xad"
    await router.forward_to_audiocodes(chunk)
    stream = ac.start_play_stream.return_value  # type: ignore[union-attr]
    stream.write_chunk.assert_called_once_with(chunk)


async def test_end_audiocodes_playback_ends_stream() -> None:
    router, ac, dg = make_router()
    await router.forward_to_audiocodes(b"\x01")
    stream = ac.start_play_stream.return_value  # type: ignore[union-attr]
    await router.end_audiocodes_playback()
    stream.end.assert_called_once()


async def test_end_audiocodes_playback_clears_state_for_next_chunk() -> None:
    router, ac, dg = make_router()
    await router.forward_to_audiocodes(b"\x01")
    await router.end_audiocodes_playback()

    # A new stream should be opened on the next forward_to_audiocodes call
    new_stream = MagicMock(spec=PlayStream)
    new_stream.write_chunk = AsyncMock()
    new_stream.end = AsyncMock()
    ac.start_play_stream.return_value = new_stream  # type: ignore[union-attr]

    await router.forward_to_audiocodes(b"\x02")
    assert ac.start_play_stream.call_count == 2  # type: ignore[union-attr]


async def test_flush_audiocodes_playback_calls_cancel() -> None:
    router, ac, dg = make_router()
    await router.forward_to_audiocodes(b"\x01")
    await router.flush_audiocodes_playback()
    ac.cancel_current_play_stream.assert_called_once()  # type: ignore[union-attr]


async def test_flush_audiocodes_playback_clears_state_for_next_chunk() -> None:
    router, ac, dg = make_router()
    await router.forward_to_audiocodes(b"\x01")
    await router.flush_audiocodes_playback()

    new_stream = MagicMock(spec=PlayStream)
    new_stream.write_chunk = AsyncMock()
    new_stream.end = AsyncMock()
    ac.start_play_stream.return_value = new_stream  # type: ignore[union-attr]

    await router.forward_to_audiocodes(b"\x02")
    assert ac.start_play_stream.call_count == 2  # type: ignore[union-attr]
