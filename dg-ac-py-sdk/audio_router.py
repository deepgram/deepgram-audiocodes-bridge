from __future__ import annotations

import logging

from .audiocodes_handler import AudioCodesHandler, PlayStream
from .deepgram_handler import DeepgramHandler

logger = logging.getLogger(__name__)


class AudioRouter:
    """Routes raw audio bytes between an AudioCodes handler and a Deepgram handler.

    Manages the ``playStream.start`` / ``playStream.chunk`` / ``playStream.stop``
    lifecycle on the AudioCodes side: opens a stream on the first chunk of a new
    agent response and closes it when Deepgram fires ``AgentAudioDone``.

    Args:
        audiocodes_handler: Handler for the inbound AudioCodes call.
        deepgram_handler: Handler for the outbound Deepgram Voice Agent connection.
    """

    def __init__(
        self,
        audiocodes_handler: AudioCodesHandler,
        deepgram_handler: DeepgramHandler,
    ) -> None:
        self._ac = audiocodes_handler
        self._dg = deepgram_handler
        self._current_play_stream: PlayStream | None = None

    async def forward_to_deepgram(self, audio_chunk: bytes) -> None:
        """Forward one decoded caller-audio buffer from AudioCodes to Deepgram."""
        await self._dg.send_audio(audio_chunk)

    async def forward_to_audiocodes(self, audio_chunk: bytes) -> None:
        """Forward one agent-audio buffer from Deepgram to AudioCodes.

        Opens a new ``playStream`` lazily on the first chunk after a flush or
        after an ``AgentAudioDone``.
        """
        if self._current_play_stream is None:
            self._current_play_stream = await self._ac.start_play_stream()
        await self._current_play_stream.write_chunk(audio_chunk)

    async def end_audiocodes_playback(self) -> None:
        """Signal that the current agent turn is complete.

        Sends ``playStream.stop`` on the active stream and clears internal
        state so the next chunk opens a new stream.
        """
        if self._current_play_stream is not None:
            stream = self._current_play_stream
            self._current_play_stream = None
            await stream.end()

    async def flush_audiocodes_playback(self) -> None:
        """Barge-in: immediately stop the active ``playStream`` and discard
        any queued TTS chunks.

        Called by :class:`DeepgramBridge` when Voice Agent emits
        ``UserStartedSpeaking``.
        """
        self._current_play_stream = None
        await self._ac.cancel_current_play_stream()
