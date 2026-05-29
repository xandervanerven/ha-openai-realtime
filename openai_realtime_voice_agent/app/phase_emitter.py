"""Emit va_client phase messages from Pipecat speaking frames.

The Home Assistant Voice PE firmware (maxmaxme `va_client` component) drives its
LED ring, mic-streaming gate and a 7 s no-speech watchdog from `phase` JSON
messages sent by the backend:

    {"type": "phase", "value": "listening" | "thinking" | "replying" | "idle"}

Without these messages the device aborts each turn after the watchdog fires, so
emitting them is required (not just cosmetic). This processor maps Pipecat's
standard speaking frames onto those phases and forwards them to the device over
the websocket as TEXT frames.

Mapping:
    UserStartedSpeakingFrame  -> listening   (server VAD heard the user)
    UserStoppedSpeakingFrame  -> thinking    (generating a response)
    BotStartedSpeakingFrame   -> replying    (TTS audio is playing)
    BotStoppedSpeakingFrame   -> idle, but DEBOUNCED (see below)

IMPORTANT — idle debounce:
    OpenAI Realtime TTS arrives in segments (per sentence, and around tool
    calls), so BotStoppedSpeakingFrame fires several times *within a single
    reply*, with sub-second gaps before the next BotStartedSpeakingFrame. If we
    emitted "idle" on every BotStoppedSpeakingFrame the device's LED would flap
    replying -> idle -> replying mid-answer, and — because the firmware only
    arms the "stop" wake word during "replying" — the user would briefly lose
    the ability to interrupt. So we do NOT go idle immediately on BotStopped:
    we arm a short timer and only emit "idle" if no further bot/user speech
    starts before it elapses (i.e. the reply has truly finished). Any
    Bot/UserStartedSpeaking cancels the pending idle.

A barge-in mid-reply surfaces as a fresh UserStartedSpeakingFrame -> "listening"
(which cancels the pending idle); the firmware uses that to flush playback.
"""
import asyncio
import logging
import os

from pipecat.frames.frames import (
    Frame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

logger = logging.getLogger(__name__)


class PhaseEmitter(FrameProcessor):
    """Forwards phase transitions to the device as JSON text frames."""

    def __init__(self, send_phase, idle_debounce_s: float = None, **kwargs):
        """
        Args:
            send_phase: async callable(value: str) that delivers the phase to
                the connected device(s).
            idle_debounce_s: seconds the bot must stay silent after a reply
                before we declare the turn idle. Defaults to the
                PHASE_IDLE_DEBOUNCE_MS env var (1500 ms) — long enough to bridge
                the inter-sentence / tool-call gaps in OpenAI Realtime TTS so the
                LED and the "stop" wake word stay active for the whole answer.
        """
        super().__init__(**kwargs)
        self._send_phase = send_phase
        if idle_debounce_s is None:
            try:
                idle_debounce_s = float(os.environ.get("PHASE_IDLE_DEBOUNCE_MS", "1500")) / 1000.0
            except (TypeError, ValueError):
                idle_debounce_s = 1.5
        self._idle_debounce_s = max(0.0, idle_debounce_s)
        self._idle_task = None
        self._current = None  # last phase actually sent, to dedupe redundant emits

    async def _emit(self, value: str) -> None:
        if value == self._current:
            return
        self._current = value
        if self._send_phase is not None:
            try:
                await self._send_phase(value)
            except Exception as e:  # never let UI signalling break the audio path
                logger.warning(f"⚠️ Failed to emit phase '{value}': {e}")

    def _cancel_pending_idle(self) -> None:
        if self._idle_task is not None and not self._idle_task.done():
            self._idle_task.cancel()
        self._idle_task = None

    async def _emit_idle_after_debounce(self) -> None:
        try:
            await asyncio.sleep(self._idle_debounce_s)
        except asyncio.CancelledError:
            return
        await self._emit("idle")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._cancel_pending_idle()
            await self._emit("listening")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._cancel_pending_idle()
            await self._emit("thinking")
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._cancel_pending_idle()
            await self._emit("replying")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # Don't go idle immediately — TTS comes in segments. Only emit idle
            # if the bot stays silent for the debounce window.
            self._cancel_pending_idle()
            self._idle_task = asyncio.create_task(self._emit_idle_after_debounce())

        await self.push_frame(frame, direction)
