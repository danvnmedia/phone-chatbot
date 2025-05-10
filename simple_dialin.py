#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

# ------------------------------------------------------------------------
# MODIFIED: May 2025 by Dan Vu
# Purpose: Assessment for Pipecat Phone Chatbot - Silence Detection & Termination
#
# - Added SilenceDetectorProcessor for silence detection and TTS prompts.
# - Graceful call termination after 3 unanswered prompts.
# - Post-call summary logging.
# - See "# --- ADDED: ..." comments for details.
# ------------------------------------------------------------------------
import argparse
import asyncio
import os
import sys
import time

from call_connection_manager import CallConfigManager, SessionManager
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndTaskFrame, TTSSpeakFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyDialinSettings, DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")


class SilenceDetectorProcessor(FrameProcessor):
    """Detects silence, plays TTS after 10s, terminates after 3 unanswered prompts, logs summary."""
    def __init__(self, tts, task, silence_prompt="Are you still there?", silence_timeout=10, max_unanswered=3):
        super().__init__()
        self.tts = tts
        self.task = task
        self.silence_prompt = silence_prompt
        self.silence_timeout = silence_timeout
        self.max_unanswered = max_unanswered
        self.last_speech_time = time.time()
        self.silence_events = 0
        self.unanswered_count = 0
        self.awaiting_response = False
        self.call_start_time = time.time()
        self.call_end_time = None
        self.terminated = False
        self.user_speaking = False
        logger.info(f"[SilenceDetector] Initialized: timeout={silence_timeout}s, max_unanswered={max_unanswered}")

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        now = time.time()
        # Track user speech
        if isinstance(frame, UserStartedSpeakingFrame):
            self.user_speaking = True
            self.last_speech_time = now
            if self.awaiting_response:
                logger.info(f"[SilenceDetector] User started speaking, reset unanswered count.")
                self.unanswered_count = 0
                self.awaiting_response = False
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.user_speaking = False
        # Track transcription
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self.last_speech_time = now
            if self.awaiting_response:
                logger.info(f"[SilenceDetector] User responded, reset unanswered count.")
                self.unanswered_count = 0
                self.awaiting_response = False
        # Detect silence
        if (not self.terminated and not self.awaiting_response and not self.user_speaking and (now - self.last_speech_time) > self.silence_timeout):
            self.silence_events += 1
            self.unanswered_count += 1
            self.awaiting_response = True
            logger.warning(f"[SilenceDetector] SILENCE DETECTED ({self.silence_timeout}s). Playing prompt #{self.unanswered_count}")
            await self.task.queue_frames([TTSSpeakFrame(f"{self.silence_prompt} This is attempt {self.unanswered_count} of {self.max_unanswered}.")])
            # Terminate if max unanswered
            if self.unanswered_count >= self.max_unanswered:
                logger.warning(f"[SilenceDetector] TERMINATING CALL: Max unanswered prompts reached!")
                self.terminated = True
                self.call_end_time = now
                await self.task.queue_frames([TTSSpeakFrame("Maximum number of unanswered prompts reached. Ending the call now. Goodbye!")])
                await self.task.queue_frames([EndTaskFrame()])
                self.log_call_summary()
        await self.push_frame(frame, direction)

    def get_stats(self):
        self.call_end_time = self.call_end_time or time.time()
        return {
            "duration": self.call_end_time - self.call_start_time,
            "silence_events": self.silence_events,
            "unanswered_prompts": self.unanswered_count,
        }

    def log_call_summary(self):
        stats = self.get_stats()
        logger.warning("=" * 60)
        logger.warning("POST-CALL SUMMARY")
        logger.warning("=" * 60)
        logger.warning(f"Call duration: {stats['duration']:.1f} seconds ({stats['duration']/60:.1f} minutes)")
        logger.warning(f"Silence events detected: {stats['silence_events']}")
        logger.warning(f"Unanswered prompts: {stats['unanswered_prompts']}")
        logger.warning(f"Call termination reason: {'Max unanswered prompts' if self.terminated else 'Normal ending'}")
        logger.warning("=" * 60)


async def main(
    room_url: str,
    token: str,
    body: dict,
):
    # ------------ CONFIGURATION AND SETUP ------------

    # Create a config manager using the provided body
    call_config_manager = CallConfigManager.from_json_string(body) if body else CallConfigManager()

    # Get important configuration values
    test_mode = call_config_manager.is_test_mode()

    # Get dialin settings if present
    dialin_settings = call_config_manager.get_dialin_settings()

    # Initialize the session manager
    session_manager = SessionManager()

    # ------------ TRANSPORT SETUP ------------

    # Set up transport parameters
    if test_mode:
        logger.info("Running in test mode")
        transport_params = DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        )
    else:
        daily_dialin_settings = DailyDialinSettings(
            call_id=dialin_settings.get("call_id"), call_domain=dialin_settings.get("call_domain")
        )
        transport_params = DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            dialin_settings=daily_dialin_settings,
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        )

    # Initialize transport with Daily
    transport = DailyTransport(
        room_url,
        token,
        "Simple Dial-in Bot",
        transport_params,
    )

    # Initialize TTS
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="b7d50908-b17c-442d-ad8d-810c63997ed9",  # Use Helpful Woman voice by default
    )

    # ------------ FUNCTION DEFINITIONS ------------

    async def terminate_call(params: FunctionCallParams):
        """Function the bot can call to terminate the call upon completion of a voicemail message."""
        if session_manager:
            # Mark that the call was terminated by the bot
            session_manager.call_flow_state.set_call_terminated()

        # Then end the call
        await params.llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

    # Define function schemas for tools
    terminate_call_function = FunctionSchema(
        name="terminate_call",
        description="Call this function to terminate the call.",
        properties={},
        required=[],
    )

    # Create tools schema
    tools = ToolsSchema(standard_tools=[terminate_call_function])

    # ------------ LLM AND CONTEXT SETUP ------------

    # Set up the system instruction for the LLM
    system_instruction = """You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself. If the user ends the conversation, **IMMEDIATELY** call the `terminate_call` function. """

    # Initialize LLM
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    # Register functions with the LLM
    llm.register_function("terminate_call", terminate_call)

    # Create system message and initialize messages list
    messages = [call_config_manager.create_system_message(system_instruction)]

    # Initialize LLM context and aggregator
    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    # ------------ PIPELINE SETUP ------------

    # Build pipeline
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    # Create pipeline task
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # Add silence detector to pipeline
    silence_detector = SilenceDetectorProcessor(tts=tts, task=task)
    pipeline.processors.insert(1, silence_detector)  # After input

    # ------------ EVENT HANDLERS ------------

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.debug(f"First participant joined: {participant['id']}")
        await transport.capture_participant_transcription(participant["id"])
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.debug(f"Participant left: {participant}, reason: {reason}")
        # Log post-call summary
        silence_detector.call_end_time = time.time()
        silence_detector.log_call_summary()
        await task.cancel()

    # ------------ RUN PIPELINE ------------

    if test_mode:
        logger.debug("Running in test mode (can be tested in Daily Prebuilt)")

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Dial-in Bot")
    parser.add_argument("-u", "--url", type=str, help="Room URL")
    parser.add_argument("-t", "--token", type=str, help="Room Token")
    parser.add_argument("-b", "--body", type=str, help="JSON configuration string")

    args = parser.parse_args()

    # Log the arguments for debugging
    logger.info(f"Room URL: {args.url}")
    logger.info(f"Token: {args.token}")
    logger.info(f"Body provided: {bool(args.body)}")

    asyncio.run(main(args.url, args.token, args.body))