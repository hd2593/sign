#!/usr/bin/env python3
"""
Integrated ASL + Speech Conversation System
- Starts both ASL and speech recording simultaneously
- Prioritizes ASL input; falls back to speech if no signs detected
- After ASL phrase, starts fresh speech recording for next turn
- Sends ASL phrases individually to cloud
- Maintains full conversation log (speech: transcribed, ASL: GPT-smoothed)

PATCHED VERSION: Memory optimizations to prevent segmentation faults
- Uses "tiny" Whisper model (lower memory footprint)
- Lazy-loads Whisper model (only when needed)
- Limits audio buffer size (prevents unbounded growth)
- Adds garbage collection before transcription
- Memory monitoring for debugging
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import os
import asyncio
import threading
import queue
import re
import gc
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
from RPLCD.i2c import CharLCD

import sounddevice as sd
import whisper
from tensorflow.keras.models import load_model
from picamera2 import Picamera2
import fastapi_poe as fp
from google.cloud import firestore
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech as cloud_speech_types

import tkinter as tk
from tkinter import scrolledtext

# Google Cloud Project ID - set via environment variable
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")


# ============================================================================
# CONFIGURATION
# ============================================================================

# --- SPEECH CONFIG ---
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 0.5
SILENCE_THRESHOLD = 0.01
SILENCE_LIMIT = 2.0

# PATCHED: Use "tiny" model instead of "base" to reduce memory usage
# Options: "tiny" (~39MB), "base" (~74MB), "small" (~244MB)
WHISPER_MODEL = "tiny"

# PATCHED: Maximum audio duration to keep in buffer (seconds)
# This prevents unbounded memory growth during long recordings
MAX_AUDIO_SECONDS = 30

# --- ASL CONFIG ---
SIGNS = ["I", "want", "drink", "water", "hi", "my", "name", "is", "James", 
         "cold", "more", "please", "thank you"]
ASL_MODEL_PATH = '/home/signbridge/Desktop/asl_model(1).h5'
SEQUENCE_LENGTH = 30
NUM_FEATURES = 126
THRESHOLD = 0.6
RECORD_TIME = 2.0
DISPLAY_TIME = 2.0
EXIT_THRESHOLD = 2.0

# --- CAMERA CONFIG ---
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PROCESS_EVERY_N_FRAMES = 2

# --- POE API CONFIG ---
POE_API_KEY = os.environ.get("POE_API_KEY", "0t84ZZMqeFE40inOvFiN17RU0iz13MhDhAKtvjOlsxI")
MODEL = "GPT-4o-Mini"

# ============================================================================
# DATA STRUCTURES
# ============================================================================

# ============================================================================
# LCD CONFIG / HELPERS
# ============================================================================

# Adjust these to match your LCD
LCD_COLS = 16
LCD_ROWS = 2

lcd = None  # will be initialized in init_lcd()


def init_lcd():
    """Initialize the I2C LCD. Non-fatal if LCD is not present."""
    global lcd
    try:
        # Change address=0x27 if your backpack uses a different address
        lcd = CharLCD(
            'PCF8574',
            address=0x27,
            cols=LCD_COLS,
            rows=LCD_ROWS
        )
        lcd.clear()
        lcd.write_string("SignBridge Ready")
        print("✓ LCD initialized")
    except Exception as e:
        lcd = None
        print(f"⚠️ LCD init failed (continuing without LCD): {e}")


def lcd_display_sentence(text: str, scroll_delay: float = 0.3):
    """
    Display a sentence on the LCD.
    - If it fits on the screen, just show it.
    - If it's longer, scroll it automatically.
    Called once per new ASL GPT-rinsed sentence.
    """
    global lcd
    if lcd is None:
        return  # LCD not available, silently skip

    if not text:
        lcd.clear()
        return

    # Collapse newlines / extra spaces so it scrolls nicely
    text = " ".join(text.split())

    total_slots = LCD_COLS * LCD_ROWS
    lcd.clear()

    # Fits without scrolling
    if len(text) <= total_slots:
        padded = text.ljust(total_slots)
        line1 = padded[:LCD_COLS]
        line2 = padded[LCD_COLS:total_slots] if LCD_ROWS > 1 else ""

        lcd.cursor_pos = (0, 0)
        lcd.write_string(line1)

        if LCD_ROWS > 1:
            lcd.cursor_pos = (1, 0)
            lcd.write_string(line2)
        return

    # Too long: scroll across both rows as one continuous window
    scroll_text = text + " " * LCD_COLS  # padding to give a clean end
    window_size = total_slots

    for i in range(len(scroll_text) - window_size + 1):
        window = scroll_text[i:i + window_size]
        line1 = window[:LCD_COLS]
        line2 = window[LCD_COLS:window_size] if LCD_ROWS > 1 else ""

        lcd.home()
        lcd.write_string(line1)

        if LCD_ROWS > 1:
            lcd.cursor_pos = (1, 0)
            lcd.write_string(line2)

        time.sleep(scroll_delay)


class InputSource(Enum):
    SPEECH = "speech"
    ASL = "asl"


@dataclass
class ConversationTurn:
    """Represents one turn in the conversation"""
    source: InputSource
    raw_text: str
    processed_text: str  # Whisper output for speech, GPT-smoothed for ASL
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationState:
    """Tracks the full conversation state"""
    turns: List[ConversationTurn] = field(default_factory=list)
    current_turn_source: Optional[InputSource] = None
    asl_phrases_sent: List[str] = field(default_factory=list)
    
    def add_turn(self, source: InputSource, raw: str, processed: str):
        turn = ConversationTurn(source=source, raw_text=raw, processed_text=processed)
        self.turns.append(turn)
        return turn
    
    def get_full_transcript(self) -> str:
        """Generate formatted conversation transcript"""
        lines = []
        for i, turn in enumerate(self.turns, 1):
            source_label = "🗣️ SPEECH" if turn.source == InputSource.SPEECH else "🤟 ASL"
            lines.append(f"Turn {i} [{source_label}]:")
            lines.append(f"  Raw: {turn.raw_text}")
            lines.append(f"  Processed: {turn.processed_text}")
            lines.append("")
        return "\n".join(lines)


# ============================================================================
# FIRESTORE INTEGRATION
# ============================================================================

db = firestore.Client()


def save_asl_phrase_to_firestore(raw_phrase: str, smooth_phrase: str):
    """Upload individual ASL phrase to Firestore"""
    timestamp = int(time.time())
    raw_words = raw_phrase.lower().split()
    
    # NEW: show the GPT-rinsed sentence on the LCD
    lcd_display_sentence(smooth_phrase)

    # Save message document
    msg_ref = db.collection("messages").add({
        "raw": raw_phrase,
        "smooth": smooth_phrase,
        "phrases": raw_words,
        "timestamp": timestamp,
        "type": "asl_phrase"
    })
    print(f"✓ Saved ASL phrase to Firestore: {msg_ref[1].id}")
    
    # Update phrase frequencies
    for p in raw_words:
        p_ref = db.collection("phrases").document(p)
        p_ref.set({"count": firestore.Increment(1)}, merge=True)
        p_ref.update({"examples": firestore.ArrayUnion([smooth_phrase])})
    
    # Update raw words
    for w in raw_words:
        w_ref = db.collection("raw_words").document(w)
        w_ref.set({"count": firestore.Increment(1)}, merge=True)
        w_ref.update({"examples": firestore.ArrayUnion([raw_phrase])})
    
    print("✓ Phrase and raw-word frequencies updated!")



def save_full_conversation_to_firestore(conversation: ConversationState):
    """Upload the complete conversation to Firestore"""
    timestamp = int(time.time())
    
    turns_data = []
    for turn in conversation.turns:
        turns_data.append({
            "source": turn.source.value,
            "raw": turn.raw_text,
            "processed": turn.processed_text,
            "timestamp": turn.timestamp
        })
    
    doc_ref = db.collection("conversations").add({
        "turns": turns_data,
        "turn_count": len(conversation.turns),
        "transcript": conversation.get_full_transcript(),
        "timestamp": timestamp
    })
    
    print(f"\n✓ Saved full conversation to Firestore: {doc_ref[1].id}")
    return doc_ref[1].id


# ============================================================================
# POE API / GPT INTEGRATION
# ============================================================================

async def query_poe_api(prompt: str) -> str:
    """Send a prompt to POE API and get response"""
    message = fp.ProtocolMessage(role="user", content=prompt)
    full_response = ""
    
    async for partial in fp.get_bot_response(
        messages=[message],
        bot_name=MODEL,
        api_key=POE_API_KEY
    ):
        full_response += partial.text
    
    return full_response.strip()


async def revise_asl_sentence(word_list: List[str]) -> str:
    """Send captured ASL words to GPT-4o-Mini for sentence completion"""
    if not word_list:
        return "No words captured."
    
    raw_sentence = ' '.join(word_list)
    
    prompt = f"""You are helping someone who used ASL (American Sign Language) recognition to capture words.
The system captured these words in sequence: {raw_sentence}

Please revise this into a complete, grammatically correct English sentence. 
The words might be missing articles (a, an, the), prepositions, or proper verb forms since ASL has different grammar than English.

Rules:
1. Keep the meaning as close as possible to the original words
2. Add necessary grammar words (articles, prepositions, etc.)
3. Fix verb tenses if needed
4. Make it a complete sentence
5. Keep it concise (1-2 sentences max)

Only provide the revised sentence, nothing else."""

    try:
        print("\n🤖 Querying GPT-4o-Mini for ASL sentence revision...")
        response = await query_poe_api(prompt)
        return response
    except Exception as e:
        print(f"❌ Error querying GPT: {e}")
        return raw_sentence


# ============================================================================
# PATCHED: MEMORY MONITORING UTILITY
# ============================================================================

def print_memory_status(label: str = ""):
    """
    PATCHED: Print current memory usage for debugging.
    Requires psutil: pip install psutil
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"📊 Memory [{label}]: {mem.percent:.1f}% used "
              f"({mem.used / 1024 / 1024:.0f}MB / {mem.total / 1024 / 1024:.0f}MB)")
    except ImportError:
        pass  # psutil not installed, skip memory monitoring


# ============================================================================
# SPEECH RECOGNITION MODULE - STREAMING WITH "OVER" STOP WORD
# ============================================================================

class SpeechDisplay:
    """
    Simple Tkinter window that shows the current speech transcript
    in a big textbox. Runs its own thread so it doesn't block the
    main program.
    """
    def __init__(self):
        self.root = None
        self.text_widget = None
        self._started = False
        self._start_gui_thread()

    def _start_gui_thread(self):
        if self._started:
            return
        self._started = True

        def run():
            try:
                self.root = tk.Tk()
                self.root.title("Speech Transcript")
                # Big-ish window; tweak as you like
                self.root.geometry("900x500")

                self.text_widget = scrolledtext.ScrolledText(
                    self.root,
                    font=("Helvetica", 28),  # big font
                    wrap="word"
                )
                self.text_widget.pack(expand=True, fill="both")
                self.text_widget.insert("1.0", "Waiting for speech...")
                self.root.mainloop()
            except Exception as e:
                print(f"⚠️ Speech display GUI failed: {e}")

        t = threading.Thread(target=run, daemon=True)
        t.start()

    def show_text(self, text: str):
        """Update the textbox with the latest transcript."""
        if not self.root or not self.text_widget:
            return

        def _update():
            self.text_widget.delete("1.0", tk.END)
            self.text_widget.insert("1.0", text if text else "(no speech detected)")

        # Schedule safely on Tkinter thread
        try:
            self.root.after(0, _update)
        except Exception as e:
            print(f"⚠️ Speech display update failed: {e}")


# Create one global instance (will open the window on first use)
try:
    speech_display = SpeechDisplay()
except Exception as e:
    print(f"⚠️ Could not start SpeechDisplay: {e}")
    speech_display = None


TRANSCRIBE_INTERVAL = 2.0  # Not used in streaming mode
STOP_WORD = "over"
STREAMING_CHUNK_DURATION = 0.1  # 100ms chunks for streaming

def contains_stop_word(text: str) -> bool:
    """
    True only if STOP_WORD appears as a separate word at the END of the text.
    e.g. 'this is my sentence over'
    """
    if not text:
        return False

    # Grab all word tokens, look only at the last one
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return False

    return tokens[-1] == STOP_WORD.lower()


# PATCHED: Calculate max blocks based on MAX_AUDIO_SECONDS
MAX_AUDIO_BLOCKS = int(MAX_AUDIO_SECONDS / BLOCK_DURATION)


def record_and_transcribe_realtime(timeout: float = 30.0) -> str:
    """
    STREAMING SPEECH RECOGNITION using Google Cloud Speech-to-Text v2.
    
    Features:
    - Real-time streaming transcription (not batch)
    - Stops automatically on 2 seconds of silence
    - Stops immediately when "over" is detected
    - Returns cleaned transcript (without "over")
    """
    if not GOOGLE_CLOUD_PROJECT:
        print("❌ Error: GOOGLE_CLOUD_PROJECT environment variable not set!")
        return ""
    
    # Threading primitives for audio streaming
    audio_queue = queue.Queue()
    stop_recording = threading.Event()
    final_transcript = []
    
    # Silence detection state
    silence_start_time = [None]  # Use list for mutability in nested function
    has_heard_speech = [False]
    
    def audio_callback(indata, frames, time_info, status):
        """Callback to capture audio and put in queue"""
        if status:
            print(f"Audio status: {status}")
        if not stop_recording.is_set():
            # Convert to 16-bit PCM bytes
            audio_int16 = np.clip(indata[:, 0], -1.0, 1.0)
            audio_int16 = (audio_int16 * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            audio_queue.put(audio_bytes)
            
            # Check for silence/speech
            rms = np.sqrt(np.mean(np.square(indata)))
            if rms > SILENCE_THRESHOLD:
                has_heard_speech[0] = True
                silence_start_time[0] = None
            else:
                if has_heard_speech[0] and silence_start_time[0] is None:
                    silence_start_time[0] = time.time()
    
    def audio_generator():
        """Generator that yields audio chunks for streaming API"""
        while not stop_recording.is_set():
            try:
                # Get audio chunk with timeout
                chunk = audio_queue.get(timeout=0.1)
                yield cloud_speech_types.StreamingRecognizeRequest(audio=chunk)
            except queue.Empty:
                # Check silence timeout
                if (has_heard_speech[0] and 
                    silence_start_time[0] is not None and 
                    time.time() - silence_start_time[0] > SILENCE_LIMIT):
                    print("\n🔇 Silence detected - stopping")
                    stop_recording.set()
                    break
                continue
    
    def request_generator():
        """Generator that yields config first, then audio chunks"""
        # Recognition config for streaming
        recognition_config = cloud_speech_types.RecognitionConfig(
            explicit_decoding_config=cloud_speech_types.ExplicitDecodingConfig(
                encoding=cloud_speech_types.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=SAMPLE_RATE,
                audio_channel_count=CHANNELS,
            ),
            language_codes=["en-US"],
            model="long",  # Use "long" for better accuracy, or "chirp_2" for Chirp
        )
        
        streaming_config = cloud_speech_types.StreamingRecognitionConfig(
            config=recognition_config,
            streaming_features=cloud_speech_types.StreamingRecognitionFeatures(
                interim_results=True,  # Get interim results for real-time feedback
            ),
        )
        
        # First request: config
        config_request = cloud_speech_types.StreamingRecognizeRequest(
            recognizer=f"projects/{GOOGLE_CLOUD_PROJECT}/locations/global/recognizers/_",
            streaming_config=streaming_config,
        )
        yield config_request
        
        # Subsequent requests: audio chunks
        yield from audio_generator()
    
    print("🎤 Listening (streaming)... say 'over' to finish")
    
    # Start audio recording in separate thread
    chunk_size = int(SAMPLE_RATE * STREAMING_CHUNK_DURATION)
    start_time = time.time()
    
    try:
        # Create the Speech client
        client = SpeechClient()
        
        # Start audio stream
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=chunk_size,
            callback=audio_callback,
        ):
            # Start streaming recognition
            responses = client.streaming_recognize(requests=request_generator())
            
            # Process streaming responses
            for response in responses:
                # Check timeout
                if time.time() - start_time > timeout:
                    print("\n⏱️ Timeout reached")
                    stop_recording.set()
                    break
                    
                
                for result in response.results:
                    if not result.alternatives:
                        continue

                    transcript = result.alternatives[0].transcript

                    if result.is_final:
                        # Final result for this segment
                        print(f"\r✓ Final: {transcript}")

                        # If stop word is at the end, strip it but keep the rest
                        if contains_stop_word(transcript):
                            cleaned = remove_stop_word(transcript)
                            if cleaned:
                                final_transcript.append(cleaned)
                            print(f"\n🛑 Stop word '{STOP_WORD}' detected!")
                            stop_recording.set()
                            break
                        else:
                            final_transcript.append(transcript)
                    else:
                        # Interim result - show in real-time
                        print(f"\r💬 {transcript}", end="", flush=True)
                        
                        try:
                            if speech_display is not None:
                                speech_display.show_text(transcript)
                        except Exception as e:
                            print(f"\n⚠️ GUI interim update error: {e}")
                
                if stop_recording.is_set():
                    break
    
    except Exception as e:
        print(f"\n❌ Streaming recognition error: {e}")
        import traceback
        traceback.print_exc()
        return ""
    
    finally:
        stop_recording.set()
    
    # Combine all final transcripts
    # Combine all final transcripts
    full_transcript = " ".join(final_transcript)
    full_transcript = remove_stop_word(full_transcript)

    print(f"\n📝 Full transcript: {full_transcript}")

    # Show the final transcript in the big textbox
    try:
        if speech_display is not None:
            speech_display.show_text(full_transcript)
    except Exception as e:
        print(f"⚠️ GUI final update error: {e}")

    return full_transcript




def remove_stop_word(transcript: str) -> str:
    if not transcript:
        return ""

    cleaned = re.sub(r'\bover\b[.!?,]*$', '', transcript, flags=re.IGNORECASE)
    return cleaned.strip()


# ============================================================================
# ASL RECOGNITION MODULE
# ============================================================================

class ASLRecognizer:
    """Handles ASL recognition using camera and MediaPipe"""
    
    # States
    STATE_WAITING = "WAITING"
    STATE_RECORDING = "RECORDING"
    STATE_RESULT = "RESULT"
    
    def __init__(self):
        print("Loading ASL model...")
        print_memory_status("before ASL model load")
        self.model = load_model(ASL_MODEL_PATH)
        print_memory_status("after ASL model load")
        
        # Verify model
        num_model_classes = self.model.output_shape[-1]
        if num_model_classes != len(SIGNS):
            print(f"⚠️ WARNING: Model expects {num_model_classes} classes, SIGNS has {len(SIGNS)}")
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        
        # Camera setup
        print("Initializing Picamera2...")
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"},
            controls={"FrameRate": 30, "AfMode": 2, "AfSpeed": 1}
        )
        self.picam2.configure(config)
        self.camera_started = False
        
        # State variables
        self.captured_words = []
        self.last_results = None
        self.frame_count = 0
    
    def _extract_landmarks(self, results) -> List[float]:
        """Extract normalized landmarks from MediaPipe results"""
        frame_data = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[0]
                for landmark in hand_landmarks.landmark:
                    frame_data.extend([
                        landmark.x - wrist.x,
                        landmark.y - wrist.y,
                        landmark.z - wrist.z
                    ])
        padding_len = NUM_FEATURES - len(frame_data)
        frame_data.extend([0.0] * padding_len)
        return frame_data[:NUM_FEATURES]
    
    def start_camera(self):
        """Start the camera if not already started"""
        if not self.camera_started:
            self.picam2.start()
            self.picam2.set_controls({"AfMode": 2, "AfTrigger": 0})
            time.sleep(1)
            self.camera_started = True
    
    def stop_camera(self):
        """Stop the camera"""
        if self.camera_started:
            self.picam2.stop()
            self.camera_started = False
            cv2.destroyAllWindows()
    
    def capture_phrase(self, stop_event: threading.Event = None) -> List[str]:
        """
        Capture a complete ASL phrase (blocking).
        Returns list of captured words.
        """
        self.captured_words = []
        self.frame_count = 0
        
        state = self.STATE_WAITING
        current_frames = []
        first_sign_completed = False
        current_prediction_text = "..."
        start_time = time.time()
        last_hand_time = time.time()
        
        self.start_camera()
        
        try:
            while True:
                if stop_event and stop_event.is_set():
                    break
                
                frame = self.picam2.capture_array()
                self.frame_count += 1
                
                image = cv2.flip(frame, 1)
                should_process = (self.frame_count % PROCESS_EVERY_N_FRAMES == 0)
                
                if should_process:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(image_rgb)
                    self.last_results = results
                else:
                    results = self.last_results
                
                hand_present = bool(results and results.multi_hand_landmarks)
                
                if hand_present:
                    last_hand_time = time.time()
                    for h in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(image, h, self.mp_hands.HAND_CONNECTIONS)
                
                elapsed = time.time() - start_time
                
                # STATE MACHINE
                if state == self.STATE_WAITING:
                    cv2.rectangle(image, (0, 0), (CAMERA_WIDTH, CAMERA_HEIGHT), (255, 0, 0), 10)
                    cv2.putText(image, "RAISE HAND", (200, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    
                    if hand_present:
                        state = self.STATE_RECORDING
                        start_time = time.time()
                        current_frames = []
                        print("ASL Recording started...")
                
                elif state == self.STATE_RECORDING:
                    cv2.rectangle(image, (0, 0), (CAMERA_WIDTH, CAMERA_HEIGHT), (0, 255, 0), 10)
                    cv2.putText(image, f"REC: {RECORD_TIME - elapsed:.1f}", (50, 450),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    if should_process:
                        current_frames.append(self._extract_landmarks(results))
                    
                    # Exit if no hands for too long after first sign
                    if first_sign_completed and (time.time() - last_hand_time > EXIT_THRESHOLD):
                        print("\n✋ No hands detected. ASL phrase complete.")
                        break
                    
                    if elapsed > RECORD_TIME:
                        # Make prediction
                        data = np.array(current_frames)
                        if len(data) > SEQUENCE_LENGTH:
                            input_data = data[:SEQUENCE_LENGTH]
                        else:
                            input_data = np.concatenate((
                                data,
                                np.zeros((SEQUENCE_LENGTH - len(data), NUM_FEATURES))
                            ))
                        
                        res = self.model.predict(np.expand_dims(input_data, axis=0), verbose=0)[0]
                        best_idx = np.argmax(res)
                        confidence = res[best_idx]
                        
                        if best_idx < len(SIGNS) and confidence > THRESHOLD:
                            word = SIGNS[best_idx]
                            self.captured_words.append(word)
                            current_prediction_text = word
                            print(f"🤟 ASL Predicted: {word} ({confidence:.2f})")
                        else:
                            current_prediction_text = "?"
                            print(f"ASL Low confidence: {confidence:.2f}")
                        
                        state = self.STATE_RESULT
                        first_sign_completed = True
                        start_time = time.time()
                
                elif state == self.STATE_RESULT:
                    cv2.rectangle(image, (0, 0), (CAMERA_WIDTH, CAMERA_HEIGHT), (0, 165, 255), 10)
                    cv2.putText(image, current_prediction_text, (200, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 3)
                    
                    if elapsed > DISPLAY_TIME:
                        state = self.STATE_RECORDING
                        start_time = time.time()
                        current_frames = []
                
                # Display captured words
                cv2.putText(image, ' '.join(self.captured_words), (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('ASL Recognition', image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("ASL quit by user")
                    break
        
        except Exception as e:
            print(f"ASL recognition error: {e}")
            import traceback
            traceback.print_exc()
        
        return self.captured_words
    
    def has_content(self) -> bool:
        """Check if any signs were captured"""
        return len(self.captured_words) > 0


# ============================================================================
# INITIAL MODE DETECTION (for first turn)
# ============================================================================

class InitialModeDetector:
    """
    Detects whether user wants to use ASL or Speech for first input.
    Uses a quick detection phase to avoid complex threading.
    
    Strategy:
    1. Start camera and audio simultaneously
    2. Check for 3 seconds: if hand detected -> ASL mode
    3. If no hand but speech detected -> Speech mode
    4. Then capture full input in the chosen mode
    """
    
    DETECTION_TIMEOUT = 3.0  # seconds to wait for initial detection
    
    def __init__(self, asl_recognizer: ASLRecognizer):
        self.asl = asl_recognizer
        self.mp_hands = asl_recognizer.mp_hands
        self.hands = asl_recognizer.hands
    
    def detect_initial_mode(self) -> InputSource:
        """
        Quick detection phase to determine ASL or Speech mode.
        Returns the detected InputSource.
        """
        print("\n🔍 Detecting input mode (raise hand for ASL, speak for voice)...")
        
        # Start camera
        self.asl.start_camera()
        
        # Also start collecting audio in background
        audio_queue = queue.Queue()
        stop_audio = threading.Event()
        has_speech = threading.Event()
        
        def audio_detector():
            """Background thread to detect if speech is happening"""
            def callback(indata, frames, time_info, status):
                audio_queue.put(indata.copy())
            
            block_size = int(SAMPLE_RATE * BLOCK_DURATION)
            try:
                with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="float32",
                    blocksize=block_size,
                    callback=callback,
                ):
                    while not stop_audio.is_set():
                        try:
                            data = audio_queue.get(timeout=0.1)
                            rms = np.sqrt(np.mean(np.square(data), dtype=np.float64))
                            if rms >= SILENCE_THRESHOLD:
                                has_speech.set()
                        except queue.Empty:
                            continue
            except Exception as e:
                print(f"Audio detector error: {e}")
        
        audio_thread = threading.Thread(target=audio_detector, daemon=True)
        audio_thread.start()
        
        # Detection loop
        start_time = time.time()
        hand_detected = False
        
        while time.time() - start_time < self.DETECTION_TIMEOUT:
            frame = self.asl.picam2.capture_array()
            image = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            hand_present = bool(results and results.multi_hand_landmarks)
            
            if hand_present:
                hand_detected = True
                # Draw hand landmarks
                for h in results.multi_hand_landmarks:
                    self.asl.mp_drawing.draw_landmarks(image, h, self.asl.mp_hands.HAND_CONNECTIONS)
            
            # Display status
            remaining = self.DETECTION_TIMEOUT - (time.time() - start_time)
            status = "HAND DETECTED!" if hand_detected else "Waiting..."
            cv2.rectangle(image, (0, 0), (CAMERA_WIDTH, CAMERA_HEIGHT), 
                         (0, 255, 0) if hand_detected else (255, 165, 0), 10)
            cv2.putText(image, f"{status} ({remaining:.1f}s)", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, "Raise hand for ASL or Speak", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            cv2.imshow('Mode Detection', image)
            
            # If hand detected, break early
            if hand_detected:
                time.sleep(0.5)  # Brief pause to confirm
                break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Stop audio detection
        stop_audio.set()
        audio_thread.join(timeout=0.5)
        
        cv2.destroyAllWindows()
        
        # Determine mode
        if hand_detected:
            print("✓ Hand detected - using ASL mode")
            return InputSource.ASL
        elif has_speech.is_set():
            print("✓ Speech detected - using Speech mode")
            self.asl.stop_camera()  # Stop camera since we're doing speech
            return InputSource.SPEECH
        else:
            # Default to ASL if nothing detected
            print("⚠️ No clear input - defaulting to ASL mode")
            return InputSource.ASL


# ============================================================================
# INTEGRATED CONVERSATION SYSTEM
# ============================================================================

class ConversationSystem:
    """Manages the integrated ASL + Speech conversation"""
    
    def __init__(self):
        print_memory_status("before system init")
        self.asl_recognizer = ASLRecognizer()
        self.conversation = ConversationState()
        self.mode_detector = InitialModeDetector(self.asl_recognizer)
        print_memory_status("after system init")
        
    async def capture_first_input(self) -> Tuple[Optional[InputSource], str, str]:
        """
        Capture initial input - first detect mode, then capture in that mode.
        Returns: (source, raw_text, processed_text)
        """
        print("\n" + "=" * 60)
        print("🎤🤟 Starting FIRST TURN - Speak OR Sign!")
        print("=" * 60)
        
        # Quick detection phase
        mode = self.mode_detector.detect_initial_mode()
        
        # Now capture full input in the detected mode
        if mode == InputSource.ASL:
            print("\n🤟 Starting ASL capture...")
            asl_words = self.asl_recognizer.capture_phrase()
            cv2.destroyAllWindows()
            
            if asl_words:
                raw_text = " ".join(asl_words)
                processed_text = await revise_asl_sentence(asl_words)
                save_asl_phrase_to_firestore(raw_text, processed_text)
                return InputSource.ASL, raw_text, processed_text
            else:
                print("⚠️ No ASL captured")
                return None, "", ""
        
        else:  # Speech mode
            print("\n🗣️ Starting Speech capture...")
            print("Say 'over' when you're done speaking.\n")
            
            transcript = record_and_transcribe_realtime(timeout=10.0)
            
            if transcript:
                return InputSource.SPEECH, transcript, transcript
            
            print("⚠️ No speech captured")
            return None, "", ""
    
    async def capture_speech_turn(self) -> Tuple[str, str]:
        """Capture a speech turn with real-time transcription. Returns (raw, processed)."""
        print("\n" + "=" * 60)
        print("🗣️ YOUR TURN TO SPEAK")
        print("=" * 60)
        print("Say 'over' when you're done speaking.\n")
        
        # Real-time transcription (stops on "over")
        transcript = record_and_transcribe_realtime(timeout=10.0)
        
        if transcript:
            return transcript, transcript
        
        print("⚠️ No speech detected")
        return "", ""
    
    async def capture_asl_turn(self) -> Tuple[str, str]:
        """Capture an ASL turn (blocking). Returns (raw, processed)."""
        print("\n" + "=" * 60)
        print("🤟 YOUR TURN TO SIGN")
        print("=" * 60)
        
        # Capture ASL phrase
        asl_words = self.asl_recognizer.capture_phrase()
        cv2.destroyAllWindows()
        
        if asl_words:
            raw_text = " ".join(asl_words)
            processed_text = await revise_asl_sentence(asl_words)
            
            # Send ASL phrase to cloud immediately
            save_asl_phrase_to_firestore(raw_text, processed_text)
            
            return raw_text, processed_text
        
        print("⚠️ No ASL detected")
        return "", ""
    
    async def run_conversation(self, num_turns: int = 5):
        """
        Run the conversation loop.
        First turn: detect mode and capture (ASL priority)
        Subsequent turns: alternate between ASL and Speech
        """
        print("\n" + "=" * 60)
        print("🗣️🤟 INTEGRATED CONVERSATION SYSTEM")
        print("=" * 60)
        print(f"Running for up to {num_turns} turns")
        print("Press 'q' in ASL window or Ctrl+C to end early\n")
        
        try:
            # ========== FIRST TURN ==========
            source, raw, processed = await self.capture_first_input()
            
            if source is None:
                print("No initial input detected. Exiting.")
                return
            
            self.conversation.add_turn(source, raw, processed)
            print(f"\n📝 Turn 1 ({source.value.upper()}):")
            print(f"   Raw: {raw}")
            print(f"   Processed: {processed}")
            
            # ========== SUBSEQUENT TURNS: ALTERNATE ==========
            # Alternate based on first turn:
            # If first was ASL -> Speech -> ASL -> Speech...
            # If first was Speech -> ASL -> Speech -> ASL...
            
            # Determine next turn type (opposite of first)
            next_is_speech = (source == InputSource.ASL)
            
            for turn_num in range(2, num_turns + 1):
                # PATCHED: Print memory status between turns
                print_memory_status(f"before turn {turn_num}")
                
                if next_is_speech:
                    # Speech turn
                    raw, processed = await self.capture_speech_turn()
                    if raw:
                        self.conversation.add_turn(InputSource.SPEECH, raw, processed)
                        print(f"\n📝 Turn {turn_num} (SPEECH): {processed}")
                    else:
                        print(f"⚠️ Skipping turn {turn_num} - no speech detected")
                else:
                    # ASL turn
                    raw, processed = await self.capture_asl_turn()
                    if raw:
                        self.conversation.add_turn(InputSource.ASL, raw, processed)
                        print(f"\n📝 Turn {turn_num} (ASL):")
                        print(f"   Raw: {raw}")
                        print(f"   Processed: {processed}")
                    else:
                        print(f"⚠️ Skipping turn {turn_num} - no ASL detected")
                
                # Alternate for next turn
                next_is_speech = not next_is_speech
                
                # PATCHED: Force garbage collection between turns
                gc.collect()
                
                # Brief pause between turns for resource cleanup
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Conversation interrupted by user")
        
        finally:
            # Cleanup
            self.asl_recognizer.stop_camera()
            cv2.destroyAllWindows()

            gc.collect()
            
            # Save and print summary
            self._print_summary()
            if self.conversation.turns:
                save_full_conversation_to_firestore(self.conversation)
    
    def _print_summary(self):
        """Print conversation summary"""
        print("\n" + "=" * 60)
        print("📜 CONVERSATION SUMMARY")
        print("=" * 60)
        if self.conversation.turns:
            print(self.conversation.get_full_transcript())
        else:
            print("No conversation recorded.")
        print("=" * 60)


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Main entry point"""
    print("=" * 60)
    print("INITIALIZING CONVERSATION SYSTEM")
    print("=" * 60)
    print_memory_status("startup")
    
    # PATCHED: Force garbage collection at startup
    gc.collect()

    # NEW: initialize the LCD (non-fatal if missing)
    init_lcd()
    
    system = ConversationSystem()
    await system.run_conversation(num_turns=9)



if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
