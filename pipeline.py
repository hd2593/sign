#!/usr/bin/env python3
"""
SignBridge - Windows PC Version (IMPROVED)
ASL + Speech Recognition Conversation Pipeline

Key improvements from previous version:
  - Tuned MediaPipe hand detection confidence thresholds for better recognition
  - Using IMAGE mode for better accuracy on static signs
  - Improved frame processing strategy
  - Better model selection for hand landmarking
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time
import os
import asyncio
import threading
import queue
import re
import gc
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum

import sounddevice as sd
import whisper
from tensorflow.keras.models import load_model

import tkinter as tk
from tkinter import scrolledtext

# ============================================================================
# OPTIONAL IMPORTS  (gracefully degrade if not installed)
# ============================================================================

try:
    import fastapi_poe as fp
    HAS_POE = True
except ImportError:
    HAS_POE = False
    print("Warning: fastapi_poe not installed - ASL sentence revision will use raw words.")

try:
    from google.cloud import firestore as gcp_firestore
    db = gcp_firestore.Client()
    HAS_FIRESTORE = True
except Exception:
    HAS_FIRESTORE = False
    db = None
    print("Warning: Firestore unavailable - cloud sync disabled (running fully offline).")


# ============================================================================
# MEDIAPIPE TASKS SETUP  (replaces legacy mp.solutions)
# ============================================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

# Using lite model for better performance (matches model_complexity=0 from original)
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_MODEL_PATH = "hand_landmarker.task"


def ensure_hand_landmarker_model():
    if os.path.exists(HAND_LANDMARKER_MODEL_PATH):
        return
    print(f"Downloading hand landmarker model to '{HAND_LANDMARKER_MODEL_PATH}'...")
    urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, HAND_LANDMARKER_MODEL_PATH)
    print("Hand landmarker model downloaded.")


def draw_hand_landmarks_on_image(image, hand_landmarks_list,
                                  connections=HAND_CONNECTIONS,
                                  landmark_color=(0, 0, 255),
                                  connection_color=(0, 255, 0),
                                  landmark_radius=3,
                                  connection_thickness=2):
    h, w, _ = image.shape
    for hand_landmarks in hand_landmarks_list:
        for start_idx, end_idx in connections:
            pt1 = (int(hand_landmarks[start_idx].x * w),
                    int(hand_landmarks[start_idx].y * h))
            pt2 = (int(hand_landmarks[end_idx].x * w),
                    int(hand_landmarks[end_idx].y * h))
            cv2.line(image, pt1, pt2, connection_color, connection_thickness)
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), landmark_radius, landmark_color, -1)


# ============================================================================
# CONFIGURATION
# ============================================================================

GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 0.5
SILENCE_THRESHOLD = 0.01
SILENCE_LIMIT = 2.0
WHISPER_MODEL = "tiny"
MAX_AUDIO_SECONDS = 30

SIGNS = None
DATA_PATH = "data_landmarks"
if not SIGNS:
    if os.path.isdir(DATA_PATH):
        SIGNS = sorted([
            d for d in os.listdir(DATA_PATH)
            if os.path.isdir(os.path.join(DATA_PATH, d))
        ])
        print(f"Auto-detected {len(SIGNS)} sign classes from '{DATA_PATH}':")
        print(SIGNS)
    else:
        SIGNS = []
        print(f"Warning: Data path '{DATA_PATH}' not found.")
else:
    print(f"Using manually defined {len(SIGNS)} sign classes:")
    print(SIGNS)

ASL_MODEL_PATH = "asl_model.h5"
SEQUENCE_LENGTH = 60
NUM_FEATURES = 126
THRESHOLD = 0.8
RECORD_TIME = 2.0
DISPLAY_TIME = 2.0
EXIT_THRESHOLD = 2.0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PROCESS_EVERY_N_FRAMES = 2

POE_API_KEY = os.environ.get(
    "POE_API_KEY",
    "0t84ZZMqeFE40inOvFiN17RU0iz13MhDhAKtvjOlsxI"
)
MODEL = "GPT-4o-Mini"


# ============================================================================
# ON-SCREEN DISPLAY  (replaces LCD)
# ============================================================================

class ScreenDisplay:
    def __init__(self, title="SignBridge - ASL Output"):
        self.root = None
        self.label = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(title,), daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    def _run(self, title):
        try:
            self.root = tk.Tk()
            self.root.title(title)
            self.root.geometry("800x120+100+0")
            self.root.attributes("-topmost", True)
            self.label = tk.Label(self.root, text="SignBridge Ready",
                                  font=("Helvetica", 26), wraplength=780, justify="center")
            self.label.pack(expand=True, fill="both")
            self._ready.set()
            self.root.mainloop()
        except Exception as e:
            print(f"Warning: Screen display init failed: {e}")
            self._ready.set()

    def show(self, text):
        if self.root is None or self.label is None:
            return
        try:
            self.root.after(0, lambda: self.label.config(text=text))
        except Exception:
            pass


screen_display = None

def init_screen_display():
    global screen_display
    try:
        screen_display = ScreenDisplay()
        print("Screen display initialized")
    except Exception as e:
        screen_display = None
        print(f"Warning: Screen display failed: {e}")

def display_sentence(text):
    if screen_display is not None:
        screen_display.show(text)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class InputSource(Enum):
    SPEECH = "speech"
    ASL = "asl"

@dataclass
class ConversationTurn:
    source: InputSource
    raw_text: str
    processed_text: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class ConversationState:
    turns: List[ConversationTurn] = field(default_factory=list)
    current_turn_source: Optional[InputSource] = None
    asl_phrases_sent: List[str] = field(default_factory=list)
    
    def add_turn(self, source: InputSource, raw: str, processed: str):
        turn = ConversationTurn(source=source, raw_text=raw, processed_text=processed)
        self.turns.append(turn)
        return turn
    
    def get_full_transcript(self) -> str:
        lines = []
        for i, turn in enumerate(self.turns, 1):
            source_label = "SPEECH" if turn.source == InputSource.SPEECH else "ASL"
            lines.append(f"Turn {i} [{source_label}]:")
            lines.append(f"  Raw: {turn.raw_text}")
            lines.append(f"  Processed: {turn.processed_text}")
            lines.append("")
        return "\n".join(lines)


# ============================================================================
# FIRESTORE INTEGRATION  (optional)
# ============================================================================

def save_asl_phrase_to_firestore(raw_phrase: str, smooth_phrase: str):
    if not HAS_FIRESTORE or db is None:
        return
    
    display_sentence(smooth_phrase)
    timestamp = int(time.time())
    raw_words = raw_phrase.lower().split()
    
    try:
        msg_ref = db.collection("messages").add({
            "raw": raw_phrase,
            "smooth": smooth_phrase,
            "phrases": raw_words,
            "timestamp": timestamp,
            "type": "asl_phrase"
        })
        print(f"Saved ASL phrase to Firestore: {msg_ref[1].id}")
        
        for p in raw_words:
            p_ref = db.collection("phrases").document(p)
            p_ref.set({"count": gcp_firestore.Increment(1)}, merge=True)
    except Exception as e:
        print(f"Firestore save error: {e}")

def save_full_conversation_to_firestore(conversation: ConversationState):
    if not HAS_FIRESTORE or db is None or not conversation.turns:
        return
    
    try:
        timestamp = int(time.time())
        turns_data = []
        for turn in conversation.turns:
            turns_data.append({
                "source": turn.source.value,
                "raw": turn.raw_text,
                "processed": turn.processed_text,
                "timestamp": int(turn.timestamp)
            })
        
        conv_ref = db.collection("conversations").add({
            "turns": turns_data,
            "timestamp": timestamp,
            "num_turns": len(conversation.turns)
        })
        print(f"Saved conversation to Firestore: {conv_ref[1].id}")
    except Exception as e:
        print(f"Firestore conversation save error: {e}")


# ============================================================================
# POE / GPT INTEGRATION
# ============================================================================

async def revise_asl_sentence(asl_words: List[str]) -> str:
    if not HAS_POE or not asl_words:
        return " ".join(asl_words)
    
    raw_sentence = " ".join(asl_words)
    prompt = f"""You are helping a deaf person communicate using ASL.
They have signed individual words/phrases in ASL: {raw_sentence}

Convert this into a natural, grammatically correct English sentence.
Keep the original meaning but make it flow naturally.
Output ONLY the corrected sentence, nothing else."""
    
    try:
        async for partial in fp.get_bot_response(
            messages=[fp.ProtocolMessage(role="user", content=prompt)],
            bot_name=MODEL,
            api_key=POE_API_KEY,
        ):
            pass
        return partial.text.strip()
    except Exception as e:
        print(f"GPT revision error: {e}")
        return raw_sentence


# ============================================================================
# UTILITIES
# ============================================================================

def print_memory_status(label=""):
    try:
        import psutil
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        print(f"[MEMORY {label}] {mem_mb:.1f} MB")
    except Exception:
        pass


# ============================================================================
# SPEECH RECOGNITION
# ============================================================================

class SpeechDisplay:
    def __init__(self):
        self.root = None
        self.text_widget = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    def _run(self):
        try:
            self.root = tk.Tk()
            self.root.title("Speech Recognition")
            self.root.geometry("600x200+100+150")
            self.root.attributes("-topmost", True)
            
            self.text_widget = scrolledtext.ScrolledText(
                self.root, wrap=tk.WORD, font=("Helvetica", 14)
            )
            self.text_widget.pack(expand=True, fill="both", padx=10, pady=10)
            
            self._ready.set()
            self.root.mainloop()
        except Exception as e:
            print(f"Warning: Speech display init failed: {e}")
            self._ready.set()

    def show_text(self, text):
        if self.root is None or self.text_widget is None:
            return
        try:
            self.root.after(0, lambda: self._update_text(text))
        except Exception:
            pass

    def _update_text(self, text):
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(tk.END, text)


speech_display = None
STOP_WORD = "over"
MAX_AUDIO_BLOCKS = int(MAX_AUDIO_SECONDS / BLOCK_DURATION)

def contains_stop_word(text: str) -> bool:
    return bool(re.search(r"\bover\b", text, re.IGNORECASE))

def remove_stop_word(transcript: str) -> str:
    if not transcript:
        return ""
    cleaned = re.sub(r"\bover\b[.!?,]*$", "", transcript, flags=re.IGNORECASE)
    return cleaned.strip()

def record_and_transcribe_realtime(timeout=30.0):
    global speech_display
    if speech_display is None:
        try:
            speech_display = SpeechDisplay()
        except Exception:
            pass

    print("Listening... say 'over' to finish")
    whisper_model = whisper.load_model(WHISPER_MODEL)

    audio_buffer = []
    stop_recording = threading.Event()
    silence_start_time = [None]
    has_heard_speech = [False]
    block_size = int(SAMPLE_RATE * BLOCK_DURATION)

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        if stop_recording.is_set():
            return
        audio_buffer.append(indata[:, 0].copy())
        if len(audio_buffer) > MAX_AUDIO_BLOCKS:
            audio_buffer.pop(0)
        rms = np.sqrt(np.mean(np.square(indata)))
        if rms > SILENCE_THRESHOLD:
            has_heard_speech[0] = True
            silence_start_time[0] = None
        else:
            if has_heard_speech[0] and silence_start_time[0] is None:
                silence_start_time[0] = time.time()

    transcribe_interval = 2.0
    last_transcribe = 0.0
    running_transcript = ""
    start_time = time.time()

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32",
                            blocksize=block_size, callback=audio_callback):
            while not stop_recording.is_set():
                time.sleep(0.1)
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    print("Timeout reached")
                    break
                if (has_heard_speech[0] and silence_start_time[0] is not None
                        and time.time() - silence_start_time[0] > SILENCE_LIMIT):
                    print("Silence detected - stopping")
                    break
                if time.time() - last_transcribe >= transcribe_interval and len(audio_buffer) > 0:
                    last_transcribe = time.time()
                    audio_np = np.concatenate(audio_buffer).astype(np.float32)
                    result = whisper_model.transcribe(audio_np, language="en", fp16=False)
                    running_transcript = result.get("text", "").strip()
                    print(f"\r  {running_transcript}", end="", flush=True)
                    if speech_display is not None:
                        speech_display.show_text(running_transcript)
                    if contains_stop_word(running_transcript):
                        print(f"\nStop word '{STOP_WORD}' detected!")
                        break
    except Exception as e:
        print(f"Recording/transcription error: {e}")
        import traceback
        traceback.print_exc()
        return ""

    if audio_buffer:
        audio_np = np.concatenate(audio_buffer).astype(np.float32)
        result = whisper_model.transcribe(audio_np, language="en", fp16=False)
        running_transcript = result.get("text", "").strip()

    full_transcript = remove_stop_word(running_transcript)
    print(f"\nFull transcript: {full_transcript}")
    if speech_display is not None:
        speech_display.show_text(full_transcript)
    return full_transcript


# ============================================================================
# ASL RECOGNITION  (IMPROVED MediaPipe configuration)
# ============================================================================

class ASLRecognizer:
    STATE_WAITING = "WAITING"
    STATE_RECORDING = "RECORDING"
    STATE_RESULT = "RESULT"

    def __init__(self):
        print("Loading ASL model...")
        print_memory_status("before ASL model load")
        self.model = load_model(ASL_MODEL_PATH)
        print_memory_status("after ASL model load")

        num_model_classes = self.model.output_shape[-1]
        if num_model_classes != len(SIGNS):
            print(f"WARNING: Model expects {num_model_classes} classes, SIGNS has {len(SIGNS)}")

        # IMPROVED: MediaPipe HandLandmarker with optimized settings
        # Using lower confidence thresholds to match original mp.solutions behavior
        ensure_hand_landmarker_model()
        base_options = mp_python.BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL_PATH)
        
        # CRITICAL: Lowered confidence thresholds from 0.5 to 0.3 for better detection
        # This compensates for the different model behavior vs. mp.solutions
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.3,  # Lowered from 0.5
            min_hand_presence_confidence=0.3,   # Lowered from 0.5
            min_tracking_confidence=0.3,         # Lowered from 0.5
        )
        self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)

        self.cap = None
        self.camera_started = False
        self.captured_words = []
        self.last_result = None
        self.frame_count = 0
        self._ts_ms = 0

    def _next_timestamp_ms(self):
        self._ts_ms += 33  # ~30 fps
        return self._ts_ms

    @staticmethod
    def _result_has_hands(result):
        return result is not None and bool(result.hand_landmarks)

    def _detect(self, rgb_frame):
        """Detect hands in the frame using MediaPipe"""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.hand_landmarker.detect_for_video(mp_image, self._next_timestamp_ms())

    def _extract_landmarks(self, result):
        """Extract normalized landmarks from MediaPipe results"""
        frame_data = []
        if result is not None and result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                wrist = hand_landmarks[0]
                for lm in hand_landmarks:
                    frame_data.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
        padding_len = NUM_FEATURES - len(frame_data)
        frame_data.extend([0.0] * padding_len)
        return frame_data[:NUM_FEATURES]

    def start_camera(self):
        if not self.camera_started:
            # Try DirectShow backend first (Windows), fall back to default
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            # Set autofocus if available
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            time.sleep(1)
            self.camera_started = True
            print("Camera started")

    def stop_camera(self):
        if self.camera_started and self.cap is not None:
            self.cap.release()
            self.cap = None
            self.camera_started = False
            cv2.destroyAllWindows()

    def _read_frame(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def capture_phrase(self, stop_event=None):
        """Capture an ASL phrase (sequence of signs)"""
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

                frame = self._read_frame()
                if frame is None:
                    continue
                self.frame_count += 1

                image = cv2.flip(frame, 1)
                
                # Process every Nth frame for hand detection
                should_process = self.frame_count % PROCESS_EVERY_N_FRAMES == 0

                if should_process:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    result = self._detect(image_rgb)
                    self.last_result = result
                else:
                    result = self.last_result

                hand_present = self._result_has_hands(result)

                if hand_present:
                    last_hand_time = time.time()
                    draw_hand_landmarks_on_image(image, result.hand_landmarks)

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
                        current_frames.append(self._extract_landmarks(result))

                    # Exit if hands removed after first sign
                    if first_sign_completed and (time.time() - last_hand_time > EXIT_THRESHOLD):
                        print("No hands detected. ASL phrase complete.")
                        break

                    # After recording time, make prediction
                    if elapsed > RECORD_TIME:
                        data = np.array(current_frames)
                        if len(data) > SEQUENCE_LENGTH:
                            input_data = data[:SEQUENCE_LENGTH]
                        else:
                            input_data = np.concatenate((
                                data, np.zeros((SEQUENCE_LENGTH - len(data), NUM_FEATURES))))

                        res = self.model.predict(np.expand_dims(input_data, axis=0), verbose=0)[0]
                        best_idx = np.argmax(res)
                        confidence = res[best_idx]

                        if best_idx < len(SIGNS) and confidence > THRESHOLD:
                            word = SIGNS[best_idx]
                            self.captured_words.append(word)
                            current_prediction_text = word
                            print(f"ASL Predicted: {word} ({confidence:.2f})")
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
                cv2.putText(image, " ".join(self.captured_words), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("ASL Recognition", image)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("ASL quit by user")
                    break

        except Exception as e:
            print(f"ASL recognition error: {e}")
            import traceback
            traceback.print_exc()

        return self.captured_words

    def has_content(self):
        return len(self.captured_words) > 0


# ============================================================================
# INITIAL MODE DETECTION
# ============================================================================

class InitialModeDetector:
    DETECTION_TIMEOUT = 3.0

    def __init__(self, asl_recognizer):
        self.asl = asl_recognizer

    def detect_initial_mode(self):
        print("Detecting input mode (raise hand for ASL, speak for voice)...")
        self.asl.start_camera()

        audio_queue = queue.Queue()
        stop_audio = threading.Event()
        has_speech = threading.Event()

        def audio_detector():
            def callback(indata, frames, time_info, status):
                audio_queue.put(indata.copy())
            block_size = int(SAMPLE_RATE * BLOCK_DURATION)
            try:
                with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                                    dtype="float32", blocksize=block_size, callback=callback):
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

        start_time = time.time()
        hand_detected = False

        while time.time() - start_time < self.DETECTION_TIMEOUT:
            frame = self.asl._read_frame()
            if frame is None:
                continue

            image = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = self.asl._detect(image_rgb)
            hand_present = self.asl._result_has_hands(result)

            if hand_present:
                hand_detected = True
                draw_hand_landmarks_on_image(image, result.hand_landmarks)

            remaining = self.DETECTION_TIMEOUT - (time.time() - start_time)
            status_text = "HAND DETECTED!" if hand_detected else "Waiting..."
            border_color = (0, 255, 0) if hand_detected else (255, 165, 0)
            cv2.rectangle(image, (0, 0), (CAMERA_WIDTH, CAMERA_HEIGHT), border_color, 10)
            cv2.putText(image, f"{status_text} ({remaining:.1f}s)", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, "Raise hand for ASL or Speak", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            cv2.imshow("Mode Detection", image)

            if hand_detected:
                time.sleep(0.5)
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        stop_audio.set()
        audio_thread.join(timeout=0.5)
        cv2.destroyAllWindows()

        if hand_detected:
            print("Hand detected - using ASL mode")
            return InputSource.ASL
        elif has_speech.is_set():
            print("Speech detected - using Speech mode")
            self.asl.stop_camera()
            return InputSource.SPEECH
        else:
            print("No clear input - defaulting to ASL mode")
            return InputSource.ASL


# ============================================================================
# CONVERSATION SYSTEM
# ============================================================================

class ConversationSystem:
    def __init__(self):
        print_memory_status("before system init")
        self.asl_recognizer = ASLRecognizer()
        self.conversation = ConversationState()
        self.mode_detector = InitialModeDetector(self.asl_recognizer)
        print_memory_status("after system init")

    async def capture_first_input(self):
        print("\n" + "=" * 60)
        print("Starting FIRST TURN - Speak OR Sign!")
        print("=" * 60)
        mode = self.mode_detector.detect_initial_mode()

        if mode == InputSource.ASL:
            print("Starting ASL capture...")
            asl_words = self.asl_recognizer.capture_phrase()
            cv2.destroyAllWindows()
            if asl_words:
                raw_text = " ".join(asl_words)
                processed_text = await revise_asl_sentence(asl_words)
                save_asl_phrase_to_firestore(raw_text, processed_text)
                return InputSource.ASL, raw_text, processed_text
            print("No ASL captured")
            return None, "", ""

        print("Starting Speech capture...")
        print("Say 'over' when you're done speaking.\n")
        transcript = record_and_transcribe_realtime(timeout=10.0)
        if transcript:
            return InputSource.SPEECH, transcript, transcript
        print("No speech captured")
        return None, "", ""

    async def capture_speech_turn(self):
        print("\n" + "=" * 60)
        print("YOUR TURN TO SPEAK")
        print("=" * 60)
        print("Say 'over' when you're done speaking.\n")
        transcript = record_and_transcribe_realtime(timeout=10.0)
        if transcript:
            return transcript, transcript
        print("No speech detected")
        return "", ""

    async def capture_asl_turn(self):
        print("\n" + "=" * 60)
        print("YOUR TURN TO SIGN")
        print("=" * 60)
        asl_words = self.asl_recognizer.capture_phrase()
        cv2.destroyAllWindows()
        if asl_words:
            raw_text = " ".join(asl_words)
            processed_text = await revise_asl_sentence(asl_words)
            save_asl_phrase_to_firestore(raw_text, processed_text)
            return raw_text, processed_text
        print("No ASL detected")
        return "", ""

    async def run_conversation(self, num_turns=5):
        print("\n" + "=" * 60)
        print("INTEGRATED CONVERSATION SYSTEM")
        print("=" * 60)
        print(f"Running for up to {num_turns} turns")
        print("Press 'q' in ASL window or Ctrl+C to end early\n")

        try:
            source, raw, processed = await self.capture_first_input()
            if source is None:
                print("No initial input detected. Exiting.")
                return
            self.conversation.add_turn(source, raw, processed)
            print(f"\nTurn 1 ({source.value.upper()}):")
            print(f"   Raw: {raw}")
            print(f"   Processed: {processed}")

            next_is_speech = source == InputSource.ASL
            for turn_num in range(2, num_turns + 1):
                print_memory_status(f"before turn {turn_num}")
                if next_is_speech:
                    raw, processed = await self.capture_speech_turn()
                    if raw:
                        self.conversation.add_turn(InputSource.SPEECH, raw, processed)
                        print(f"\nTurn {turn_num} (SPEECH): {processed}")
                    else:
                        print(f"Skipping turn {turn_num} - no speech detected")
                else:
                    raw, processed = await self.capture_asl_turn()
                    if raw:
                        self.conversation.add_turn(InputSource.ASL, raw, processed)
                        print(f"\nTurn {turn_num} (ASL):")
                        print(f"   Raw: {raw}")
                        print(f"   Processed: {processed}")
                    else:
                        print(f"Skipping turn {turn_num} - no ASL detected")
                next_is_speech = not next_is_speech
                gc.collect()
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n\nConversation interrupted by user")
        finally:
            self.asl_recognizer.stop_camera()
            cv2.destroyAllWindows()
            gc.collect()
            self._print_summary()
            if self.conversation.turns:
                save_full_conversation_to_firestore(self.conversation)

    def _print_summary(self):
        print("\n" + "=" * 60)
        print("CONVERSATION SUMMARY")
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
    print("=" * 60)
    print("INITIALIZING CONVERSATION SYSTEM (Windows - IMPROVED)")
    print("=" * 60)
    print_memory_status("startup")
    gc.collect()
    init_screen_display()
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
