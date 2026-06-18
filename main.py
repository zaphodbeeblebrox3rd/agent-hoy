#!/usr/bin/env python3
"""
Real-time Speech Transcription with Keyword-based Topic Explanations

This application listens to microphone input, transcribes speech in real-time,
and allows users to click on keywords to get topic explanations, technical
challenges, and command examples.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import speech_recognition as sr
import threading
import queue
import re
import json
import time
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional
import requests
import os
from datetime import datetime, timedelta

from config_loader import (
    load_question_patterns,
    load_question_type_patterns,
    load_tech_keywords,
    load_topic_explanations,
    load_transcription_corrections,
)
from transcription_config import (
    AI_ANALYSIS_THROTTLE_SECONDS,
    HIGHLIGHT_DEBOUNCE_MS,
    LISTEN_PHRASE_TIME_LIMIT,
    LISTEN_TIMEOUT,
    QUESTION_COMPLETION_TIMEOUT,
    WHISPER_BEAM_SIZE,
    WHISPER_CONDITION_ON_PREVIOUS,
    WHISPER_FAST_MODE,
    WHISPER_INITIAL_PROMPT,
    WHISPER_LANGUAGE,
    WHISPER_SAMPLE_RATE,
    get_whisper_model,
    is_cuda_available,
    should_prefer_whisper,
)

# OpenAI integration
from openai_config import (
    AVAILABLE_MODELS,
    MODEL_ID_BY_LABEL,
    openai_config,
    save_persisted_model,
)
try:
    from openai_integration import openai_analyzer
    OPENAI_AVAILABLE = True
except ImportError:
    print("OpenAI integration not available - using template fallback")
    openai_analyzer = None
    OPENAI_AVAILABLE = False

class SpeechTranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Speech Transcription with Topic Explorer")
        self.root.geometry("1200x900")  # Increased height for better pane visibility
        
        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        self.microphone = None  # Initialize later in setup_speech_recognition
        self.is_listening = False
        self.is_paused = False
        self.audio_queue = queue.Queue()
        
        # Audio processing settings for robustness
        self.min_audio_duration = 0.2  # Minimum 0.2 seconds (reduced)
        self.min_audio_energy = 20   # Minimum RMS energy (much lower)
        self.max_audio_duration = 30  # Maximum 30 seconds
        
        # FLAC availability tracking
        self.flac_available = None  # Will be checked on first use
        self.google_fallback_enabled = True
        
        # Font size tracking
        self.current_font_size = 12  # Default font size
        
        # Audio validation settings
        self.audio_validation_enabled = True
        self.validation_failures = 0
        
        # AI analysis throttling
        self.last_ai_analysis_time = 0
        self.ai_analysis_throttle_seconds = AI_ANALYSIS_THROTTLE_SECONDS

        # Question completion detection
        self.question_completion_timeout = QUESTION_COMPLETION_TIMEOUT
        self.last_speech_time = 0
        self.pending_analysis = None
        
        # Dynamic question type evolution
        self.current_question_type = None
        self.question_type_history = []
        self.context_evolution_threshold = 0.3  # Minimum confidence to change question type
        
        # Load transcription corrections from config file
        self.transcription_corrections = load_transcription_corrections()
        
        # Track analyzed keywords to prevent duplicate analysis
        self.analyzed_keywords = set()
        self.last_analyzed_transcription = ""
        
        # AI analysis lock to prevent concurrent analysis
        self.ai_analysis_lock = threading.Lock()
        self.ai_analysis_running = False
        
        # Transcription buffer and queue for AI analysis
        self.transcription_buffer = ""
        self.pending_ai_analysis = False
        
        # Offline capabilities
        self.use_offline = False
        self.offline_recognizer = None
        self.cache_dir = "cache"
        self.cache_file = os.path.join(self.cache_dir, "topic_cache.pkl")
        self.session_cache = {}
        self.last_network_check = None
        self.network_online = True
        self.network_check_ttl = int(os.getenv("NETWORK_CHECK_TTL_SECONDS", "30"))
        self.verbose = os.getenv("VERBOSE", "").lower() in ("1", "true", "yes", "on")
        if is_cuda_available():
            recognition_workers = 1
        else:
            recognition_workers = int(os.getenv("RECOGNITION_MAX_WORKERS", "2"))
        self.recognition_executor = ThreadPoolExecutor(
            max_workers=recognition_workers,
            thread_name_prefix="stt",
        )
        self.recognition_generation = 0
        self.last_whisper_text = ""
        self.whisper_device = "cpu"
        self.whisper_fp16 = False
        self.whisper_model_name = get_whisper_model()
        self._highlight_debounce_id = None
        
        # Transcription data
        self.current_transcription = ""
        self.transcription_history = []
        
        # Initialize caching
        self.setup_caching()
        
        # Question/keyword config loaded from conf/*.conf (see conf/*.conf.example)
        self.question_patterns = load_question_patterns()
        self.question_type_patterns = load_question_type_patterns()
        self.tech_keywords = load_tech_keywords()
        self.topic_explanations = load_topic_explanations()
        
        self.setup_ui()
        self.setup_speech_recognition()
        self.setup_offline_recognition()
        
        # Bind audio status refresh to microphone changes
        self.root.bind("<FocusIn>", lambda e: self.update_audio_status())
        
    def setup_ui(self):
        """Set up the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel - organized in two rows
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # First row - Action buttons and font controls
        top_row = ttk.Frame(control_frame)
        top_row.pack(fill=tk.X, pady=(0, 5))
        
        self.listen_button = ttk.Button(top_row, text="Start Listening", 
                                      command=self.toggle_listening)
        self.listen_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.pause_button = ttk.Button(top_row, text="Pause Listening", 
                                     command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_button = ttk.Button(top_row, text="Clear Text", 
                                     command=self.clear_transcription)
        self.clear_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Cost reset button
        self.cost_reset_button = ttk.Button(top_row, text="Reset Cost", 
                                           command=self.reset_session_cost)
        self.cost_reset_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Font size controls
        font_frame = ttk.Frame(top_row)
        font_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        ttk.Label(font_frame, text="Font Size:").pack(side=tk.LEFT)
        
        self.decrease_font_button = ttk.Button(font_frame, text="A-", 
                                             command=self.decrease_font_size, width=3)
        self.decrease_font_button.pack(side=tk.LEFT, padx=(5, 2))
        
        self.increase_font_button = ttk.Button(font_frame, text="A+", 
                                             command=self.increase_font_size, width=3)
        self.increase_font_button.pack(side=tk.LEFT, padx=(2, 0))

        # OpenAI model selector
        model_frame = ttk.Frame(top_row)
        model_frame.pack(side=tk.LEFT, padx=(20, 0))

        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT)

        self.model_var = tk.StringVar(value=openai_config.get_model_label())
        self.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=[label for label, _ in AVAILABLE_MODELS],
            state="readonly"
            if OPENAI_AVAILABLE and openai_analyzer and openai_analyzer.is_available()
            else "disabled",
            width=22,
        )
        self.model_combo.pack(side=tk.LEFT, padx=(5, 0))
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_selected)
        
        # Second row - Status labels and settings
        bottom_row = ttk.Frame(control_frame)
        bottom_row.pack(fill=tk.X)
        
        self.status_label = ttk.Label(bottom_row, text="Status: Ready")
        self.status_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # Audio status display
        self.audio_status_label = ttk.Label(bottom_row, text="Audio Status: Default Microphone")
        self.audio_status_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # AI toggle checkbox
        self.ai_enabled_var = tk.BooleanVar(value=True)
        self.ai_checkbox = ttk.Checkbutton(
            bottom_row, 
            text="Enable AI Analysis", 
            variable=self.ai_enabled_var,
            command=self.on_ai_toggle
        )
        self.ai_checkbox.pack(side=tk.LEFT, padx=(0, 20))
        
        # OpenAI status indicator
        self.openai_status_label = ttk.Label(bottom_row, text=self._openai_status_text())
        self.openai_status_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # Cost tracking
        self.session_cost = 0.0
        self.cost_label = ttk.Label(bottom_row, text="Session Cost: $0.00")
        self.cost_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # API call tracking
        self.api_call_count = 0
        self.api_counter_label = ttk.Label(bottom_row, text="API Calls: 0")
        self.api_counter_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # AI analysis status indicator
        self.ai_status_label = ttk.Label(bottom_row, text="AI Status: Ready")
        self.ai_status_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # Question type indicator
        self.question_type_label = ttk.Label(bottom_row, text="Type: None", 
                                           font=("Arial", 9), foreground="green")
        self.question_type_label.pack(side=tk.LEFT, padx=(0, 0))
        
        # Create resizable paned window for transcription and topic areas
        self.paned_window = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        self.paned_window.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Top pane - Transcription display
        transcription_frame = ttk.LabelFrame(self.paned_window, text="Live Transcription", padding="5")
        self.paned_window.add(transcription_frame, weight=1)
        transcription_frame.columnconfigure(0, weight=1)
        transcription_frame.rowconfigure(0, weight=1)
        
        self.transcription_text = scrolledtext.ScrolledText(
            transcription_frame, 
            wrap=tk.WORD, 
            height=15,
            font=("Arial", self.current_font_size)
        )
        self.transcription_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bottom pane - Split into topic explanation and AI output
        bottom_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(bottom_frame, weight=1)
        
        # Create horizontal PanedWindow for bottom split
        self.bottom_paned = ttk.PanedWindow(bottom_frame, orient=tk.HORIZONTAL)
        self.bottom_paned.pack(fill=tk.BOTH, expand=True)
        
        # Left side - AI-driven output panel
        ai_frame = ttk.LabelFrame(self.bottom_paned, text="AI-Driven Analysis & Suggestions", padding="5")
        self.bottom_paned.add(ai_frame, weight=1)
        ai_frame.columnconfigure(0, weight=1)
        ai_frame.rowconfigure(0, weight=1)
        
        self.ai_text = scrolledtext.ScrolledText(
            ai_frame,
            wrap=tk.WORD,
            height=15,
            font=("Arial", max(8, self.current_font_size - 2)),
            state=tk.DISABLED
        )
        self.ai_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Right side - Topic explanation panel
        topic_frame = ttk.LabelFrame(self.bottom_paned, text="Topic Explanation & Troubleshooting", padding="5")
        self.bottom_paned.add(topic_frame, weight=1)
        topic_frame.columnconfigure(0, weight=1)
        topic_frame.rowconfigure(0, weight=1)
        
        self.topic_text = scrolledtext.ScrolledText(
            topic_frame,
            wrap=tk.WORD,
            height=15,
            font=("Arial", max(8, self.current_font_size - 2)),
            state=tk.DISABLED
        )
        self.topic_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bind click events to transcription text
        self.transcription_text.bind("<Button-1>", self.on_text_click)
        
        # Set initial pane positions (50/50 splits)
        self.paned_window.pane(0, weight=1)
        self.paned_window.pane(1, weight=1)
        self.bottom_paned.pane(0, weight=1)
        self.bottom_paned.pane(1, weight=1)
        
        # Bind pane resize events to save user preferences
        self.paned_window.bind("<ButtonRelease-1>", self.on_vertical_pane_resize)
        self.bottom_paned.bind("<ButtonRelease-1>", self.on_horizontal_pane_resize)
        
        # Restore saved pane positions after a short delay
        self.root.after(100, self.restore_pane_positions)
        
    def setup_caching(self):
        """Initialize caching system"""
        try:
            # Create cache directory if it doesn't exist
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            
            # Load existing cache
            self.load_cache()
        except Exception as e:
            print(f"Cache setup failed: {e}")
            # Continue without caching
            self.cache_dir = None

    def _verbose(self, message: str):
        if self.verbose:
            print(message)
    
    def setup_speech_recognition(self):
        """Initialize speech recognition with microphone"""
        try:
            print("Initializing microphone...")
            # Initialize microphone
            self.microphone = sr.Microphone()
            print("Microphone object created successfully")
            
            # Optimize recognizer settings for better performance
            self.recognizer.energy_threshold = 300  # Lower threshold for better sensitivity
            self.recognizer.dynamic_energy_threshold = True  # Auto-adjust to ambient noise
            self.recognizer.pause_threshold = 0.8  # Shorter pause detection
            self.recognizer.phrase_threshold = 0.3  # Faster phrase detection
            self.recognizer.non_speaking_duration = 0.5  # Shorter non-speaking detection
            print("Recognizer settings configured")
            
            print("Adjusting for ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Faster calibration
            print("Ambient noise adjustment completed")
            
            self.status_label.config(text="Status: Microphone ready")
            print("Microphone setup completed successfully")
            
            # Update audio status now that microphone is initialized
            self.update_audio_status()
        except Exception as e:
            print(f"Microphone setup failed: {e}")
            messagebox.showerror("Error", f"Failed to initialize microphone: {str(e)}")
            self.status_label.config(text="Status: Microphone error")
    
    def setup_offline_recognition(self):
        """Setup offline speech recognition fallback"""
        try:
            import whisper

            if is_cuda_available():
                self.whisper_device = "cuda"
                self.whisper_fp16 = True
            else:
                self.whisper_device = "cpu"
                self.whisper_fp16 = False

            self.whisper_model = whisper.load_model(self.whisper_model_name, device=self.whisper_device)
            device_label = "CUDA" if self.whisper_device == "cuda" else "CPU"
            prefer = "yes" if self.whisper_device == "cuda" else "when offline"
            fast = "on" if WHISPER_FAST_MODE else "off"
            print(
                f"Whisper ready (model: {self.whisper_model_name}, device: {device_label}, "
                f"stt_priority: {prefer}, fast_mode: {fast})"
            )
            self.status_label.config(
                text=f"Status: Whisper {self.whisper_model_name} on {device_label}"
            )
        except ImportError:
            print("Whisper not available - install Full profile: uv sync --extra whisper")
            self.whisper_model = None
        except Exception as e:
            print(f"Offline recognition setup failed: {e}")
            self.whisper_model = None
    
    def toggle_listening(self):
        """Start or stop listening for speech"""
        if not self.is_listening:
            self.start_listening()
        else:
            self.stop_listening()
    
    def toggle_pause(self):
        """Pause or resume listening for speech"""
        if not self.is_paused:
            self.pause_listening()
        else:
            self.resume_listening()
    
    def start_listening(self):
        """Start the speech recognition thread"""
        self.is_listening = True
        self.is_paused = False
        self.listen_button.config(text="Stop Listening")
        self.pause_button.config(text="Pause Listening", state=tk.NORMAL)
        self.status_label.config(text="Status: Listening...")
        
        # Start background thread for speech recognition
        self.listen_thread = threading.Thread(target=self.listen_continuously, daemon=True)
        self.listen_thread.start()
    
    def stop_listening(self):
        """Stop the speech recognition"""
        self.is_listening = False
        self.is_paused = False
        self.listen_button.config(text="Start Listening")
        self.pause_button.config(text="Pause Listening", state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")
        print("Stopped listening for speech")
    
    def _make_ai_stream_callback(self):
        """Return a callback that streams partial AI text to the UI thread."""
        def callback(partial_text):
            self.root.after(0, lambda content=partial_text: self.show_ai_analysis(content))
        return callback
    
    def pause_listening(self):
        """Pause listening but allow AI analysis to continue."""
        if self.is_listening and not self.is_paused:
            self.is_paused = True
            self.recognition_generation += 1
            self.pause_button.config(text="Resume Listening")
            self.status_label.config(text="Status: Paused (AI analysis continues)")
            print("Paused listening - discarding in-flight speech recognition")
    
    def resume_listening(self):
        """Resume listening after pause."""
        if self.is_listening and self.is_paused:
            self.is_paused = False
            self.recognition_generation += 1
            self.pause_button.config(text="Pause Listening")
            self.status_label.config(text="Status: Listening...")
            print("Resumed listening...")
    
    def listen_continuously(self):
        """Continuously listen for speech in a separate thread"""
        print("Starting continuous listening...")
        while self.is_listening:
            try:
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                self._verbose("Listening for audio...")
                with self.microphone as source:
                    audio = self.recognizer.listen(
                        source,
                        timeout=LISTEN_TIMEOUT,
                        phrase_time_limit=LISTEN_PHRASE_TIME_LIMIT,
                    )
                
                if self.is_paused or not self.is_listening:
                    continue
                
                self._verbose("Audio captured, processing...")
                generation = self.recognition_generation
                try:
                    self.recognition_executor.submit(
                        self.process_audio, audio, generation
                    )
                except RuntimeError:
                    self._verbose("Recognition worker pool saturated, dropping audio chunk")
                
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"Error in speech recognition: {e}")
                time.sleep(0.1)
        print("Stopped continuous listening")
    
    def check_network_connection(self):
        """Check if internet connection is available (cached)."""
        now = time.time()
        if self.last_network_check is not None:
            if now - self.last_network_check < self.network_check_ttl:
                return self.network_online

        try:
            response = requests.get("https://www.google.com", timeout=3)
            self.network_online = response.status_code == 200
        except requests.RequestException:
            self.network_online = False

        self.last_network_check = now
        return self.network_online
    
    def is_audio_valid(self, audio_data):
        """Check if audio data is valid for processing"""
        try:
            import numpy as np
            import wave
            import io
            
            # Check minimum length
            if len(audio_data) < 2000:  # Less than ~0.2 seconds
                print(f"Audio too short: {len(audio_data)} bytes")
                return False
            
            # Convert audio data to numpy array for analysis
            with io.BytesIO(audio_data) as wav_buffer:
                with wave.open(wav_buffer, "rb") as wav_file:
                    sample_rate = wav_file.getframerate()
                    frames = wav_file.readframes(wav_file.getnframes())
                    audio_array = np.frombuffer(frames, dtype=np.int16)

            if len(audio_array) == 0:
                print("Audio array is empty")
                return False

            rms_energy = np.sqrt(np.mean(audio_array**2))

            if rms_energy < self.min_audio_energy:
                print(f"Audio too quiet: RMS energy {rms_energy:.2f} < {self.min_audio_energy}")
                return False

            duration = len(audio_array) / sample_rate
            if duration > self.max_audio_duration:
                print(f"Audio too long: {duration:.2f}s > {self.max_audio_duration}s")
                return False

            self._verbose(
                f"Audio valid: {len(audio_array)} samples, {duration:.2f}s, RMS {rms_energy:.2f}"
            )
            return True
            
        except Exception as e:
            print(f"Audio validation failed: {e}")
            return False
    
    def check_flac_availability(self):
        """Check if FLAC is available for Google Speech Recognition"""
        if self.flac_available is not None:
            return self.flac_available
        
        try:
            # Try to get FLAC converter to test availability
            from speech_recognition.audio import get_flac_converter
            converter = get_flac_converter()
            self.flac_available = converter is not None
            if not self.flac_available:
                print("FLAC not available - disabling Google Speech Recognition fallback")
                self.google_fallback_enabled = False
                # Update status to inform user
                if hasattr(self, 'status_label'):
                    self.status_label.config(text="Status: Whisper only - Google Speech disabled (FLAC unavailable)")
            return self.flac_available
        except Exception as e:
            print(f"FLAC check failed: {e}")
            self.flac_available = False
            self.google_fallback_enabled = False
            return False
    
    def _resample_audio(self, audio_array, sample_rate: int):
        """Resample mono audio to Whisper's expected 16 kHz."""
        import numpy as np

        if sample_rate == WHISPER_SAMPLE_RATE or len(audio_array) == 0:
            return audio_array

        duration = len(audio_array) / sample_rate
        target_length = max(1, int(duration * WHISPER_SAMPLE_RATE))
        indices = np.linspace(0, len(audio_array) - 1, target_length)
        return np.interp(indices, np.arange(len(audio_array)), audio_array).astype(np.float32)

    def _should_apply_recognition(self, generation: int) -> bool:
        return (
            self.is_listening
            and not self.is_paused
            and generation == self.recognition_generation
        )

    def _apply_transcription_if_current(self, generation: int, text: str):
        if self._should_apply_recognition(generation) and text.strip():
            self.update_transcription(text)

    def _recognize_google(self, audio) -> str:
        if not self.check_flac_availability():
            return ""

        text = self.recognizer.recognize_google(
            audio,
            language="en-US",
            show_all=False,
            with_confidence=False,
        )
        return self.correct_transcription_errors(text.strip())

    def _normalize_transcription_text(self, full_transcription: str) -> str:
        """Strip timestamps and normalize transcript text for keyword matching."""
        text_parts = []
        for line in full_transcription.splitlines():
            stripped = line.strip()
            if stripped.startswith("[") and "] " in stripped:
                stripped = stripped.split("] ", 1)[1]
            if stripped:
                text_parts.append(stripped)
        return " ".join(text_parts).lower()

    def _keyword_matches(self, text: str, keyword: str) -> bool:
        """Match whole words for single tokens and phrases for multi-word keywords."""
        keyword = keyword.lower()
        if " " in keyword:
            return keyword in text
        pattern = r"\b" + re.escape(keyword) + r"\b"
        return re.search(pattern, text) is not None

    def _find_keyword_matches(self, full_transcription: str) -> list[tuple[str, str]]:
        """Return category/keyword matches, preferring longer phrases first."""
        text = self._normalize_transcription_text(full_transcription)
        matches = []
        for category, keywords in self.tech_keywords.items():
            for keyword in keywords:
                if self._keyword_matches(text, keyword):
                    matches.append((category, keyword, len(keyword)))
        matches.sort(key=lambda item: item[2], reverse=True)

        seen_categories = set()
        result = []
        for category, keyword, _length in matches:
            if category not in seen_categories:
                seen_categories.add(category)
                result.append((category, keyword))
        return result

    def _analysis_signature(self, category: str, keyword: str, full_transcription: str) -> str:
        """Context-aware signature so new discussion triggers a fresh OpenAI query."""
        tail = self._normalize_transcription_text(full_transcription)[-300:]
        return hashlib.md5(f"{category}:{keyword}:{tail}".encode()).hexdigest()

    def _run_contextual_analysis(self, category: str, keyword: str, full_transcription: str) -> bool:
        """Run OpenAI contextual analysis for a category/keyword match."""
        explanations = self.get_topic_explanations()
        if category not in explanations:
            self._verbose(f"No topic explanation for category '{category}'")
            return False

        explanation = explanations[category]
        if not self.ai_enabled_var.get():
            self.root.after(
                0,
                lambda: self.show_ai_analysis(
                    "AI Analysis disabled. Enable the checkbox to see AI-enhanced insights."
                ),
            )
            return True

        ai_content = self.generate_contextual_ai_analysis(
            category, explanation, full_transcription, keyword
        )
        self.root.after(0, lambda ai=ai_content: self.show_ai_analysis(ai))
        self.root.after(0, lambda cat=category: self.show_topic_explanation(cat))
        self.last_ai_analysis_time = time.time()
        self.last_analyzed_transcription = full_transcription
        print(f"AI analysis triggered for keyword '{keyword}' in category '{category}'")
        return True

    def _analyze_transcription_for_keywords(self, full_transcription: str) -> bool:
        """Find the best new keyword match and trigger contextual analysis."""
        for category, keyword in self._find_keyword_matches(full_transcription):
            signature = self._analysis_signature(category, keyword, full_transcription)
            if signature in self.analyzed_keywords:
                continue

            self.analyzed_keywords.add(signature)
            return self._run_contextual_analysis(category, keyword, full_transcription)

        return False

    def process_audio(self, audio, generation: int):
        """Process audio and update transcription with fallback."""
        try:
            if not self._should_apply_recognition(generation):
                return

            self._verbose("Processing audio...")

            online = self.check_network_connection()
            self.use_offline = not online
            if not online:
                self.root.after(
                    0,
                    lambda: self.status_label.config(
                        text="Status: Offline mode - using local recognition"
                    ),
                )

            audio_data = audio.get_wav_data()
            if self.audio_validation_enabled and not self.is_audio_valid(audio_data):
                self.validation_failures += 1
                self._verbose(
                    f"Audio validation failed - skipping chunk (failures: {self.validation_failures})"
                )
                if self.validation_failures > 10:
                    print("Too many validation failures - disabling audio validation temporarily")
                    self.audio_validation_enabled = False
                return

            if not self._should_apply_recognition(generation):
                return

            text = ""
            whisper_available = hasattr(self, "whisper_model") and self.whisper_model is not None
            prefer_whisper = should_prefer_whisper(whisper_available, online)

            if prefer_whisper and whisper_available:
                self._verbose("Using Whisper (preferred - CUDA available)...")
                text = self.recognize_offline(
                    audio,
                    audio_data=audio_data,
                    skip_validation=True,
                )

            if not text.strip() and online and self.google_fallback_enabled and not prefer_whisper:
                self._verbose("Using Google Speech Recognition...")
                try:
                    text = self._recognize_google(audio)
                except sr.UnknownValueError:
                    text = ""
                except sr.RequestError as e:
                    print(f"Google Speech Recognition request failed: {e}")
                    text = ""

            if not text.strip() and whisper_available and not prefer_whisper:
                self._verbose("Using Whisper fallback...")
                text = self.recognize_offline(
                    audio,
                    audio_data=audio_data,
                    skip_validation=True,
                )

            if not text.strip() and online and self.google_fallback_enabled and prefer_whisper:
                self._verbose("Trying Google Speech Recognition fallback...")
                try:
                    text = self._recognize_google(audio)
                except (sr.UnknownValueError, sr.RequestError):
                    text = ""

            if text.strip() and self._should_apply_recognition(generation):
                self.root.after(0, lambda t=text, g=generation: self._apply_transcription_if_current(g, t))

        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            print(f"Could not request results from speech recognition service: {e}")
            self.use_offline = True
            if not self._should_apply_recognition(generation):
                return
            try:
                text = self.recognize_offline(audio)
                if text.strip():
                    self.root.after(
                        0,
                        lambda t=text, g=generation: self._apply_transcription_if_current(g, t),
                    )
            except Exception as offline_error:
                print(f"Offline recognition also failed: {offline_error}")
    
    def recognize_offline(self, audio, audio_data=None, skip_validation=False):
        """Offline speech recognition fallback using Whisper"""
        try:
            if not (hasattr(self, "whisper_model") and self.whisper_model is not None):
                return ""

            import io
            import wave

            import numpy as np

            if audio_data is None:
                audio_data = audio.get_wav_data()

            if not skip_validation and not self.is_audio_valid(audio_data):
                return ""

            with io.BytesIO(audio_data) as wav_buffer:
                with wave.open(wav_buffer, "rb") as wav_file:
                    sample_rate = wav_file.getframerate()
                    frames = wav_file.readframes(wav_file.getnframes())
                    audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio_array) == 0:
                return ""

            audio_array = self._resample_audio(audio_array, sample_rate)

            prompt = WHISPER_INITIAL_PROMPT
            if WHISPER_CONDITION_ON_PREVIOUS and self.last_whisper_text:
                prompt = f"{WHISPER_INITIAL_PROMPT} Previous text: {self.last_whisper_text[-120:]}"

            result = self.whisper_model.transcribe(
                audio_array,
                language=WHISPER_LANGUAGE,
                task="transcribe",
                fp16=self.whisper_fp16,
                verbose=False,
                beam_size=WHISPER_BEAM_SIZE,
                best_of=1,
                temperature=0,
                without_timestamps=True,
                condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS,
                initial_prompt=prompt,
            )

            if result and "text" in result:
                text = result["text"].strip()
                if text:
                    self.last_whisper_text = text
                    return self.correct_transcription_errors(text)
            return ""

        except Exception as e:
            print(f"Whisper recognition failed: {e}")
            return ""
    
    def correct_transcription_errors(self, text):
        """Correct common transcription errors for technical terms using config file"""
        corrected_text = text
        
        # Apply corrections from config file
        for wrong, right in self.transcription_corrections.items():
            corrected_text = corrected_text.replace(wrong, right)

        self._verbose(f"Transcription correction - Original: '{text}' -> Corrected: '{corrected_text}'")
        return corrected_text
    
    def update_transcription(self, text):
        """Update the transcription display with new text"""
        # Append new text with timestamp
        timestamp = time.strftime("%H:%M:%S")
        formatted_text = f"[{timestamp}] {text}\n"
        
        # Insert at end and scroll to bottom
        self.transcription_text.insert(tk.END, formatted_text)
        self.transcription_text.see(tk.END)
        
        # Check for questions and provide troubleshooting suggestions
        # Run this asynchronously to not block transcription
        threading.Thread(
            target=self.check_questions_async,
            args=(text,),
            daemon=True
        ).start()
        
        # Update last speech time and schedule delayed analysis
        self.last_speech_time = time.time()
        
        # Cancel any pending analysis and schedule new one
        if self.pending_analysis:
            self.root.after_cancel(self.pending_analysis)
        
        # Schedule analysis after question completion timeout
        self.pending_analysis = self.root.after(
            int(self.question_completion_timeout * 1000),
            lambda: self.schedule_delayed_analysis()
        )

        # Debounce keyword highlighting on the UI thread (tkinter is not thread-safe)
        if self._highlight_debounce_id:
            self.root.after_cancel(self._highlight_debounce_id)
        self._highlight_debounce_id = self.root.after(
            HIGHLIGHT_DEBOUNCE_MS,
            self.highlight_keywords,
        )

        # Update current transcription
        self.current_transcription += text + " "
    
    def schedule_delayed_analysis(self):
        """Schedule AI analysis after question completion timeout"""
        try:
            # Check if enough time has passed since last speech
            current_time = time.time()
            if current_time - self.last_speech_time >= self.question_completion_timeout:
                self._verbose("Question completion detected, scheduling AI analysis...")
                # Run keyword analysis in a separate thread
                threading.Thread(
                    target=self.check_and_analyze_keywords_throttled,
                    args=("",),  # Empty text since we'll get full transcription
                    daemon=True
                ).start()
            else:
                # Reschedule for remaining time
                remaining_time = int((self.question_completion_timeout - (current_time - self.last_speech_time)) * 1000)
                if remaining_time > 0:
                    self.pending_analysis = self.root.after(remaining_time, lambda: self.schedule_delayed_analysis())
        except Exception as e:
            print(f"Error in delayed analysis scheduling: {e}")
    
    def check_questions_async(self, text):
        """Asynchronously check for questions without blocking transcription"""
        try:
            if self.detect_question(text):
                self.provide_troubleshooting_suggestions(text)
        except Exception as e:
            print(f"Error in async question checking: {e}")
    
    def highlight_keywords(self):
        """Highlight detected keywords in the transcription (debounced, UI thread only)."""
        self._highlight_debounce_id = None
        try:
            tags_to_remove = [
                tag for tag in self.transcription_text.tag_names()
                if tag.startswith("keyword")
            ]
            for tag in tags_to_remove:
                self.transcription_text.tag_remove(tag, "1.0", tk.END)

            self.transcription_text.tag_configure(
                "keyword",
                background="yellow",
                foreground="black",
                underline=True,
            )

            full_text = self.transcription_text.get("1.0", tk.END)
            if not full_text.strip():
                return

            # Only scan the recent tail of the transcript to keep highlighting fast.
            search_text = full_text[-4000:]
            search_offset = max(0, len(full_text) - len(search_text))

            for category, keywords in self.tech_keywords.items():
                for keyword in keywords:
                    pattern = r"\b" + re.escape(keyword) + r"\b"
                    for match in re.finditer(pattern, search_text, re.IGNORECASE):
                        try:
                            start_pos = f"1.0+{search_offset + match.start()}c"
                            end_pos = f"1.0+{search_offset + match.end()}c"
                            self.transcription_text.tag_add("keyword", start_pos, end_pos)
                            self.transcription_text.tag_add(
                                f"keyword_{category}", start_pos, end_pos
                            )
                        except tk.TclError:
                            continue
        except Exception as e:
            print(f"Error highlighting keywords: {e}")
    
    def on_text_click(self, event):
        """Handle clicks on transcription text"""
        # Get the character position of the click
        char_index = self.transcription_text.index(f"@{event.x},{event.y}")
        
        # Find which tag (if any) was clicked
        clicked_tags = self.transcription_text.tag_names(char_index)
        keyword_tags = [tag for tag in clicked_tags if tag.startswith("keyword_")]
        
        if keyword_tags:
            # Extract category from tag name
            category = keyword_tags[0].replace("keyword_", "")
            self.show_topic_explanation(category)
    
    def on_vertical_pane_resize(self, event):
        """Handle vertical pane resize events (transcription vs bottom area)"""
        try:
            sash_pos = self.paned_window.sashpos(0)
            window_height = self.paned_window.winfo_height()
            if window_height > 0:
                relative_pos = sash_pos / window_height
                self.session_cache['vertical_pane_position'] = relative_pos
        except Exception as e:
            print(f"Error saving vertical pane position: {e}")
    
    def on_horizontal_pane_resize(self, event):
        """Handle horizontal pane resize events (topic vs AI)"""
        try:
            sash_pos = self.bottom_paned.sashpos(0)
            window_width = self.bottom_paned.winfo_width()
            if window_width > 0:
                relative_pos = sash_pos / window_width
                self.session_cache['horizontal_pane_position'] = relative_pos
        except Exception as e:
            print(f"Error saving horizontal pane position: {e}")
    
    def restore_pane_positions(self):
        """Restore saved pane positions from cache"""
        try:
            # Restore vertical pane position
            if 'vertical_pane_position' in self.session_cache:
                relative_pos = self.session_cache['vertical_pane_position']
                window_height = self.paned_window.winfo_height()
                if window_height > 0:
                    sash_pos = int(relative_pos * window_height)
                    self.paned_window.sashpos(0, sash_pos)
            
            # Restore horizontal pane position
            if 'horizontal_pane_position' in self.session_cache:
                relative_pos = self.session_cache['horizontal_pane_position']
                window_width = self.bottom_paned.winfo_width()
                if window_width > 0:
                    sash_pos = int(relative_pos * window_width)
                    self.bottom_paned.sashpos(0, sash_pos)
        except Exception as e:
            print(f"Error restoring pane positions: {e}")
    
    def _openai_status_text(self) -> str:
        if OPENAI_AVAILABLE and openai_analyzer and openai_analyzer.is_available():
            return f"OpenAI: {openai_config.model}"
        return "OpenAI: Template Mode"

    def on_model_selected(self, _event=None):
        """Handle model combobox selection."""
        try:
            label = self.model_var.get()
            model_id = MODEL_ID_BY_LABEL.get(label)
            if not model_id:
                print(f"Unknown model label selected: {label}")
                return

            openai_config.set_model(model_id)
            save_persisted_model(model_id)
            self.openai_status_label.config(text=self._openai_status_text())
            self.update_ai_status(f"Model changed to {model_id}")
            print(f"OpenAI model set to {model_id}")
        except Exception as e:
            print(f"Error changing OpenAI model: {e}")

    def on_ai_toggle(self):
        """Handle AI toggle checkbox state change"""
        try:
            ai_enabled = self.ai_enabled_var.get()
            if ai_enabled:
                print("AI Analysis enabled")
                # Show AI pane if it was hidden
                self.show_ai_analysis("AI Analysis enabled. Click on keywords or ask questions to see AI-enhanced insights.")
            else:
                print("AI Analysis disabled")
                # Clear AI pane and show disabled message
                self.show_ai_analysis("AI Analysis disabled. Enable the checkbox to see AI-enhanced insights.")
        except Exception as e:
            print(f"Error handling AI toggle: {e}")
    
    def show_ai_analysis(self, content):
        """Display AI-driven analysis in the AI pane"""
        try:
            self.ai_text.config(state=tk.NORMAL)
            self.ai_text.delete("1.0", tk.END)
            self.ai_text.insert("1.0", content)
            self.ai_text.config(state=tk.DISABLED)
            self.ai_text.see(tk.END)
        except Exception as e:
            print(f"Error displaying AI analysis: {e}")
    
    def update_audio_status(self):
        """Update the audio status display with current microphone info"""
        try:
            # Get microphone device info
            mic_name = "Unknown"
            if hasattr(self, 'microphone') and self.microphone:
                try:
                    # Try to get device name from microphone
                    device_index = self.microphone.device_index
                    if device_index is not None:
                        import pyaudio
                        p = pyaudio.PyAudio()
                        try:
                            device_info = p.get_device_info_by_index(device_index)
                            mic_name = device_info.get('name', 'Default Microphone')
                        except:
                            mic_name = f"Device {device_index}"
                        finally:
                            p.terminate()
                    else:
                        mic_name = "Default Microphone"
                except Exception as e:
                    print(f"Error getting microphone info: {e}")
                    mic_name = "Default Microphone"
            
            # Update the status label
            self.audio_status_label.config(text=f"Audio Status: {mic_name}")
            print(f"Audio status updated: {mic_name}")
            
        except Exception as e:
            print(f"Error updating audio status: {e}")
            self.audio_status_label.config(text="Audio Status: Unknown")
    
    def update_cost_display(self):
        """Update the cost display in the UI"""
        try:
            self.cost_label.config(text=f"Session Cost: ${self.session_cost:.4f}")
        except Exception as e:
            print(f"Error updating cost display: {e}")
    
    def update_api_counter_display(self):
        """Update the API counter display in the UI"""
        try:
            self.api_counter_label.config(text=f"API Calls: {self.api_call_count}")
        except Exception as e:
            print(f"Error updating API counter display: {e}")
    
    def add_to_session_cost(self, cost):
        """Add cost to session total and update display"""
        try:
            self.session_cost += cost
            self.update_cost_display()
            print(f"Added ${cost:.4f} to session cost. Total: ${self.session_cost:.4f}")
        except Exception as e:
            print(f"Error updating session cost: {e}")
    
    def increment_api_counter(self):
        """Increment API call counter and update display"""
        try:
            self.api_call_count += 1
            self.update_api_counter_display()
            print(f"API call #{self.api_call_count} made")
        except Exception as e:
            print(f"Error updating API counter: {e}")
    
    def update_ai_status(self, status):
        """Update the AI analysis status display"""
        try:
            self.ai_status_label.config(text=f"AI Status: {status}")
            print(f"AI Status updated: {status}")
        except Exception as e:
            print(f"Error updating AI status: {e}")
    
    def update_question_type_display(self, question_type):
        """Update the question type indicator"""
        try:
            self.question_type_label.config(text=f"Type: {question_type.title()}")
            print(f"Question type updated: {question_type}")
        except Exception as e:
            print(f"Error updating question type: {e}")
    
    def reset_session_cost(self):
        """Reset session cost and API counter to zero"""
        try:
            self.session_cost = 0.0
            self.api_call_count = 0
            self.update_cost_display()
            self.update_api_counter_display()
            print("Session cost and API counter reset to 0")
        except Exception as e:
            print(f"Error resetting session cost and API counter: {e}")
    
    def generate_ai_analysis(self, category, explanation):
        """Generate AI-driven analysis for a topic"""
        try:
            # Update AI status to processing
            self.root.after(0, lambda: self.update_ai_status("Processing..."))
            
            # Use OpenAI if available and enabled
            if OPENAI_AVAILABLE and openai_analyzer and openai_analyzer.is_available():
                print(f"Using OpenAI for topic analysis: {category}")
                print(f"OpenAI_AVAILABLE: {OPENAI_AVAILABLE}")
                print(f"openai_analyzer.is_available(): {openai_analyzer.is_available()}")
                self.root.after(0, lambda: self.update_ai_status("Calling OpenAI..."))
                ai_content, cost = openai_analyzer.generate_topic_analysis(
                    category,
                    explanation,
                    stream_callback=self._make_ai_stream_callback(),
                )
                if cost > 0:
                    # Schedule cost and API counter updates on main thread to prevent hanging
                    self.root.after(0, lambda: self.add_to_session_cost(cost))
                    self.root.after(0, lambda: self.increment_api_counter())
                self.root.after(0, lambda: self.update_ai_status("Completed"))
                return ai_content
            else:
                # Fallback to template-based analysis
                print(f"Using template fallback for topic analysis: {category}")
                self.root.after(0, lambda: self.update_ai_status("Using Template"))
                ai_content = self._get_template_ai_analysis(category, explanation)
                self.root.after(0, lambda: self.update_ai_status("Completed"))
                return ai_content
            
        except Exception as e:
            print(f"AI analysis generation failed: {e}")
            self.root.after(0, lambda: self.update_ai_status("Error"))
            return f"AI analysis generation failed: {e}"
    
    def _get_template_ai_analysis(self, category, explanation):
        """Template-based AI analysis fallback"""
        ai_content = f"🤖 AI-Enhanced Analysis: {explanation['title']}\n\n"
        
        # Include boot process explanation if available (for Linux topic)
        if 'boot_process' in explanation:
            ai_content += f"🚀 **Boot Process Overview:**\n"
            ai_content += f"{explanation['boot_process']}\n\n"
        
        ai_content += f"📊 **Advanced Insights:**\n"
        ai_content += f"• This topic is commonly encountered in {category} environments\n"
        ai_content += f"• Key performance indicators to monitor\n"
        ai_content += f"• Best practices for optimization\n\n"
        
        ai_content += f"🔧 **Advanced Commands:**\n"
        ai_content += f"• Performance monitoring: `htop`, `iostat`, `netstat`\n"
        ai_content += f"• Debugging: `strace`, `gdb`, `valgrind`\n"
        ai_content += f"• Log analysis: `grep`, `awk`, `sed`\n\n"
        
        ai_content += f"⚠️ **Common Pitfalls:**\n"
        ai_content += f"• Memory leaks and resource management\n"
        ai_content += f"• Security vulnerabilities to watch for\n"
        ai_content += f"• Performance bottlenecks\n\n"
        
        ai_content += f"🚀 **Next Steps:**\n"
        ai_content += f"• Consider implementing monitoring\n"
        ai_content += f"• Review security best practices\n"
        ai_content += f"• Plan for scalability\n\n"
        
        ai_content += f"💡 **AI Suggestion:**\n"
        ai_content += f"Based on the topic '{category}', consider exploring related technologies "
        ai_content += f"and implementing automated testing and monitoring solutions."
        
        return ai_content
    
    def generate_ai_troubleshooting(self, question_text, suggestions):
        """Generate AI-driven troubleshooting analysis"""
        try:
            # Update AI status to processing
            self.root.after(0, lambda: self.update_ai_status("Processing..."))
            
            # Use OpenAI if available and enabled
            if OPENAI_AVAILABLE and openai_analyzer and openai_analyzer.is_available():
                print(f"Using OpenAI for troubleshooting analysis")
                print(f"OpenAI_AVAILABLE: {OPENAI_AVAILABLE}")
                print(f"openai_analyzer.is_available(): {openai_analyzer.is_available()}")
                self.root.after(0, lambda: self.update_ai_status("Calling OpenAI..."))
                ai_content, cost = openai_analyzer.generate_troubleshooting_analysis(
                    question_text,
                    suggestions,
                    stream_callback=self._make_ai_stream_callback(),
                )
                if cost > 0:
                    # Schedule cost and API counter updates on main thread to prevent hanging
                    self.root.after(0, lambda: self.add_to_session_cost(cost))
                    self.root.after(0, lambda: self.increment_api_counter())
                self.root.after(0, lambda: self.update_ai_status("Completed"))
                return ai_content
            else:
                # Fallback to template-based analysis
                print(f"Using template fallback for troubleshooting analysis")
                self.root.after(0, lambda: self.update_ai_status("Using Template"))
                ai_content = self._get_template_troubleshooting_analysis(question_text, suggestions)
                self.root.after(0, lambda: self.update_ai_status("Completed"))
                return ai_content
            
        except Exception as e:
            print(f"AI troubleshooting analysis generation failed: {e}")
            self.root.after(0, lambda: self.update_ai_status("Error"))
            return f"AI troubleshooting analysis generation failed: {e}"
    
    def _get_template_troubleshooting_analysis(self, question_text, suggestions):
        """Template-based troubleshooting analysis fallback"""
        ai_content = f"🤖 AI Troubleshooting Analysis\n\n"
        ai_content += f"📝 **Question Analysis:**\n"
        ai_content += f"• Detected question type: Technical troubleshooting\n"
        ai_content += f"• Complexity level: Intermediate to Advanced\n"
        ai_content += f"• Context: {question_text[:100]}...\n\n"
        
        ai_content += f"🎯 **AI-Enhanced Approach:**\n"
        ai_content += f"• Systematic debugging methodology\n"
        ai_content += f"• Root cause analysis techniques\n"
        ai_content += f"• Performance optimization strategies\n\n"
        
        ai_content += f"🔍 **Advanced Diagnostics:**\n"
        ai_content += f"• Log analysis with `grep`, `awk`, `sed`\n"
        ai_content += f"• System monitoring with `htop`, `iostat`\n"
        ai_content += f"• Network analysis with `netstat`, `tcpdump`\n\n"
        
        ai_content += f"⚡ **Quick Wins:**\n"
        ai_content += f"• Check system resources first\n"
        ai_content += f"• Verify configuration files\n"
        ai_content += f"• Test with minimal configuration\n\n"
        
        ai_content += f"🚀 **Long-term Solutions:**\n"
        ai_content += f"• Implement monitoring and alerting\n"
        ai_content += f"• Document the resolution process\n"
        ai_content += f"• Create runbooks for future reference\n\n"
        
        ai_content += f"💡 **AI Recommendation:**\n"
        ai_content += f"Consider implementing automated testing and monitoring to prevent similar issues in the future."
        
        return ai_content
    
    def generate_contextual_ai_analysis(self, category, explanation, full_transcription, detected_keyword):
        """Generate AI analysis based on keywords AND full transcription context"""
        try:
            print(f"DEBUG: generate_contextual_ai_analysis called with category='{category}', keyword='{detected_keyword}'")
            print(f"DEBUG: Full transcription: '{full_transcription}'")
            
            # Update AI status to processing
            self.root.after(0, lambda: self.update_ai_status("Processing..."))
            
            # Detect question type with evolution tracking
            question_type = self.detect_question_type_evolution(full_transcription, full_transcription)
            print(f"Current question type: {question_type}")
            print(f"DEBUG: Full transcription for analysis: '{full_transcription}'")
            print(f"DEBUG: Question type history: {self.question_type_history}")
            
            # Update question type display
            self.root.after(0, lambda: self.update_question_type_display(question_type))
            
            # Use OpenAI if available and enabled
            if OPENAI_AVAILABLE and openai_analyzer and openai_analyzer.is_available():
                print(f"Using OpenAI for contextual analysis: {category} (type: {question_type})")
                print(f"OpenAI_AVAILABLE: {OPENAI_AVAILABLE}")
                print(f"openai_analyzer.is_available(): {openai_analyzer.is_available()}")
                self.root.after(0, lambda: self.update_ai_status("Calling OpenAI..."))
                
                # Create enhanced context for OpenAI with question type
                enhanced_explanation = explanation.copy()
                
                # Format the context more clearly to preserve the complete question
                context_parts = []
                context_parts.append("COMPLETE USER QUESTION:")
                context_parts.append(full_transcription)
                context_parts.append("")
                context_parts.append("DETECTED KEYWORD:")
                context_parts.append(detected_keyword)
                context_parts.append("")
                context_parts.append("QUESTION TYPE:")
                context_parts.append(question_type)
                context_parts.append("")
                context_parts.append("IMPORTANT: Focus ONLY on the user's actual question. Do not inject irrelevant content.")
                context_parts.append("Please provide a comprehensive response that addresses the ENTIRE question above.")
                
                enhanced_explanation['context'] = "\n".join(context_parts)
                enhanced_explanation['detected_keyword'] = detected_keyword
                enhanced_explanation['question_type'] = question_type
                enhanced_explanation['session_context'] = f"This is part of an ongoing technical discussion session. Question type: {question_type}"
                
                ai_content, cost = openai_analyzer.generate_topic_analysis(
                    category,
                    enhanced_explanation,
                    stream_callback=self._make_ai_stream_callback(),
                )
                if cost > 0:
                    # Schedule cost and API counter updates on main thread to prevent hanging
                    self.root.after(0, lambda: self.add_to_session_cost(cost))
                    self.root.after(0, lambda: self.increment_api_counter())
                
                # Update status to completed
                self.root.after(0, lambda: self.update_ai_status("Completed"))
                return ai_content
            else:
                # Fallback to template-based analysis with context and question type
                print(f"Using template fallback for contextual analysis: {category} (type: {question_type})")
                print(f"OpenAI_AVAILABLE: {OPENAI_AVAILABLE}")
                print(f"openai_analyzer.is_available(): {openai_analyzer.is_available() if OPENAI_AVAILABLE else 'N/A'}")
                self.root.after(0, lambda: self.update_ai_status("Using Template"))
                ai_content = self._get_contextual_template_analysis(category, explanation, full_transcription, detected_keyword, question_type)
                self.root.after(0, lambda: self.update_ai_status("Completed"))
                return ai_content
            
        except Exception as e:
            print(f"Contextual AI analysis generation failed: {e}")
            self.root.after(0, lambda: self.update_ai_status("Error"))
            return f"Contextual AI analysis generation failed: {e}"
    
    def _get_contextual_template_analysis(self, category, explanation, full_transcription, detected_keyword, question_type='troubleshooting'):
        """Template-based contextual analysis fallback with adaptive question types"""
        ai_content = f"🤖 AI-Enhanced Analysis: {explanation['title']}\n\n"
        ai_content += f"📊 **Context-Aware Insights:**\n"
        ai_content += f"• Detected keyword: '{detected_keyword}' in category '{category}'\n"
        ai_content += f"• Question type: {question_type.title()}\n"
        ai_content += f"• Session context: {len(full_transcription.split())} words transcribed\n"
        ai_content += f"• Complete question: {full_transcription}\n"
        
        # Adaptive content based on question type
        if question_type == 'architecture':
            ai_content += f"🏗️ **System Architecture & Design Framework:**\n"
            ai_content += f"• System architecture patterns and design principles for {category}\n"
            ai_content += f"• Scalability and performance design strategies\n"
            ai_content += f"• Component interaction and service boundaries\n"
            ai_content += f"• Technology stack and tool recommendations\n"
            ai_content += f"• Integration patterns and best practices\n"
            ai_content += f"• Monitoring and observability architecture\n"
            ai_content += f"• Security architecture and design considerations\n"
            ai_content += f"• Deployment and infrastructure patterns\n"
            ai_content += f"• Data flow and processing architecture\n"
            ai_content += f"• Fault tolerance and resilience patterns\n"
            ai_content += f"• High availability and disaster recovery\n"
            ai_content += f"• User experience and interface design considerations\n"
            ai_content += f"• Workflow and process design optimization\n"
            ai_content += f"• Testing and validation design strategies\n"
            ai_content += f"• Compliance and regulatory considerations\n\n"
            
            
        elif question_type == 'policy':
            ai_content += f"• Policy and governance considerations for {category}\n"
            ai_content += f"• Compliance and regulatory requirements\n"
            ai_content += f"• Best practices and standards\n\n"
            
            ai_content += f"📋 **Policy Guidance:**\n"
            ai_content += f"• Governance frameworks: ITIL, COBIT, NIST\n"
            ai_content += f"• Compliance standards: GDPR, HIPAA, SOX, PCI-DSS\n"
            ai_content += f"• Documentation requirements: Policies, Procedures, Guidelines\n"
            ai_content += f"• Approval workflows: Change management, Risk assessment\n\n"
            
        elif question_type == 'security':
            ai_content += f"• Security considerations for {category} implementations\n"
            ai_content += f"• Threat modeling and risk assessment\n"
            ai_content += f"• Security controls and monitoring\n\n"
            
            ai_content += f"🔒 **Security Guidance:**\n"
            ai_content += f"• Security frameworks: OWASP, NIST Cybersecurity Framework\n"
            ai_content += f"• Authentication: MFA, SSO, OAuth, SAML\n"
            ai_content += f"• Encryption: TLS, AES, RSA, Key management\n"
            ai_content += f"• Monitoring: SIEM, IDS/IPS, Vulnerability scanning\n\n"
            
        else:  # troubleshooting (default)
            ai_content += f"• This topic is commonly encountered in {category} environments\n"
            ai_content += f"• Key performance indicators to monitor\n"
            ai_content += f"• Best practices for optimization\n\n"
            
            ai_content += f"🔧 **Troubleshooting Commands:**\n"
            ai_content += f"• Performance monitoring: `htop`, `iostat`, `netstat`\n"
            ai_content += f"• Debugging: `strace`, `gdb`, `valgrind`\n"
            ai_content += f"• Log analysis: `grep`, `awk`, `sed`\n\n"
        
        ai_content += f"⚠️ **Common Considerations:**\n"
        if question_type == 'architecture':
            ai_content += f"• System complexity and maintainability\n"
            ai_content += f"• Performance bottlenecks and scalability limits\n"
            ai_content += f"• Integration challenges and dependencies\n"
        elif question_type == 'design':
            ai_content += f"• User experience and usability issues\n"
            ai_content += f"• Accessibility and inclusive design\n"
            ai_content += f"• Performance impact on user interactions\n"
        elif question_type == 'policy':
            ai_content += f"• Compliance gaps and regulatory risks\n"
            ai_content += f"• Policy enforcement and monitoring\n"
            ai_content += f"• Change management and approval processes\n"
        elif question_type == 'security':
            ai_content += f"• Security vulnerabilities and attack vectors\n"
            ai_content += f"• Access control and privilege escalation\n"
            ai_content += f"• Data protection and privacy concerns\n"
        else:
            ai_content += f"• Memory leaks and resource management\n"
            ai_content += f"• Security vulnerabilities to watch for\n"
            ai_content += f"• Performance bottlenecks\n"
        
        ai_content += f"\n🚀 **Next Steps:**\n"
        if question_type == 'architecture':
            ai_content += f"• Create architectural diagrams and documentation\n"
            ai_content += f"• Evaluate technology stack and dependencies\n"
            ai_content += f"• Plan for scalability and performance testing\n"
        elif question_type == 'design':
            ai_content += f"• Create wireframes and prototypes\n"
            ai_content += f"• Conduct user research and testing\n"
            ai_content += f"• Iterate on design based on feedback\n"
        elif question_type == 'policy':
            ai_content += f"• Review compliance requirements\n"
            ai_content += f"• Document policies and procedures\n"
            ai_content += f"• Establish approval workflows\n"
        elif question_type == 'security':
            ai_content += f"• Conduct security assessment and testing\n"
            ai_content += f"• Implement security controls and monitoring\n"
            ai_content += f"• Establish incident response procedures\n"
        else:
            ai_content += f"• Consider implementing monitoring\n"
            ai_content += f"• Review security best practices\n"
            ai_content += f"• Plan for scalability\n"
        
        ai_content += f"\n💡 **AI Suggestion:**\n"
        if question_type == 'architecture':
            ai_content += f"Based on your {question_type} question about {category}, consider exploring architectural patterns, "
            ai_content += f"design principles, and scalability strategies that align with your system requirements."
        elif question_type == 'design':
            ai_content += f"Based on your {question_type} question about {category}, consider user-centered design approaches, "
            ai_content += f"prototyping methodologies, and usability testing to create effective solutions."
        elif question_type == 'policy':
            ai_content += f"Based on your {question_type} question about {category}, consider governance frameworks, "
            ai_content += f"compliance requirements, and best practices for establishing effective policies."
        elif question_type == 'security':
            ai_content += f"Based on your {question_type} question about {category}, consider security frameworks, "
            ai_content += f"threat modeling, and defense-in-depth strategies for robust security implementation."
        else:
            ai_content += f"Based on the keyword '{detected_keyword}' in your discussion about {category}, "
            ai_content += f"consider exploring related technologies and implementing automated testing and monitoring solutions."
        
        return ai_content
    
    def check_and_analyze_keywords_throttled(self, text):
        """Throttled version of keyword analysis to prevent UI freezing"""
        try:
            current_time = time.time()
            
            # Check if enough time has passed since last analysis
            if current_time - self.last_ai_analysis_time < self.ai_analysis_throttle_seconds:
                return  # Skip analysis if too soon
            
            # Get current full transcription
            current_transcription = self.transcription_text.get("1.0", tk.END).strip()
            
            # Only analyze if transcription has changed since last analysis
            # After Clear Text, last_analyzed_transcription is "", so any new content should trigger analysis
            if current_transcription == self.last_analyzed_transcription:
                return  # No new content to analyze
            
            # If transcription is empty or whitespace-only, don't analyze
            # Note: current_transcription is already stripped, so just check if it's empty
            if not current_transcription:
                return
            
            # Check if AI analysis is already running
            with self.ai_analysis_lock:
                if self.ai_analysis_running:
                    # Buffer the new content for processing after current analysis completes
                    self.transcription_buffer = current_transcription
                    self.pending_ai_analysis = True
                    print(f"AI analysis already running, buffering new content ({len(text)} chars)")
                    return
                self.ai_analysis_running = True
            
            # Run keyword analysis in a separate thread to prevent UI blocking
            # Use a more aggressive approach - just start the thread and don't wait
            analysis_thread = threading.Thread(
                target=self.check_and_analyze_keywords_async,
                args=(text, current_transcription),
                daemon=True
            )
            analysis_thread.start()
            
        except Exception as e:
            print(f"Error in throttled keyword analysis: {e}")
            with self.ai_analysis_lock:
                self.ai_analysis_running = False
    
    def check_and_analyze_keywords_async(self, text, full_transcription):
        """Run keyword analysis without releasing the concurrency lock early."""
        try:
            self.check_and_analyze_keywords(text, full_transcription)
        except Exception as e:
            print(f"Error in async keyword analysis: {e}")
            with self.ai_analysis_lock:
                self.ai_analysis_running = False
    
    def check_and_analyze_keywords(self, text, full_transcription):
        """Check for keywords in text and automatically trigger AI analysis"""
        try:
            if self._analyze_transcription_for_keywords(full_transcription):
                return

            # No new keyword context; still refresh tracking for throttling.
            self.last_analyzed_transcription = full_transcription

        except Exception as e:
            print(f"Error in keyword analysis: {e}")
        finally:
            # Always release the lock when done
            with self.ai_analysis_lock:
                self.ai_analysis_running = False
                
                # Check if there's buffered content to process
                if self.pending_ai_analysis and self.transcription_buffer:
                    print("Processing buffered transcription content...")
                    # Process the buffered content in a new thread
                    buffered_thread = threading.Thread(
                        target=self.process_buffered_content,
                        daemon=True
                    )
                    buffered_thread.start()
    
    def process_buffered_content(self):
        """Process buffered transcription content after AI analysis completes"""
        try:
            # Reset the pending flag and get the buffered content
            self.pending_ai_analysis = False
            buffered_transcription = self.transcription_buffer
            self.transcription_buffer = ""
            
            if not buffered_transcription:
                return

            if self._analyze_transcription_for_keywords(buffered_transcription):
                return

            self.last_analyzed_transcription = buffered_transcription
            print("Buffered content processed, no new keyword context found")

        except Exception as e:
            print(f"Error processing buffered content: {e}")
    
    def show_topic_explanation(self, category):
        """Display topic explanation for the clicked keyword category"""
        # Check cache first
        cached_explanation = self.get_cached_explanation(category)
        if cached_explanation:
            # Use cached explanation
            self.topic_text.config(state=tk.NORMAL)
            self.topic_text.delete("1.0", tk.END)
            self.topic_text.insert("1.0", cached_explanation)
            self.topic_text.config(state=tk.DISABLED)
            return
        
        explanations = self.get_topic_explanations()
        
        if category in explanations:
            explanation = explanations[category]
            
            # Clear and update topic text
            self.topic_text.config(state=tk.NORMAL)
            self.topic_text.delete("1.0", tk.END)
            
            # Format the explanation
            formatted_text = f"Topic: {explanation['title']}\n\n"
            formatted_text += f"Summary: {explanation['summary']}\n\n"
            
            # Include boot process explanation if available (for Linux topic)
            if 'boot_process' in explanation:
                formatted_text += f"{explanation['boot_process']}\n\n"
            
            formatted_text += f"Technical Challenges:\n{explanation['challenges']}\n\n"
            formatted_text += f"Useful Commands:\n{explanation['commands']}"
            
            self.topic_text.insert("1.0", formatted_text)
            self.topic_text.config(state=tk.DISABLED)
            
            # Cache the explanation
            self.cache_explanation(category, formatted_text)
            
            # Show AI analysis for the topic (if enabled)
            if self.ai_enabled_var.get():
                ai_content = self.generate_ai_analysis(category, explanation)
                self.show_ai_analysis(ai_content)
            else:
                self.show_ai_analysis("AI Analysis disabled. Enable the checkbox to see AI-enhanced insights.")
    
    def get_topic_explanations(self):
        """Get topic explanations loaded from conf/topic_explanations.conf"""
        return self.topic_explanations

    def detect_question(self, text):
        """Detect if the text contains a question"""
        text_lower = text.lower()
        
        # Check for question patterns
        for pattern in self.question_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        # Check for question words at the beginning
        question_words = ['how', 'what', 'why', 'when', 'where', 'which', 'who', 'can', 'could', 'should', 'would', 'do', 'does', 'did', 'will', 'is', 'are', 'was', 'were']
        words = text_lower.split()
        if words and words[0] in question_words:
            return True
        
        # Enhanced detection for computational job issues
        computational_indicators = [
            'job', 'jobs', 'slurm', 'squeue', 'sbatch', 'srun', 'sacct', 'scontrol',
            'open ondemand', 'ondemand', 'ood', 'web interface', 'hpc portal', 'cluster portal',
            'pbs', 'torque', 'qstat', 'qsub', 'qdel', 'pbsnodes', 'showq', 'maui', 'moab',
            'queue', 'pending', 'running', 'failed', 'cancelled', 'timeout', 'hung', 'stuck',
            'slow', 'performance', 'bottleneck', 'resource', 'memory', 'cpu', 'disk',
            'computational', 'compute', 'hpc', 'cluster', 'batch', 'parallel', 'mpi', 'openmp',
            'walltime', 'wall time', 'time limit', 'allocation', 'priority', 'partition', 'node', 'nodes',
            'quantum', 'qubit', 'quantum algorithm', 'quantum optimization', 'quantum annealing', 'quantum supremacy', 'quantum advantage', 'quantum circuit', 'quantum gate', 'quantum error correction', 'quantum coherence', 'quantum entanglement', 'quantum superposition'
        ]
        
        # Check if any computational indicators are present
        if any(indicator in text_lower for indicator in computational_indicators):
            return True
            
        return False
    
    def detect_question_type(self, text):
        """Detect the type of question being asked for adaptive AI analysis"""
        text_lower = text.lower()
        
        # Check for specific "design" patterns first (higher priority)
        design_patterns = [
            r'\b(design a|design an|design the|designing a|designing an|designing the)\b',
            r'\b(system design|architecture design|solution design|infrastructure design)\b'
        ]
        
        print(f"DEBUG: Checking design patterns against text: '{text_lower}'")
        for pattern in design_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            print(f"DEBUG: Pattern '{pattern}' match: {match}")
            if match:
                print(f"DEBUG: Detected question type 'architecture' with high-priority design pattern '{pattern}'")
                return 'architecture'
        
        # Check each question type pattern
        print(f"DEBUG: Checking all question type patterns against text: '{text_lower}'")
        for question_type, patterns in self.question_type_patterns.items():
            print(f"DEBUG: Checking {question_type} patterns...")
            for pattern in patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                print(f"DEBUG: Pattern '{pattern}' match: {match}")
                if match:
                    print(f"DEBUG: Detected question type '{question_type}' with pattern '{pattern}'")
                    return question_type
        
        # Default to troubleshooting if no specific type detected
        print(f"DEBUG: No specific question type detected, defaulting to 'troubleshooting'")
        return 'troubleshooting'
    
    def detect_question_type_evolution(self, text, full_transcription):
        """Detect if question type should evolve based on context changes"""
        print(f"DEBUG: Evolution detection - text: '{text[:100]}...', full: '{full_transcription[:100]}...'")
        
        # Get current detected type
        detected_type = self.detect_question_type(text)
        print(f"DEBUG: Basic detection result: '{detected_type}'")
        
        # If no previous type, use detected type
        if self.current_question_type is None:
            self.current_question_type = detected_type
            self.question_type_history.append(detected_type)
            print(f"DEBUG: Initial question type set to '{detected_type}'")
            return detected_type
        
        # Calculate confidence scores for each question type
        type_scores = {}
        text_lower = text.lower()
        full_lower = full_transcription.lower()
        
        for question_type, patterns in self.question_type_patterns.items():
            score = 0
            total_patterns = len(patterns)
            
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    score += 1
                if re.search(pattern, full_lower, re.IGNORECASE):
                    score += 0.5  # Lower weight for full context
            
            type_scores[question_type] = score / total_patterns if total_patterns > 0 else 0
        
        # Find the highest scoring type
        best_type = max(type_scores, key=type_scores.get)
        best_score = type_scores[best_type]
        
        print(f"DEBUG: Question type scores: {type_scores}")
        print(f"DEBUG: Best type: '{best_type}' with score: {best_score:.2f}")
        
        # Check if we should evolve the question type
        if (best_type != self.current_question_type and 
            best_score >= self.context_evolution_threshold):
            
            print(f"DEBUG: Question type evolving from '{self.current_question_type}' to '{best_type}'")
            self.current_question_type = best_type
            self.question_type_history.append(best_type)
            
            # Keep history manageable (last 5 types)
            if len(self.question_type_history) > 5:
                self.question_type_history = self.question_type_history[-5:]
            
            return best_type
        
        # No evolution needed
        print(f"DEBUG: Question type remains '{self.current_question_type}'")
        return self.current_question_type
    
    def provide_troubleshooting_suggestions(self, question_text):
        """Provide troubleshooting suggestions based on the question"""
        # Check cache first
        question_hash = hashlib.md5(question_text.encode()).hexdigest()
        cached_suggestions = self.get_cached_troubleshooting(question_hash)
        
        if cached_suggestions:
            # Use cached suggestions
            self.topic_text.config(state=tk.NORMAL)
            self.topic_text.delete("1.0", tk.END)
            self.topic_text.insert("1.0", cached_suggestions)
            self.topic_text.config(state=tk.DISABLED)
            return
        
        suggestions = self.get_troubleshooting_suggestions(question_text)
        
        if suggestions:
            # Clear and update topic text with troubleshooting suggestions
            self.topic_text.config(state=tk.NORMAL)
            self.topic_text.delete("1.0", tk.END)
            
            # Format the suggestions
            formatted_text = f"🔧 Troubleshooting Suggestions:\n\n"
            formatted_text += f"Question: {question_text}\n\n"
            formatted_text += f"Approach: {suggestions['approach']}\n\n"
            formatted_text += f"Steps to Try:\n{suggestions['steps']}\n\n"
            formatted_text += f"Commands to Run:\n{suggestions['commands']}\n\n"
            formatted_text += f"Additional Resources:\n{suggestions['resources']}"
            
            self.topic_text.insert("1.0", formatted_text)
            self.topic_text.config(state=tk.DISABLED)
            
            # Cache the suggestions
            self.cache_troubleshooting(question_hash, formatted_text)
            
            # Show AI analysis for troubleshooting (if enabled)
            if self.ai_enabled_var.get():
                # Use the new contextual AI analysis instead of old troubleshooting
                ai_content = self.generate_contextual_ai_analysis("troubleshooting", suggestions, question_text, "troubleshooting")
                self.show_ai_analysis(ai_content)
            else:
                self.show_ai_analysis("AI Analysis disabled. Enable the checkbox to see AI-enhanced insights.")
    
    def get_troubleshooting_suggestions(self, question_text):
        """Get troubleshooting suggestions based on question content"""
        question_lower = question_text.lower()
        
        # Network/Connectivity issues
        if any(word in question_lower for word in ['network', 'connect', 'ping', 'dns', 'firewall', 'port', 'connection']):
            return {
                'approach': 'Network connectivity troubleshooting',
                'steps': '• Check network connectivity with ping\n• Verify DNS resolution\n• Test specific ports and services\n• Check firewall rules\n• Review network configuration',
                'commands': '• ping -c 4 target_host\n• nslookup domain.com\n• telnet host port\n• netstat -tulpn\n• iptables -L',
                'resources': '• Check network logs: /var/log/syslog\n• Review firewall configuration\n• Test with different network paths'
            }
        
        # PBS/Torque job scheduling issues
        elif any(word in question_lower for word in ['pbs', 'torque', 'qstat', 'qsub', 'qdel', 'pbsnodes', 'showq', 'maui', 'moab', 'qhold', 'qrls', 'qalter']):
            return {
                'approach': 'PBS/Torque job scheduling troubleshooting',
                'steps': '• Check job status and queue position\n• Review job logs and error messages\n• Verify resource requirements and availability\n• Check node status and health\n• Analyze job dependencies and priorities\n• Review user quotas and limits\n• Test with smaller resource requests',
                'commands': 'PBS/TORQUE COMMANDS:\n• qstat -u username\n• qstat -f job_id\n• pbsnodes -a\n• showq -u username\n• qdel job_id\n• qsub job_script.sh\n• qstat -Q (show queues)\n• qstat -B (show server status)\n• qalter -l walltime=2:00:00 job_id\n• qhold job_id\n• qrls job_id\n\nSLURM EQUIVALENTS:\n• squeue -u username (vs qstat -u username)\n• sbatch job_script.sh (vs qsub job_script.sh)\n• scancel job_id (vs qdel job_id)\n• sinfo -N -l (vs pbsnodes -a)\n• sacct -j job_id (vs qstat -f job_id)',
                'resources': 'PBS/TORQUE RESOURCES:\n• Check job logs: /var/spool/pbs/server_logs/\n• Review user quotas: qstat -u username\n• Check node status: pbsnodes -a\n• Review job accounting: qstat -f job_id\n• Maui scheduler logs: /var/log/maui/\n• Moab scheduler logs: /var/log/moab/\n\nSLURM EQUIVALENTS:\n• /var/log/slurm/ (vs /var/spool/pbs/server_logs/)\n• sacctmgr show user username (vs qstat -u username)\n• sinfo -N -l (vs pbsnodes -a)\n• sacct -j job_id (vs qstat -f job_id)'
            }
        
        # Slurm/Job scheduling issues
        elif any(word in question_lower for word in ['slurm', 'squeue', 'sbatch', 'srun', 'sacct', 'job', 'jobs', 'queue', 'pending', 'running', 'failed', 'cancelled', 'timeout', 'hung', 'stuck', 'open ondemand', 'ondemand', 'ood', 'web interface', 'hpc portal', 'cluster portal']):
            return {
                'approach': 'Job scheduler troubleshooting (Slurm & Open OnDemand)',
                'steps': '• Check job status and queue position\n• Review job logs and error messages\n• Verify resource requirements and availability\n• Check node status and health\n• Analyze job dependencies and priorities\n• Review user quotas and limits\n• Test with smaller resource requests\n• Check Open OnDemand web interface accessibility\n• Verify interactive application configurations\n• Review user authentication and permissions',
                'commands': 'SLURM COMMANDS:\n• squeue -u username -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"\n• sacct -j job_id --format=JobID,State,ExitCode,Start,End,Elapsed\n• scontrol show job job_id\n• sinfo -N -l\n• scontrol show partition partition_name\n• sacct -u username --starttime=YYYY-MM-DD\n\nOPEN ONDEMAND:\n• Access web portal: https://cluster.domain.edu\n• Interactive Apps: Jupyter, RStudio, MATLAB, VSCode\n• File Manager: Upload/download files\n• Job Composer: Create and submit jobs\n• Active Jobs: Monitor running jobs\n• Shell Access: Terminal access to compute nodes\n\nPBS/TORQUE COMMANDS:\n• qstat -u username\n• qstat -f job_id\n• pbsnodes -a\n• showq -u username\n• qdel job_id\n• qsub job_script.sh\n• qstat -Q (show queues)\n• qstat -B (show server status)',
                'resources': 'SLURM RESOURCES:\n• Check job logs: /var/log/slurm/\n• Review user quotas: sacctmgr show user username\n• Check node status: sinfo -N -l\n• Review job accounting: sacct -j job_id\n\nOPEN ONDEMAND RESOURCES:\n• Web interface: https://cluster.domain.edu\n• Interactive Apps: Jupyter, RStudio, MATLAB, VSCode\n• File Manager: Upload/download files\n• Job Composer: Create and submit jobs\n• Active Jobs: Monitor running jobs\n• Shell Access: Terminal access to compute nodes\n\nPBS/TORQUE RESOURCES:\n• Check job logs: /var/spool/pbs/server_logs/\n• Review user quotas: qstat -u username\n• Check node status: pbsnodes -a\n• Review job accounting: qstat -f job_id'
            }
        
        # Programming and debugging issues
        elif any(word in question_lower for word in ['code', 'coding', 'programming', 'scripting', 'development', 'debug', 'debugging', 'bug', 'error', 'exception', 'crash', 'compile', 'compilation', 'build', 'make', 'link', 'syntax', 'semantic', 'logic', 'algorithm', 'memory', 'leak', 'segmentation', 'fault', 'core dump', 'performance', 'optimization', 'profiling', 'benchmark', 'function', 'variable', 'loop', 'condition', 'recursion', 'api', 'library', 'framework', 'dependency', 'package']):
            return {
                'approach': 'Programming and debugging troubleshooting',
                'steps': '• Identify the programming language and environment\n• Check for syntax and compilation errors\n• Analyze runtime errors and exceptions\n• Review memory usage and potential leaks\n• Test with debugging tools and profilers\n• Verify dependencies and library versions\n• Check for logical errors and algorithm issues',
                'commands': 'DEBUGGING COMMANDS:\n• gdb ./program (C/C++)\n• python -m pdb script.py (Python)\n• node --inspect script.js (JavaScript)\n• ruby -r debug script.rb (Ruby)\n• bash -x script.sh (Shell)\n• lua -l debug script.lua (Lua)\n• powershell -Command "Get-Error" (PowerShell)\n\nCOMPILATION COMMANDS:\n• gcc -Wall -Wextra -g -o program source.c\n• make -f Makefile\n• npm run build\n• bundle exec rake\n• lua -p script.lua\n\nPROFILING COMMANDS:\n• valgrind --leak-check=full ./program\n• perf record ./program\n• strace ./program\n• time ./program',
                'resources': '• Check compiler/interpreter error messages\n• Review stack traces and exception details\n• Use debugging tools: gdb, pdb, browser dev tools\n• Profile memory usage: valgrind, heaptrack\n• Analyze performance: perf, gprof, profiler tools\n• Review documentation and API references\n• Test with minimal reproducible examples'
            }
        
        # Computational job performance issues
        elif any(word in question_lower for word in ['slow', 'performance', 'latency', 'cpu', 'memory', 'disk', 'bottleneck', 'computational', 'compute', 'hpc', 'cluster', 'batch', 'parallel', 'mpi', 'openmp']):
            return {
                'approach': 'Computational job performance analysis',
                'steps': '• Monitor job resource utilization\n• Identify performance bottlenecks\n• Check for memory leaks or excessive I/O\n• Analyze parallel scaling efficiency\n• Review job configuration and resource requests\n• Test with different resource allocations\n• Profile application performance',
                'commands': '• squeue -u username -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R" - Show detailed job queue info\n• sacct -j job_id --format=JobID,State,ExitCode,Start,End,Elapsed - Check job accounting\n• scontrol show job job_id - Display comprehensive job information\n• sinfo -N -l - List all compute nodes with detailed status\n• srun --ntasks=4 --cpus-per-task=2 ./program - Run parallel job with 4 tasks, 2 CPUs each\n• sbatch --time=2:00:00 --mem=8G job_script.sh - Submit job with 2-hour limit, 8GB memory\n• scancel job_id - Cancel running or pending job\n• salloc --time=1:00:00 --nodes=1 --ntasks=4 - Allocate resources for interactive session',
                'resources': '• Check job logs for performance issues\n• Review resource utilization reports\n• Analyze parallel scaling efficiency\n• Consider profiling tools: gprof, valgrind'
            }
        
        # Application/Service issues
        elif any(word in question_lower for word in ['service', 'application', 'daemon', 'process', 'start', 'stop', 'restart']):
            return {
                'approach': 'Service and application troubleshooting',
                'steps': '• Check service status and logs\n• Verify configuration files\n• Test service dependencies\n• Check resource availability\n• Review error messages',
                'commands': '• systemctl status service_name\n• journalctl -u service_name -f\n• ps aux | grep process_name\n• lsof -i :port\n• strace -p process_id',
                'resources': '• Check application logs\n• Review configuration files\n• Test with minimal configuration'
            }
        
        # Database issues
        elif any(word in question_lower for word in ['database', 'sql', 'mysql', 'postgres', 'mongodb', 'query', 'connection']):
            return {
                'approach': 'Database troubleshooting',
                'steps': '• Check database connectivity\n• Review query performance\n• Check database logs\n• Verify user permissions\n• Test database configuration',
                'commands': '• mysql -u user -p -e "SHOW PROCESSLIST;"\n• psql -U user -d database -c "SELECT * FROM pg_stat_activity;"\n• mongosh --eval "db.runCommand({serverStatus: 1})"\n• EXPLAIN SELECT query;',
                'resources': '• Check database logs\n• Review slow query logs\n• Monitor database metrics'
            }
        
        # Security issues
        elif any(word in question_lower for word in ['security', 'permission', 'access', 'authentication', 'authorization', 'ssl', 'certificate']):
            return {
                'approach': 'Security and access troubleshooting',
                'steps': '• Check file permissions and ownership\n• Verify SSL/TLS certificates\n• Review authentication logs\n• Check firewall rules\n• Test access controls',
                'commands': '• ls -la file_path\n• openssl x509 -in cert.pem -text -noout\n• tail -f /var/log/auth.log\n• iptables -L\n• getfacl file_path',
                'resources': '• Review security logs\n• Check certificate validity\n• Test with different users'
            }
        
        # Docker/Container issues
        elif any(word in question_lower for word in ['docker', 'container', 'image', 'kubernetes', 'pod', 'deployment']):
            return {
                'approach': 'Container and orchestration troubleshooting',
                'steps': '• Check container status and logs\n• Verify image availability\n• Check resource limits\n• Review container configuration\n• Test network connectivity',
                'commands': '• docker ps -a\n• docker logs container_id\n• docker exec -it container_id /bin/bash\n• kubectl get pods\n• kubectl describe pod pod_name',
                'resources': '• Check container logs\n• Review orchestration logs\n• Test with simple containers'
            }
        
        # iPad/Tablet repair issues
        elif any(word in question_lower for word in ['ipad', 'tablet', 'android tablet', 'samsung tablet', 'galaxy tab', 'pixel tablet', 'fire tablet', 'screen', 'display', 'digitizer', 'touchscreen', 'battery', 'charging', 'charging port', 'usb-c', 'lightning', 'home button', 'face id', 'touch id', 'camera', 'speaker', 'microphone', 'water damage', 'cracked', 'broken screen', 'dead pixel', 'ghost touch', 'unresponsive', 'not charging', 'battery drain', 'overheating', 'boot loop', 'won\'t turn on', 'stuck on logo', 'dfu mode', 'recovery mode', 'factory reset', 'apple pencil', 'smart keyboard', 'magic keyboard']):
            return {
                'approach': 'iPad and tablet repair troubleshooting',
                'steps': '• Identify the specific issue (screen, battery, charging, software, etc.)\n• Check for physical damage (cracks, water damage, dents)\n• Test basic functions (power, volume, buttons, touch response)\n• Check software status (iOS/iPadOS version, update availability)\n• Verify charging accessories and ports\n• Test in safe mode or recovery mode if needed\n• Check for warranty status before attempting repairs\n• Document symptoms and error messages',
                'commands': 'IPAD/IPHONE DIAGNOSTICS:\n• Settings > General > About (check model, iOS version, serial number)\n• Settings > Battery (check battery health and usage)\n• Settings > Privacy & Security > Analytics & Improvements > Analytics Data (check crash logs)\n• Force restart: Press and release Volume Up, then Volume Down, then hold Power button\n• DFU Mode: Connect to computer, hold Power + Home (or Volume Down on newer models) for 10 seconds\n• Recovery Mode: Connect to computer, hold Power + Home (or Volume Down) until recovery screen appears\n• iTunes/Finder: Use to restore, update, or backup device\n\nANDROID TABLET DIAGNOSTICS:\n• Settings > About tablet (check model, Android version, build number)\n• Settings > Battery (check battery usage and health)\n• Settings > Developer Options > USB Debugging (enable for advanced diagnostics)\n• Recovery Mode: Power off, then hold Power + Volume Down (varies by manufacturer)\n• Fastboot Mode: Power off, then hold Power + Volume Down (for some devices)\n• ADB commands: adb devices, adb logcat, adb shell\n• Factory Reset: Settings > System > Reset > Factory data reset\n\nCOMMON REPAIR STEPS:\n• Screen replacement: Remove broken screen, disconnect cables, install new screen\n• Battery replacement: Remove old battery, install new battery, calibrate\n• Charging port repair: Clean port, check for debris, replace if damaged\n• Water damage: Power off immediately, remove from water, dry thoroughly, check for corrosion',
                'resources': '• Check device warranty status (Apple Support, Samsung Support, etc.)\n• Review repair guides: iFixit.com, YouTube repair tutorials\n• Check for known issues: Apple Support Communities, XDA Developers, Reddit\n• Test with different charging cables and adapters\n• Check for software updates: Settings > General > Software Update\n• Backup device before attempting repairs: iCloud, iTunes, or Android backup\n• Use genuine parts when possible for best compatibility\n• Consider professional repair for complex issues (logic board, Face ID, etc.)'
            }
        
        # General troubleshooting
        else:
            return {
                'approach': 'General troubleshooting methodology',
                'steps': '• Gather information about the issue\n• Check system logs and error messages\n• Test with minimal configuration\n• Isolate the problem scope\n• Document findings and solutions',
                'commands': '• journalctl -f\n• dmesg | tail -20\n• systemctl status\n• ps aux\n• netstat -tulpn',
                'resources': '• Check relevant log files\n• Review system documentation\n• Test in isolated environment'
            }
    
    def load_cache(self):
        """Load cached topic explanations and session data"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.session_cache = cache_data.get('session_cache', {})
                    print(f"Loaded cache with {len(self.session_cache)} entries")
        except Exception as e:
            print(f"Failed to load cache: {e}")
            self.session_cache = {}
    
    def save_cache(self):
        """Save topic explanations and session data to cache"""
        try:
            cache_data = {
                'session_cache': self.session_cache,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print("Cache saved successfully")
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def get_cached_explanation(self, topic_key):
        """Get cached topic explanation"""
        cache_key = hashlib.md5(topic_key.encode()).hexdigest()
        if cache_key in self.session_cache:
            cached_data = self.session_cache[cache_key]
            # Check if cache is still valid (24 hours)
            if datetime.now() - cached_data['timestamp'] < timedelta(hours=24):
                return cached_data['explanation']
        return None
    
    def cache_explanation(self, topic_key, explanation):
        """Cache topic explanation"""
        cache_key = hashlib.md5(topic_key.encode()).hexdigest()
        self.session_cache[cache_key] = {
            'explanation': explanation,
            'timestamp': datetime.now()
        }
        # Save cache periodically
        if len(self.session_cache) % 10 == 0:
            self.save_cache()
    
    def get_cached_troubleshooting(self, question_hash):
        """Get cached troubleshooting suggestions"""
        if question_hash in self.session_cache:
            cached_data = self.session_cache[question_hash]
            # Check if cache is still valid (1 hour for troubleshooting)
            if datetime.now() - cached_data['timestamp'] < timedelta(hours=1):
                return cached_data['suggestions']
        return None
    
    def cache_troubleshooting(self, question_hash, suggestions):
        """Cache troubleshooting suggestions"""
        self.session_cache[question_hash] = {
            'suggestions': suggestions,
            'timestamp': datetime.now()
        }
    
    def clear_transcription(self):
        """Clear the transcription text and all analysis panes"""
        self.transcription_text.delete("1.0", tk.END)
        
        # Clear all keyword tags
        tags_to_remove = [tag for tag in self.transcription_text.tag_names() 
                         if tag.startswith("keyword")]
        for tag in tags_to_remove:
            self.transcription_text.tag_remove(tag, "1.0", tk.END)
        
        # Clear topic explanation pane
        self.topic_text.config(state=tk.NORMAL)
        self.topic_text.delete("1.0", tk.END)
        self.topic_text.config(state=tk.DISABLED)
        
        # Clear AI analysis pane
        self.ai_text.config(state=tk.NORMAL)
        self.ai_text.delete("1.0", tk.END)
        self.ai_text.config(state=tk.DISABLED)
        
        self.current_transcription = ""
        
        # Reset analyzed keywords to allow fresh analysis
        self.analyzed_keywords.clear()
        self.last_analyzed_transcription = ""
        
        # Reset AI analysis throttle to allow immediate analysis after clearing
        self.last_ai_analysis_time = 0
        
        # Reset AI analysis lock and buffer
        with self.ai_analysis_lock:
            self.ai_analysis_running = False
            self.transcription_buffer = ""
            self.pending_ai_analysis = False
    
    def increase_font_size(self):
        """Increase font size for all text widgets"""
        if self.current_font_size < 24:  # Maximum font size limit
            self.current_font_size += 1
            self.update_all_fonts()
    
    def decrease_font_size(self):
        """Decrease font size for all text widgets"""
        if self.current_font_size > 8:  # Minimum font size limit
            self.current_font_size -= 1
            self.update_all_fonts()
    
    def update_all_fonts(self):
        """Update font size for all text widgets"""
        # Update main transcription text
        self.transcription_text.config(font=("Arial", self.current_font_size))
        
        # Update topic and AI text with slightly smaller font
        smaller_font_size = max(8, self.current_font_size - 2)
        self.topic_text.config(font=("Arial", smaller_font_size))
        self.ai_text.config(font=("Arial", smaller_font_size))
        
        # Cancel any pending analysis
        if self.pending_analysis:
            self.root.after_cancel(self.pending_analysis)
            self.pending_analysis = None
        
        # Reset AI status
        self.update_ai_status("Ready")
        
        # Reset pause state
        self.is_paused = False
        if self.is_listening:
            self.pause_button.config(text="Pause Listening")
        
        # Reset question type evolution
        self.current_question_type = None
        self.question_type_history = []
        self.update_question_type_display("None")
        
        print("Cleared transcription, topic explanation, and AI analysis panes")

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = SpeechTranscriptionApp(root)
    
    # Handle window closing
    def on_closing():
        app.stop_listening()
        app.save_cache()  # Save cache before closing
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
