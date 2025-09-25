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
from typing import Dict, List, Tuple, Optional
import requests
import os
from datetime import datetime, timedelta

# OpenAI integration
try:
    from openai_integration import openai_analyzer
    from openai_config import openai_config
    OPENAI_AVAILABLE = True
except ImportError:
    print("OpenAI integration not available - using template fallback")
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
        
        # Audio validation settings
        self.audio_validation_enabled = True
        self.validation_failures = 0
        
        # AI analysis throttling
        self.last_ai_analysis_time = 0
        self.ai_analysis_throttle_seconds = 5  # Minimum 5 seconds between AI analyses
        
        # Question completion detection
        self.question_completion_timeout = 3  # Wait 3 seconds after last speech before analyzing
        self.last_speech_time = 0
        self.pending_analysis = None
        
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
        
        # Transcription data
        self.current_transcription = ""
        self.transcription_history = []
        
        # Initialize caching
        self.setup_caching()
        
        # Question detection patterns
        self.question_patterns = [
            r'\b(how|what|why|when|where|which|who)\b.*\?',
            r'\b(can you|could you|would you|should i|do you)\b.*\?',
            r'\b(is there|are there|does|did|will|would)\b.*\?',
            r'\b(help|troubleshoot|debug|fix|solve|resolve)\b.*\?',
            r'\b(problem|issue|error|bug|failing|broken)\b.*\?',
            r'\b(not working|doesn\'t work|won\'t start|can\'t connect)\b.*\?',
            r'\b(how do i|how can i|how to|what\'s the best way)\b.*\?',
            r'\b(any ideas|suggestions|recommendations)\b.*\?',
            # Computational job specific patterns
            r'\b(job|jobs|slurm|squeue|sbatch|srun|sacct)\b.*\?',
            r'\b(open ondemand|ondemand|ood|web interface|hpc portal|cluster portal)\b.*\?',
            r'\b(pbs|torque|qstat|qsub|qdel|pbsnodes|showq|maui|moab)\b.*\?',
            r'\b(slow|timeout|hung|stuck|running too long)\b.*\?',
            r'\b(failed|failure|error|exit code|killed)\b.*\?',
            r'\b(queue|pending|waiting|priority)\b.*\?',
            r'\b(resource|memory|cpu|node|partition)\b.*\?',
            r'\b(why is my|why did my|what happened to)\b.*\?',
            # Programming and scripting patterns
            r'\b(code|coding|programming|scripting|development)\b.*\?',
            r'\b(debug|debugging|bug|error|exception|crash)\b.*\?',
            r'\b(compile|compilation|build|make|link)\b.*\?',
            r'\b(syntax|semantic|logic|algorithm)\b.*\?',
            r'\b(memory|leak|segmentation|fault|core dump)\b.*\?',
            r'\b(performance|optimization|profiling|benchmark)\b.*\?',
            r'\b(function|variable|loop|condition|recursion)\b.*\?',
            r'\b(api|library|framework|dependency|package)\b.*\?',
            # Quantum computing patterns
            r'\b(quantum|qubit|quantum algorithm|quantum optimization|quantum annealing|quantum supremacy|quantum advantage|quantum circuit|quantum gate|quantum error correction|quantum coherence|quantum entanglement|quantum superposition)\b.*\?'
        ]
        
        # Question type detection patterns for adaptive AI analysis
        self.question_type_patterns = {
            'architecture': [
                r'\b(architecture|architectural|design|designing|structure|framework|pattern|patterns)\b',
                r'\b(how should|what\'s the best|recommend|recommendation|approach|strategy)\b',
                r'\b(scalability|scalable|performance|optimization|efficiency)\b',
                r'\b(microservices|monolith|distributed|centralized|decentralized)\b',
                r'\b(component|components|module|modules|service|services)\b'
            ],
            'design': [
                r'\b(design|designing|ui|ux|interface|user experience|usability)\b',
                r'\b(layout|layout|wireframe|mockup|prototype|prototyping)\b',
                r'\b(workflow|process|procedure|methodology|method)\b',
                r'\b(requirements|specification|spec|documentation)\b',
                r'\b(blueprint|plan|planning|roadmap)\b'
            ],
            'policy': [
                r'\b(policy|policies|governance|compliance|regulation|regulatory)\b',
                r'\b(standards|standard|guidelines|guideline|best practices)\b',
                r'\b(approval|approve|review|audit|auditing)\b',
                r'\b(process|procedure|workflow|workflow)\b',
                r'\b(security|security|access|permission|authorization)\b'
            ],
            'security': [
                r'\b(security|secure|securing|vulnerability|vulnerabilities)\b',
                r'\b(authentication|authorization|access|permission|permissions)\b',
                r'\b(encryption|encrypt|decrypt|cipher|cryptography)\b',
                r'\b(firewall|firewalls|intrusion|detection|prevention)\b',
                r'\b(compliance|audit|auditing|penetration|penetration testing)\b'
            ],
            'troubleshooting': [
                r'\b(problem|issue|error|bug|failing|broken|not working)\b',
                r'\b(troubleshoot|debug|fix|solve|resolve|repair)\b',
                r'\b(help|stuck|blocked|can\'t|won\'t|doesn\'t work)\b',
                r'\b(crash|crashed|hang|hanging|timeout|timeout)\b',
                r'\b(slow|performance|bottleneck|optimization)\b'
            ]
        }
        
        # Keyword detection patterns
        self.tech_keywords = {
            'python': ['python', 'py', 'pip', 'virtualenv', 'conda'],
            'docker': ['docker', 'container', 'dockerfile', 'kubernetes', 'k8s'],
            'aws': ['aws', 'amazon web services', 'ec2', 's3', 'lambda', 'cloudformation'],
            'linux': ['linux', 'ubuntu', 'centos', 'bash', 'shell', 'terminal'],
            'git': ['git', 'github', 'version control', 'commit', 'branch', 'merge'],
            'database': ['database', 'sql', 'mysql', 'postgresql', 'mongodb', 'redis'],
            'networking': ['network', 'tcp', 'udp', 'http', 'https', 'dns', 'firewall'],
            'security': ['security', 'encryption', 'ssl', 'tls', 'authentication', 'authorization'],
            'monitoring': ['monitoring', 'logging', 'metrics', 'prometheus', 'grafana', 'elk'],
            'ci_cd': ['ci', 'cd', 'jenkins', 'github actions', 'gitlab', 'pipeline'],
            'hft': ['hft', 'high frequency trading', 'algorithmic trading', 'market making', 'latency', 'tick data', 'order book', 'market data', 'trading algorithm'],
            'hpc': ['hpc', 'high performance computing', 'supercomputing', 'parallel computing', 'cluster computing', 'mpi', 'openmp', 'cuda', 'gpu computing', 'distributed computing'],
            'network_storage': ['network storage', 'nas', 'san', 'iscsi', 'nfs', 'cifs', 'samba', 'glusterfs', 'ceph', 'distributed storage', 'object storage', 'block storage'],
            'performance': ['performance', 'latency', 'throughput', 'bandwidth', 'fpga', 'rdma', 'roce', 'infiniband', 'kernel bypass', 'dpdk', 'spdk', 'zero copy', 'memory mapping', 'cpu affinity', 'numa', 'cache optimization', 'vectorization', 'simd', 'avx', 'optimization', 'profiling', 'benchmarking', 'quantum computing', 'quantum', 'qubit', 'quantum algorithm', 'quantum optimization', 'quantum annealing', 'quantum supremacy', 'quantum advantage', 'quantum circuit', 'quantum gate', 'quantum error correction', 'quantum coherence', 'quantum entanglement', 'quantum superposition'],
            'slurm': ['slurm', 'squeue', 'sbatch', 'srun', 'sacct', 'scontrol', 'sinfo', 'scancel', 'salloc', 'job', 'jobs', 'queue', 'partition', 'node', 'nodes', 'walltime', 'wall time', 'time limit', 'resource', 'resources', 'allocation', 'priority', 'qos', 'account', 'user', 'group', 'open ondemand', 'ondemand', 'ood', 'web interface', 'hpc portal', 'cluster portal'],
            'pbs_torque': ['pbs', 'torque', 'qstat', 'qsub', 'qdel', 'qhold', 'qrls', 'pbsnodes', 'showq', 'maui', 'moab', 'qalter', 'qselect', 'qrerun', 'qmove', 'qrun', 'qstop', 'qstart', 'qterm', 'qconfig', 'pbs_server', 'pbs_mom', 'pbs_sched'],
            'computational_jobs': ['computational', 'compute', 'hpc', 'cluster', 'batch', 'batch job', 'parallel', 'mpi', 'openmp', 'threading', 'multiprocessing', 'distributed', 'workload', 'scheduler', 'scheduling', 'queue', 'pending', 'running', 'completed', 'failed', 'cancelled', 'timeout', 'hung', 'stuck', 'slow', 'performance', 'bottleneck', 'resource', 'memory', 'cpu', 'disk', 'io', 'network', 'bandwidth', 'latency', 'throughput'],
            'programming': ['programming', 'coding', 'scripting', 'development', 'software', 'application', 'program', 'code', 'function', 'variable', 'loop', 'condition', 'algorithm', 'data structure', 'debugging', 'compilation', 'execution', 'runtime', 'syntax', 'semantic', 'library', 'framework', 'api', 'sdk'],
            'c_programming': ['c', 'c programming', 'gcc', 'clang', 'make', 'makefile', 'compiler', 'linker', 'header', 'stdio', 'stdlib', 'string', 'pointer', 'array', 'struct', 'union', 'enum', 'malloc', 'free', 'memory management', 'segmentation fault', 'core dump', 'gdb', 'valgrind'],
            'javascript': ['javascript', 'js', 'node', 'nodejs', 'npm', 'yarn', 'react', 'vue', 'angular', 'jquery', 'typescript', 'es6', 'es2015', 'async', 'await', 'promise', 'callback', 'closure', 'prototype', 'json', 'ajax', 'dom', 'browser', 'v8', 'webpack', 'babel'],
            'ruby': ['ruby', 'rails', 'gem', 'bundle', 'rake', 'irb', 'erb', 'haml', 'sass', 'scss', 'coffeescript', 'sinatra', 'rack', 'activerecord', 'migration', 'controller', 'model', 'view', 'route', 'middleware', 'rspec', 'cucumber', 'capybara'],
            'shell_scripting': ['bash', 'zsh', 'shell', 'scripting', 'command line', 'terminal', 'cli', 'alias', 'function', 'export', 'source', 'exec', 'fork', 'pipe', 'redirect', 'stdin', 'stdout', 'stderr', 'environment', 'variable', 'subshell', 'job control', 'signal', 'trap', 'cron', 'at', 'systemd'],
            'lua': ['lua', 'luajit', 'coroutine', 'metatable', 'closure', 'upvalue', 'require', 'module', 'package', 'table', 'string', 'math', 'io', 'os', 'debug', 'lua c api', 'embedding', 'scripting language'],
            'powershell': ['powershell', 'ps1', 'cmdlet', 'module', 'pipeline', 'object', 'variable', 'function', 'script', 'parameter', 'switch', 'foreach', 'where', 'select', 'sort', 'group', 'measure', 'format', 'export', 'import', 'get', 'set', 'new', 'remove', 'invoke']
        }
        
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
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.listen_button = ttk.Button(control_frame, text="Start Listening", 
                                      command=self.toggle_listening)
        self.listen_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.pause_button = ttk.Button(control_frame, text="Pause Listening", 
                                     command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_button = ttk.Button(control_frame, text="Clear Text", 
                                     command=self.clear_transcription)
        self.clear_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Cost reset button
        self.cost_reset_button = ttk.Button(control_frame, text="Reset Cost", 
                                           command=self.reset_session_cost)
        self.cost_reset_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.status_label = ttk.Label(control_frame, text="Status: Ready")
        self.status_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Audio status display
        self.audio_status_label = ttk.Label(control_frame, text="Audio Status: Default Microphone")
        self.audio_status_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # AI toggle checkbox
        self.ai_enabled_var = tk.BooleanVar(value=True)
        self.ai_checkbox = ttk.Checkbutton(
            control_frame, 
            text="Enable AI Analysis", 
            variable=self.ai_enabled_var,
            command=self.on_ai_toggle
        )
        self.ai_checkbox.pack(side=tk.LEFT, padx=(20, 0))
        
        # OpenAI status indicator
        openai_status = "OpenAI: Configured" if (OPENAI_AVAILABLE and openai_analyzer.is_available()) else "OpenAI: Template Mode"
        self.openai_status_label = ttk.Label(control_frame, text=openai_status)
        self.openai_status_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Cost tracking
        self.session_cost = 0.0
        self.cost_label = ttk.Label(control_frame, text="Session Cost: $0.00")
        self.cost_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # API call tracking
        self.api_call_count = 0
        self.api_counter_label = ttk.Label(control_frame, text="API Calls: 0")
        self.api_counter_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # AI analysis status indicator
        self.ai_status_label = ttk.Label(control_frame, text="AI Status: Ready")
        self.ai_status_label.pack(side=tk.LEFT, padx=(20, 0))
        
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
            font=("Arial", 12)
        )
        self.transcription_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bottom pane - Split into topic explanation and AI output
        bottom_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(bottom_frame, weight=1)
        
        # Create horizontal PanedWindow for bottom split
        self.bottom_paned = ttk.PanedWindow(bottom_frame, orient=tk.HORIZONTAL)
        self.bottom_paned.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Topic explanation panel
        topic_frame = ttk.LabelFrame(self.bottom_paned, text="Topic Explanation & Troubleshooting", padding="5")
        self.bottom_paned.add(topic_frame, weight=1)
        topic_frame.columnconfigure(0, weight=1)
        topic_frame.rowconfigure(0, weight=1)
        
        self.topic_text = scrolledtext.ScrolledText(
            topic_frame,
            wrap=tk.WORD,
            height=15,
            font=("Arial", 10),
            state=tk.DISABLED
        )
        self.topic_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Right side - AI-driven output panel
        ai_frame = ttk.LabelFrame(self.bottom_paned, text="AI-Driven Analysis & Suggestions", padding="5")
        self.bottom_paned.add(ai_frame, weight=1)
        ai_frame.columnconfigure(0, weight=1)
        ai_frame.rowconfigure(0, weight=1)
        
        self.ai_text = scrolledtext.ScrolledText(
            ai_frame,
            wrap=tk.WORD,
            height=15,
            font=("Arial", 10),
            state=tk.DISABLED
        )
        self.ai_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
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
            # Try to use Whisper for offline recognition
            import whisper
            self.whisper_model = whisper.load_model("base")  # Start with base model for speed
            print("Whisper offline recognition setup completed")
            self.status_label.config(text="Status: Whisper ready - offline recognition available")
        except ImportError:
            print("Whisper not available, using basic offline recognition")
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
    
    def pause_listening(self):
        """Pause listening but allow AI analysis to continue"""
        if self.is_listening and not self.is_paused:
            self.is_paused = True
            self.pause_button.config(text="Resume Listening")
            self.status_label.config(text="Status: Paused (AI analysis continues)")
            print("Paused listening - AI analysis will continue...")
    
    def resume_listening(self):
        """Resume listening after pause"""
        if self.is_listening and self.is_paused:
            self.is_paused = False
            self.pause_button.config(text="Pause Listening")
            self.status_label.config(text="Status: Listening...")
            print("Resumed listening...")
    
    def listen_continuously(self):
        """Continuously listen for speech in a separate thread"""
        print("Starting continuous listening...")
        while self.is_listening:
            try:
                # Skip audio processing if paused
                if self.is_paused:
                    time.sleep(0.1)  # Short sleep when paused
                    continue
                
                print("Listening for audio...")
                with self.microphone as source:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                
                print("Audio captured, processing...")
                # Recognize speech in a separate thread to avoid blocking
                recognition_thread = threading.Thread(
                    target=self.process_audio, 
                    args=(audio,), 
                    daemon=True
                )
                recognition_thread.start()
                
            except sr.WaitTimeoutError:
                # Timeout is normal, continue listening
                continue
            except Exception as e:
                print(f"Error in speech recognition: {e}")
                time.sleep(0.1)
        print("Stopped continuous listening")
    
    def check_network_connection(self):
        """Check if internet connection is available"""
        try:
            # Quick network check
            response = requests.get("https://www.google.com", timeout=3)
            return response.status_code == 200
        except:
            return False
    
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
                with wave.open(wav_buffer, 'rb') as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    audio_array = np.frombuffer(frames, dtype=np.int16)
            
            # Check if audio array is not empty
            if len(audio_array) == 0:
                print("Audio array is empty")
                return False
            
            # Calculate RMS energy
            rms_energy = np.sqrt(np.mean(audio_array**2))
            
            # Check if audio has sufficient energy (not silence)
            if rms_energy < self.min_audio_energy:
                print(f"Audio too quiet: RMS energy {rms_energy:.2f} < {self.min_audio_energy}")
                return False
            
            # Check for maximum duration (avoid processing very long audio)
            duration = len(audio_array) / 16000  # Assuming 16kHz sample rate
            if duration > self.max_audio_duration:
                print(f"Audio too long: {duration:.2f}s > {self.max_audio_duration}s")
                return False
            
            print(f"Audio valid: {len(audio_array)} samples, {duration:.2f}s, RMS {rms_energy:.2f}")
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
    
    def process_audio(self, audio):
        """Process audio and update transcription with fallback"""
        try:
            print("Processing audio...")  # Debug output
            
            # Check network connection first
            if not self.check_network_connection():
                self.use_offline = True
                self.status_label.config(text="Status: Offline mode - using local recognition")
            
            # Check if audio is valid for processing
            audio_data = audio.get_wav_data()
            if self.audio_validation_enabled and not self.is_audio_valid(audio_data):
                self.validation_failures += 1
                print(f"Audio validation failed - skipping audio chunk (failures: {self.validation_failures})")
                
                # If too many failures, disable validation temporarily
                if self.validation_failures > 10:
                    print("Too many validation failures - disabling audio validation temporarily")
                    self.audio_validation_enabled = False
                return  # Skip invalid audio (too short, too quiet, etc.)
            
            if self.use_offline:
                # Try offline recognition with Whisper
                print("Using offline Whisper recognition...")
                text = self.recognize_offline(audio)
                print(f"Offline Whisper result: '{text}'")
            else:
                # Try Whisper first if available (better accuracy)
                if hasattr(self, 'whisper_model') and self.whisper_model is not None:
                    print("Using Whisper for recognition...")
                    text = self.recognize_offline(audio)
                    print(f"Whisper result: '{text}'")
                    # If Whisper fails or returns empty, try Google as fallback (only if FLAC is available)
                    if not text.strip() and self.google_fallback_enabled:
                        # Check FLAC availability before trying Google
                        if self.check_flac_availability():
                            print("Whisper returned empty result, trying Google...")
                            try:
                                text = self.recognizer.recognize_google(
                                    audio, 
                                    language='en-US',
                                    show_all=False,
                                    with_confidence=False
                                )
                            except Exception as e:
                                print(f"Google fallback also failed: {e}")
                                text = ""
                        else:
                            print("Google fallback disabled - FLAC not available")
                else:
                    # Check FLAC availability before trying Google
                    if self.check_flac_availability():
                        try:
                            # Use Google with language hints and better settings
                            text = self.recognizer.recognize_google(
                                audio, 
                                language='en-US',
                                show_all=False,
                                with_confidence=False
                            )
                        except OSError as e:
                            if "FLAC" in str(e):
                                print("FLAC not available, disabling Google fallback")
                                self.google_fallback_enabled = False
                                text = ""
                            else:
                                raise e
                    else:
                        print("Google Speech Recognition disabled - FLAC not available")
                        text = ""
            
            if text.strip():
                # Update transcription in main thread
                self.root.after(0, self.update_transcription, text)
                
        except sr.UnknownValueError:
            # Speech was unintelligible
            pass
        except sr.RequestError as e:
            print(f"Could not request results from speech recognition service: {e}")
            # Fallback to offline recognition
            self.use_offline = True
            try:
                text = self.recognize_offline(audio)
                if text.strip():
                    self.root.after(0, self.update_transcription, text)
            except Exception as offline_error:
                print(f"Offline recognition also failed: {offline_error}")
    
    def recognize_offline(self, audio):
        """Offline speech recognition fallback using Whisper"""
        try:
            if hasattr(self, 'whisper_model') and self.whisper_model is not None:
                # Convert audio to format Whisper can use
                import io
                import wave
                import numpy as np
                
                # Get audio data
                audio_data = audio.get_wav_data()
                
                # Check if audio is valid for processing
                if not self.is_audio_valid(audio_data):
                    return ""  # Skip invalid audio
                
                # Create a temporary file for Whisper
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_file_path = temp_file.name
                
                try:
                    # Use Whisper to transcribe with better error handling
                    result = self.whisper_model.transcribe(
                        temp_file_path,
                        language='en',
                        task='transcribe',
                        fp16=False,  # Use fp32 for better compatibility
                        verbose=False,
                        condition_on_previous_text=False,  # Don't depend on previous text
                        initial_prompt=None  # No initial prompt
                    )
                    
                    # Check if result is valid
                    if result and "text" in result:
                        text = result["text"].strip()
                        # Only return non-empty results
                        return text if text else ""
                    else:
                        return ""
                        
                except Exception as e:
                    print(f"Whisper transcription error: {e}")
                    return ""
                        
                finally:
                    # Clean up temporary file
                    import os
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
            else:
                return "Whisper model not available"
        except Exception as e:
            print(f"Whisper recognition failed: {e}")
            # Don't return error message, just return empty string
            return ""
    
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
        
        # Highlight keywords asynchronously
        threading.Thread(
            target=self.highlight_keywords,
            daemon=True
        ).start()
        
        # Update current transcription
        self.current_transcription += text + " "
    
    def schedule_delayed_analysis(self):
        """Schedule AI analysis after question completion timeout"""
        try:
            # Check if enough time has passed since last speech
            current_time = time.time()
            if current_time - self.last_speech_time >= self.question_completion_timeout:
                print(f"Question completion detected, scheduling AI analysis...")
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
        """Highlight detected keywords in the transcription"""
        # Get current text
        current_text = self.transcription_text.get("1.0", tk.END)
        
        # Clear existing tags
        for tag in self.transcription_text.tag_names():
            self.transcription_text.tag_remove(tag, "1.0", tk.END)
        
        # Configure keyword highlighting
        self.transcription_text.tag_configure("keyword", 
                                            background="yellow", 
                                            foreground="black",
                                            underline=True)
        
        # Find and highlight keywords
        for category, keywords in self.tech_keywords.items():
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = re.finditer(pattern, current_text, re.IGNORECASE)
                for match in matches:
                    start_pos = f"1.0+{match.start()}c"
                    end_pos = f"1.0+{match.end()}c"
                    self.transcription_text.tag_add("keyword", start_pos, end_pos)
                    self.transcription_text.tag_add(f"keyword_{category}", start_pos, end_pos)
    
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
            if OPENAI_AVAILABLE and openai_analyzer.is_available():
                print(f"Using OpenAI for topic analysis: {category}")
                print(f"OpenAI_AVAILABLE: {OPENAI_AVAILABLE}")
                print(f"openai_analyzer.is_available(): {openai_analyzer.is_available()}")
                self.root.after(0, lambda: self.update_ai_status("Calling OpenAI..."))
                ai_content, cost = openai_analyzer.generate_topic_analysis(category, explanation)
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
            if OPENAI_AVAILABLE and openai_analyzer.is_available():
                print(f"Using OpenAI for troubleshooting analysis")
                print(f"OpenAI_AVAILABLE: {OPENAI_AVAILABLE}")
                print(f"openai_analyzer.is_available(): {openai_analyzer.is_available()}")
                self.root.after(0, lambda: self.update_ai_status("Calling OpenAI..."))
                ai_content, cost = openai_analyzer.generate_troubleshooting_analysis(question_text, suggestions)
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
            # Update AI status to processing
            self.root.after(0, lambda: self.update_ai_status("Processing..."))
            
            # Detect question type for adaptive analysis
            question_type = self.detect_question_type(full_transcription)
            print(f"Detected question type: {question_type}")
            
            # Use OpenAI if available and enabled
            if OPENAI_AVAILABLE and openai_analyzer.is_available():
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
                context_parts.append("Please provide a comprehensive response that addresses the ENTIRE question above.")
                
                enhanced_explanation['context'] = "\n".join(context_parts)
                enhanced_explanation['detected_keyword'] = detected_keyword
                enhanced_explanation['question_type'] = question_type
                enhanced_explanation['session_context'] = f"This is part of an ongoing technical discussion session. Question type: {question_type}"
                
                ai_content, cost = openai_analyzer.generate_topic_analysis(category, enhanced_explanation)
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
            ai_content += f"• Architectural considerations for {category} systems\n"
            ai_content += f"• Scalability and performance design patterns\n"
            ai_content += f"• Component interaction and dependencies\n\n"
            
            ai_content += f"🏗️ **Architectural Guidance:**\n"
            ai_content += f"• Design patterns: Microservices, Event-driven, CQRS\n"
            ai_content += f"• Scalability strategies: Horizontal vs vertical scaling\n"
            ai_content += f"• Integration patterns: API Gateway, Service Mesh\n"
            ai_content += f"• Data architecture: CQRS, Event Sourcing, Caching\n\n"
            
        elif question_type == 'design':
            ai_content += f"• Design principles and best practices for {category}\n"
            ai_content += f"• User experience and interface considerations\n"
            ai_content += f"• Workflow and process optimization\n\n"
            
            ai_content += f"🎨 **Design Guidance:**\n"
            ai_content += f"• UX/UI principles: Usability, Accessibility, Responsive\n"
            ai_content += f"• Design patterns: MVC, MVP, MVVM, Observer\n"
            ai_content += f"• Prototyping tools: Figma, Sketch, Adobe XD\n"
            ai_content += f"• User research methods: Interviews, Surveys, A/B Testing\n\n"
            
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
            if current_transcription == self.last_analyzed_transcription:
                return  # No new content to analyze
            
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
        """Completely asynchronous keyword analysis that doesn't block UI"""
        try:
            # Immediately release the lock to allow UI to continue
            with self.ai_analysis_lock:
                self.ai_analysis_running = False
            
            # Run the actual analysis in a completely separate thread
            analysis_thread = threading.Thread(
                target=self.check_and_analyze_keywords,
                args=(text, full_transcription),
                daemon=True
            )
            analysis_thread.start()
            
        except Exception as e:
            print(f"Error in async keyword analysis: {e}")
    
    def check_and_analyze_keywords(self, text, full_transcription):
        """Check for keywords in text and automatically trigger AI analysis"""
        try:
            text_lower = text.lower()
            found_keywords = []
            
            # Check all keyword categories for new keywords
            for category, keywords in self.tech_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        # Create a unique identifier for this keyword in this context
                        keyword_id = f"{category}:{keyword.lower()}"
                        
                        # Only analyze if we haven't seen this keyword before
                        if keyword_id not in self.analyzed_keywords:
                            found_keywords.append((category, keyword))
                            self.analyzed_keywords.add(keyword_id)
                            break  # Only need one match per category
            
            # If new keywords found, generate AI analysis
            if found_keywords:
                # Get the most relevant category (first match)
                category, keyword = found_keywords[0]
                
                # Get topic explanation for context
                explanations = self.get_topic_explanations()
                if category in explanations:
                    explanation = explanations[category]
                    
                    # Generate AI analysis with full context
                    if self.ai_enabled_var.get():
                        ai_content = self.generate_contextual_ai_analysis(category, explanation, full_transcription, keyword)
                        
                        # Schedule UI updates on main thread to prevent hanging
                        self.root.after(0, lambda: self.show_ai_analysis(ai_content))
                        self.root.after(0, lambda: self.show_topic_explanation(category))
                        
                        # Update timestamp and transcription tracking
                        self.last_ai_analysis_time = time.time()
                        self.last_analyzed_transcription = full_transcription
                        
                        print(f"AI analysis triggered for new keyword: '{keyword}' in category: '{category}'")
                    else:
                        # Schedule UI update on main thread
                        self.root.after(0, lambda: self.show_ai_analysis("AI Analysis disabled. Enable the checkbox to see AI-enhanced insights."))
            else:
                # No new keywords found, but update the transcription tracking
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
            
            # Check if we have new keywords in the buffered content
            buffered_lower = buffered_transcription.lower()
            found_keywords = []
            
            # Check all keyword categories for new keywords
            for category, keywords in self.tech_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in buffered_lower:
                        # Create a unique identifier for this keyword in this context
                        keyword_id = f"{category}:{keyword.lower()}"
                        
                        # Only analyze if we haven't seen this keyword before
                        if keyword_id not in self.analyzed_keywords:
                            found_keywords.append((category, keyword))
                            self.analyzed_keywords.add(keyword_id)
                            break  # Only need one match per category
            
            # If new keywords found, generate AI analysis
            if found_keywords:
                # Get the most relevant category (first match)
                category, keyword = found_keywords[0]
                
                # Get topic explanation for context
                explanations = self.get_topic_explanations()
                if category in explanations:
                    explanation = explanations[category]
                    
                    # Generate AI analysis with full context
                    if self.ai_enabled_var.get():
                        ai_content = self.generate_contextual_ai_analysis(category, explanation, buffered_transcription, keyword)
                        
                        # Schedule UI updates on main thread to prevent hanging
                        self.root.after(0, lambda: self.show_ai_analysis(ai_content))
                        self.root.after(0, lambda: self.show_topic_explanation(category))
                        
                        # Update timestamp and transcription tracking
                        self.last_ai_analysis_time = time.time()
                        self.last_analyzed_transcription = buffered_transcription
                        
                        print(f"Buffered AI analysis triggered for new keyword: '{keyword}' in category: '{category}'")
                    else:
                        # Schedule UI update on main thread
                        self.root.after(0, lambda: self.show_ai_analysis("AI Analysis disabled. Enable the checkbox to see AI-enhanced insights."))
            else:
                # No new keywords found, but update the transcription tracking
                self.last_analyzed_transcription = buffered_transcription
                print("Buffered content processed, no new keywords found")
                
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
        """Get topic explanations, challenges, and commands"""
        return {
            'python': {
                'title': 'Python Programming',
                'summary': 'Python is a high-level, interpreted programming language known for its simplicity and versatility. It\'s widely used in web development, data science, machine learning, automation, and system administration.',
                'challenges': '• Performance optimization for CPU-intensive tasks\n• Memory management for large datasets\n• Package dependency conflicts\n• Version compatibility issues\n• Debugging complex asynchronous code',
                'commands': '• pip install package_name - Install Python packages from PyPI\n• python -m venv venv_name - Create isolated virtual environment\n• python -m pytest tests/ - Run unit tests with pytest framework\n• pip freeze > requirements.txt - Generate dependency list for project\n• python -c "import sys; print(sys.version)" - Check Python version from command line\n• python -m pip install --upgrade pip - Update pip to latest version\n• python -m pdb script.py - Debug Python script interactively\n• python -m profile script.py - Profile script performance and bottlenecks'
            },
            'docker': {
                'title': 'Docker Containerization',
                'summary': 'Docker is a containerization platform that packages applications and their dependencies into lightweight, portable containers. It enables consistent deployment across different environments.',
                'challenges': '• Container security and isolation\n• Resource management and optimization\n• Networking between containers\n• Persistent data storage\n• Container orchestration at scale',
                'commands': '• docker build -t image_name . - Build Docker image from Dockerfile in current directory\n• docker run -d -p 8080:80 image_name - Run container in background, map port 8080 to 80\n• docker ps -a - List all containers (running and stopped)\n• docker logs container_id - View container output and error logs\n• docker-compose up -d - Start multi-container application in background\n• docker exec -it container_id /bin/bash - Get interactive shell in running container\n• docker stop container_id - Gracefully stop running container\n• docker rm container_id - Remove stopped container from system'
            },
            'aws': {
                'title': 'Amazon Web Services (AWS)',
                'summary': 'AWS is a comprehensive cloud computing platform offering over 200 services including compute, storage, databases, networking, and security. It\'s the leading cloud provider globally.',
                'challenges': '• Cost optimization and resource management\n• Security and compliance\n• Multi-region deployment\n• Service integration complexity\n• Monitoring and troubleshooting distributed systems',
                'commands': '• aws s3 ls s3://bucket-name - List objects in S3 bucket\n• aws ec2 describe-instances - Show all EC2 instances and their status\n• aws lambda list-functions - List all Lambda functions in your account\n• aws cloudformation describe-stacks - Show CloudFormation stack information\n• aws logs describe-log-groups - List CloudWatch log groups\n• aws s3 cp file.txt s3://bucket/ - Upload file to S3 bucket\n• aws ec2 start-instances --instance-ids i-1234567890abcdef0 - Start specific EC2 instance\n• aws configure list - Check current AWS configuration settings'
            },
            'linux': {
                'title': 'Linux System Administration',
                'summary': 'Linux is an open-source Unix-like operating system kernel. It\'s widely used in servers, embedded systems, and development environments. System administration involves managing users, processes, services, and system resources.',
                'challenges': '• System security hardening\n• Performance tuning and optimization\n• Service dependency management\n• Log analysis and troubleshooting\n• Backup and disaster recovery',
                'commands': '• sudo systemctl status service_name - Check if systemd service is running\n• tail -f /var/log/syslog - Monitor system log in real-time\n• ps aux | grep process_name - Find running processes by name\n• df -h - Show disk space usage in human-readable format\n• netstat -tulpn - Display network connections and listening ports\n• sudo systemctl restart service_name - Restart a systemd service\n• top - Display running processes and system resource usage\n• find /path -name "*.log" - Search for files by name pattern'
            },
            'git': {
                'title': 'Git Version Control',
                'summary': 'Git is a distributed version control system that tracks changes in source code during software development. It enables collaboration, branching, merging, and maintaining project history.',
                'challenges': '• Merge conflicts resolution\n• Branch management strategies\n• Large repository performance\n• Access control and permissions\n• Backup and disaster recovery',
                'commands': '• git clone repository_url - Download repository to local machine\n• git add . && git commit -m "message" - Stage all changes and commit with message\n• git push origin branch_name - Upload local commits to remote repository\n• git pull origin main - Download and merge changes from remote main branch\n• git log --oneline - Show commit history in compact format\n• git status - Check which files are modified, staged, or untracked\n• git branch -a - List all local and remote branches\n• git checkout -b new_branch - Create and switch to new branch'
            },
            'database': {
                'title': 'Database Management',
                'summary': 'Databases are organized collections of data that can be easily accessed, managed, and updated. They support various data models including relational, NoSQL, and in-memory databases.',
                'challenges': '• Query optimization and performance\n• Data consistency and ACID properties\n• Scalability and sharding\n• Backup and recovery procedures\n• Security and access control',
                'commands': '• SELECT * FROM table_name WHERE condition; - Query data from database table\n• CREATE INDEX idx_name ON table_name(column); - Create index to speed up queries\n• SHOW PROCESSLIST; - Display currently running database processes\n• mysqldump -u user -p database_name > backup.sql - Create database backup file\n• EXPLAIN SELECT query; - Analyze query execution plan and performance\n• SHOW TABLES; - List all tables in current database\n• DESCRIBE table_name; - Show table structure and column information\n• GRANT SELECT ON database.* TO user@localhost; - Grant database permissions to user'
            },
            'networking': {
                'title': 'Computer Networking',
                'summary': 'Computer networking involves connecting computers and devices to share resources and information. It includes protocols, routing, switching, and network security.',
                'challenges': '• Network security and firewalls\n• Bandwidth optimization\n• Latency and packet loss\n• Protocol compatibility\n• Troubleshooting connectivity issues',
                'commands': '• ping hostname_or_ip - Test network connectivity and latency\n• traceroute destination - Trace network path to destination\n• netstat -an - Show all network connections and listening ports\n• tcpdump -i interface - Capture and analyze network packets\n• nmap -sS target_ip - Scan target for open ports using SYN scan\n• ssh user@hostname - Connect to remote server securely\n• curl -I http://example.com - Check if website is accessible (HEAD request)\n• dig domain.com - Query DNS records for domain name'
            },
            'security': {
                'title': 'Information Security',
                'summary': 'Information security involves protecting digital information from unauthorized access, use, disclosure, disruption, modification, or destruction. It includes encryption, authentication, and access control.',
                'challenges': '• Threat detection and prevention\n• Identity and access management\n• Encryption key management\n• Compliance and auditing\n• Incident response and forensics',
                'commands': '• openssl genrsa -out private.key 2048 - Generate 2048-bit RSA private key\n• ssh-keygen -t rsa -b 4096 - Create SSH key pair for secure authentication\n• nmap -sV target_ip - Scan target and identify service versions\n• fail2ban-client status - Check status of intrusion prevention system\n• certbot --nginx -d domain.com - Obtain SSL certificate for domain\n• chmod 600 private.key - Set secure permissions (owner read/write only)\n• sudo ufw status - Check firewall rules and status\n• openssl x509 -in cert.pem -text -noout - Examine SSL certificate details'
            },
            'monitoring': {
                'title': 'System Monitoring',
                'summary': 'System monitoring involves observing and measuring system performance, availability, and health. It includes logging, metrics collection, alerting, and visualization.',
                'challenges': '• Log aggregation and analysis\n• Metric collection at scale\n• Alert fatigue and noise\n• Performance impact of monitoring\n• Data retention and storage',
                'commands': '• tail -f /var/log/application.log - Monitor application logs in real-time\n• htop - Interactive process viewer with resource usage\n• iostat -x 1 - Monitor disk I/O statistics every second\n• prometheus --config.file=prometheus.yml - Start metrics collection server\n• grafana-server - Launch web-based monitoring dashboard\n• journalctl -u service_name -f - Monitor systemd service logs\n• free -h - Display memory usage in human-readable format\n• sar -u 1 10 - Monitor CPU usage every second for 10 iterations'
            },
            'ci_cd': {
                'title': 'Continuous Integration/Continuous Deployment',
                'summary': 'CI/CD is a set of practices that automate the integration and deployment of code changes. It includes automated testing, building, and deployment pipelines.',
                'challenges': '• Pipeline complexity and maintenance\n• Test environment management\n• Deployment rollback strategies\n• Security in CI/CD pipelines\n• Performance and scalability',
                'commands': '• git push origin feature-branch - Trigger CI pipeline by pushing code\n• docker build -t app:latest . - Build application container image\n• kubectl apply -f deployment.yaml - Deploy application to Kubernetes cluster\n• helm install release-name chart/ - Install application using Helm package manager\n• jenkins build job-name - Trigger Jenkins CI/CD pipeline manually\n• docker-compose up -d - Start multi-container application stack\n• kubectl get pods - List all running pods in Kubernetes cluster\n• helm list - Show all installed Helm releases'
            },
            'hft': {
                'title': 'High-Frequency Trading (HFT)',
                'summary': 'High-Frequency Trading uses sophisticated algorithms and ultra-fast computer systems to execute trades in milliseconds or microseconds. It relies on speed, low latency, and market data analysis to capitalize on small price differences.',
                'challenges': '• Ultra-low latency requirements (microseconds)\n• Market data processing at high speeds\n• Risk management in volatile markets\n• Regulatory compliance and reporting\n• Infrastructure costs and co-location\n• Algorithm stability under market stress',
                'commands': '• ping -c 100 exchange-server.com - Test latency to exchange (100 packets)\n• tcpdump -i eth0 -n "port 443" - Capture HTTPS traffic on network interface\n• strace -p $(pgrep trading_engine) - Trace system calls of trading process\n• perf record -g ./trading_algorithm - Profile trading algorithm performance\n• netstat -i | grep -E "(RX|TX)" - Monitor network interface statistics\n• taskset -c 0-3 ./trading_app - Pin trading app to specific CPU cores\n• ethtool -K eth0 gro off tso off - Disable network optimizations for low latency\n• numactl --cpunodebind=0 --membind=0 ./app - Bind to specific NUMA node'
            },
            'hpc': {
                'title': 'High-Performance Computing (HPC)',
                'summary': 'HPC involves using supercomputers and parallel processing techniques to solve complex computational problems. It includes cluster computing, GPU acceleration, and distributed computing for scientific, engineering, and research applications.',
                'challenges': '• Parallel algorithm design and optimization\n• Load balancing across compute nodes\n• Memory bandwidth and cache optimization\n• Inter-node communication bottlenecks\n• Fault tolerance in large-scale systems\n• Power consumption and cooling requirements',
                'commands': '• mpirun -np 64 ./parallel_program - Run MPI program on 64 processes\n• nvidia-smi -l 1 - Monitor GPU usage every second\n• htop -d 1 - Display system resources every second\n• sacct -j job_id --format=JobID,State,ExitCode - Check Slurm job status and exit code\n• squeue -u username - Show user\'s jobs in Slurm queue\n• srun --gres=gpu:1 --pty bash - Get interactive shell with GPU access\n• sinfo -N -l - List all compute nodes and their status\n• scontrol show job job_id - Display detailed job information'
            },
            'network_storage': {
                'title': 'Network Storage Technologies',
                'summary': 'Network storage systems provide shared storage resources over a network, including NAS (Network Attached Storage), SAN (Storage Area Network), and distributed storage solutions. They enable centralized data management and high availability.',
                'challenges': '• Network latency and bandwidth optimization\n• Data consistency across distributed systems\n• Backup and disaster recovery strategies\n• Storage capacity planning and scaling\n• Security and access control\n• Performance tuning for different workloads',
                'commands': '• mount -t nfs server:/path /local/mount - Mount NFS share to local directory\n• iscsiadm -m discovery -t st -p target_ip - Discover iSCSI targets on server\n• df -h | grep nfs - Check NFS mount points and disk usage\n• iostat -x 1 - Monitor storage I/O performance every second\n• ceph status - Check Ceph distributed storage cluster health\n• gluster volume info - Display GlusterFS volume information\n• smbclient -L //server -U username - List SMB/CIFS shares on server\n• showmount -e server - Show NFS exports available on server'
            },
            'performance': {
                'title': 'Performance Optimization & Quantum Computing',
                'summary': 'Performance optimization focuses on achieving maximum throughput and minimum latency in computing systems. This includes FPGA acceleration, RDMA networking, kernel bypass techniques, and hardware-optimized programming for trading, HPC, and real-time applications. Quantum computing represents the next frontier in computational performance, offering exponential speedups for specific problem classes through quantum algorithms and quantum advantage.',
                'challenges': '• Ultra-low latency requirements (nanoseconds to microseconds)\n• Hardware-software co-design complexity\n• Memory hierarchy optimization and cache efficiency\n• Network stack bypass and user-space networking\n• FPGA programming and verification\n• NUMA topology and CPU affinity management\n• Real-time system constraints and determinism\n• Quantum error correction and coherence maintenance\n• Quantum algorithm design and optimization\n• Quantum-classical hybrid system integration',
                'commands': 'TRADITIONAL PERFORMANCE:\n• numactl --cpunodebind=0 --membind=0 ./app - Bind app to specific NUMA node\n• taskset -c 0-3 ./process - Pin process to CPU cores 0-3\n• perf record -g -e cycles,instructions ./program - Profile with call graphs\n• rdma_cm -s -p 12345 - Start RDMA connection manager on port 12345\n• ibv_devinfo -v - Display InfiniBand device information\n• fpga-load-accel -a accelerator.bit - Load FPGA accelerator bitstream\n• dpdk-testpmd -l 0-3 -n 4 -- -i - Test DPDK packet processing\n• ethtool -K eth0 gro off tso off - Disable network optimizations for low latency\n\nQUANTUM COMPUTING:\n• qiskit --version - Check IBM Qiskit quantum framework version\n• cirq --version - Check Google Cirq quantum framework version\n• qiskit-aer --version - Check Qiskit Aer simulator version\n• python -c "import qiskit; print(qiskit.__version__)" - Verify Qiskit installation\n• qiskit transpile --optimization_level=3 circuit - Optimize quantum circuit\n• qiskit execute --shots=1024 circuit - Run quantum circuit with 1024 shots\n• qiskit optimize --backend=backend circuit - Optimize for specific quantum backend'
            },
            'slurm': {
                'title': 'Slurm Workload Manager & Open OnDemand',
                'summary': 'Slurm (Simple Linux Utility for Resource Management) is an open-source job scheduler for Linux clusters. It manages job queues, allocates resources, and provides job accounting. Open OnDemand is a web-based interface for HPC clusters that integrates with Slurm, providing user-friendly access to cluster resources, job management, and interactive applications.',
                'challenges': '• Job queue management and priority handling\n• Resource allocation and scheduling optimization\n• Job dependency and workflow management\n• Fair share and user/group quotas\n• Node failure handling and job recovery\n• Performance monitoring and accounting\n• Integration with storage and network systems\n• Web interface configuration and security\n• Interactive application management\n• User authentication and authorization',
                'commands': 'SLURM COMMANDS:\n• squeue -u username - Show user\'s jobs in queue\n• sbatch job_script.sh - Submit batch job script to queue\n• srun --pty bash - Get interactive shell on compute node\n• sacct -j job_id --format=JobID,State,ExitCode - Check job status and exit code\n• scontrol show job job_id - Display detailed job information\n• sinfo -N -l - List all compute nodes and their status\n• scancel job_id - Cancel or delete job from queue\n• salloc --time=1:00:00 --nodes=1 - Allocate resources for interactive session\n\nOPEN ONDEMAND:\n• Access via web browser: https://cluster.domain.edu - Login to web portal\n• Interactive Apps: Jupyter, RStudio, MATLAB, VSCode - Launch development environments\n• File Manager: Upload/download files - Manage files through web interface\n• Job Composer: Create and submit jobs - Build job scripts visually\n• Active Jobs: Monitor running jobs - Track job progress and status\n• Shell Access: Terminal access to compute nodes - Get command line access'
            },
            'pbs_torque': {
                'title': 'PBS/Torque with Maui/Moab Scheduler',
                'summary': 'PBS (Portable Batch System) and Torque are job scheduling systems for HPC clusters, often paired with Maui or Moab schedulers for advanced scheduling policies. They provide job queuing, resource management, and workload distribution across compute nodes.',
                'challenges': '• Job queue management and priority handling\n• Resource allocation and scheduling optimization\n• Job dependency and workflow management\n• Fair share and user/group quotas\n• Node failure handling and job recovery\n• Performance monitoring and accounting\n• Integration with storage and network systems',
                'commands': '• qstat -u username - Show user\'s jobs in PBS queue\n• qsub job_script.sh - Submit batch job script to PBS queue\n• qdel job_id - Delete or cancel PBS job\n• pbsnodes -a - List all compute nodes and their status\n• showq -u username - Display user\'s jobs in Maui/Moab queue\n• qstat -f job_id - Show detailed PBS job information\n• qalter -l walltime=2:00:00 job_id - Modify job walltime to 2 hours\n• qhold job_id - Hold job to prevent execution\n• qrls job_id - Release held job to allow execution'
            },
            'computational_jobs': {
                'title': 'Computational Job Management',
                'summary': 'Computational job management involves running, monitoring, and troubleshooting batch jobs, parallel computations, and distributed workloads. This includes job scheduling, resource allocation, performance optimization, and failure handling for scientific and engineering computations.',
                'challenges': '• Job scheduling and resource allocation\n• Performance optimization and bottleneck identification\n• Memory management and out-of-memory errors\n• I/O optimization and storage bottlenecks\n• Parallel scaling and load balancing\n• Job failure diagnosis and recovery\n• Resource contention and queue management',
                'commands': '• squeue -u username -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"\n• sacct -j job_id --format=JobID,State,ExitCode,Start,End,Elapsed\n• scontrol show job job_id\n• sinfo -N -l\n• scancel job_id\n• srun --ntasks=4 --cpus-per-task=2 ./program\n• sbatch --time=2:00:00 --mem=8G job_script.sh'
            },
            'programming': {
                'title': 'General Programming & Software Development',
                'summary': 'Programming involves writing, testing, and maintaining code to create software applications. It encompasses various programming languages, development methodologies, debugging techniques, and software engineering practices for building reliable and efficient applications.',
                'challenges': '• Algorithm design and optimization\n• Memory management and resource allocation\n• Debugging complex logic errors\n• Performance profiling and optimization\n• Code maintainability and documentation\n• Testing and quality assurance\n• Version control and collaboration',
                'commands': '• git init && git add . && git commit -m "Initial commit"\n• gcc -o program source.c\n• python -m py_compile script.py\n• node --version && npm --version\n• ruby -c script.rb\n• bash -n script.sh\n• lua -l script.lua\n• powershell -Command "Get-Command"'
            },
            'c_programming': {
                'title': 'C Programming Language',
                'summary': 'C is a low-level, compiled programming language known for its efficiency and direct hardware access. It\'s widely used in system programming, embedded systems, and performance-critical applications. C provides manual memory management and direct pointer manipulation.',
                'challenges': '• Manual memory management and memory leaks\n• Pointer arithmetic and segmentation faults\n• Buffer overflows and security vulnerabilities\n• Platform-specific code and portability\n• Debugging complex pointer issues\n• Performance optimization and profiling\n• Cross-compilation and build systems',
                'commands': '• gcc -Wall -Wextra -o program source.c\n• gcc -g -o program source.c && gdb ./program\n• valgrind --leak-check=full ./program\n• make -f Makefile\n• gcc -O2 -march=native -o program source.c\n• objdump -d program\n• nm program | grep function_name\n• strace ./program'
            },
            'javascript': {
                'title': 'JavaScript & Node.js Development',
                'summary': 'JavaScript is a high-level, interpreted programming language primarily used for web development. Node.js extends JavaScript to server-side development. It features asynchronous programming, dynamic typing, and extensive ecosystem of libraries and frameworks.',
                'challenges': '• Asynchronous programming and callback hell\n• Memory leaks and garbage collection\n• Cross-browser compatibility issues\n• Performance optimization and bundle size\n• Security vulnerabilities (XSS, CSRF)\n• Debugging asynchronous code\n• Dependency management and version conflicts',
                'commands': '• node script.js\n• npm init && npm install package\n• npm run build\n• yarn add package\n• npx create-react-app my-app\n• npm test\n• node --inspect script.js\n• npm audit && npm audit fix'
            },
            'ruby': {
                'title': 'Ruby & Ruby on Rails',
                'summary': 'Ruby is a dynamic, object-oriented programming language known for its elegant syntax and developer productivity. Ruby on Rails is a web application framework that follows convention over configuration, providing rapid development capabilities for web applications.',
                'challenges': '• Performance optimization and memory usage\n• Database query optimization (N+1 problems)\n• Security vulnerabilities and best practices\n• Testing and test-driven development\n• Deployment and production configuration\n• Gem dependency management\n• Scaling and performance monitoring',
                'commands': '• ruby script.rb\n• rails new myapp\n• bundle install\n• rails server\n• rake db:migrate\n• rails console\n• rspec spec/\n• bundle exec rubocop\n• gem install gem_name'
            },
            'shell_scripting': {
                'title': 'Shell Scripting (Bash/Zsh)',
                'summary': 'Shell scripting involves writing scripts using command-line interpreters like Bash or Zsh to automate system tasks, process data, and manage system operations. It\'s essential for system administration, DevOps, and automation workflows.',
                'challenges': '• Portability across different shell environments\n• Error handling and exit codes\n• Variable scoping and subshell issues\n• Performance optimization for large datasets\n• Security vulnerabilities and input validation\n• Debugging complex script logic\n• Cross-platform compatibility',
                'commands': '• bash script.sh\n• chmod +x script.sh\n• source ~/.bashrc\n• export VARIABLE=value\n• ps aux | grep process\n• find /path -name "*.txt"\n• awk \'{print $1}\' file.txt\n• sed \'s/old/new/g\' file.txt\n• cron -e'
            },
            'lua': {
                'title': 'Lua Scripting Language',
                'summary': 'Lua is a lightweight, embeddable scripting language designed for extensibility and performance. It\'s commonly used in game development, embedded systems, and as an extension language for applications. Lua features coroutines, metatables, and a simple C API.',
                'challenges': '• Memory management and garbage collection\n• Performance optimization for embedded systems\n• C API integration and binding\n• Debugging embedded Lua scripts\n• Error handling and exception management\n• Threading and concurrency\n• Cross-platform compatibility',
                'commands': '• lua script.lua\n• luajit script.lua\n• lua -l module_name\n• lua -e "print(\'Hello World\')"\n• lua -i (interactive mode)\n• lua -v (version)\n• lua -p script.lua (syntax check)\n• lua -l debug script.lua'
            },
            'powershell': {
                'title': 'PowerShell Scripting',
                'summary': 'PowerShell is a task automation and configuration management framework from Microsoft. It provides a command-line shell and scripting language built on .NET, with object-oriented capabilities and extensive system management features for Windows and cross-platform environments.',
                'challenges': '• Execution policy and security restrictions\n• Object pipeline and data manipulation\n• Error handling and exception management\n• Performance optimization for large datasets\n• Cross-platform compatibility (PowerShell Core)\n• Module management and dependencies\n• Debugging complex scripts',
                'commands': '• powershell -Command "Get-Process"\n• powershell -File script.ps1\n• Get-ExecutionPolicy\n• Set-ExecutionPolicy RemoteSigned\n• Import-Module ModuleName\n• Get-Command -Module ModuleName\n• Get-Help CommandName\n• Test-Path "C:\\path"'
            }
        }
    
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
        
        # Check each question type pattern
        for question_type, patterns in self.question_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return question_type
        
        # Default to troubleshooting if no specific type detected
        return 'troubleshooting'
    
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
                ai_content = self.generate_ai_troubleshooting(question_text, suggestions)
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
        
        # Reset AI analysis lock and buffer
        with self.ai_analysis_lock:
            self.ai_analysis_running = False
            self.transcription_buffer = ""
            self.pending_ai_analysis = False
        
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
