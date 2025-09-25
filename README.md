# Real-time Speech Transcription with Topic Explorer

A Python application that provides real-time speech transcription with interactive keyword-based topic explanations. Click on highlighted technical terms to get instant summaries, technical challenges, and useful commands.

## Features

- **Real-time Speech Recognition**: Continuous microphone listening with live transcription
- **Interactive Keywords**: Click on highlighted technical terms to get detailed explanations
- **Topic Explanations**: Get summaries, technical challenges, and command examples for various tech topics
- **Modern GUI**: Clean, responsive interface built with tkinter
- **Keyword Categories**: Covers Python, Docker, AWS, Linux, Git, Databases, Networking, Security, Monitoring, CI/CD, HFT, HPC, Network Storage, Performance, Slurm, PBS/Torque, Computational Jobs, Programming, C, JavaScript, Ruby, Shell Scripting, Lua, and PowerShell
- **AI-Enhanced Analysis**: OpenAI integration for contextual analysis and adaptive question type detection
- **Transcription Corrections**: Configurable post-processing corrections for common speech-to-text errors
- **Dynamic Question Types**: Intelligent detection of architecture, design, security, and troubleshooting questions

## Installation

### Prerequisites
- **Miniconda** or **Anaconda** (recommended for easy dependency management)
- **Python 3.9-3.12** (3.11 recommended)
- **FLAC audio converter** (required for Google Speech Recognition)

### Quick Start with Conda

**Option 1: Automated Setup (Recommended)**
```bash
# Linux/macOS
git clone <repository-url>
cd agent-hoy
./setup_conda.sh

# Windows
git clone <repository-url>
cd agent-hoy
setup_conda.bat
```

**Option 2: Manual Setup**
```bash
# Clone the repository
git clone <repository-url>
cd agent-hoy

# Create and activate conda environment
conda env create -f environment.yaml
conda activate agent-hoy

# Run the application
python main.py
```

**Note:** FLAC is required for Google Speech Recognition. If you encounter FLAC errors:
- **Windows**: Install via Chocolatey: `choco install flac`
- **Linux**: `sudo apt-get install flac`
- **macOS**: `brew install flac`

The application will fallback to alternative recognition methods if FLAC is not available. 

### Alternative: Manual Installation

If you prefer pip installation:

1. **Install system dependencies first**:
   - **Ubuntu/Debian**: `sudo apt-get install portaudio19-dev python3-pyaudio`
   - **CentOS/RHEL**: `sudo yum install portaudio-devel`
   - **macOS**: `brew install portaudio`
   - **Windows**: PyAudio should install automatically

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the Application

1. **Activate the conda environment**:
   ```bash
   conda activate agent-hoy
   ```

2. **Run the application**:
   ```bash
   python main.py
   ```

### Using the Application

1. **Start listening**:
   - Click "Start Listening" to begin speech recognition
   - Speak clearly into your microphone
   - Watch as your speech is transcribed in real-time

2. **Explore topics**:
   - Technical keywords will be highlighted in yellow
   - Click on any highlighted keyword to see:
     - Topic summary and explanation
     - Common technical challenges
     - Useful commands and examples

3. **Controls**:
   - **Start/Stop Listening**: Toggle speech recognition
   - **Clear Text**: Clear the transcription and topic panels
   - **Status**: Shows current application state

### Offline Capabilities

The application automatically handles offline scenarios:
- **Cached Content**: Previously viewed explanations are available offline
- **Offline Recognition**: Falls back to local speech recognition when internet is unavailable
- **Status Indication**: Shows "Offline mode" when using local recognition

## Transcription Corrections

The application includes configurable post-processing corrections to improve transcription accuracy for technical terms. Corrections are loaded from `corrections.conf` at startup.

### Configuration File

Create or edit `corrections.conf` to customize transcription corrections:

```conf
# Transcription Corrections Configuration
# Format: original_text = corrected_text
# Lines starting with # are comments

# Technical term corrections
computer environment = compute environment
on premise = on-premise
camera = containerized
in Fennaband = Infiniband
created storage later = federated storage layers
bare metal = bare-metal
split brain = split-brain
cloud burst = cloud-burst
open source = open-source
fault tolerant = fault-tolerant

# Common speech-to-text errors
Design of = Design a
for us = for a
or solution = Your solution
and show = and ensure
This includes = for specialized hardware

# Technical abbreviations
HPC = HPC
GPU = GPU
API = API
CLI = CLI
Docker = Docker
Kubernetes = Kubernetes
AWS = AWS
Azure = Azure
```

### Adding Custom Corrections

1. **Edit the config file**: Open `corrections.conf` in a text editor
2. **Add your corrections**: Use the format `original = corrected`
3. **Restart the application**: Changes take effect on next startup
4. **Test your corrections**: Speak the original text to verify corrections work

### Default Corrections

If `corrections.conf` is not found, the application uses built-in default corrections for common technical terms and speech-to-text errors.

## Supported Topics

The application recognizes and provides explanations for:

- **Python**: Programming language, packages, virtual environments
- **Docker**: Containerization, images, orchestration
- **AWS**: Cloud services, EC2, S3, Lambda, CloudFormation
- **Linux**: System administration, commands, services
- **Git**: Version control, branching, collaboration
- **Database**: SQL, MySQL, PostgreSQL, MongoDB
- **Networking**: TCP/UDP, HTTP/HTTPS, DNS, firewalls
- **Security**: Encryption, SSL/TLS, authentication
- **Monitoring**: Logging, metrics, Prometheus, Grafana
- **CI/CD**: Jenkins, GitHub Actions, deployment pipelines
- **HFT**: High-frequency trading, algorithmic trading, market making, latency optimization
- **HPC**: High-performance computing, supercomputing, parallel computing, GPU computing
- **Network Storage**: NAS, SAN, iSCSI, NFS, CIFS, distributed storage, Ceph, GlusterFS
- **Performance**: FPGA, RDMA, RoCE, InfiniBand, kernel bypass, DPDK, low-latency optimization, quantum computing
- **Slurm**: Job scheduling, queue management, resource allocation, job monitoring, cluster administration, Open OnDemand web interface
- **PBS/Torque**: PBS/Torque job scheduling, Maui/Moab schedulers, qstat/qsub commands, job management
- **Computational Jobs**: HPC workloads, batch processing, parallel computing, job troubleshooting, performance optimization
- **Programming**: General programming concepts, software development, debugging, algorithms, data structures
- **C Programming**: C language, GCC, debugging, memory management, pointers, compilation, profiling
- **JavaScript**: JavaScript, Node.js, npm, frameworks, asynchronous programming, web development
- **Ruby**: Ruby, Rails, gems, testing, web development, MVC architecture
- **Shell Scripting**: Bash, Zsh, command-line automation, system administration, DevOps
- **Lua**: Lua scripting, embedded systems, game development, C API integration
- **PowerShell**: PowerShell scripting, Windows automation, .NET integration, cross-platform scripting

## Technical Details

### Architecture
- **Speech Recognition**: Uses Google Speech Recognition API (free, no API key required)
- **GUI Framework**: tkinter for cross-platform compatibility
- **Threading**: Background speech processing to maintain responsive UI
- **Keyword Detection**: Pattern matching with regex for real-time highlighting

### Dependencies
- `speech_recognition`: Speech-to-text conversion
- `pyaudio`: Microphone audio capture
- `tkinter`: GUI framework (included with Python)
- `threading`: Background processing
- `queue`: Thread-safe communication

## Troubleshooting

### Microphone Issues
- Ensure your microphone is working and not muted
- Check system audio permissions
- Try adjusting microphone sensitivity in system settings

### Speech Recognition Issues
- Speak clearly and at moderate volume
- Reduce background noise
- Check internet connection (Google Speech Recognition requires internet)

### Installation Issues
- On Linux, you may need to install additional audio libraries
- On macOS, you might need to install Xcode command line tools
- On Windows, ensure you have a compatible Python installation

## Troubleshooting

### Common Issues

**Environment Creation Fails**:
```bash
# Update conda first
conda update conda
conda env create -f environment.yaml
```

**PyAudio Issues**:
```bash
# Conda should handle this automatically, but if issues persist:
conda install -c conda-forge pyaudio
```

**Microphone Not Working**:
- Check system microphone permissions
- Ensure microphone is not being used by other applications
- Try running: `python -c "import pyaudio; print('PyAudio working')"`

**Offline Mode Not Working**:
- Check internet connection
- Verify vosk and pocketsphinx are installed: `conda list | grep -E "(vosk|pocketsphinx)"`

### Environment Management

**Remove Environment**:
```bash
conda env remove -n agent-hoy
```

**Update Environment**:
```bash
conda env update -f environment.yaml
```

**Export Environment**:
```bash
conda env export > environment.yaml
```

## Customization

### Adding New Keywords
Edit the `tech_keywords` dictionary in `main.py` to add new keyword categories:

```python
self.tech_keywords = {
    'your_category': ['keyword1', 'keyword2', 'keyword3'],
    # ... existing categories
}
```

### Adding Topic Explanations
Update the `get_topic_explanations()` method to include new topics:

```python
'your_category': {
    'title': 'Your Topic Title',
    'summary': 'Brief description...',
    'challenges': '• Challenge 1\n• Challenge 2',
    'commands': '• command1\n• command2'
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Future Enhancements

- [ ] Custom keyword categories
- [ ] Export transcription to file
- [ ] Integration with external APIs for enhanced explanations
- [ ] Voice commands for application control
- [ ] Real-time collaboration features