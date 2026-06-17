#!/usr/bin/env python3
"""
Transcription Configuration Options

This file contains configuration options for different speech recognition services
and their respective settings for optimal transcription quality.
"""

import os

# Whisper model: tiny, base, small, medium, large. Leave unset for auto (base on CPU, small on GPU).
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
WHISPER_SAMPLE_RATE = 16000
WHISPER_INITIAL_PROMPT = (
    "This is a technical discussion about computer systems, HPC clusters, "
    "cloud computing, containers, orchestration, and infrastructure. "
    "Use technical terminology accurately."
)

# Fast decoding defaults for live microphone chunks (override for accuracy)
WHISPER_FAST_MODE = os.getenv("WHISPER_FAST_MODE", "true").lower() in ("1", "true", "yes", "on")
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
WHISPER_CONDITION_ON_PREVIOUS = os.getenv(
    "WHISPER_CONDITION_ON_PREVIOUS",
    "false" if WHISPER_FAST_MODE else "true",
).lower() in ("1", "true", "yes", "on")


def get_whisper_model() -> str:
    """Pick a model balanced for the available hardware."""
    explicit = os.getenv("WHISPER_MODEL", "").strip()
    if explicit:
        return explicit
    if WHISPER_FAST_MODE and not is_cuda_available():
        return "base"
    return "small" if is_cuda_available() else "base"

# When online without CUDA, Google STT is usually faster and more accurate on short chunks.
# When CUDA is available, Whisper on GPU is preferred for low-latency local transcription.
STT_PREFER_GOOGLE = os.getenv("STT_PREFER_GOOGLE", "").lower() in ("1", "true", "yes", "on")
STT_PREFER_WHISPER = os.getenv("STT_PREFER_WHISPER", "").lower() in ("1", "true", "yes", "on")

# Microphone capture tuning (lower phrase_time_limit = faster chunks and pause response)
LISTEN_TIMEOUT = float(os.getenv("LISTEN_TIMEOUT", "0.5"))
LISTEN_PHRASE_TIME_LIMIT = float(os.getenv("LISTEN_PHRASE_TIME_LIMIT", "3"))

# AI pipeline latency tuning
QUESTION_COMPLETION_TIMEOUT = float(os.getenv("QUESTION_COMPLETION_TIMEOUT", "1.5"))
AI_ANALYSIS_THROTTLE_SECONDS = float(os.getenv("AI_ANALYSIS_THROTTLE_SECONDS", "2"))
HIGHLIGHT_DEBOUNCE_MS = int(os.getenv("HIGHLIGHT_DEBOUNCE_MS", "400"))

_cuda_available = None


def is_cuda_available() -> bool:
    """Return True when PyTorch can use a CUDA GPU."""
    global _cuda_available
    if _cuda_available is not None:
        return _cuda_available

    try:
        import torch

        _cuda_available = bool(torch.cuda.is_available())
    except ImportError:
        _cuda_available = False

    return _cuda_available


def should_prefer_whisper(whisper_available: bool, online: bool) -> bool:
    """
    Choose Whisper first when a GPU is available for speed, unless explicitly overridden.
    Without CUDA, prefer Google while online because CPU Whisper is comparatively slow.
    """
    if not whisper_available:
        return False
    if STT_PREFER_GOOGLE:
        return False
    if STT_PREFER_WHISPER:
        return True
    if is_cuda_available():
        return True
    return not online

# Google Speech Recognition Settings
GOOGLE_CONFIG = {
    'language': 'en-US',
    'show_all': False,
    'with_confidence': False,
    'timeout': 10,
    'phrase_time_limit': 5
}

# Alternative recognition services (for future integration)
RECOGNITION_SERVICES = {
    'google': {
        'name': 'Google Speech Recognition',
        'free': True,
        'accuracy': 'High',
        'speed': 'Fast',
        'offline': False,
        'api_key_required': False
    },
    'whisper': {
        'name': 'OpenAI Whisper',
        'free': True,
        'accuracy': 'Very High',
        'speed': 'Medium',
        'offline': True,
        'api_key_required': False,
        'model_sizes': ['tiny', 'base', 'small', 'medium', 'large']
    },
    'azure': {
        'name': 'Azure Speech Services',
        'free': False,
        'accuracy': 'Very High',
        'speed': 'Very Fast',
        'offline': False,
        'api_key_required': True
    },
    'aws_transcribe': {
        'name': 'AWS Transcribe',
        'free': False,
        'accuracy': 'High',
        'speed': 'Fast',
        'offline': False,
        'api_key_required': True
    },
    'anthropic': {
        'name': 'Anthropic Claude',
        'free': False,
        'accuracy': 'Very High',
        'speed': 'Fast',
        'offline': False,
        'api_key_required': True,
        'note': 'Requires text input, not direct audio'
    }
}

# Audio processing settings
AUDIO_SETTINGS = {
    'energy_threshold': 300,
    'dynamic_energy_threshold': True,
    'pause_threshold': 0.8,
    'phrase_threshold': 0.3,
    'non_speaking_duration': 0.5,
    'calibration_duration': 0.5,
    'timeout': 1,
    'phrase_time_limit': None
}

# Language models for better accuracy
LANGUAGE_MODELS = {
    'english_technical': {
        'language': 'en-US',
        'phrases': [
            # Core Python & Programming
            'python', 'pip', 'conda', 'anaconda', 'virtualenv', 'venv', 'jupyter', 'ipython', 'pypi',
            'flask', 'django', 'fastapi', 'requests', 'pytest', 'unittest', 'sqlalchemy',
            
            # Data Science & ML Libraries
            'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly', 'scikit-learn', 'sklearn',
            'tensorflow', 'pytorch', 'keras', 'xgboost', 'lightgbm', 'catboost', 'hugging face',
            'transformers', 'opencv', 'pillow', 'spacy', 'nltk', 'gensim', 'textblob',
            'polars', 'pyarrow', 'dask', 'modin', 'bokeh', 'altair', 'streamlit', 'gradio',
            'statsmodels', 'pymc', 'prophet', 'mlflow', 'wandb', 'tensorboard',
            
            # Infrastructure & DevOps
            'docker', 'dockerfile', 'docker-compose', 'kubernetes', 'aws', 'ec2', 's3', 'lambda',
            'cloudformation', 'cloudwatch', 'sagemaker', 'azure', 'linux', 'ubuntu',
            'git', 'github', 'gitlab', 'jenkins', 'ci', 'cd', 'devops',
            
            # Databases & Storage
            'database', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
            
            # Networking & Security
            'networking', 'tcp', 'udp', 'http', 'https', 'ssl', 'tls', 'api', 'rest', 'graphql',
            'security', 'encryption', 'authentication', 'authorization', 'oauth', 'jwt',
            
            # Monitoring & Observability
            'monitoring', 'logging', 'metrics', 'alerting', 'prometheus', 'grafana', 'kibana',
            
            # HPC & Advanced Computing
            'hpc', 'slurm', 'pbs', 'torque', 'maui', 'moab', 'condor', 'htcondor', 'sge', 'lsf',
            'mpi', 'openmpi', 'mpich', 'openmp', 'omp', 'cuda', 'hip', 'rocm', 'opencl',
            'nvidia', 'amd', 'gpu', 'parallel computing', 'distributed computing',
            'infiniband', 'ethernet', 'rdma', 'roce', 'mellanox',
            'lustre', 'gpfs', 'beegfs', 'ceph', 'glusterfs', 'nfs',
            'vtune', 'perf', 'gprof', 'valgrind', 'advisor', 'tau', 'scorep',
            'linpack', 'hpl', 'hpcg', 'npb', 'nas', 'stream', 'bandwidth',
            
            # Hybrid Cloud & Cloud Native
            'hybrid-cloud', 'multi-cloud', 'cloudbursting', 'on-premise', 'edge computing',
            'istio', 'linkerd', 'consul', 'envoy', 'microservices', 'service mesh',
            'terraform', 'ansible', 'chef', 'puppet', 'helm', 'kustomize',
            'cloud-native', 'cncf', 'prometheus', 'grafana', 'jaeger', 'fluent',
            'nginx', 'haproxy', 'f5', 'cdn', 'load balancer',
            'elasticsearch', 'kibana', 'logstash', 'datadog', 'newrelic', 'splunk',
            'appdynamics', 'dynatrace',
            
            # Data Formats & Files
            'json', 'yaml', 'xml', 'csv', 'parquet', 'avro', 'pickle', 'hdf5',
            
            # HFT & Low-Latency Computing
            'fpga', 'asic', 'smartnic', 'tick-to-trade', 'cpu pinning', 'numa',
            'kernel bypass', 'userspace', 'dpdk', 'sr-iov', 'cut-through',
            'packet pacing', 'qos', 'iwarp', 'omnipath', '100gig', '400gig',
            'connectx', 'ddr4', 'ddr5', 'lrdimm', 'rdimm', 'ecc', 'nvme',
            'iops', 'optane', 'xeon', 'epyc', 'skylake', 'avx', 'simd',
            'hyperthreading', 'smt', 'turbo boost',
            
            # Datacenter Infrastructure
            '1u', '2u', '4u', 'top-of-rack', 'tor', 'spine-leaf', 'fat-tree',
            'clos', 'non-blocking', 'bisection bandwidth', 'pdu', 'ups',
            'hot-swap', 'n+1', 'liquid cooling', 'immersion cooling', 'crac',
            'in-row cooling', 'raised floor', 'hot aisle', 'cold aisle',
            
            # System Management & Monitoring
            'ipmi', 'idrac', 'ilo', 'bmc', 'out-of-band', 'oob', 'snmp',
            'nagios', 'zabbix', 'ganglia',
            
            # Trading & Performance
            'microsecond', 'nanosecond', 'picosecond', 'latency', 'jitter',
            'deterministic', 'real-time', 'colocation', 'colo', 'dark pool',
            'matching engine', 'feed handler', 'market making', 'arbitrage',
            'algo trading',
            
            # Network Protocols
            'tcp', 'udp', 'multicast', 'unicast', 'igmp', 'pim', 'vlan',
            'vxlan', 'bgp', 'ospf',
            
            # Memory & Optimization
            'shared memory', 'memory-mapped', 'mmap', 'zero-copy', 'lock-free',
            'wait-free', 'atomic operations', 'compare-and-swap', 'cas',
            'memory barriers', 'cache coherency', 'false sharing',
            
            # System Tuning
            'cpu governor', 'performance governor', 'powersave', 'ondemand',
            'dvfs', 'frequency scaling', 'interrupt affinity', 'irq', 'softirq',
            'tasklet', 'napi',
            
            # Clustering & High Availability
            'split-brain', 'quorum', 'witness', 'tiebreaker', 'consensus',
            'pacemaker', 'corosync', 'heartbeat', 'stonith', 'fencing',
            'failover', 'failback', 'active-passive', 'active-active',
            'standby', 'floating ip', 'virtual ip', 'cluster ip',
            
            # Distributed Systems
            'cap theorem', 'consistency', 'availability', 'partition tolerance',
            'acid', 'atomicity', 'isolation', 'durability', 'raft', 'paxos',
            'byzantine', 'leader election', 'consensus algorithm',
            
            # Load Balancing & Networking
            'load balancer', 'round-robin', 'least connections', 'sticky sessions',
            'session affinity', 'upstream', 'downstream', 'bonding', 'lacp',
            'link aggregation', 'nic teaming', 'active-backup',
            
            # Storage & File Systems
            'shared storage', 'cluster file system', 'gfs', 'gfs2', 'ocfs',
            'cluster lvm', 'multipath', 'alua', 'raw device', 'block device',
            'file transfer protocol', 'ftp', 'sftp', 'scp', 'rsync', 'nfs',
            'cifs', 'smb',
            
            # Monitoring & SLA
            'sla', 'slo', 'sli', 'mttr', 'mtbf', 'uptime', 'downtime',
            'incident', 'outage', 'runbook', 'playbook', 'escalation'
        ]
    }
}

# Performance optimization settings
PERFORMANCE_SETTINGS = {
    'chunk_size': 1024,
    'sample_rate': 16000,
    'channels': 1,
    'format': 'int16',
    'buffer_size': 4096
}

def get_optimal_settings(service='google', use_case='general'):
    """
    Get optimal settings for a specific service and use case
    
    Args:
        service: Recognition service ('google', 'whisper', 'azure', etc.)
        use_case: Use case ('general', 'technical', 'meeting', 'interview')
    
    Returns:
        dict: Optimized settings for the service and use case
    """
    base_settings = AUDIO_SETTINGS.copy()
    
    if service == 'google':
        base_settings.update(GOOGLE_CONFIG)
        if use_case == 'technical':
            base_settings['language'] = 'en-US'
            # Add technical phrases for better recognition
            base_settings['phrases'] = LANGUAGE_MODELS['english_technical']['phrases']
    
    elif service == 'whisper':
        base_settings.update({
            'model': 'base',  # Start with base model for speed
            'language': 'en',
            'task': 'transcribe',
            'temperature': 0.0
        })
    
    return base_settings

def get_service_recommendations():
    """
    Get recommendations for different use cases
    
    Returns:
        dict: Service recommendations for different scenarios
    """
    return {
        'best_accuracy': 'whisper',
        'fastest': 'google',
        'offline': 'whisper',
        'enterprise': 'azure',
        'cost_effective': 'google',
        'technical_content': 'whisper',
        'real_time': 'google'
    }
