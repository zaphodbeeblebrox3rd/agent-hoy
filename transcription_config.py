#!/usr/bin/env python3
"""
Transcription Configuration Options

This file contains configuration options for different speech recognition services
and their respective settings for optimal transcription quality.
"""

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
            'python', 'docker', 'kubernetes', 'aws', 'azure', 'linux', 'ubuntu',
            'git', 'github', 'gitlab', 'jenkins', 'ci', 'cd', 'devops',
            'database', 'mysql', 'postgresql', 'mongodb', 'redis',
            'networking', 'tcp', 'udp', 'http', 'https', 'ssl', 'tls',
            'security', 'encryption', 'authentication', 'authorization',
            'monitoring', 'logging', 'metrics', 'alerting', 'prometheus',
            'hpc', 'slurm', 'pbs', 'torque', 'maui', 'moab',
            'quantum', 'qubit', 'quantum computing', 'quantum algorithm',
            'fpga', 'rdma', 'roce', 'infiniBand', 'kernel bypass'
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
