# openai_integration.py
# OpenAI API integration for AI analysis

import openai
import json
import hashlib
import time
import os
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from openai_config import openai_config

class OpenAIAnalyzer:
    """OpenAI API integration for generating AI analysis"""
    
    def __init__(self):
        self.client = None
        self.rate_limit_tracker = {
            'requests': [],
            'tokens': []
        }
        
        # Initialize OpenAI client if configured
        if openai_config.is_configured():
            try:
                self.client = openai.OpenAI(api_key=openai_config.api_key)
                print("OpenAI client initialized successfully")
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            print("OpenAI not configured - using template fallback")
    
    def is_available(self) -> bool:
        """Check if OpenAI is available and configured"""
        return self.client is not None and openai_config.is_configured()
    
    def _check_rate_limits(self) -> bool:
        """Check if we're within rate limits"""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old entries
        self.rate_limit_tracker['requests'] = [
            req_time for req_time in self.rate_limit_tracker['requests'] 
            if req_time > minute_ago
        ]
        self.rate_limit_tracker['tokens'] = [
            token_time for token_time in self.rate_limit_tracker['tokens'] 
            if token_time > minute_ago
        ]
        
        # Check limits
        if len(self.rate_limit_tracker['requests']) >= openai_config.max_requests_per_minute:
            return False
        
        if len(self.rate_limit_tracker['tokens']) >= openai_config.max_tokens_per_minute:
            return False
        
        return True
    
    def _record_request(self, tokens_used: int):
        """Record a request for rate limiting"""
        now = time.time()
        self.rate_limit_tracker['requests'].append(now)
        for _ in range(tokens_used):
            self.rate_limit_tracker['tokens'].append(now)
    
    def _get_cache_key(self, prompt: str, context: str) -> str:
        """Generate cache key for response"""
        content = f"{prompt}:{context}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available and not expired"""
        if not openai_config.cache_responses:
            return None
        
        try:
            cache_file = os.path.join('cache', f'openai_{cache_key}.json')
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check expiration
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                if datetime.now() - cache_time < timedelta(hours=openai_config.cache_expiration_hours):
                    return cache_data['response']
        except Exception as e:
            print(f"Error reading cache: {e}")
        
        return None
    
    def _cache_response(self, cache_key: str, response: str):
        """Cache the response"""
        if not openai_config.cache_responses:
            return
        
        try:
            os.makedirs('cache', exist_ok=True)
            cache_file = os.path.join('cache', f'openai_{cache_key}.json')
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'response': response
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"Error caching response: {e}")
    
    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost based on token usage for gpt-4o-mini"""
        # gpt-4o-mini pricing (as of 2024)
        input_cost_per_1k = 0.00015  # $0.15 per 1M tokens
        output_cost_per_1k = 0.0006   # $0.60 per 1M tokens
        
        input_cost = (prompt_tokens / 1000) * input_cost_per_1k
        output_cost = (completion_tokens / 1000) * output_cost_per_1k
        
        total_cost = input_cost + output_cost
        return round(total_cost, 6)  # Round to 6 decimal places
    
    def generate_topic_analysis(self, category: str, explanation: Dict[str, str]) -> tuple[str, float]:
        """Generate AI analysis for a topic with adaptive question type support"""
        if not self.is_available():
            return self._get_fallback_topic_analysis(category, explanation), 0.0
        
        # Get question type from explanation if available
        question_type = explanation.get('question_type', 'troubleshooting')
        print(f"DEBUG: OpenAI prompt generation - category: '{category}', question_type: '{question_type}'")
        
        # Create adaptive prompt based on question type
        if question_type == 'architecture':
            prompt = f"""
            You are a system architect. Analyze the following technical topic and provide architectural guidance:
            
            Topic: {explanation.get('title', category)}
            Summary: {explanation.get('summary', '')}
            Challenges: {explanation.get('challenges', '')}
            Commands: {explanation.get('commands', '')}
            
            User Question: {explanation.get('context', '')}
            
            Please provide architectural guidance focusing on:
            1. System architecture patterns and design principles
            2. Scalability and performance design strategies
            3. Component interaction and service boundaries
            4. Technology stack and tool recommendations
            5. Integration patterns and best practices
            6. Security architecture considerations
            7. Deployment and infrastructure patterns
            8. Fault tolerance and resilience patterns
            9. High availability and disaster recovery
            10. Compliance and regulatory considerations
            
            Format the response with clear architectural sections and design recommendations.
            """
        elif question_type == 'security':
            prompt = f"""
            You are a security expert. Analyze the following technical topic and provide security guidance:
            
            Topic: {explanation.get('title', category)}
            Summary: {explanation.get('summary', '')}
            Challenges: {explanation.get('challenges', '')}
            Commands: {explanation.get('commands', '')}
            
            User Question: {explanation.get('context', '')}
            
            Please provide security guidance focusing on:
            1. Security architecture and design principles
            2. Threat modeling and risk assessment
            3. Access control and authentication strategies
            4. Data protection and encryption
            5. Network security considerations
            6. Compliance and regulatory requirements
            7. Security monitoring and incident response
            8. Vulnerability management
            9. Security testing and validation
            10. Security best practices and standards
            
            Format the response with clear security sections and actionable recommendations.
            """
        else:  # Default to troubleshooting
            prompt = f"""
            Analyze the following technical topic and provide enhanced troubleshooting insights:
            
            Topic: {explanation.get('title', category)}
            Summary: {explanation.get('summary', '')}
            Challenges: {explanation.get('challenges', '')}
            Commands: {explanation.get('commands', '')}
            
            User Question: {explanation.get('context', '')}
            
            Please provide:
            1. Advanced insights and best practices
            2. Performance optimization strategies
            3. Common pitfalls and how to avoid them
            4. Recommended tools and monitoring approaches
            5. Next steps for implementation
            
            Format the response with clear sections and actionable advice.
            """
        
        print(f"DEBUG: Using {question_type} prompt for OpenAI API call")
        return self._make_api_call(prompt, f"topic:{category}")
    
    def generate_troubleshooting_analysis(self, question: str, suggestions: Dict[str, str]) -> tuple[str, float]:
        """Generate AI analysis for troubleshooting"""
        if not self.is_available():
            return self._get_fallback_troubleshooting_analysis(question, suggestions), 0.0
        
        prompt = f"""
        Analyze this technical troubleshooting scenario and provide enhanced guidance:
        
        Question: {question}
        Current Approach: {suggestions.get('approach', '')}
        Steps: {suggestions.get('steps', '')}
        Commands: {suggestions.get('commands', '')}
        
        Please provide:
        1. Root cause analysis methodology
        2. Advanced diagnostic techniques
        3. Performance optimization strategies
        4. Prevention strategies for future issues
        5. Monitoring and alerting recommendations
        
        Focus on systematic debugging and long-term solutions.
        """
        
        return self._make_api_call(prompt, f"troubleshooting:{question[:50]}")
    
    def _make_api_call(self, prompt: str, context: str) -> tuple[str, float]:
        """Make API call to OpenAI"""
        if not self._check_rate_limits():
            return "⚠️ Rate limit exceeded. Please wait before making more requests.", 0.0
        
        cache_key = self._get_cache_key(prompt, context)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response, 0.0  # Cached responses have no cost
        
        try:
            response = self.client.chat.completions.create(
                model=openai_config.model,
                messages=[
                    {"role": "system", "content": "You are a technical expert providing detailed analysis and recommendations for software development, system administration, and troubleshooting."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=openai_config.max_tokens,
                temperature=openai_config.temperature
            )
            
            ai_response = response.choices[0].message.content
            
            # Calculate cost based on token usage
            usage = response.usage
            cost = self._calculate_cost(usage.prompt_tokens, usage.completion_tokens)
            
            self._record_request(usage.total_tokens)
            
            # Cache the response
            self._cache_response(cache_key, ai_response)
            
            return ai_response, cost
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            if openai_config.use_fallback_on_error:
                fallback_response = self._get_fallback_response(prompt, context)
                return fallback_response, 0.0
            else:
                return f"❌ OpenAI API error: {str(e)}", 0.0
    
    def _get_fallback_response(self, prompt: str, context: str) -> str:
        """Get fallback response when OpenAI is unavailable"""
        if openai_config.fallback_to_template:
            return self._get_template_response(context)
        else:
            return "🤖 AI Analysis temporarily unavailable. Please check your OpenAI configuration."
    
    def _get_template_response(self, context: str) -> str:
        """Get template-based response as fallback"""
        if "topic:" in context:
            category = context.replace("topic:", "")
            return f"""
🤖 AI-Enhanced Analysis: {category.title()}

📊 **Advanced Insights:**
• This topic is commonly encountered in {category} environments
• Key performance indicators to monitor
• Best practices for optimization

🔧 **Advanced Commands:**
• Performance monitoring: `htop`, `iostat`, `netstat`
• Debugging: `strace`, `gdb`, `valgrind`
• Log analysis: `grep`, `awk`, `sed`

⚠️ **Common Pitfalls:**
• Memory leaks and resource management
• Security vulnerabilities to watch for
• Performance bottlenecks

🚀 **Next Steps:**
• Consider implementing monitoring
• Review security best practices
• Plan for scalability

💡 **AI Suggestion:**
Based on the topic '{category}', consider exploring related technologies and implementing automated testing and monitoring solutions.
            """
        else:
            return """
🤖 AI Troubleshooting Analysis

📝 **Question Analysis:**
• Detected question type: Technical troubleshooting
• Complexity level: Intermediate to Advanced

🎯 **AI-Enhanced Approach:**
• Systematic debugging methodology
• Root cause analysis techniques
• Performance optimization strategies

🔍 **Advanced Diagnostics:**
• Log analysis with `grep`, `awk`, `sed`
• System monitoring with `htop`, `iostat`
• Network analysis with `netstat`, `tcpdump`

⚡ **Quick Wins:**
• Check system resources first
• Verify configuration files
• Test with minimal configuration

🚀 **Long-term Solutions:**
• Implement monitoring and alerting
• Document the resolution process
• Create runbooks for future reference

💡 **AI Recommendation: Consider implementing automated testing and monitoring to prevent similar issues in the future.
            """
    
    def _get_fallback_topic_analysis(self, category: str, explanation: Dict[str, str]) -> str:
        """Fallback topic analysis"""
        return self._get_template_response(f"topic:{category}")
    
    def _get_fallback_troubleshooting_analysis(self, question: str, suggestions: Dict[str, str]) -> str:
        """Fallback troubleshooting analysis"""
        return self._get_template_response("troubleshooting")

# Global analyzer instance
openai_analyzer = OpenAIAnalyzer()
