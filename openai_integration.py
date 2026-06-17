# openai_integration.py
# OpenAI API integration for AI analysis

import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from typing import Callable, Optional

import openai

from openai_config import openai_config


class OpenAIAnalyzer:
    """OpenAI API integration for generating AI analysis."""

    def __init__(self):
        self.client = None
        self.rate_limit_tracker = {
            "request_times": [],
            "token_count": 0,
            "token_window_start": time.time(),
        }

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
        return self.client is not None and openai_config.is_configured()

    def _check_rate_limits(self, estimated_tokens: int = 0) -> bool:
        now = time.time()
        minute_ago = now - 60

        self.rate_limit_tracker["request_times"] = [
            req_time
            for req_time in self.rate_limit_tracker["request_times"]
            if req_time > minute_ago
        ]

        if now - self.rate_limit_tracker["token_window_start"] > 60:
            self.rate_limit_tracker["token_count"] = 0
            self.rate_limit_tracker["token_window_start"] = now

        if len(self.rate_limit_tracker["request_times"]) >= openai_config.max_requests_per_minute:
            return False

        if self.rate_limit_tracker["token_count"] + estimated_tokens > openai_config.max_tokens_per_minute:
            return False

        return True

    def _record_request(self, tokens_used: int):
        now = time.time()
        self.rate_limit_tracker["request_times"].append(now)
        self.rate_limit_tracker["token_count"] += tokens_used

    def _get_cache_key(self, prompt: str, context: str) -> str:
        content = f"{prompt}:{context}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        if not openai_config.cache_responses:
            return None

        try:
            cache_file = os.path.join("cache", f"openai_{cache_key}.json")
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)

                cache_time = datetime.fromisoformat(cache_data["timestamp"])
                if datetime.now() - cache_time < timedelta(hours=openai_config.cache_expiration_hours):
                    return cache_data["response"]
        except Exception as e:
            print(f"Error reading cache: {e}")

        return None

    def _cache_response(self, cache_key: str, response: str):
        if not openai_config.cache_responses:
            return

        try:
            os.makedirs("cache", exist_ok=True)
            cache_file = os.path.join("cache", f"openai_{cache_key}.json")
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "response": response,
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"Error caching response: {e}")

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = openai_config.get_model_pricing()
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 6)

    def _debug(self, message: str):
        if os.getenv("VERBOSE", "").lower() in ("1", "true", "yes", "on"):
            print(message)

    def generate_topic_analysis(
        self,
        category: str,
        explanation: dict,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, float]:
        if not self.is_available():
            return self._get_fallback_topic_analysis(category, explanation), 0.0

        question_type = explanation.get("question_type", "troubleshooting")
        self._debug(
            f"OpenAI prompt generation - category: '{category}', question_type: '{question_type}'"
        )

        boot_process_info = ""
        if "boot_process" in explanation:
            boot_process_info = f"\nBoot Process Details:\n{explanation.get('boot_process', '')}\n"

        if question_type == "architecture":
            prompt = f"""
            You are a system architect. Analyze the following technical topic and provide architectural guidance:

            Topic: {explanation.get('title', category)}
            Summary: {explanation.get('summary', '')}
            Challenges: {explanation.get('challenges', '')}
            Commands: {explanation.get('commands', '')}
            {boot_process_info}
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
        elif question_type == "security":
            prompt = f"""
            You are a security expert. Analyze the following technical topic and provide security guidance:

            Topic: {explanation.get('title', category)}
            Summary: {explanation.get('summary', '')}
            Challenges: {explanation.get('challenges', '')}
            Commands: {explanation.get('commands', '')}
            {boot_process_info}
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
        else:
            prompt = f"""
            Analyze the following technical topic and provide enhanced troubleshooting insights:

            Topic: {explanation.get('title', category)}
            Summary: {explanation.get('summary', '')}
            Challenges: {explanation.get('challenges', '')}
            Commands: {explanation.get('commands', '')}
            {boot_process_info}
            User Question: {explanation.get('context', '')}

            Please provide:
            1. Advanced insights and best practices
            2. Performance optimization strategies
            3. Common pitfalls and how to avoid them
            4. Recommended tools and monitoring approaches
            5. Next steps for implementation

            Format the response with clear sections and actionable advice.
            """

        self._debug(f"Using {question_type} prompt for OpenAI API call")
        return self._make_api_call(prompt, f"topic:{category}", stream_callback)

    def generate_troubleshooting_analysis(
        self,
        question: str,
        suggestions: dict,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, float]:
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

        return self._make_api_call(prompt, f"troubleshooting:{question[:50]}", stream_callback)

    def _extract_response_text(self, response) -> str:
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text

        text_parts = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for content in getattr(item, "content", []) or []:
                    if getattr(content, "type", None) == "output_text":
                        text_parts.append(content.text)
        return "".join(text_parts)

    def _make_responses_api_call(
        self,
        prompt: str,
        stream_callback: Optional[Callable[[str], None]],
    ) -> tuple[str, float, int]:
        use_stream = openai_config.stream_responses and stream_callback is not None

        if use_stream:
            stream = self.client.responses.create(
                model=openai_config.model,
                instructions=openai_config.system_instructions,
                input=prompt,
                max_output_tokens=openai_config.max_tokens,
                temperature=openai_config.temperature,
                store=False,
                stream=True,
            )

            ai_response = ""
            for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        ai_response += delta
                        stream_callback(ai_response)
                elif event_type == "response.completed":
                    usage = getattr(getattr(event, "response", None), "usage", None)
                    if usage:
                        cost = self._calculate_cost(
                            getattr(usage, "input_tokens", 0),
                            getattr(usage, "output_tokens", 0),
                        )
                        total_tokens = getattr(usage, "total_tokens", 0)
                        return ai_response, cost, total_tokens

            estimated_tokens = max(len(prompt.split()), len(ai_response.split())) + 100
            return ai_response, 0.0, estimated_tokens

        response = self.client.responses.create(
            model=openai_config.model,
            instructions=openai_config.system_instructions,
            input=prompt,
            max_output_tokens=openai_config.max_tokens,
            temperature=openai_config.temperature,
            store=False,
        )

        ai_response = self._extract_response_text(response)
        usage = response.usage
        cost = self._calculate_cost(usage.input_tokens, usage.output_tokens)
        return ai_response, cost, usage.total_tokens

    def _make_chat_completions_call(self, prompt: str) -> tuple[str, float, int]:
        response = self.client.chat.completions.create(
            model=openai_config.model,
            messages=[
                {"role": "system", "content": openai_config.system_instructions},
                {"role": "user", "content": prompt},
            ],
            max_tokens=openai_config.max_tokens,
            temperature=openai_config.temperature,
        )

        ai_response = response.choices[0].message.content or ""
        usage = response.usage
        cost = self._calculate_cost(usage.prompt_tokens, usage.completion_tokens)
        return ai_response, cost, usage.total_tokens

    def _make_api_call(
        self,
        prompt: str,
        context: str,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, float]:
        if not self._check_rate_limits(estimated_tokens=openai_config.max_tokens):
            return "Rate limit exceeded. Please wait before making more requests.", 0.0

        cache_key = self._get_cache_key(prompt, context)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            if stream_callback:
                stream_callback(cached_response)
            return cached_response, 0.0

        try:
            if openai_config.use_responses_api:
                ai_response, cost, total_tokens = self._make_responses_api_call(
                    prompt, stream_callback
                )
            else:
                ai_response, cost, total_tokens = self._make_chat_completions_call(prompt)
                if stream_callback:
                    stream_callback(ai_response)

            self._record_request(total_tokens)
            self._cache_response(cache_key, ai_response)
            return ai_response, cost

        except Exception as e:
            print(f"OpenAI API error: {e}")
            if openai_config.use_fallback_on_error:
                return self._get_fallback_response(prompt, context), 0.0
            return f"OpenAI API error: {str(e)}", 0.0

    def _get_fallback_response(self, prompt: str, context: str) -> str:
        if openai_config.fallback_to_template:
            return self._get_template_response(context)
        return "AI Analysis temporarily unavailable. Please check your OpenAI configuration."

    def _get_template_response(self, context: str) -> str:
        if "topic:" in context:
            category = context.replace("topic:", "")
            return f"""
AI-Enhanced Analysis: {category.title()}

Advanced Insights:
- This topic is commonly encountered in {category} environments
- Key performance indicators to monitor
- Best practices for optimization

Advanced Commands:
- Performance monitoring: htop, iostat, netstat
- Debugging: strace, gdb, valgrind
- Log analysis: grep, awk, sed

Common Pitfalls:
- Memory leaks and resource management
- Security vulnerabilities to watch for
- Performance bottlenecks

Next Steps:
- Consider implementing monitoring
- Review security best practices
- Plan for scalability

AI Suggestion:
Based on the topic '{category}', consider exploring related technologies and implementing automated testing and monitoring solutions.
            """
        return """
AI Troubleshooting Analysis

Question Analysis:
- Detected question type: Technical troubleshooting
- Complexity level: Intermediate to Advanced

AI-Enhanced Approach:
- Systematic debugging methodology
- Root cause analysis techniques
- Performance optimization strategies

Advanced Diagnostics:
- Log analysis with grep, awk, sed
- System monitoring with htop, iostat
- Network analysis with netstat, tcpdump

Quick Wins:
- Check system resources first
- Verify configuration files
- Test with minimal configuration

Long-term Solutions:
- Implement monitoring and alerting
- Document the resolution process
- Create runbooks for future reference

AI Recommendation: Consider implementing automated testing and monitoring to prevent similar issues in the future.
            """

    def _get_fallback_topic_analysis(self, category: str, explanation: dict) -> str:
        return self._get_template_response(f"topic:{category}")

    def _get_fallback_troubleshooting_analysis(self, question: str, suggestions: dict) -> str:
        return self._get_template_response("troubleshooting")


openai_analyzer = OpenAIAnalyzer()
