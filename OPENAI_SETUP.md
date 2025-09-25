# OpenAI Integration Setup Guide

This guide explains how to configure OpenAI integration for enhanced AI analysis in the speech transcription application.

## Prerequisites

- OpenAI API account with credits
- OpenAI API key
- Internet connection for API calls

## Setup Options

### Option 1: Environment Variable (Recommended)

Set the `OPENAI_API_KEY` environment variable:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**Linux/macOS:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Option 2: Configuration File

Create a file named `openai_key.txt` in the project root:

```
your-api-key-here
```

**Note:** This file will be ignored by git for security.

## Getting Your OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (it starts with `sk-`)

## Cost Management

The application uses the `gpt-4o-mini` model by default, which is cost-effective:
- ~$0.15 per 1M input tokens
- ~$0.60 per 1M output tokens
- Typical analysis: ~$0.001-0.01 per request

## Features

### AI Analysis Features
- **Topic Analysis**: Enhanced insights for technical topics
- **Troubleshooting Analysis**: Advanced debugging guidance
- **Caching**: Responses are cached to reduce API calls
- **Rate Limiting**: Built-in protection against API limits
- **Fallback**: Template-based responses when API is unavailable

### Configuration Options

Edit `openai_config.py` to customize:

```python
# Model settings
self.model = "gpt-4o-mini"  # Change model
self.max_tokens = 1000     # Adjust response length
self.temperature = 0.7     # Adjust creativity (0.0-1.0)

# Rate limiting
self.max_requests_per_minute = 20
self.max_tokens_per_minute = 40000

# Caching
self.cache_responses = True
self.cache_expiration_hours = 24
```

## Testing the Integration

1. **Check Status**: Look for "OpenAI: Configured" in the status bar
2. **Test Analysis**: Click on keywords to see AI-enhanced analysis
3. **Monitor Usage**: Check OpenAI dashboard for API usage

## Troubleshooting

### Common Issues

**"OpenAI: Template Mode"**
- API key not configured
- Check environment variable or config file
- Verify API key is valid

**Rate Limit Errors**
- Too many requests in a short time
- Wait a minute before trying again
- Adjust rate limits in config

**Network Errors**
- Check internet connection
- Verify OpenAI API is accessible
- Check firewall settings

### Debug Mode

Enable debug output by setting:
```python
# In openai_config.py
self.debug = True
```

## Security Notes

- Never commit API keys to version control
- Use environment variables for production
- Monitor API usage regularly
- Rotate API keys periodically

## Cost Optimization

- **Caching**: Responses are cached for 24 hours
- **Rate Limiting**: Prevents excessive API calls
- **Efficient Model**: Uses cost-effective gpt-4o-mini
- **Smart Fallback**: Uses templates when API is unavailable

## Support

For issues with OpenAI integration:
1. Check the console output for error messages
2. Verify API key configuration
3. Test with a simple API call
4. Check OpenAI service status

The application will gracefully fall back to template-based analysis if OpenAI is unavailable.
