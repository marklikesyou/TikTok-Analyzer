# TikTok Video Content Analyzer

A flexible, AI-powered tool that analyzes TikTok video content across any domain or topic. Using GPT-4 and advanced speech recognition, it provides comprehensive content analysis regardless of the subject matter.

## Features

- Topic-agnostic video content analysis
- Flexible search and download of TikTok videos
- High-quality audio transcription using OpenAI's Whisper
- Advanced content analysis using GPT-4o, including:
  - Theme identification
  - Key points extraction
  - Content style analysis
  - Mood/tone detection
  - Notable mentions and references
- Asynchronous video processing
- Comprehensive error handling

## Requirements

- Python 3.10+
- OpenAI API key
- RapidAPI key for TikTok endpoints

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your API keys:
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `RAPIDAPI_KEY`: Your RapidAPI key

## Usage

1. Configure analysis parameters in `tiktok_analyzer.py`:
   ```python
   SEARCH_QUERY = "your search query"  # your main query
   MAX_VIDEOS = 5  # number of videos from the SERP
   ```

2. Run the analyzer:
   ```python
   python tiktok_analyzer.py
   ```

## Output

The analyzer provides detailed insights for each video:
- Comprehensive themes and topics
- Content style analysis
- Emotional tone and mood
- Key points and takeaways
- Notable mentions and references

## Technical Details

- Uses OpenAI's GPT-4o for intelligent content analysis
- Implements asynchronous processing for better performance
- Provides robust error handling and recovery
- Supports flexible query processing

## Notes

- API costs depend on video length and analysis depth
- Processing time varies based on video count and length
- Internet connection required for API calls
- Temporary files are automatically cleaned up

## Error Handling

The system includes comprehensive error handling for:
- API failures
- Network issues
- Invalid video URLs
- Transcription errors
- Content analysis failures
