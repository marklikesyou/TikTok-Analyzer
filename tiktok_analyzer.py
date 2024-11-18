import os
import asyncio
import requests
import whisper
import openai
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import json
from moviepy.editor import VideoFileClip
import aiohttp
from datetime import datetime
import nest_asyncio
import soundfile as sf
import numpy as np
import re

#everything begins here ;)
SEARCH_QUERY = "pizza recipe"
MAX_VIDEOS = 5

nest_asyncio.apply()

class TikTokAnalyzer:
    def __init__(self, api_key=None, rapidapi_key=None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.rapidapi_key = rapidapi_key or os.getenv('RAPIDAPI_KEY')
        self.masterquery = None
        
        self.openai_client = openai.AsyncOpenAI(api_key=self.api_key)
        self.whisper_model = whisper.load_model("base")
        self.data_dir = "downloaded_videos"
        os.makedirs(self.data_dir, exist_ok=True)

    def filter_videos(self, videos):
        filtered_videos = []
        for video in videos:
            if (video.get('duration', 0) > 0 and 
                video.get('play_count', 0) > 0 and 
                video.get('digg_count', 0) > 0):  
                filtered_videos.append(video)
        
        return filtered_videos

    def search_videos(self, query, max_videos=5):
        url = "https://tiktok-scraper7.p.rapidapi.com/feed/search"
        
        querystring = {
            "keywords": query,
            "region": "US",
            "count": str(max_videos),
            "cursor": "0",
            "publish_time": "0",
            "sort_type": "0"
        }
        
        headers = {
            "X-RapidAPI-Key": os.getenv('RAPIDAPI_KEY'),
            "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
        }

        try:
            response = requests.get(url, headers=headers, params=querystring)
            response.raise_for_status()
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'videos' in data['data']:
                    videos = data['data']['videos']
                    return self.filter_videos(videos)
                else:
                    return []
            return []
        except requests.exceptions.RequestException as e:
            print(f"Error in video search: {str(e)}")
            if hasattr(e.response, 'text'):
                print(f"Response text: {e.response.text}")
            return []
        except Exception as e:
            print(f"Unexpected error in video search: {str(e)}")
            return []

    async def download_video(self, video_url, video_id):
        output_path = os.path.join(self.data_dir, f"{video_id}.mp4")
        if os.path.exists(output_path):
            return output_path
            
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(video_url, headers=headers) as response:
                    if response.status == 200:
                        with open(output_path, 'wb') as f:
                            while True:
                                chunk = await response.content.read(8192)
                                if not chunk:
                                    break
                                f.write(chunk)
                        return output_path
                    else:
                        print(f"Failed to download video: HTTP {response.status}")
                        print(f"Response headers: {response.headers}")
                        return None
                        
        except Exception as e:
            print(f"Error downloading video: {str(e)}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return None

    def extract_audio(self, video_path):
        try:
            video_path = os.path.abspath(video_path)
            base_name = os.path.splitext(video_path)[0]
            output_path = f"{base_name}.wav"
            
            if not os.path.exists(video_path):
                return None
            
            video = VideoFileClip(video_path)
            if video.audio is None:
                video.close()
                return None
                
            video.audio.write_audiofile(output_path)
            
            if not os.path.exists(output_path):
                video.close()
                return None
            
            video.close()
            
            return output_path
        except Exception as e:
            print(f"Error extracting audio: {str(e)}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            if 'video' in locals():
                video.close()
            return None

    def transcribe_video(self, audio_path):
        try:
            if not audio_path or not os.path.exists(audio_path):
                return ""
                
            audio_data, sample_rate = sf.read(audio_path)
            
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            audio_data = audio_data.astype(np.float32)
            
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            if sample_rate != 16000:
                audio_length = len(audio_data)
                new_length = int(audio_length * 16000 / sample_rate)
                audio_data = np.interp(
                    np.linspace(0, audio_length, new_length),
                    np.arange(audio_length),
                    audio_data
                ).astype(np.float32)  
            
            result = self.whisper_model.transcribe(
                audio_data,
                language=None,  
                task="transcribe",
                fp16=False  
            )
            
            try:
                os.remove(audio_path)
            except Exception as e:
                print(f"Error cleaning up audio file: {str(e)}")
            
            return result["text"]
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            return ""

    async def analyze_content(self, video_data, transcription):
        try:
            prompt = f"""Analyze this TikTok video and respond with ONLY valid JSON:
            Title: {video_data.get('title', 'Unknown')}
            Transcription: {transcription}
            Query Context: {self.masterquery}

            Extract the following information and respond in JSON format:
            1. Main themes and topics discussed
            2. Key points or takeaways
            3. Content style and format
            4. Overall mood/tone
            5. Notable mentions or references

            Format as JSON with these exact keys:
            {{
                "themes": ["list of main themes"],
                "key_points": ["list of main points"],
                "content_style": "string",
                "mood": "string",
                "notable_mentions": ["list of important references"]
            }}"""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            try:
                content = response.choices[0].message.content.strip()
                content = content.replace("```json", "").replace("```", "").strip()
                return json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON response: {content}")
                return {
                    "themes": [],
                    "key_points": [],
                    "content_style": "unknown",
                    "mood": "unknown",
                    "notable_mentions": []
                }
        except Exception as e:
            print(f"Error in content analysis: {str(e)}")
            return {
                "themes": [],
                "key_points": [],
                "content_style": "unknown",
                "mood": "unknown",
                "notable_mentions": []
            }

    async def analyze_results(self, results):
        all_analyses = [r for r in results if r.get('analysis')]
        
        if not all_analyses:
            return "No valid analyses found."

        analyses_text = "\n".join([
            f"Video {i+1}:\n" +
            f"Themes: {analysis['analysis'].get('themes', [])}\n" +
            f"Key Points: {analysis['analysis'].get('key_points', [])}\n" +
            f"Style: {analysis['analysis'].get('content_style', '')}\n" +
            f"Mood: {analysis['analysis'].get('mood', '')}\n"
            for i, analysis in enumerate(all_analyses)
        ])

        prompt = f"""Based on the analysis of {len(all_analyses)} TikTok videos about "{self.masterquery}", provide a brief summary (max 100 words) covering:
        1. Common themes and patterns
        2. Overall content style and mood
        3. Key takeaways
        
        Videos analyzed:
        {analyses_text}"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            
            print("\n=== Content Summary ===")
            print(response.choices[0].message.content.strip())
            print("=====================")
            
        except Exception as e:
            print(f"Error in summary generation: {str(e)}")

    async def process_videos(self, query, max_videos=5):
        self.masterquery = query
        videos = self.search_videos(query, max_videos * 2)
        if not videos:
            return []
        
        filtered_videos = self.filter_videos(videos)[:max_videos]
        analyzed_results = []
        
        for video in filtered_videos:
            try:
                video_path = await self.download_video(video['play'], video['video_id'])
                if not video_path:
                    continue
                    
                audio_path = self.extract_audio(video_path)
                if not audio_path:
                    continue
                    
                transcription = self.transcribe_video(audio_path)
                if not transcription:
                    continue
                
                analysis = await self.analyze_content(video, transcription)
                
                analyzed_results.append({
                    'video_id': video['video_id'],
                    'title': video.get('title', ''),
                    'analysis': analysis
                })
                
                print(f"Successfully processed video {video['video_id']}\n")
                
            except Exception as e:
                print(f"Error processing video {video.get('video_id', 'unknown')}: {str(e)}")
                continue
                
        if analyzed_results:
            await self.analyze_results(analyzed_results)
            
        return analyzed_results

    def analyze_videos(self, query, max_videos=5):
        self.masterquery = query
        print(f"\nAnalyzing TikTok videos about: {query}")
        return self.process_videos(query, max_videos)

async def main():
    load_dotenv()
    analyzer = TikTokAnalyzer()
    
    print(f"\nAnalyzing TikTok videos about: {SEARCH_QUERY}")
    try:
        async with openai.AsyncOpenAI() as client:
            analyzer.openai_client = client
            await analyzer.analyze_videos(SEARCH_QUERY, max_videos=MAX_VIDEOS)
    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        if hasattr(analyzer, 'openai_client'):
            await analyzer.openai_client.close()

if __name__ == "__main__":
    asyncio.run(main())
