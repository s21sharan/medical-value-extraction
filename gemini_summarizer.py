import os
import json
import requests
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class GeminiSummarizer:
    """
    A class for generating value summaries using the Gemini API
    """
    
    def __init__(self, api_key=None):
        """Initialize the GeminiSummarizer with an API key."""
        # Use provided API key or get from environment variables
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided. Please set GEMINI_API_KEY in .env file or pass as parameter.")
        
        # Use the correct model name for v1beta
        self.model_name = "gemini-1.5-pro-latest"
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
    
    def generate_summary(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate a one-sentence summary of a person's values using Gemini API.
        
        Args:
            analysis_results: Dictionary containing value analysis results
            
        Returns:
            A one-sentence summary of the person's values
        """
        # Extract key information from analysis results
        top_values = analysis_results.get('top_values', [])
        value_stances = analysis_results.get('value_stances', {})
        value_trends = analysis_results.get('value_trends', {})
        
        # Prepare detailed information about values
        value_details = []
        for value, count in top_values:
            stances = value_stances.get(value, {})
            # Calculate the sentiment ratio
            promotes = stances.get(1, 0)
            reduces = stances.get(-1, 0)
            neutral = stances.get(0, 0)
            
            sentiment = "positive" if promotes > reduces else "negative" if reduces > promotes else "neutral"
            value_details.append({
                "value": value,
                "count": count,
                "sentiment": sentiment,
                "promotes": promotes,
                "reduces": reduces,
                "neutral": neutral
            })
        
        # Extract trend information if available
        trend_info = []
        if value_trends:
            segments = sorted(value_trends.keys())
            for i, segment in enumerate(segments):
                if i > 0:  # Skip first segment as we're looking for changes
                    changes = value_trends[segment].get('changes_from_previous', [])
                    if changes:
                        trend_info.extend(changes)
        
        # Construct prompt for Gemini
        prompt = {
            "contents": [{
                "parts": [{
                    "text": f"""
I need a ONE SENTENCE summary of a person's values based on the following analysis. 
The summary should mention their primary values and indicate if their values have changed over time.

Value Details:
{json.dumps(value_details, indent=2)}

Value Trends:
{json.dumps(trend_info, indent=2)}

Generate a concise, insightful one-sentence summary that captures both their core values and any significant changes in those values over time.
The summary should be in the format: "This person [value statement] and [change statement if applicable]."
"""
                }]
            }]
        }
        
        # Make request to Gemini API
        response = self._make_api_request(prompt)
        
        if response and 'candidates' in response:
            # Extract the summary text from response
            text = response['candidates'][0]['content']['parts'][0]['text'].strip()
            
            # Clean up the response - we only want one sentence
            sentences = text.split('.')
            summary = sentences[0].strip() + '.'
            
            return summary
        
        # Fallback if API fails
        if top_values:
            # Simple fallback summary
            top_value = top_values[0][0].lower()
            return f"This person primarily values {top_value}."
        else:
            return "No clear value patterns could be identified in this conversation."
    
    def _make_api_request(self, prompt: Dict) -> Dict:
        """Make a request to the Gemini API."""
        headers = {
            "Content-Type": "application/json"
        }
        
        params = {
            "key": self.api_key
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                params=params,
                json=prompt
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error calling Gemini API: {response.status_code}")
                print(response.text)
                return None
                
        except Exception as e:
            print(f"Exception when calling Gemini API: {e}")
            return None 