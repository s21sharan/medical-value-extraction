#!/usr/bin/env python3
"""
Script to create a .env file with the Gemini API key.
Run this script first to set up the API key for the values framework.
"""

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Create .env file with Gemini API key")
    parser.add_argument("--api_key", type=str, required=True,
                        help="Gemini API key to save in .env file")
    return parser.parse_args()

def create_env_file(api_key):
    """Create a .env file with the given Gemini API key."""
    env_content = f"GEMINI_API_KEY={api_key}\n"
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print(f".env file created successfully with Gemini API key.")

def main():
    args = parse_args()
    create_env_file(args.api_key)

if __name__ == "__main__":
    main() 