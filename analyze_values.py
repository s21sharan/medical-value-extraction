#!/usr/bin/env python3
from values_framework import ValuesFramework
import pandas as pd
import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from dotenv import load_dotenv
from gemini_summarizer import GeminiSummarizer

# Load environment variables
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze values in conversations using ValuesNet")
    parser.add_argument("--model_type", type=str, default="tfidf", choices=["tfidf", "transformer"],
                        help="Model type to use (tfidf or transformer)")
    parser.add_argument("--data_dir", type=str, default="valuesNet-dataset",
                        help="Directory containing ValuesNet dataset")
    parser.add_argument("--conversation_file", type=str,
                        help="JSON file containing conversation data (optional)")
    parser.add_argument("--analyze_sample", action="store_true",
                        help="Analyze a sample conversation")
    parser.add_argument("--use_gemini", action="store_true",
                        help="Use Gemini API for summarization")
    parser.add_argument("--gemini_api_key", type=str,
                        help="Gemini API key (optional, can also be set in .env file)")
    return parser.parse_args()

def visualize_value_distribution(results):
    """Visualize the distribution of values detected in the conversation."""
    # Extract value counts
    values = []
    counts = []
    for value, count in results['values_detected'].items():
        values.append(value)
        counts.append(count)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({"Value": values, "Count": counts})
    df = df.sort_values("Count", ascending=False)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Value", y="Count", data=df)
    plt.title("Distribution of Values in Conversation")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("value_distribution.png")
    plt.close()
    
    print(f"Value distribution visualization saved to value_distribution.png")

def visualize_value_changes(results):
    """Visualize how values change over time in the conversation."""
    if not results['value_trends']:
        return
    
    # Prepare data for plotting
    segments = list(results['value_trends'].keys())
    segment_indices = range(len(segments))
    
    # Get all unique values across segments
    all_values = set()
    for segment in segments:
        for value, _ in results['value_trends'][segment]['top_values']:
            all_values.add(value)
    
    # Create a dict to store value counts per segment
    value_counts = {value: [0] * len(segments) for value in all_values}
    
    # Fill in the counts
    for i, segment in enumerate(segments):
        segment_values = dict(results['value_trends'][segment]['top_values'])
        for value in all_values:
            if value in segment_values:
                value_counts[value][i] = segment_values[value]
    
    # Create DataFrame for plotting
    data = []
    for value, counts in value_counts.items():
        for i, count in enumerate(counts):
            data.append({"Segment": segments[i], "Value": value, "Count": count})
    
    df = pd.DataFrame(data)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    ax = sns.lineplot(x="Segment", y="Count", hue="Value", data=df, marker='o', linewidth=2)
    plt.title("Changes in Value Expression Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("value_changes.png")
    plt.close()
    
    print(f"Value changes visualization saved to value_changes.png")

def get_gemini_summary(results, api_key=None):
    """Generate a summary of values using Gemini API."""
    try:
        # Initialize the GeminiSummarizer
        summarizer = GeminiSummarizer(api_key=api_key)
        
        # Generate the summary
        summary = summarizer.generate_summary(results)
        return summary
    except Exception as e:
        print(f"Error using Gemini API: {e}")
        # Fallback to the basic summary
        return results['values_summary']

def analyze_user_conversations(conversation_file, use_gemini=False, gemini_api_key=None):
    """Analyze conversations from a JSON file."""
    # Load conversation data
    with open(conversation_file, 'r') as f:
        conversations = json.load(f)
    
    # Initialize framework
    framework = ValuesFramework(model_type="tfidf")
    framework.load_valuesnet_data()
    framework.train_values_classifier()
    framework.train_label_classifier()
    
    # Analyze each user's conversation
    all_results = {}
    for user_id, texts in conversations.items():
        print(f"\nAnalyzing conversation for user {user_id}...")
        results = framework.analyze_conversation(texts, user_id=user_id)
        all_results[user_id] = results
        
        # Get value summary (using Gemini if requested)
        if use_gemini:
            results['gemini_summary'] = get_gemini_summary(results, gemini_api_key)
            print(f"Gemini values summary for user {user_id}: {results['gemini_summary']}")
        else:
            print(f"Values summary for user {user_id}: {results['values_summary']}")
        
        # Print detailed results
        print(f"Top values: {', '.join([f'{v} ({c})' for v, c in results['top_values']])}")
        
        # Optional: save individual user results
        with open(f"user_{user_id}_values.json", 'w') as f:
            # Convert Counter objects to dictionaries for JSON serialization
            serializable_results = results.copy()
            serializable_results['values_detected'] = dict(results['values_detected'])
            for value, stances in results['value_stances'].items():
                serializable_results['value_stances'][value] = dict(stances)
            json.dump(serializable_results, f, indent=2)
    
    return all_results

def analyze_sample_conversation(use_gemini=False, gemini_api_key=None):
    """Analyze a sample conversation to demonstrate the framework."""
    # Initialize framework
    framework = ValuesFramework(model_type="tfidf")
    framework.load_valuesnet_data()
    framework.train_values_classifier()
    framework.train_label_classifier()
    
    # Sample conversation showing value change over time
    conversation = [
        # Early posts - emphasis on achievement and conformity
        "I'm working hard to get a promotion at my job. It's all about showing results.",
        "I always try to follow the rules and meet expectations. It's important to fit in.",
        "I believe that if you work hard enough, you'll always succeed.",
        "My parents taught me that respecting authority is critical to success.",
        
        # Middle posts - starting to question conformity, more emphasis on self-direction
        "Sometimes I wonder if following all these social norms is really making me happy.",
        "I've started to think more about what I actually want, not what others expect.",
        "I accomplished a lot at work, but I'm not sure if that's what matters most anymore.",
        "I've been exploring new hobbies that let me express myself more freely.",
        
        # Recent posts - strong emphasis on self-direction and universalism
        "I've decided to take a sabbatical to travel and find what truly matters to me.",
        "I care more about making a difference in the world than climbing the corporate ladder.",
        "Everyone should have the freedom to define success on their own terms.",
        "I believe we need to work together to create a more equitable society for everyone."
    ]
    
    # Analyze the conversation
    results = framework.analyze_conversation(conversation, user_id="sample_user")
    
    # Get value summary (using Gemini if requested)
    if use_gemini:
        results['gemini_summary'] = get_gemini_summary(results, gemini_api_key)
        print("\nGemini Values Summary:")
        print(results['gemini_summary'])
    
    # Print standard summary
    print("\nStandard Values Summary:")
    print(results['values_summary'])
    
    print("\nValue Analysis by Utterance:")
    for i, utterance in enumerate(results['utterances']):
        period = "Early" if i < 4 else "Middle" if i < 8 else "Recent"
        print(f"\n[{period}] {utterance['text']}")
        for value_info in utterance['values']:
            print(f"  - {value_info['value_type']} ({value_info['stance_label']}, confidence: {value_info['confidence']:.2f})")
    
    print("\nValue Trends:")
    if results['value_trends']:
        for segment, trend in results['value_trends'].items():
            print(f"\n{segment.replace('_', ' ').title()}:")
            print(f"  Top values: {', '.join([f'{v[0]} ({v[1]})' for v in trend['top_values']])}")
            if trend['changes_from_previous']:
                print(f"  Changes: {', '.join(trend['changes_from_previous'])}")
    
    # Create visualizations
    visualize_value_distribution(results)
    visualize_value_changes(results)
    
    return results

def analyze_valuesnet_dataset(model_type="tfidf", data_dir="valuesNet-dataset", use_gemini=False, gemini_api_key=None):
    """
    Analyze all utterances from the ValuesNet dataset directly instead of using sample conversations.
    This provides more comprehensive insights into value expressions across the entire dataset.
    """
    # Initialize framework
    framework = ValuesFramework(model_type=model_type)
    
    print(f"Loading ValuesNet dataset from {data_dir}...")
    framework.load_valuesnet_data(data_dir=data_dir)
    
    print(f"Training values classifier using {model_type} model...")
    framework.train_values_classifier()
    
    print("Training label classifier...")
    framework.train_label_classifier()
    
    # Extract all utterances from the dataset for analysis
    print("\nExtracting utterances from ValuesNet dataset for analysis...")
    all_utterances = []
    
    # Collect utterances from train, test, and eval sets
    for dataset_name, dataset in [
        ("Training", framework.train_data), 
        ("Test", framework.test_data), 
        ("Evaluation", framework.eval_data)
    ]:
        print(f"Processing {dataset_name} set ({len(dataset)} utterances)...")
        for _, row in dataset.iterrows():
            all_utterances.append(row["utterance"])
    
    print(f"\nAnalyzing {len(all_utterances)} utterances from ValuesNet dataset...")
    
    # Split utterances into smaller batches to avoid memory issues
    batch_size = 1000
    batches = [all_utterances[i:i+batch_size] for i in range(0, len(all_utterances), batch_size)]
    
    # Process each batch
    all_values = Counter()
    all_value_stances = defaultdict(Counter)
    all_utterance_results = []
    
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)}...")
        results = framework.analyze_conversation(batch, user_id=f"valuesnet_batch_{i}")
        
        # Collect statistics
        all_values.update(results['values_detected'])
        for value, stances in results['value_stances'].items():
            all_value_stances[value].update(stances)
        all_utterance_results.extend(results['utterances'])
    
    # Compile results
    compiled_results = {
        'utterances': all_utterance_results,
        'values_detected': all_values,
        'value_stances': all_value_stances,
        'top_values': all_values.most_common(10)
    }
    
    # Generate summary
    compiled_results['values_summary'] = framework.generate_values_summary(compiled_results)
    
    # Get Gemini summary if requested
    if use_gemini:
        compiled_results['gemini_summary'] = get_gemini_summary(compiled_results, gemini_api_key)
        print("\nGemini Values Summary for ValuesNet dataset:")
        print(compiled_results['gemini_summary'])
    
    # Print standard summary
    print("\nStandard Values Summary for ValuesNet dataset:")
    print(compiled_results['values_summary'])
    
    # Print top values with counts
    print("\nTop Values in ValuesNet dataset:")
    for value, count in compiled_results['top_values']:
        print(f"  - {value}: {count} occurrences")
    
    # Print stance statistics
    print("\nValue Stance Statistics:")
    for value, stances in compiled_results['value_stances'].items():
        promotes = stances.get(1, 0)
        reduces = stances.get(-1, 0)
        neutral = stances.get(0, 0)
        total = promotes + reduces + neutral
        
        if total > 0:
            print(f"  - {value}: Promotes: {promotes} ({promotes/total:.1%}), "
                  f"Reduces: {reduces} ({reduces/total:.1%}), "
                  f"Neutral: {neutral} ({neutral/total:.1%})")
    
    # Create visualizations
    visualize_value_distribution(compiled_results)
    
    # Save results to file
    with open("valuesnet_analysis_results.json", 'w') as f:
        # Convert Counter objects to dictionaries for JSON serialization
        serializable_results = compiled_results.copy()
        serializable_results['values_detected'] = dict(compiled_results['values_detected'])
        
        # Convert value stances with proper key conversion for JSON serialization
        serializable_stances = {}
        for value, stances in compiled_results['value_stances'].items():
            # Convert keys from numpy types to native Python types
            serializable_stances[value] = {int(k): int(v) for k, v in stances.items()}
        
        serializable_results['value_stances'] = serializable_stances
        
        # Remove utterances to keep file size manageable
        serializable_results.pop('utterances', None)
        json.dump(serializable_results, f, indent=2)
    
    print("\nAnalysis results saved to valuesnet_analysis_results.json")
    return compiled_results

def main():
    args = parse_args()
    
    # Get Gemini API key from arguments or environment
    gemini_api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
    
    if args.conversation_file:
        all_results = analyze_user_conversations(
            args.conversation_file, 
            use_gemini=args.use_gemini, 
            gemini_api_key=gemini_api_key
        )
    elif args.analyze_sample:
        results = analyze_sample_conversation(
            use_gemini=args.use_gemini, 
            gemini_api_key=gemini_api_key
        )
    else:
        # Analyze ValuesNet dataset directly
        results = analyze_valuesnet_dataset(
            model_type=args.model_type,
            data_dir=args.data_dir,
            use_gemini=args.use_gemini,
            gemini_api_key=gemini_api_key
        )

if __name__ == "__main__":
    main() 