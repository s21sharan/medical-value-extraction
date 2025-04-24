# Values Detection Framework

A tool for identifying values-related utterances in conversations, tracking changes over time, and summarizing a person's values.

## Key Features

- **Value Detection**: Identifies expressions of Schwartz's 10 basic human values.
- **Stance Classification**: Determines if values are promoted, reduced, or neutral.
- **Temporal Analysis**: Tracks changes in values over time.
- **Summarization**: Generates concise summaries of values, with optional Gemini API integration for nuanced summaries.

## Models

The framework supports two types of models:

1. **TF-IDF + Logistic Regression**: Uses TF-IDF vectorization with a logistic regression classifier. It is faster and simpler, suitable for quick analyses.
2. **Transformer Model**: Leverages pretrained language models, specifically DistilBERT, fine-tuned on the ValuesNet dataset. It is more accurate but slower.

## Usage

### Basic Example

```python
from values_framework import ValuesFramework

framework = ValuesFramework(model_type="tfidf")
framework.load_valuesnet_data()
framework.train_values_classifier()

conversation = [
    "Helping others is crucial.",
    "I strive for personal success.",
    "Freedom is my top priority."
]

results = framework.analyze_conversation(conversation)
print(results['values_summary'])
```

### Complex Example in a Medical Context

```python
from values_framework import ValuesFramework

framework = ValuesFramework(model_type="transformer")
framework.load_valuesnet_data()
framework.train_values_classifier()

conversation = [
    "Patient: I've been feeling anxious about my health lately.",
    "Doctor: It's important to focus on maintaining a balanced lifestyle.",
    "Patient: I used to prioritize work over everything, but now I see the value in taking care of my well-being.",
    "Doctor: That's a positive change. Prioritizing your health can lead to better outcomes in the long run."
]

results = framework.analyze_conversation(conversation)
print(results['values_summary'])
```

### Command Line

```bash
# Analyze a sample conversation
python analyze_values.py --analyze_sample

# Use Gemini API for enhanced summaries
python analyze_values.py --analyze_sample --use_gemini
```

## Example Output

- **Built-in Summarizer**: "The patient initially valued work and achievement but has shifted towards valuing health and well-being."
- **Gemini API Summarizer**: "This patient has transitioned from prioritizing work to embracing a holistic approach to health, emphasizing the importance of balance and self-care."

## Designing a Framework for Long Conversations

- **Dataset Selection**: Choose datasets with diverse topics and speaker metadata.
- **Scalability**: Implement batch processing for large data volumes.
- **Contextual Understanding**: Use NLP techniques to maintain context in long conversations.

## Challenges

- **Dataset Availability**: Finding suitable datasets can be difficult.
- **Complexity of Values**: Values are context-dependent, complicating detection and summarization.

## Conclusion

This framework offers a starting point for analyzing values in conversations, with advanced summarization capabilities for deeper insights. Good luck in expanding this framework for broader applications! 