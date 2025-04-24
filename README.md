# Values Detection Framework

A framework for identifying values-related utterances in conversations, tracking value changes over time, and summarizing a person's values into concise statements.

## Overview

This framework is designed to:

1. Detect value expressions in individual utterances or longer conversations
2. Identify which of Schwartz's 10 basic values are being expressed
3. Determine whether each value is being promoted, reduced, or is unrelated
4. Track how values change over time in conversation history
5. Generate single-sentence summaries of a person's values
6. Use Google's Gemini API to generate more nuanced value summaries (optional)

The framework leverages the [ValuesNet dataset](https://github.com/SRussell-CASA/ValuesNet), a corpus of annotated statements with human value expressions.

## Features

- **Value Identification**: Detects expressions of 10 basic human values (Achievement, Benevolence, Conformity, Hedonism, Power, Security, Self-Direction, Stimulation, Tradition, Universalism)
- **Stance Classification**: Determines if a statement promotes, reduces, or is unrelated to a value
- **Temporal Analysis**: Tracks how values change over time in conversation history
- **Value Summarization**: Generates concise summaries of a person's key values
- **Multiple Model Options**: Supports both TF-IDF and transformer-based models
- **Visualization**: Includes tools to visualize value distributions and changes
- **Gemini AI Integration**: Optional integration with Google's Gemini API for more nuanced value summaries

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/values-detection.git
cd values-detection

# Install dependencies
pip install -r requirements.txt

# Download ValuesNet dataset if not already available
# The dataset should be placed in a folder named 'valuesNet-dataset'

# Set up Gemini API (optional)
python create_env.py --api_key YOUR_GEMINI_API_KEY
```

## Usage

### Basic Usage

```python
from values_framework import ValuesFramework

# Initialize the framework
framework = ValuesFramework(model_type="tfidf")  # or "transformer"

# Load and process the ValuesNet dataset
framework.load_valuesnet_data()

# Train the classifiers
framework.train_values_classifier()
framework.train_label_classifier()

# Analyze a conversation
conversation = [
    "I think it's important to help others in need during tough times.",
    "I worked really hard to get the promotion at work.",
    "I value my freedom to make my own choices above all else."
]

results = framework.analyze_conversation(conversation)
print(results['values_summary'])
```

### Command Line Usage

The repository includes a command-line script for analyzing conversations:

```bash
# Analyze a sample conversation to demonstrate the framework
python analyze_values.py --analyze_sample

# Use the transformer model instead of TF-IDF
python analyze_values.py --model_type transformer

# Analyze conversations from a JSON file
python analyze_values.py --conversation_file conversations.json

# Use Gemini API for value summarization
python analyze_values.py --analyze_sample --use_gemini

# Specify Gemini API key directly (alternative to .env file)
python analyze_values.py --analyze_sample --use_gemini --gemini_api_key YOUR_API_KEY
```

### Using the ValuesNet Dataset Directly

The framework can analyze all utterances from the ValuesNet dataset to provide comprehensive insights:

```bash
# Use the default ValuesNet dataset path with TF-IDF model (faster)
python analyze_values.py

# Specify a different dataset directory
python analyze_values.py --data_dir custom_valuesnet_path

# Use the transformer model for potentially higher accuracy
python analyze_values.py --model_type transformer

# Include Gemini API for enhanced summary generation
python analyze_values.py --use_gemini
```

This will:
1. Process all utterances from the train, test, and evaluation sets
2. Identify values and stances across the entire dataset
3. Generate comprehensive statistics and visualizations
4. Save the analysis results to `valuesnet_analysis_results.json`
5. Create visualizations showing value distributions

### Using Gemini API for Value Summarization

The framework can use Google's Gemini API to generate more nuanced, one-sentence summaries of a person's values with a focus on detecting value changes over time.

To use this feature:

1. Set up your Gemini API key:
   ```bash
   python create_env.py --api_key YOUR_GEMINI_API_KEY
   ```

2. Run analysis with the `--use_gemini` flag:
   ```bash
   python analyze_values.py --analyze_sample --use_gemini
   ```

The Gemini API will generate summaries that capture both core values and value changes, such as:
"This person initially prioritized achievement and conformity but has shifted towards valuing self-direction and universalism, reflecting a journey from traditional success metrics to finding deeper meaning and personal freedom."

### Input Format for Conversation Files

The `conversations.json` file should have the following format:

```json
{
  "user1": [
    "utterance 1",
    "utterance 2",
    "..."
  ],
  "user2": [
    "utterance 1",
    "utterance 2",
    "..."
  ]
}
```

## Methodology

### Value Categories

The framework uses Schwartz's 10 basic human values:

1. **Achievement**: Personal success through demonstrating competence
2. **Benevolence**: Preserving and enhancing the welfare of those with whom one is in frequent personal contact
3. **Conformity**: Restraint of actions likely to upset or harm others and violate social norms
4. **Hedonism**: Pleasure and sensuous gratification for oneself
5. **Power**: Social status and prestige, control or dominance over people and resources
6. **Security**: Safety, harmony, and stability of society, relationships, and self
7. **Self-Direction**: Independent thought and action
8. **Stimulation**: Excitement, novelty, and challenge in life
9. **Tradition**: Respect, commitment, and acceptance of customs and ideas
10. **Universalism**: Understanding, appreciation, tolerance, and protection for the welfare of all people and for nature

### Models

The framework offers two types of models:

1. **TF-IDF + Logistic Regression** (faster, simpler): Uses TF-IDF vectorization with a logistic regression classifier
2. **Transformer** (more accurate, slower): Leverages pretrained language models (DistilBERT) fine-tuned on the ValuesNet dataset

### Tracking Value Changes

The framework divides a user's conversation history into segments and analyzes how value expressions change across these segments. This allows for detecting trends such as:

- Values that have become more or less prominent over time
- Values that have emerged or disappeared
- Changes in stance toward particular values

## Example Output

For each conversation, the framework generates:

1. A concise summary of the person's values
2. Detailed analysis of values expressed in each utterance
3. Temporal trends showing how values have changed over time
4. Visualizations of value distributions and changes

### Value Summary Examples

**Built-in summarizer:**
```
The person primarily values benevolence and rejects conformity, and has increasingly emphasized self-direction.
```

**Gemini API summarizer:**
```
This person demonstrates a strong commitment to universalism, valuing environmental protection and collective action for the greater good, while consistently rejecting individualistic values that prioritize personal gain over community welfare.
```

## License

[MIT License](LICENSE)

## Citation

If you use this framework in your research, please cite:

```
@misc{values-detection-framework,
  author = {Your Name},
  title = {Values Detection Framework},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/yourusername/values-detection}
}
```

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- pytorch
- transformers
- nltk
- matplotlib
- seaborn
- requests
- python-dotenv

## Framework Overview

This framework is designed to identify values-related utterances in conversations and track changes in a person's values over time. It leverages the ValuesNet dataset and integrates with Google's Gemini API to generate concise summaries of a person's values.

### Key Features
- **Values Detection**: The framework can detect expressions of values in conversations, identifying which values are being promoted, reduced, or remain neutral.
- **Change Tracking**: It tracks changes in values over time, providing insights into how a person's values evolve.
- **Summarization**: Using the Gemini API, the framework can generate a one-sentence summary of a person's values, highlighting core values and significant changes.

### Usage
- **Analyzing Conversations**: The framework can analyze both sample conversations and user-provided JSON files containing conversation data.
- **Command-Line Interface**: Users can run the analysis via command-line scripts, with options to use the Gemini API for enhanced summarization.

### Designing a Framework for Long Conversations
To design a framework capable of analyzing long and general conversations or a user's posting history, consider the following:
- **Dataset Selection**: Finding a suitable dataset can be challenging. Look for datasets that capture a wide range of conversational topics and include metadata about the speakers.
- **Scalability**: Ensure the framework can handle large volumes of data efficiently, possibly by implementing batch processing or parallelization.
- **Contextual Understanding**: Develop methods to maintain context across long conversations, which may involve natural language processing techniques to track topics and sentiments.

### Challenges
- **Dataset Availability**: Suitable datasets for values analysis may be scarce, requiring custom data collection or augmentation.
- **Complexity of Values**: Values are complex and context-dependent, making it challenging to accurately detect and summarize them.

### Conclusion
This framework provides a robust starting point for analyzing values in conversations. By integrating advanced summarization capabilities, it offers a comprehensive tool for understanding how values are expressed and evolve over time. Good luck in your efforts to expand and refine this framework for broader applications! 