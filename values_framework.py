import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Try to download NLTK resources, handling the case where they might already exist
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    pass


class ValuesFramework:
    """
    A framework for identifying values-related utterances in conversations
    and tracking value changes over time.
    """
    
    # Schwartz's 10 basic values
    VALUE_TYPES = [
        'ACHIEVEMENT', 'BENEVOLENCE', 'CONFORMITY', 'HEDONISM', 
        'POWER', 'SECURITY', 'SELF-DIRECTION', 'STIMULATION', 
        'TRADITION', 'UNIVERSALISM'
    ]
    
    # Value labels
    LABEL_MAP = {
        1: "promotes",
        0: "unrelated",
        -1: "reduces"
    }
    
    def __init__(self, model_type="transformer"):
        """Initialize the values framework with the specified model type."""
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.values_model = None
        self.label_model = None
        self.tfidf = None
        self.value_keywords = self._initialize_value_keywords()
    
    def _initialize_value_keywords(self):
        """Initialize keywords associated with each value type."""
        # These are simplified - in practice, you'd want a more comprehensive list
        return {
            'ACHIEVEMENT': ['success', 'accomplish', 'achieve', 'win', 'ambitious', 'capable', 'competent'],
            'BENEVOLENCE': ['help', 'care', 'forgive', 'honest', 'loyal', 'responsible', 'friendship', 'love'],
            'CONFORMITY': ['obey', 'polite', 'discipline', 'respect', 'obedient', 'comply', 'follow'],
            'HEDONISM': ['pleasure', 'enjoy', 'fun', 'desire', 'gratification', 'indulge', 'delight'],
            'POWER': ['control', 'wealth', 'authority', 'status', 'influence', 'dominate', 'prestige'],
            'SECURITY': ['safety', 'protect', 'stable', 'clean', 'secure', 'health', 'harmony'],
            'SELF-DIRECTION': ['freedom', 'independent', 'choose', 'curious', 'creativity', 'explore'],
            'STIMULATION': ['excitement', 'novelty', 'challenge', 'adventure', 'daring', 'variety'],
            'TRADITION': ['tradition', 'custom', 'religious', 'modest', 'humble', 'devout', 'respect'],
            'UNIVERSALISM': ['equality', 'justice', 'peace', 'tolerance', 'protect', 'environment', 'unity']
        }
    
    def load_valuesnet_data(self, data_dir="valuesNet-dataset"):
        """Load the ValuesNet dataset."""
        # Load main datasets
        self.train_data = pd.read_csv(f"{data_dir}/train.csv")
        self.test_data = pd.read_csv(f"{data_dir}/test.csv")
        self.eval_data = pd.read_csv(f"{data_dir}/eval.csv")
        
        # Extract value type from scenario text in the format "[VALUE] utterance"
        self.train_data['value_type'] = self.train_data['scenario'].apply(
            lambda x: re.match(r'\[(.*?)\]', x).group(1) if re.match(r'\[(.*?)\]', x) else None
        )
        
        # Extract the actual utterance without the value prefix
        self.train_data['utterance'] = self.train_data['scenario'].apply(
            lambda x: re.sub(r'\[(.*?)\]\s*', '', x)
        )
        
        # Do the same for test and eval data
        self.test_data['value_type'] = self.test_data['scenario'].apply(
            lambda x: re.match(r'\[(.*?)\]', x).group(1) if re.match(r'\[(.*?)\]', x) else None
        )
        self.test_data['utterance'] = self.test_data['scenario'].apply(
            lambda x: re.sub(r'\[(.*?)\]\s*', '', x)
        )
        
        self.eval_data['value_type'] = self.eval_data['scenario'].apply(
            lambda x: re.match(r'\[(.*?)\]', x).group(1) if re.match(r'\[(.*?)\]', x) else None
        )
        self.eval_data['utterance'] = self.eval_data['scenario'].apply(
            lambda x: re.sub(r'\[(.*?)\]\s*', '', x)
        )
        
        print(f"Loaded {len(self.train_data)} training samples")
        print(f"Loaded {len(self.test_data)} test samples")
        print(f"Loaded {len(self.eval_data)} evaluation samples")
        
        return self.train_data, self.test_data, self.eval_data
    
    def train_values_classifier(self):
        """Train a model to identify which value is expressed in an utterance."""
        if self.model_type == "transformer":
            return self._train_transformer_values_classifier()
        else:
            return self._train_tfidf_values_classifier()
    
    def _train_tfidf_values_classifier(self):
        """Train a TF-IDF based model to identify which value is expressed."""
        # Create TF-IDF features
        self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train = self.tfidf.fit_transform(self.train_data['utterance'])
        
        # Create one-hot encodings for the value types
        y_train = pd.get_dummies(self.train_data['value_type'])
        
        # Train a multi-output classifier
        base_clf = LogisticRegression(max_iter=1000)
        self.values_model = MultiOutputClassifier(base_clf)
        self.values_model.fit(X_train, y_train)
        
        # Store the column names for later use
        self.value_columns = y_train.columns.tolist()
        
        # Evaluate on test data
        X_test = self.tfidf.transform(self.test_data['utterance'])
        y_test = pd.get_dummies(self.test_data['value_type'])
        
        y_pred = self.values_model.predict(X_test)
        
        # Convert predictions to DataFrame for easier analysis
        y_pred_df = pd.DataFrame(y_pred, columns=y_test.columns)
        
        # Print classification report
        for col in y_test.columns:
            print(f"Value: {col}")
            print(classification_report(y_test[col], y_pred_df[col]))
            
        return self.values_model
    
    def _train_transformer_values_classifier(self):
        """Train a transformer model to identify which value is expressed."""
        # Load pretrained model and tokenizer
        model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=len(self.VALUE_TYPES)
        )
        
        # Prepare dataset
        train_encodings = self.tokenizer(
            self.train_data['utterance'].tolist(), 
            truncation=True, 
            padding=True
        )
        
        # Convert value types to numeric labels
        label_map = {value: i for i, value in enumerate(self.VALUE_TYPES)}
        train_labels = [label_map[val] for val in self.train_data['value_type']]
        
        # Create PyTorch dataset
        train_dataset = ValueDataset(train_encodings, train_labels)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
        )
        
        # Create data collator for dynamic padding
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        trainer.train()
        
        # Save model and tokenizer
        self.model.save_pretrained("./value_classifier_model")
        self.tokenizer.save_pretrained("./value_classifier_tokenizer")
        
        return self.model
    
    def train_label_classifier(self):
        """Train a model to predict if an utterance promotes/reduces/is unrelated to a value."""
        if self.model_type == "transformer":
            return self._train_transformer_label_classifier()
        else:
            return self._train_tfidf_label_classifier()
    
    def _train_tfidf_label_classifier(self):
        """Train a TF-IDF based model to classify value promotion/reduction."""
        # Create TF-IDF features
        if self.tfidf is None:
            self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X_train = self.tfidf.fit_transform(self.train_data['utterance'])
        else:
            X_train = self.tfidf.transform(self.train_data['utterance'])
        
        # Get the labels
        y_train = self.train_data['label']
        
        # Train logistic regression
        self.label_model = LogisticRegression(max_iter=1000)
        self.label_model.fit(X_train, y_train)
        
        # Evaluate on test data
        X_test = self.tfidf.transform(self.test_data['utterance'])
        y_test = self.test_data['label']
        
        y_pred = self.label_model.predict(X_test)
        
        # Print classification report
        print("Label Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["reduces", "unrelated", "promotes"]))
            
        return self.label_model
    
    def _train_transformer_label_classifier(self):
        """Train a transformer model to classify value promotion/reduction."""
        # Load pretrained model and tokenizer
        model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=3  # -1, 0, 1
        )
        
        # Prepare dataset
        train_encodings = self.tokenizer(
            self.train_data['utterance'].tolist(), 
            truncation=True, 
            padding=True
        )
        
        # Adjust labels to be 0, 1, 2 instead of -1, 0, 1
        train_labels = [label + 1 for label in self.train_data['label']]
        
        # Create PyTorch dataset
        train_dataset = ValueDataset(train_encodings, train_labels)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir="./results_label",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs_label",
            logging_steps=10,
        )
        
        # Create data collator for dynamic padding
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.label_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        trainer.train()
        
        # Save model and tokenizer
        self.label_model.save_pretrained("./label_classifier_model")
        self.tokenizer.save_pretrained("./label_classifier_tokenizer")
        
        return self.label_model
    
    def identify_values_in_text(self, text):
        """Identify values expressed in a given text."""
        if self.model_type == "transformer" and self.model is not None:
            # Tokenize the input text
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predictions = torch.softmax(logits, dim=1)
                
            # Get the top values
            top_values_indices = torch.topk(predictions[0], 3).indices.tolist()
            top_values = [self.VALUE_TYPES[idx] for idx in top_values_indices]
            confidence_scores = torch.topk(predictions[0], 3).values.tolist()
            
            return list(zip(top_values, confidence_scores))
        
        elif self.values_model is not None:
            # Transform text using TF-IDF
            X = self.tfidf.transform([text])
            
            # Get predictions
            pred_proba = self.values_model.predict_proba(X)
            
            # Combine value types with their probabilities
            value_scores = []
            for i, estimator in enumerate(self.values_model.estimators_):
                # Only proceed if we have this column
                if i < len(self.value_columns):
                    value = self.value_columns[i]
                    # Get probability of the positive class for this value
                    proba = pred_proba[i][0][1]  # Probability of positive class
                    value_scores.append((value, proba))
            
            # Sort by probability and get top 3
            value_scores.sort(key=lambda x: x[1], reverse=True)
            return value_scores[:3]
        
        else:
            # Fallback method using keyword matching
            value_scores = []
            
            # Tokenize text
            words = word_tokenize(text.lower())
            words = [w for w in words if w not in stopwords.words('english')]
            
            # Count occurrences of value-related keywords
            for value, keywords in self.value_keywords.items():
                score = sum(1 for word in words if word in keywords)
                value_scores.append((value, score))
            
            # Sort by score and get top 3
            value_scores.sort(key=lambda x: x[1], reverse=True)
            value_scores = [(value, score / max(1, len(words))) for value, score in value_scores]
            return value_scores[:3]
    
    def classify_value_stance(self, text, value_type):
        """Classify if the text promotes, reduces, or is unrelated to the given value."""
        if self.model_type == "transformer" and self.label_model is not None:
            # Prepend the value to the text
            input_text = f"[{value_type}] {text}"
            
            # Tokenize the input text
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.label_model(**inputs)
                logits = outputs.logits
                predictions = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(predictions, dim=1).item()
            
            # Convert back from 0,1,2 to -1,0,1
            predicted_class = predicted_class - 1
            
            return predicted_class, self.LABEL_MAP[predicted_class]
        
        elif self.label_model is not None:
            # Transform text using TF-IDF
            input_text = f"[{value_type}] {text}"
            X = self.tfidf.transform([input_text])
            
            # Get prediction
            pred = self.label_model.predict(X)[0]
            
            return pred, self.LABEL_MAP[pred]
        
        else:
            # Fallback method using keyword matching
            words = word_tokenize(text.lower())
            words = [w for w in words if w not in stopwords.words('english')]
            
            # Check if any promoting terms are present
            promoting_terms = ['good', 'great', 'important', 'essential', 'support', 'advocate']
            reducing_terms = ['bad', 'wrong', 'harmful', 'avoid', 'against', 'reject']
            
            promoting_count = sum(1 for word in words if word in promoting_terms)
            reducing_count = sum(1 for word in words if word in reducing_terms)
            
            # Determine stance based on counts
            if promoting_count > reducing_count:
                return 1, self.LABEL_MAP[1]
            elif reducing_count > promoting_count:
                return -1, self.LABEL_MAP[-1]
            else:
                return 0, self.LABEL_MAP[0]
    
    def analyze_conversation(self, conversation_texts, user_id=None):
        """
        Analyze a list of utterances from a conversation to identify values.
        
        Args:
            conversation_texts: List of text utterances from a conversation
            user_id: Optional identifier for the user
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            'user_id': user_id,
            'utterances': [],
            'values_detected': Counter(),
            'value_stances': defaultdict(Counter)
        }
        
        for i, text in enumerate(conversation_texts):
            utterance_analysis = {
                'text': text,
                'index': i,
                'values': []
            }
            
            # Identify values in the utterance
            values = self.identify_values_in_text(text)
            
            for value, confidence in values:
                if confidence > 0.1:  # Only consider values with confidence > 0.1
                    stance, stance_label = self.classify_value_stance(text, value)
                    
                    results['values_detected'][value] += 1
                    results['value_stances'][value][stance] += 1
                    
                    utterance_analysis['values'].append({
                        'value_type': value,
                        'confidence': confidence,
                        'stance': stance,
                        'stance_label': stance_label
                    })
            
            results['utterances'].append(utterance_analysis)
        
        # Calculate the most prominent values
        results['top_values'] = results['values_detected'].most_common(3)
        
        # Analyze value change over time (simple approach)
        results['value_trends'] = self._analyze_value_trends(results['utterances'])
        
        # Generate summary of the person's values
        results['values_summary'] = self.generate_values_summary(results)
        
        return results
    
    def _analyze_value_trends(self, utterances):
        """Analyze how values change over the conversation."""
        # Simple approach: divide conversation into 3 equal parts and compare value distributions
        if len(utterances) < 3:
            return None
        
        segment_size = max(1, len(utterances) // 3)
        segments = [
            utterances[:segment_size],
            utterances[segment_size:2*segment_size],
            utterances[2*segment_size:]
        ]
        
        segment_values = []
        for segment in segments:
            # Count values in this segment
            value_counts = Counter()
            for utterance in segment:
                for value_info in utterance['values']:
                    value_counts[value_info['value_type']] += 1
            
            # Get top values in this segment
            top_segment_values = value_counts.most_common(3)
            segment_values.append(top_segment_values)
        
        # Check for changes in top values across segments
        value_trends = {}
        for i, segment in enumerate(segment_values):
            segment_name = f"segment_{i+1}"
            value_trends[segment_name] = {
                'top_values': segment,
                'changes_from_previous': []
            }
            
            # Compare with previous segment
            if i > 0:
                prev_values = dict(segment_values[i-1])
                current_values = dict(segment)
                
                # Identify values that increased or decreased in prominence
                for value, count in current_values.items():
                    if value in prev_values:
                        if count > prev_values[value]:
                            value_trends[segment_name]['changes_from_previous'].append(
                                f"{value} increased"
                            )
                        elif count < prev_values[value]:
                            value_trends[segment_name]['changes_from_previous'].append(
                                f"{value} decreased"
                            )
                    else:
                        value_trends[segment_name]['changes_from_previous'].append(
                            f"{value} emerged"
                        )
                
                for value in prev_values:
                    if value not in current_values:
                        value_trends[segment_name]['changes_from_previous'].append(
                            f"{value} disappeared"
                        )
        
        return value_trends
    
    def generate_values_summary(self, analysis_results):
        """Generate a summary of a person's values based on analysis results."""
        top_values = analysis_results['top_values']
        value_stances = analysis_results['value_stances']
        
        if not top_values:
            return "No clear values detected in the conversation."
        
        # Get top 2 values and their dominant stance
        value_summary_parts = []
        for value, count in top_values[:2]:
            # Skip values with very low counts
            if count < 2:
                continue
                
            stances = value_stances[value]
            if stances[1] > stances[-1]:  # More promotes than reduces
                value_summary_parts.append(f"values {value.lower()}")
            elif stances[-1] > stances[1]:  # More reduces than promotes
                value_summary_parts.append(f"rejects {value.lower()}")
            else:
                value_summary_parts.append(f"has mixed feelings about {value.lower()}")
        
        # Add value trends if available
        trend_insight = ""
        if 'value_trends' in analysis_results and analysis_results['value_trends']:
            trends = analysis_results['value_trends']
            last_segment = f"segment_{len(trends)}"
            
            if 'changes_from_previous' in trends[last_segment] and trends[last_segment]['changes_from_previous']:
                recent_changes = trends[last_segment]['changes_from_previous']
                if recent_changes:
                    change = recent_changes[0]
                    if "increased" in change:
                        value = change.split()[0].lower()
                        trend_insight = f" and has increasingly emphasized {value}"
                    elif "emerged" in change:
                        value = change.split()[0].lower()
                        trend_insight = f" and has recently started to express {value}"
        
        # Combine parts into a summary sentence
        if value_summary_parts:
            summary = f"The person primarily {' and '.join(value_summary_parts)}{trend_insight}."
            return summary
        else:
            return "No clear pattern of values detected in the conversation."


class ValueDataset(torch.utils.data.Dataset):
    """PyTorch dataset for value classification."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Example usage
if __name__ == "__main__":
    # Initialize the framework
    framework = ValuesFramework(model_type="tfidf")  # or "transformer"
    
    # Load data
    framework.load_valuesnet_data()
    
    # Train models
    framework.train_values_classifier()
    framework.train_label_classifier()
    
    # Example conversation analysis
    conversation = [
        "I think it's important to help others in need, especially during tough times.",
        "I worked really hard to get the promotion at work, I'm proud of my achievements.",
        "I believe rules are meant to be followed, and social norms should be respected.",
        "I'm not sure if I agree with traditional family structures anymore.",
        "I value my freedom to make my own choices above all else.",
        "I used to care more about what others thought, but now I prioritize my own happiness."
    ]
    
    results = framework.analyze_conversation(conversation, user_id="user123")
    
    # Print the values summary
    print("\nValues Summary:")
    print(results['values_summary'])
    
    # Print detected values for each utterance
    print("\nValue Analysis by Utterance:")
    for utterance in results['utterances']:
        print(f"\nText: {utterance['text']}")
        for value_info in utterance['values']:
            print(f"  - {value_info['value_type']} ({value_info['stance_label']}, confidence: {value_info['confidence']:.2f})")
    
    # Print value trends
    print("\nValue Trends:")
    if results['value_trends']:
        for segment, trend in results['value_trends'].items():
            print(f"\n{segment.replace('_', ' ').title()}:")
            print(f"  Top values: {', '.join([f'{v[0]} ({v[1]})' for v in trend['top_values']])}")
            if trend['changes_from_previous']:
                print(f"  Changes: {', '.join(trend['changes_from_previous'])}") 