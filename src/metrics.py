"""
Metrics for measuring response diversity, novelty, and complexity
All metrics based on peer-reviewed literature
"""

import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from typing import List, Dict, Optional
import logging
from sentence_transformers import SentenceTransformer

from config import METRICS_CONFIG

logger = logging.getLogger(__name__)

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)


class ResponseMetrics:
    """Calculate various metrics on LLM responses"""
    
    def __init__(self):
        """Initialize metrics calculator"""
        # Load sentence embedding model for semantic novelty
        model_name = METRICS_CONFIG["semantic_novelty"]["model"]
        self.embedding_model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")
    
    def calculate_all(self, response: str, previous_response: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate all metrics for a response
        
        Args:
            response: Current response text
            previous_response: Previous response for novelty calculation
        
        Returns:
            Dictionary of metric names and values
        """
        metrics = {
            "response_length": len(response),
            "token_count": self.token_count(response),
            "entropy": self.shannon_entropy(response),
            "ttr": self.type_token_ratio(response),
            "mtld": self.mtld(response),
        }
        
        if previous_response:
            metrics["semantic_novelty"] = self.semantic_novelty(response, previous_response)
        
        return metrics
    
    @staticmethod
    def token_count(text: str) -> int:
        """Count tokens in text"""
        return len(word_tokenize(text.lower()))
    
    def shannon_entropy(self, text: str, normalize: bool = True) -> float:
        """
        Calculate Shannon entropy of response
        
        Based on: Shannon, C. E. (1948). A mathematical theory of communication.
        
        H(X) = -Σ p(x_i) log₂ p(x_i)
        
        Args:
            text: Input text
            normalize: If True, normalize by max possible entropy
        
        Returns:
            Entropy value (bits)
        """
        tokens = word_tokenize(text.lower())
        
        if len(tokens) == 0:
            return 0.0
        
        # Calculate token probabilities
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        probabilities = np.array([count / total_tokens for count in token_counts.values()])
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # Normalize by maximum possible entropy (uniform distribution)
        if normalize:
            max_entropy = np.log2(len(token_counts))
            entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return float(entropy)
    
    def type_token_ratio(self, text: str) -> float:
        """
        Calculate Type-Token Ratio (TTR)
        
        TTR = |unique tokens| / |total tokens|
        
        Simple lexical diversity measure, sensitive to text length
        
        Args:
            text: Input text
        
        Returns:
            TTR value between 0 and 1
        """
        tokens = word_tokenize(text.lower())
        
        if len(tokens) == 0:
            return 0.0
        
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        
        return unique_tokens / total_tokens
    
    def mtld(self, text: str, threshold: float = 0.72) -> float:
        """
        Calculate Measure of Textual Lexical Diversity (MTLD)
        
        Based on: McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: 
        A validation study of sophisticated approaches to lexical diversity assessment.
        
        More robust to text length than TTR
        
        Args:
            text: Input text
            threshold: TTR threshold for factor counting
        
        Returns:
            MTLD value
        """
        tokens = word_tokenize(text.lower())
        
        if len(tokens) < METRICS_CONFIG["lexical_diversity"]["min_length"]:
            logger.debug(f"Text too short for reliable MTLD ({len(tokens)} tokens)")
            return self.type_token_ratio(text)  # Fall back to TTR
        
        def mtld_forward(tokens, threshold):
            """Calculate MTLD in forward direction"""
            factor = 0
            types = set()
            token_count = 0
            
            for token in tokens:
                types.add(token)
                token_count += 1
                
                if token_count > 0:
                    ttr = len(types) / token_count
                    if ttr <= threshold:
                        factor += 1
                        types = set()
                        token_count = 0
            
            # Partial factor
            if token_count > 0:
                ttr = len(types) / token_count
                factor += (1 - ttr) / (1 - threshold)
            
            return len(tokens) / factor if factor > 0 else len(tokens)
        
        # Calculate bidirectionally and average
        forward_mtld = mtld_forward(tokens, threshold)
        backward_mtld = mtld_forward(tokens[::-1], threshold)
        
        return (forward_mtld + backward_mtld) / 2
    
    def semantic_novelty(self, current_text: str, previous_text: str) -> float:
        """
        Calculate semantic novelty as cosine distance between embeddings
        
        novelty = 1 - cos(embed(current), embed(previous))
        
        Higher values indicate greater semantic departure
        
        Args:
            current_text: Current response
            previous_text: Previous response
        
        Returns:
            Novelty score between 0 and 2 (typically 0-1)
        """
        try:
            # Generate embeddings
            current_emb = self.embedding_model.encode(current_text, convert_to_numpy=True)
            previous_emb = self.embedding_model.encode(previous_text, convert_to_numpy=True)
            
            # Calculate cosine similarity
            cos_sim = np.dot(current_emb, previous_emb) / (
                np.linalg.norm(current_emb) * np.linalg.norm(previous_emb)
            )
            
            # Convert to distance (novelty)
            novelty = 1 - cos_sim
            
            return float(novelty)
            
        except Exception as e:
            logger.error(f"Error calculating semantic novelty: {e}")
            return 0.0
    
    def calculate_batch(
        self, 
        responses: List[str],
        calculate_novelty: bool = True
    ) -> List[Dict[str, float]]:
        """
        Calculate metrics for a batch of responses
        
        Args:
            responses: List of response texts
            calculate_novelty: Whether to calculate novelty between consecutive responses
        
        Returns:
            List of metric dictionaries
        """
        metrics_list = []
        
        for i, response in enumerate(responses):
            previous = responses[i-1] if i > 0 and calculate_novelty else None
            metrics = self.calculate_all(response, previous)
            metrics_list.append(metrics)
        
        return metrics_list


def calculate_aggregate_statistics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate aggregate statistics across multiple responses
    
    Args:
        metrics_list: List of metric dictionaries
    
    Returns:
        Dictionary of statistics for each metric
    """
    if not metrics_list:
        return {}
    
    # Get all metric names
    metric_names = metrics_list[0].keys()
    
    statistics = {}
    for metric_name in metric_names:
        values = [m[metric_name] for m in metrics_list if metric_name in m]
        
        if values:
            statistics[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "median": np.median(values),
                "min": np.min(values),
                "max": np.max(values),
                "n": len(values)
            }
    
    return statistics


def compare_conditions(
    condition_a_metrics: List[Dict[str, float]],
    condition_b_metrics: List[Dict[str, float]],
    metric_name: str = "entropy"
) -> Dict[str, float]:
    """
    Compare a metric between two conditions
    
    Args:
        condition_a_metrics: Metrics from condition A
        condition_b_metrics: Metrics from condition B
        metric_name: Which metric to compare
    
    Returns:
        Dictionary with comparison statistics
    """
    from scipy import stats
    
    values_a = [m[metric_name] for m in condition_a_metrics if metric_name in m]
    values_b = [m[metric_name] for m in condition_b_metrics if metric_name in m]
    
    # Independent samples t-test
    t_stat, p_value = stats.ttest_ind(values_a, values_b)
    
    # Effect size (Cohen's d)
    mean_a, mean_b = np.mean(values_a), np.mean(values_b)
    std_pooled = np.sqrt((np.var(values_a) + np.var(values_b)) / 2)
    cohens_d = (mean_a - mean_b) / std_pooled if std_pooled > 0 else 0
    
    return {
        "metric": metric_name,
        "condition_a_mean": mean_a,
        "condition_b_mean": mean_b,
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "significant": p_value < 0.05,
        "n_a": len(values_a),
        "n_b": len(values_b),
    }


if __name__ == "__main__":
    # Test metrics
    logging.basicConfig(level=logging.INFO)
    
    test_response_1 = """
    Habituation is a form of learning where an organism decreases its response
    to a repeated stimulus over time. This adaptive mechanism allows organisms
    to ignore irrelevant stimuli and focus on novel or important information.
    """
    
    test_response_2 = """
    Classical conditioning involves learning associations between stimuli.
    Unlike habituation, classical conditioning creates new behavioral responses
    through repeated pairings of neutral and meaningful stimuli.
    """
    
    print("\n🧪 Testing Metrics Module\n")
    
    metrics = ResponseMetrics()
    
    # Test on single response
    result_1 = metrics.calculate_all(test_response_1)
    print("Response 1 metrics:")
    for metric, value in result_1.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test novelty between responses
    result_2 = metrics.calculate_all(test_response_2, test_response_1)
    print(f"\nResponse 2 metrics (with novelty):")
    for metric, value in result_2.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n✅ Metrics module test complete!")