"""
Study 3: Novelty Gradient

Tests whether multi-turn conversations show steering toward increasing complexity/novelty.

H3: novelty(turn_t) > novelty(turn_{t-1})
    semantic_distance(turn_1, turn_n) increases with n

Design:
- N=30 multi-turn conversations (10 turns each)
- Open-ended, model continues conversation
- Measure topic divergence and complexity trajectory
"""

import logging
from pathlib import Path
from typing import List
import argparse
from datetime import datetime
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from llm_interface import create_interface
from metrics import ResponseMetrics
from config import STUDY_PARAMS, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Study3NoveltyGradient:
    """Study 3: Novelty gradient in multi-turn conversations"""
    
    def __init__(self, pilot: bool = False):
        self.pilot = pilot
        self.params = STUDY_PARAMS["study_3"]
        
        self.n_conversations = self.params["pilot_n"] if pilot else self.params["n_conversations"]
        self.turns_per_conversation = self.params["turns_per_conversation"]
        
        self.metrics_calculator = ResponseMetrics()
        logger.info(f"Initialized Study 3 ({'PILOT' if pilot else 'FULL'})")
    
    def get_initial_prompts(self) -> List[str]:
        """Generate neutral, open-ended initial prompts"""
        return [
            "Tell me about an interesting phenomenon in nature.",
            "What's something surprising about how the world works?",
            "Describe an important concept from any field you find fascinating.",
            "What's a common misconception that people have?",
            "Explain something counterintuitive you've learned.",
        ] * 10  # Repeat to get enough for all conversations
    
    def run_conversation(self, model_name: str, initial_prompt: str, conversation_id: int) -> pd.DataFrame:
        """
        Run one multi-turn conversation
        
        Returns DataFrame with metrics for each turn
        """
        interface = create_interface(model_name)
        
        # Start conversation
        current_prompt = initial_prompt
        conversation_history = []
        metrics_list = []
        
        for turn in range(self.turns_per_conversation):
            response = interface.generate(current_prompt, temperature=1.0, max_tokens=300)
            conversation_history.append(response.response)
            
            # Calculate metrics
            previous = conversation_history[-2] if len(conversation_history) > 1 else None
            metrics = self.metrics_calculator.calculate_all(response.response, previous)
            
            # Calculate distance from first turn
            if turn > 0:
                first_turn = conversation_history[0]
                distance_from_start = self.metrics_calculator.semantic_novelty(
                    response.response, first_turn
                )
                metrics["distance_from_start"] = distance_from_start
            
            metrics.update({
                "model": model_name,
                "conversation_id": conversation_id,
                "turn": turn,
                "prompt": current_prompt,
            })
            metrics_list.append(metrics)
            
            # Generate follow-up prompt (model-steered)
            if turn < self.turns_per_conversation - 1:
                current_prompt = f"Continue discussing related ideas or explore a connected topic."
        
        return pd.DataFrame(metrics_list)
    
    def run_full_study(self, models: List[str] = ["claude"]):
        """Run complete Study 3"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results = []
        
        initial_prompts = self.get_initial_prompts()
        
        for model_name in models:
            logger.info(f"\n{'='*60}\nTesting model: {model_name}\n{'='*60}")
            
            for conv_id in tqdm(range(self.n_conversations), desc=f"{model_name} conversations"):
                initial = initial_prompts[conv_id]
                conv_df = self.run_conversation(model_name, initial, conv_id)
                all_results.append(conv_df)
        
        # Save results
        final_df = pd.concat(all_results, ignore_index=True)
        output_file = PROCESSED_DATA_DIR / f"study3_results_{timestamp}.csv"
        final_df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")
        
        self.print_summary(final_df)
        return final_df
    
    def print_summary(self, df: pd.DataFrame):
        """Print summary of novelty trends"""
        print("\n" + "="*80)
        print("STUDY 3 NOVELTY GRADIENT SUMMARY")
        print("="*80 + "\n")
        
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            print(f"\nModel: {model}")
            print("-" * 40)
            
            # Average semantic novelty by turn
            if 'semantic_novelty' in model_df.columns:
                turn_novelty = model_df.groupby('turn')['semantic_novelty'].mean()
                print("\nAverage semantic novelty by turn:")
                for turn, novelty in turn_novelty.items():
                    print(f"  Turn {turn}: {novelty:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run Study 3: Novelty Gradient")
    parser.add_argument("--pilot", action="store_true", help="Run pilot version")
    parser.add_argument("--models", nargs="+", default=["claude"],
                       help="Models to test")
    args = parser.parse_args()
    
    study = Study3NoveltyGradient(pilot=args.pilot)
    study.run_full_study(models=args.models)


if __name__ == "__main__":
    main()