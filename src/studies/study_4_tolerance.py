"""
Study 4: Tolerance Patterns

Tests whether habituation accelerates with repeated exposure sessions (tolerance analog).

H4: d(H)/d(trial) | session_k > d(H)/d(trial) | session_{k-1}

Design:
- N=20 sessions over time
- Each session = 50 trials (repetitive prompts)
- Measure rate of entropy decline per session
- Test if decline steepens (tolerance)
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


class Study4Tolerance:
    """Study 4: Tolerance/sensitization patterns"""
    
    def __init__(self, pilot: bool = False):
        self.pilot = pilot
        self.params = STUDY_PARAMS["study_4"]
        
        if pilot:
            self.n_sessions = self.params["pilot_n"]
            self.trials_per_session = 10  # Reduced for pilot
        else:
            self.n_sessions = self.params["n_sessions"]
            self.trials_per_session = self.params["trials_per_session"]
        
        self.metrics_calculator = ResponseMetrics()
        logger.info(f"Initialized Study 4 ({'PILOT' if pilot else 'FULL'})")
    
    def generate_session_prompts(self, session_id: int) -> List[str]:
        """Generate prompts for one session (same structure, varied content)"""
        concepts = [
            "thermodynamics", "game theory", "cellular respiration", "tectonic plates",
            "market dynamics", "neural plasticity", "quantum entanglement", "RNA transcription",
            "geological time", "opportunity cost"
        ] * 10
        
        prompts = [
            f"Explain the concept of {concept} to a beginner."
            for concept in concepts[:self.trials_per_session]
        ]
        
        return prompts
    
    def run_session(self, model_name: str, session_id: int) -> pd.DataFrame:
        """
        Run one session (multiple trials with same structure)
        
        Returns DataFrame with metrics for each trial
        """
        interface = create_interface(model_name)
        prompts = self.generate_session_prompts(session_id)
        
        metrics_list = []
        responses = []
        
        for trial_id, prompt in enumerate(tqdm(prompts, desc=f"Session {session_id}", leave=False)):
            response = interface.generate(prompt, temperature=1.0, max_tokens=300)
            responses.append(response)
            
            previous = responses[-2].response if len(responses) > 1 else None
            metrics = self.metrics_calculator.calculate_all(response.response, previous)
            
            metrics.update({
                "model": model_name,
                "session_id": session_id,
                "trial_id": trial_id,
                "prompt": prompt,
            })
            metrics_list.append(metrics)
        
        return pd.DataFrame(metrics_list)
    
    def run_full_study(self, models: List[str] = ["claude"]):
        """Run complete Study 4 across multiple sessions"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results = []
        
        for model_name in models:
            logger.info(f"\n{'='*60}\nTesting model: {model_name}\n{'='*60}")
            
            for session_id in tqdm(range(self.n_sessions), desc=f"{model_name} sessions"):
                session_df = self.run_session(model_name, session_id)
                all_results.append(session_df)
        
        # Save results
        final_df = pd.concat(all_results, ignore_index=True)
        output_file = PROCESSED_DATA_DIR / f"study4_results_{timestamp}.csv"
        final_df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")
        
        self.print_summary(final_df)
        return final_df
    
    def print_summary(self, df: pd.DataFrame):
        """Print summary showing tolerance patterns"""
        print("\n" + "="*80)
        print("STUDY 4 TOLERANCE PATTERNS SUMMARY")
        print("="*80 + "\n")
        
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            print(f"\nModel: {model}")
            print("-" * 40)
            
            # Average entropy by session
            session_entropy = model_df.groupby('session_id')['entropy'].mean()
            print("\nAverage entropy by session:")
            for session, entropy in list(session_entropy.items())[:5]:  # Show first 5
                print(f"  Session {session}: {entropy:.4f}")
            
            if len(session_entropy) > 5:
                print(f"  ... ({len(session_entropy) - 5} more sessions)")


def main():
    parser = argparse.ArgumentParser(description="Run Study 4: Tolerance Patterns")
    parser.add_argument("--pilot", action="store_true", help="Run pilot version")
    parser.add_argument("--models", nargs="+", default=["claude"],
                       help="Models to test")
    args = parser.parse_args()
    
    study = Study4Tolerance(pilot=args.pilot)
    study.run_full_study(models=args.models)


if __name__ == "__main__":
    main()