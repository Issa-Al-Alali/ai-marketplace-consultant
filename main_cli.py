"""
Command Line Interface for Batch Processing
Use this for large-scale vendor analysis without browser limitations
"""

import argparse
import json
import sys
import os
from tqdm import tqdm
import pandas as pd
from data_engine import load_and_preprocess_data, get_vendor_summary
from agent import MarketplaceAgent
from benchmark import BenchmarkEvaluator
from config import GOOGLE_API_KEY

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def main():
    parser = argparse.ArgumentParser(description='Marketplace Vendor Analysis CLI')
    parser.add_argument('--mode', choices=['batch', 'benchmark', 'single'], required=True,
                        help='Analysis mode')
    parser.add_argument('--technique', choices=['cot', 'few_shot', 'evaluator', 'voting', 'all'],
                        default='cot', help='Agentic technique to use')
    parser.add_argument('--vendor-id', type=str, help='Specific vendor ID for single analysis')
    parser.add_argument('--batch-size', type=int, default=100, help='Number of vendors to process')
    parser.add_argument('--output', type=str, default='results.json', help='Output file path')
    
    args = parser.parse_args()
    
    # Initialize
    print("[*] Initializing Marketplace AI Agent...")
    agent = MarketplaceAgent(GOOGLE_API_KEY)
    
    if args.mode == 'single':
        print(f"[*] Analyzing vendor: {args.vendor_id}")
        df = load_and_preprocess_data()
        vendor_data = get_vendor_summary(df, args.vendor_id)
        
        result = run_analysis(agent, vendor_data, args.technique)
        
        print("\n" + "="*60)
        print(json.dumps(result, indent=2))
        print("="*60)
        
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n[OK] Results saved to {args.output}")
    
    elif args.mode == 'batch':
        print(f"[*] Batch processing {args.batch_size} vendors...")
        df = load_and_preprocess_data()
        
        vendors = df.head(args.batch_size)
        results = []
        
        for _, vendor in tqdm(vendors.iterrows(), total=len(vendors), desc="Processing"):
            vendor_data = get_vendor_summary(df, vendor['seller_id'])
            result = run_analysis(agent, vendor_data, args.technique)
            
            results.append({
                'seller_id': vendor['seller_id'],
                'analysis': result,
                'actual_tier': vendor['tier'],
                'actual_score': float(vendor['performance_score'])
            })
        
        # Calculate accuracy
        tier_matches = sum(1 for r in results 
                          if r['analysis'].get('tier') == r['actual_tier'])
        tier_accuracy = (tier_matches / len(results)) * 100
        
        score_errors = [abs(r['analysis'].get('overall_score', 
                                              r['analysis'].get('score', 0)) - 
                           r['actual_score']) for r in results]
        avg_score_error = sum(score_errors) / len(score_errors)
        
        summary = {
            'total_processed': len(results),
            'technique': args.technique,
            'tier_accuracy': tier_accuracy,
            'avg_score_error': avg_score_error,
            'results': results
        }
        
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[OK] Processed {len(results)} vendors")
        print(f"📊 Tier Accuracy: {tier_accuracy:.2f}%")
        print(f"📊 Avg Score Error: {avg_score_error:.2f}")
        print(f"💾 Results saved to {args.output}")
    
    elif args.mode == 'benchmark':
        print("[*] Running benchmark evaluation...")
        evaluator = BenchmarkEvaluator()
        
        if args.technique == 'all':
            methods = ['chain_of_thought', 'few_shot', 'evaluator_optimizer']
        else:
            methods = [map_technique(args.technique)]
        
        comparison_df = evaluator.compare_methods(agent, methods=methods)
        
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print(comparison_df.to_string())
        print("="*60)
        
        comparison_df.to_csv(args.output.replace('.json', '.csv'), index=False)
        print(f"\n[OK] Results saved to {args.output.replace('.json', '.csv')}")
    
    print(f"\n[*] Total LLM calls: {agent.get_stats()['total_llm_calls']}")

def run_analysis(agent, vendor_data, technique):
    """Run analysis with specified technique"""
    if technique == 'cot':
        return agent.run_chain_of_thought(vendor_data)
    elif technique == 'few_shot':
        return agent.run_few_shot_cot(vendor_data)
    elif technique == 'evaluator':
        result = agent.run_evaluator_optimizer(vendor_data)
        return result.get('final', result)
    elif technique == 'voting':
        result = agent.run_voting(vendor_data)
        return result.get('consensus', result)
    else:
        return agent.run_chain_of_thought(vendor_data)

def map_technique(short_name):
    """Map CLI technique names to internal names"""
    mapping = {
        'cot': 'chain_of_thought',
        'few_shot': 'few_shot',
        'evaluator': 'evaluator_optimizer',
        'voting': 'voting'
    }
    return mapping.get(short_name, 'chain_of_thought')

if __name__ == '__main__':
    main()