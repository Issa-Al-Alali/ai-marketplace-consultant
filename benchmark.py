import json
import pandas as pd
import numpy as np

class BenchmarkDataset:
    def __init__(self):
        self.benchmark_cases = []
        self.load_benchmark_data()
    
    def load_benchmark_data(self):
        # Case 1: Excellent Performer
        self.benchmark_cases.append({
            'case_id': 'vendor_001',
            'difficulty': 'Easy',
            'description': 'High-revenue, excellent reviews',
            'metrics': {
                'seller_id': 'benchmark_001',
                'total_revenue': 85000.50,
                'avg_order_value': 425.25,
                'order_count': 200,
                'avg_review_score': 4.8,
                'review_count': 180,
                'avg_delivery_delay': -1.5,
                'on_time_delivery_rate': 95.0,
                'unique_products': 25,
                'category_diversity': 3,
                'top_category': 'electronics',
                'sales_velocity': 1.2,
                'customer_retention': 1.4,
                'days_active': 365
            },
            'ground_truth': {
                'tier': 'A',
                'overall_score': 92,
                'key_strengths': ['Exceptional satisfaction', 'Strong revenue', 'Reliable delivery'],
                'key_weaknesses': ['Could diversify catalog'],
                'key_recommendations': ['Expand marketing', 'Premium product line'],
                'risk_level': 'Low'
            }
        })
        
        # Case 2: Struggling Vendor
        self.benchmark_cases.append({
            'case_id': 'vendor_002',
            'difficulty': 'Easy',
            'description': 'Low revenue, poor reviews',
            'metrics': {
                'seller_id': 'benchmark_002',
                'total_revenue': 3500.75,
                'avg_order_value': 35.00,
                'order_count': 100,
                'avg_review_score': 2.8,
                'review_count': 95,
                'avg_delivery_delay': 8.5,
                'on_time_delivery_rate': 45.0,
                'unique_products': 5,
                'category_diversity': 1,
                'top_category': 'home_decor',
                'sales_velocity': 0.15,
                'customer_retention': 1.05,
                'days_active': 450
            },
            'ground_truth': {
                'tier': 'C',
                'overall_score': 35,
                'key_strengths': ['Consistent presence'],
                'key_weaknesses': ['Severe delays', 'Poor satisfaction', 'Limited variety'],
                'key_recommendations': ['URGENT: Fix delivery', 'Improve quality', 'Expand catalog'],
                'risk_level': 'High'
            }
        })
        
        # Case 3: Mid-Tier
        self.benchmark_cases.append({
            'case_id': 'vendor_003',
            'difficulty': 'Medium',
            'description': 'Solid performance',
            'metrics': {
                'seller_id': 'benchmark_003',
                'total_revenue': 28000.00,
                'avg_order_value': 175.00,
                'order_count': 160,
                'avg_review_score': 4.1,
                'review_count': 140,
                'avg_delivery_delay': 2.0,
                'on_time_delivery_rate': 75.0,
                'unique_products': 40,
                'category_diversity': 5,
                'top_category': 'fashion',
                'sales_velocity': 0.6,
                'customer_retention': 1.2,
                'days_active': 280
            },
            'ground_truth': {
                'tier': 'B',
                'overall_score': 68,
                'key_strengths': ['Good diversity', 'Solid reviews'],
                'key_weaknesses': ['Minor delays', 'Inconsistent'],
                'key_recommendations': ['Optimize logistics', 'Focus on best sellers'],
                'risk_level': 'Medium'
            }
        })
        
        # Case 4: Niche Specialist
        self.benchmark_cases.append({
            'case_id': 'vendor_004',
            'difficulty': 'Hard',
            'description': 'Low volume, high value',
            'metrics': {
                'seller_id': 'benchmark_004',
                'total_revenue': 45000.00,
                'avg_order_value': 1500.00,
                'order_count': 30,
                'avg_review_score': 4.9,
                'review_count': 28,
                'avg_delivery_delay': -0.5,
                'on_time_delivery_rate': 100.0,
                'unique_products': 8,
                'category_diversity': 1,
                'top_category': 'furniture',
                'sales_velocity': 0.25,
                'customer_retention': 1.1,
                'days_active': 120
            },
            'ground_truth': {
                'tier': 'B',
                'overall_score': 75,
                'key_strengths': ['Exceptional quality', 'Perfect delivery', 'High AOV'],
                'key_weaknesses': ['Very low volume', 'Limited range', 'Scalability concerns'],
                'key_recommendations': ['Maintain quality while growing volume', 'Build brand awareness'],
                'risk_level': 'Medium'
            }
        })
        
        # Case 5: Declining
        self.benchmark_cases.append({
            'case_id': 'vendor_005',
            'difficulty': 'Very Hard',
            'description': 'Declining metrics',
            'metrics': {
                'seller_id': 'benchmark_005',
                'total_revenue': 18000.00,
                'avg_order_value': 120.00,
                'order_count': 150,
                'avg_review_score': 3.5,
                'review_count': 145,
                'avg_delivery_delay': 4.5,
                'on_time_delivery_rate': 60.0,
                'unique_products': 55,
                'category_diversity': 8,
                'top_category': 'toys',
                'sales_velocity': 0.4,
                'customer_retention': 1.08,
                'days_active': 520
            },
            'ground_truth': {
                'tier': 'C',
                'overall_score': 48,
                'key_strengths': ['Diverse catalog', 'Long presence'],
                'key_weaknesses': ['Declining satisfaction', 'Deteriorating delivery', 'Over-diversified'],
                'key_recommendations': ['CRITICAL: Analyze decline', 'Narrow focus', 'Audit supply chain'],
                'risk_level': 'High'
            }
        })
    
    def get_case(self, case_id):
        for case in self.benchmark_cases:
            if case['case_id'] == case_id:
                return case
        return None
    
    def get_all_cases(self):
        return self.benchmark_cases
    
    def get_cases_by_difficulty(self, difficulty):
        return [c for c in self.benchmark_cases if c['difficulty'] == difficulty]


class BenchmarkEvaluator:
    def __init__(self):
        self.results = []
    
    def evaluate_prediction(self, predicted, ground_truth):
        metrics = {
            'tier_correct': predicted.get('tier') == ground_truth.get('tier'),
            'score_error': abs(predicted.get('overall_score', predicted.get('score', 50)) - ground_truth['overall_score']),
            'score_within_10': abs(predicted.get('overall_score', predicted.get('score', 50)) - ground_truth['overall_score']) <= 10,
            'score_within_5': abs(predicted.get('overall_score', predicted.get('score', 50)) - ground_truth['overall_score']) <= 5
        }
        
        pred_recs = predicted.get('recommendations', [])
        true_recs = ground_truth['key_recommendations']
        
        if isinstance(pred_recs, list):
            pred_actions = set()
            for rec in pred_recs:
                if isinstance(rec, dict):
                    pred_actions.add(rec.get('action', '').lower())
                else:
                    pred_actions.add(str(rec).lower())
            
            true_keywords = set()
            for rec in true_recs:
                true_keywords.update(rec.lower().split())
            
            matches = sum(1 for action in pred_actions if any(kw in action for kw in true_keywords))
            metrics['recommendation_overlap_score'] = matches / len(true_recs) if true_recs else 0
        
        return metrics
    
    def run_benchmark(self, agent, method='chain_of_thought'):
        dataset = BenchmarkDataset()
        results = []
        
        for case in dataset.get_all_cases():
            print(f"Evaluating {case['case_id']} ({case['difficulty']})...")
            
            if method == 'chain_of_thought':
                prediction = agent.run_chain_of_thought(case['metrics'])
            elif method == 'few_shot':
                prediction = agent.run_few_shot_cot(case['metrics'])
            elif method == 'evaluator_optimizer':
                result = agent.run_evaluator_optimizer(case['metrics'])
                prediction = result.get('final', result)
            elif method == 'voting':
                result = agent.run_voting(case['metrics'])
                prediction = result.get('consensus', result)
            elif method == 'self_consistency':
                prediction = agent.run_self_consistency(case['metrics'])
            
            eval_metrics = self.evaluate_prediction(prediction, case['ground_truth'])
            
            results.append({
                'case_id': case['case_id'],
                'difficulty': case['difficulty'],
                'prediction': prediction,
                'ground_truth': case['ground_truth'],
                'metrics': eval_metrics
            })
        
        return self.summarize_results(results, method)
    
    def summarize_results(self, results, method_name):
        df = pd.DataFrame([r['metrics'] for r in results])
        
        summary = {
            'method': method_name,
            'total_cases': len(results),
            'tier_accuracy': df['tier_correct'].mean() * 100,
            'avg_score_error': df['score_error'].mean(),
            'score_within_10_pct': df['score_within_10'].mean() * 100,
            'score_within_5_pct': df['score_within_5'].mean() * 100,
            'avg_recommendation_quality': df['recommendation_overlap_score'].mean() * 100,
            'detailed_results': results
        }
        
        return summary
    
    def compare_methods(self, agent, methods=['chain_of_thought', 'few_shot', 'evaluator_optimizer']):
        comparisons = []
        
        for method in methods:
            print(f"\n{'='*60}")
            print(f"Benchmarking Method: {method}")
            print(f"{'='*60}")
            summary = self.run_benchmark(agent, method=method)
            comparisons.append(summary)
        
        return self.create_comparison_table(comparisons)
    
    def create_comparison_table(self, comparisons):
        data = []
        for comp in comparisons:
            data.append({
                'Method': comp['method'],
                'Tier Accuracy (%)': round(comp['tier_accuracy'], 2),
                'Avg Score Error': round(comp['avg_score_error'], 2),
                'Score ±10 (%)': round(comp['score_within_10_pct'], 2),
                'Score ±5 (%)': round(comp['score_within_5_pct'], 2),
                'Rec Quality (%)': round(comp['avg_recommendation_quality'], 2)
            })
        
        return pd.DataFrame(data)