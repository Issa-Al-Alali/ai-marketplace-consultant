import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_engine import load_and_preprocess_data, get_vendor_summary
from agent import MarketplaceAgent
from benchmark import BenchmarkDataset, BenchmarkEvaluator
from config import GROQ_API_KEY
import json

st.set_page_config(
    page_title="AI Marketplace Consultant Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize
@st.cache_data
def get_data():
    return load_and_preprocess_data()

@st.cache_resource
def get_agent():
    return MarketplaceAgent(GROQ_API_KEY)

# Load data
try:
    df = get_data()
    agent = get_agent()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Header
st.title("🚀 AI-Powered Marketplace Vendor Analysis")
st.markdown("**Advanced Agentic System for Olist Brazilian E-commerce Platform**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Analysis Mode
    analysis_mode = st.radio(
        "Analysis Mode",
        ["Single Vendor Analysis", "Benchmark Evaluation", "Batch Processing"]
    )
    
    st.divider()
    
    if analysis_mode == "Single Vendor Analysis":
        st.subheader("Vendor Selection")
        
        tier_filter = st.multiselect(
            "Filter by Tier",
            options=['A', 'B', 'C'],
            default=['A', 'B', 'C']
        )
        
        filtered_df = df[df['tier'].isin(tier_filter)]
        
        vendor_id = st.selectbox(
            "Select Vendor (Seller ID)",
            options=filtered_df['seller_id'].head(200).tolist(),
            format_func=lambda x: f"{x} (Tier {filtered_df[filtered_df['seller_id']==x]['tier'].iloc[0]})"
        )
        
        st.divider()
        
        st.subheader("Agentic Technique")
        technique = st.selectbox(
            "Choose Analysis Method",
            [
                "Chain-of-Thought",
                "Few-Shot CoT",
                "Evaluator-Optimizer",
                "Voting (Ensemble)",
                "Self-Consistency",
                "Compare All"
            ]
        )
    
    elif analysis_mode == "Benchmark Evaluation":
        st.subheader("Benchmark Settings")
        eval_techniques = st.multiselect(
            "Techniques to Evaluate",
            ["Chain-of-Thought", "Few-Shot CoT", "Evaluator-Optimizer"],
            default=["Chain-of-Thought", "Few-Shot CoT"]
        )
    
    else:
        st.subheader("Batch Settings")
        batch_size = st.slider("Number of Vendors", 5, 50, 10)
        batch_technique = st.selectbox(
            "Technique",
            ["Chain-of-Thought", "Few-Shot CoT", "Evaluator-Optimizer"]
        )

# Main Content
if analysis_mode == "Single Vendor Analysis":
    
    vendor_metrics = df[df['seller_id'] == vendor_id].iloc[0].to_dict()
    vendor_summary = get_vendor_summary(df, vendor_id)
    
    st.header(f"📊 Vendor Profile: {vendor_id}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Performance Tier", vendor_summary['tier'])
    col2.metric("Overall Score", f"{vendor_summary['performance_score']:.0f}/100")
    col3.metric("Total Revenue", f"R$ {vendor_summary['financial']['total_revenue']:,.2f}")
    col4.metric("Orders", vendor_summary['operations']['order_count'])
    
    with st.expander("📈 Detailed Metrics", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Financial**")
            st.write(f"• Avg Order: R$ {vendor_summary['financial']['avg_order_value']:.2f}")
            st.write(f"• Rev/Product: R$ {vendor_summary['financial']['revenue_per_product']:.2f}")
        
        with col2:
            st.markdown("**Performance**")
            st.write(f"• Review: {vendor_summary['performance']['avg_review_score']:.1f}/5.0 ⭐")
            st.write(f"• On-Time: {vendor_summary['performance']['on_time_delivery_rate']:.1f}%")
        
        with col3:
            st.markdown("**Catalog**")
            st.write(f"• Products: {vendor_summary['catalog']['unique_products']}")
            st.write(f"• Categories: {vendor_summary['catalog']['category_diversity']}")
    
    st.divider()
    st.header("🤖 AI Analysis")
    
    if technique != "Compare All":
        if st.button("🔍 Generate Analysis", type="primary", use_container_width=True):
            with st.spinner(f"Running {technique} analysis..."):
                
                if technique == "Chain-of-Thought":
                    result = agent.run_chain_of_thought(vendor_summary)
                elif technique == "Few-Shot CoT":
                    result = agent.run_few_shot_cot(vendor_summary)
                elif technique == "Evaluator-Optimizer":
                    result = agent.run_evaluator_optimizer(vendor_summary)
                    result = result.get('final', result)
                elif technique == "Voting (Ensemble)":
                    result = agent.run_voting(vendor_summary, num_voters=3)
                    result = result.get('consensus', result)
                elif technique == "Self-Consistency":
                    result = agent.run_self_consistency(vendor_summary)
                
                st.success("Analysis Complete!")
                
                if isinstance(result, dict) and 'error' not in result:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("📋 Assessment")
                        st.metric("Assigned Tier", result.get('tier', 'N/A'))
                        st.metric("Score", f"{result.get('overall_score', result.get('score', 0))}/100")
                        
                        if 'strengths' in result:
                            st.markdown("**💪 Strengths:**")
                            for s in result['strengths']:
                                st.write(f"• {s}")
                        
                        if 'weaknesses' in result:
                            st.markdown("**⚠️ Weaknesses:**")
                            for w in result['weaknesses']:
                                st.write(f"• {w}")
                    
                    with col2:
                        if 'growth_score' in result:
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=result['growth_score'],
                                title={'text': "Growth Potential"},
                                gauge={'axis': {'range': [0, 100]},
                                       'bar': {'color': "green"}}
                            ))
                            st.plotly_chart(fig, use_container_width=True)
                    
                    if 'recommendations' in result:
                        st.subheader("💡 Strategic Recommendations")
                        for rec in result['recommendations']:
                            if isinstance(rec, dict):
                                priority = rec.get('priority', 'medium')
                                icon = "🔴" if priority == "high" else "🟡" if priority == "medium" else "🟢"
                                st.markdown(f"{icon} **{rec.get('action')}**")
                                if 'expected_impact' in rec:
                                    st.caption(rec['expected_impact'])
                            else:
                                st.write(f"• {rec}")
                    
                    with st.expander("🔍 View Raw JSON"):
                        st.json(result)
                else:
                    st.error("Analysis failed")
                    st.json(result)
    
    else:  # Compare All
        if st.button("🔍 Run All Techniques", type="primary", use_container_width=True):
            techniques = ["Chain-of-Thought", "Few-Shot CoT", "Evaluator-Optimizer"]
            results = {}
            
            for tech in techniques:
                with st.spinner(f"Running {tech}..."):
                    if tech == "Chain-of-Thought":
                        results[tech] = agent.run_chain_of_thought(vendor_summary)
                    elif tech == "Few-Shot CoT":
                        results[tech] = agent.run_few_shot_cot(vendor_summary)
                    elif tech == "Evaluator-Optimizer":
                        full_result = agent.run_evaluator_optimizer(vendor_summary)
                        results[tech] = full_result.get('final', full_result)
            
            st.success("All analyses complete!")
            
            comparison_data = []
            for tech, result in results.items():
                if isinstance(result, dict):
                    comparison_data.append({
                        'Technique': tech,
                        'Tier': result.get('tier', 'N/A'),
                        'Score': result.get('overall_score', result.get('score', 0)),
                        'Recommendations': len(result.get('recommendations', []))
                    })
            
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
            
            tabs = st.tabs(techniques)
            for i, tech in enumerate(techniques):
                with tabs[i]:
                    st.json(results[tech])

elif analysis_mode == "Benchmark Evaluation":
    st.header("🎯 Benchmark Evaluation")
    st.markdown("Evaluate agent performance against expert-labeled test cases")
    
    benchmark = BenchmarkDataset()
    cases_df = pd.DataFrame([
        {
            'Case ID': c['case_id'],
            'Difficulty': c['difficulty'],
            'Description': c['description'],
            'True Tier': c['ground_truth']['tier'],
            'True Score': c['ground_truth']['overall_score']
        }
        for c in benchmark.get_all_cases()
    ])
    
    st.subheader("📚 Benchmark Dataset")
    st.dataframe(cases_df, use_container_width=True)
    
    if st.button("🚀 Run Benchmark", type="primary"):
        evaluator = BenchmarkEvaluator()
        
        technique_map = {
            "Chain-of-Thought": "chain_of_thought",
            "Few-Shot CoT": "few_shot",
            "Evaluator-Optimizer": "evaluator_optimizer"
        }
        
        methods_to_eval = [technique_map[t] for t in eval_techniques]
        
        with st.spinner("Running benchmark..."):
            comparison_df = evaluator.compare_methods(agent, methods=methods_to_eval)
        
        st.success("Benchmark Complete!")
        
        st.subheader("📊 Results Comparison")
        st.dataframe(comparison_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(comparison_df, x='Method', y='Tier Accuracy (%)', 
                        title="Tier Prediction Accuracy")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(comparison_df, x='Method', y='Avg Score Error', 
                        title="Average Score Error")
            st.plotly_chart(fig, use_container_width=True)

else:  # Batch Processing
    st.header("⚡ Batch Processing")
    
    if st.button("🚀 Start Batch Analysis", type="primary"):
        vendors_to_process = df.head(batch_size)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        for idx, (_, vendor) in enumerate(vendors_to_process.iterrows()):
            status_text.text(f"Processing {idx+1}/{batch_size}...")
            progress_bar.progress((idx + 1) / batch_size)
            
            vendor_data = get_vendor_summary(df, vendor['seller_id'])
            
            if batch_technique == "Chain-of-Thought":
                result = agent.run_chain_of_thought(vendor_data)
            elif batch_technique == "Few-Shot CoT":
                result = agent.run_few_shot_cot(vendor_data)
            else:
                result = agent.run_evaluator_optimizer(vendor_data)
                result = result.get('final', result)
            
            results.append({
                'seller_id': vendor['seller_id'],
                'predicted_tier': result.get('tier', 'N/A'),
                'predicted_score': result.get('overall_score', result.get('score', 0)),
                'actual_tier': vendor['tier'],
                'actual_score': vendor['performance_score']
            })
        
        status_text.text("Complete!")
        
        results_df = pd.DataFrame(results)
        st.subheader("📊 Batch Results")
        st.dataframe(results_df, use_container_width=True)
        
        tier_accuracy = (results_df['predicted_tier'] == results_df['actual_tier']).mean() * 100
        score_mae = (results_df['predicted_score'] - results_df['actual_score']).abs().mean()
        
        col1, col2 = st.columns(2)
        col1.metric("Tier Accuracy", f"{tier_accuracy:.1f}%")
        col2.metric("Score MAE", f"{score_mae:.2f}")
        
        csv = results_df.to_csv(index=False)
        st.download_button("📥 Download Results", csv, "batch_results.csv", "text/csv")

st.divider()
st.caption("💡 For large-scale processing, use `python main_cli.py`")
st.caption("⚙️ Powered by Groq LLama 3.3 70B | Data: Olist Brazilian E-commerce")