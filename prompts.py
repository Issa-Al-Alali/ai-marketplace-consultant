SYSTEM_PROMPT = """You are an expert AI Marketplace Growth Consultant specializing in e-commerce vendor performance analysis.
Your role is to analyze vendor metrics from the Olist Brazilian marketplace and provide actionable, strategic recommendations.
You excel at identifying growth opportunities, operational inefficiencies, and competitive advantages."""

# ============================================================================
# Technique 1: Chain-of-Thought (CoT) Prompting
# ============================================================================
COT_PROMPT = """Analyze this marketplace vendor systematically:

VENDOR DATA:
{metrics}

ANALYSIS FRAMEWORK - Think step-by-step:

1. FINANCIAL HEALTH ASSESSMENT:
   - Evaluate total revenue and profit margins
   - Compare average order value to marketplace benchmarks
   - Assess revenue per product efficiency

2. OPERATIONAL PERFORMANCE:
   - Analyze sales velocity (orders/day)
   - Review delivery performance and delays
   - Evaluate customer satisfaction (review scores)

3. MARKET POSITIONING:
   - Assess product catalog diversity
   - Identify category competitiveness
   - Evaluate customer retention patterns

4. GROWTH POTENTIAL:
   - Identify expansion opportunities
   - Detect operational bottlenecks
   - Recommend strategic improvements

OUTPUT FORMAT (JSON):
{{
  "tier": "A/B/C",
  "overall_score": 0-100,
  "strengths": ["strength1", "strength2", "strength3"],
  "weaknesses": ["weakness1", "weakness2"],
  "recommendations": [
    {{"action": "specific action", "priority": "high/medium/low", "expected_impact": "description"}}
  ],
  "risk_factors": ["risk1", "risk2"],
  "growth_score": 0-100
}}"""

# ============================================================================
# Technique 2: Few-Shot Chain-of-Thought Prompting
# ============================================================================
FEW_SHOT_COT_PROMPT = """You are analyzing a marketplace vendor. Here are examples of high-quality analysis:

EXAMPLE 1 - High Performer:
Vendor: seller_xyz
Metrics: {{revenue: 50000, orders: 200, avg_review: 4.8, delivery_delay: -2 days}}

Analysis:
Step 1: Financial - Strong revenue ($50K), excellent AOV ($250)
Step 2: Performance - Outstanding reviews (4.8/5), early deliveries indicate operational excellence
Step 3: Position - Likely premium product category, high customer trust
Output: {{"tier": "A", "score": 92, "recommendation": "Scale marketing budget, expand product line"}}

EXAMPLE 2 - Struggling Vendor:
Vendor: seller_abc
Metrics: {{revenue: 5000, orders: 100, avg_review: 3.2, delivery_delay: 7 days}}

Analysis:
Step 1: Financial - Low revenue despite decent volume, suggests low-margin products
Step 2: Performance - Poor reviews (3.2/5) and late deliveries are red flags
Step 3: Position - Operational issues damaging reputation
Output: {{"tier": "C", "score": 45, "recommendation": "Urgent: Fix delivery process, improve product quality"}}

EXAMPLE 3 - Mid-tier with Potential:
Vendor: seller_def
Metrics: {{revenue: 20000, orders: 80, avg_review: 4.3, delivery_delay: 1 day}}

Analysis:
Step 1: Financial - Good revenue, high AOV ($250) suggests premium positioning
Step 2: Performance - Solid reviews, minor delivery delays acceptable
Step 3: Position - Strong foundation, ready for growth
Output: {{"tier": "B", "score": 72, "recommendation": "Increase inventory, optimize delivery partner"}}

---

NOW ANALYZE THIS VENDOR:
{metrics}

Follow the same step-by-step reasoning pattern and provide structured JSON output."""

# ============================================================================
# Technique 3: Evaluator-Optimizer (Two-Stage Refinement)
# ============================================================================
EVALUATOR_PROMPT = """You are a CRITICAL EVALUATOR reviewing a vendor analysis draft.

ORIGINAL VENDOR DATA:
{metrics}

DRAFT ANALYSIS:
{draft}

EVALUATION CRITERIA:
1. ACCURACY: Are the conclusions supported by the data?
2. DEPTH: Does it identify root causes, not just symptoms?
3. ACTIONABILITY: Are recommendations specific and implementable?
4. RISK AWARENESS: Are potential risks properly identified?
5. SCORING FAIRNESS: Is the tier/score justified?

Critique the draft by:
- Identifying logical gaps or unsupported claims
- Checking if important metrics were overlooked
- Evaluating whether recommendations are realistic
- Assessing if the tier assignment is too generous or harsh

Provide your critique in this format:
{{
  "accuracy_issues": ["issue1", "issue2"],
  "missing_insights": ["what was missed"],
  "recommendation_improvements": ["how to improve"],
  "score_adjustment": "+5 or -10 points and why"
}}"""

OPTIMIZER_PROMPT = """You are an ANALYSIS OPTIMIZER refining a vendor assessment.

VENDOR DATA:
{metrics}

ORIGINAL DRAFT:
{draft}

EVALUATOR CRITIQUE:
{critique}

Your task: Produce an IMPROVED, REFINED analysis that addresses all critique points.

Requirements:
- Fix accuracy issues
- Add missing insights
- Strengthen recommendations with specific actions
- Adjust tier/score if critique justified it
- Maintain objectivity and data-driven conclusions

Output the final refined analysis in JSON format matching the original structure."""

# ============================================================================
# Technique 4: Parallel Voting (Ensemble Decision Making)
# ============================================================================
VOTING_PROMPT = """You are a CONSENSUS BUILDER evaluating multiple vendor analyses.

VENDOR DATA:
{metrics}

THREE INDEPENDENT ANALYSES:

Analysis 1:
{result1}

Analysis 2:
{result2}

Analysis 3:
{result3}

Your task: Synthesize these three perspectives into ONE balanced, optimal analysis.

Process:
1. Identify CONSENSUS points (where 2+ analyses agree)
2. Resolve CONFLICTS (where analyses disagree)
3. Select the BEST INSIGHTS from each
4. Determine the most FAIR tier/score
5. Combine the most ACTIONABLE recommendations

Prioritize:
- Data-supported conclusions over speculation
- Specific recommendations over generic advice
- Balanced assessment over extreme judgments

Output the final synthesized analysis in JSON format."""

# ============================================================================
# Technique 5: Self-Consistency with Reasoning Paths
# ============================================================================
SELF_CONSISTENCY_PROMPT = """Analyze this vendor using MULTIPLE REASONING PATHS, then converge to a consistent conclusion.

VENDOR DATA:
{metrics}

Generate THREE independent reasoning approaches:

PATH 1 - FINANCIAL-FIRST ANALYSIS:
Start with revenue/profitability, then work outward to operations and market position.

PATH 2 - CUSTOMER-CENTRIC ANALYSIS:
Start with customer satisfaction/reviews, then examine what drives those ratings.

PATH 3 - OPERATIONAL-EFFICIENCY ANALYSIS:
Start with delivery performance and volume metrics, then assess business health.

After completing all three paths:
- Note where conclusions ALIGN (high confidence)
- Note where conclusions DIVERGE (requires deeper thought)
- Synthesize into ONE COHERENT assessment

Output format:
{{
  "reasoning_paths": {{
    "financial_first": "conclusion",
    "customer_centric": "conclusion",
    "operational_efficiency": "conclusion"
  }},
  "consensus_points": ["point1", "point2"],
  "divergent_points": ["point1"],
  "final_analysis": {{standard JSON format}}
}}"""

# ============================================================================
# Technique 6: Comparative Benchmark Analysis
# ============================================================================
COMPARATIVE_PROMPT = """Analyze this vendor WITH PEER COMPARISON context.

TARGET VENDOR:
{metrics}

MARKETPLACE BENCHMARKS:
{benchmarks}

Analysis Framework:
1. Compare against TIER AVERAGES (A, B, C tier vendors)
2. Identify PERCENTILE RANKINGS for key metrics
3. Detect OUTLIER PERFORMANCE (exceptionally good/bad)
4. Assess COMPETITIVE POSITIONING within category

Provide:
- Relative performance vs. peers
- Competitive advantages/disadvantages
- Market share potential
- Category-specific insights

Output comparative analysis in JSON with benchmark comparisons."""