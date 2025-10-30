# AutoCO: Automated Constraint Optimization via LLM-driven Bidirectional Coevolution

## 🎯 Core Innovation

Large Language Model (LLM)-based optimization has recently shown promise for autonomous problem solving, yet most approaches still cast LLMs as passive constraint checkers rather than proactive strategy designers, limiting their effectiveness on complex Constraint Optimization Problems (COPs).

**The Fundamental Challenge**: Most real-world COPs are NP-hard, with complexity arising from hard constraints that create conflicting decisions and global dependencies. These constraints fragment the feasible region, complicating the solution process and underscoring the necessity for efficient and high-quality solutions. As hard constraints increase, obtaining feasible solutions becomes more difficult, resulting in ineffective feedback that can lead to unguided search processes producing suboptimal designs.

### Our Solution: AutoCO Framework

To address these challenges, we present **AutoCO**, an end-to-end **Auto**mated **C**onstraint **O**ptimization method that tightly couples operations-research principles of constraint relaxation with LLM reasoning. AutoCO enables LLMs to explicitly explore and optimize constraint relaxation strategies as integral components in the algorithm design process, positioning them as proactive strategy designers rather than passive validators.

**Transformative Approach**: Unlike existing LLM methods that primarily rely on feedback from feasible solutions and lack detailed problem analysis, AutoCO integrates the world knowledge, reasoning, and strategy search capabilities of LLMs with constraint relaxation principles through three complementary innovations:

## 🚀 Key Contributions

### 1. Triple-Representation Scheme
A core innovation is a **unified triple-representation** that maintains synchronized evolution across three coupled components:
- **Relaxation Strategies**: Constraint-specific relaxation coefficients and modification approaches
- **Algorithmic Principles**: High-level algorithm design concepts and optimization methodologies  
- **Executable Codes**: Fully implemented solver implementations ready for execution

This design enables the LLM to synthesize, justify, and instantiate relaxation strategies that are both principled and executable. The unified representation ensures that each generated solver explicitly employs its corresponding constraint relaxation strategy to solve the optimization problem, bridging the structural modeling gap in conventional approaches.

### 2. Bidirectional Global-Local Coevolution Mechanism
To navigate fragmented solution spaces and address the vast strategy decision space, AutoCO employs a **bidirectional coevolution mechanism**, synergistically coupling:

- **Global Exploration**: Monte Carlo Tree Search (MCTS) for systematic relaxation-trajectory exploration and global strategy guidance
- **Local Intensification**: Evolutionary Algorithms (EAs) for fine-grained solution refinement and local optimization

This continuous exchange of priors and feedback explicitly balances diversification and intensification, thus preventing premature convergence. The mechanism delegates fine-grained optimization of strategy-concepts-code triples to the local EA layer, while the global MCTS layer explores the global strategy space to identify promising directions.

### 3. LLM-Driven Constraint Relaxation Methodology
We propose a structured three-step relaxation method that transforms expert-dependent constraint handling into an LLM-driven automated approach:

- **Constraint Importance Analysis**: LLM-powered parsing to identify constraints and assign context-aware importance weights
- **Relaxation Range Suggestion**: Adaptive determination of relaxation factor ranges based on constraint criticality
- **Strategy Generation**: Systematic synthesis of diverse relaxation strategies adhering to constraint structures

## 🏗 System Architecture
<div align="center">
<img src="figure/figure_ours.jpg" width="1000" alt="AutoCO Architecture">
</div>
<p align="center">
<em>Figure 1: Architecture of AutoCO. Initially, we use LLMs to parse user-input problems and generate initial constraint relaxation strategies. Next, the bidirectional coevolution mechanism combining local EA and global MCTS explores and optimizes strategies and codes. Finally, we evaluate the generated algorithms on problem instances and provide individual fitness feedback.</em>
</p>

## 📊 Method Comparison
<div align="center">
<img src="figure/figure_Intro.jpg" width="800" alt="Method Comparison">
</div>
<p align="center">
<em>Figure 2: Comparisons of human/LLM-based solutions for COPs. (A) Expert-designed method leverages human analysis. (B) Current LLM methods focus on code generation. (C) Our AutoCO combines human-inspired relaxation strategies with automation to effectively discover feasible solutions.</em>
</p>

## 🏆 Experimental Results

Extensive experiments on three challenging COP benchmarks validate AutoCO's consistent effectiveness and superior performance, especially in hard regimes where current methods degrade.

### Performance Highlights
- **24.7% average optimality gap reduction** vs state-of-the-art LLM methods
- **Effective constraint handling** in VRPTW, VRPTW-fuel, and Safety Facility Layout problems
- **Rapid feasible solution generation** even under hard constraints
- **Superior performance** where exact solvers like Gurobi struggle with larger instances
- 
<div align="center">
<img src="figure/figure_plot.jpg" width="400" height="250" alt="Performance Comparison">
</div>
<p align="center">
<em>Figure 3: AutoCO (blue) vs. Current LLM method (green) performance on VRPTW-fuel problems over 100 iterations. The autonomous constraint relaxation temporarily expands feasible regions, enhancing optimization feedback.</em>
</p>


## 🔬 Technical Approach

### Three-Phase Architecture
1. **Problem Analysis & Strategy Design**: LLM-driven constraint importance analysis and relaxation range suggestion
2. **Optimal Strategy Search**: Bidirectional coevolution with MCTS-EA coordination  
3. **Code Execution & Evaluation**: Secure execution with iterative repair mechanisms

### Constraint Relaxation Strategy
- **Intelligent Constraint Weighting**: LLM assigns importance weights to constraints
- **Adaptive Relaxation Ranges**: Context-aware relaxation factor determination
- **Structured Strategy Generation**: Systematic sampling of relaxation coefficients

## 💡 Significance

Results highlight AutoCO as a principled and effective path toward **proactive, verifiable LLM-driven optimization**, transforming LLMs from passive validators into strategic constraint relaxation designers. The framework demonstrates consistent effectiveness across diverse constraint optimization domains, achieving significant performance improvements particularly in challenging regimes where current methods degrade.
