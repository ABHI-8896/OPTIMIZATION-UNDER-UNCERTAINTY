# Optimization Under Uncertainty

## Overview

This project explores the impact of uncertainty in optimization problems, specifically focusing on scheduling and production planning. The study utilizes various modeling approaches, including stochastic programming and robust optimization, to address the inherent complexity in decision-making under uncertainty.

## Key Features

- **Stochastic Modeling**: Models uncertainty using random variables with known or estimated probability distributions.
- **Robust Optimization**: Provides solutions that remain feasible under uncertain conditions, with a balance between optimality and robustness.
- **Comparative Analysis**: Compares a naive approach with robust optimization, highlighting the trade-offs between computational efficiency and solution reliability.

## Mathematical Formulation

### Production Planning Problem (PPP)

The project models a production planning scenario where the goal is to maximize profit under uncertain availability of raw materials. The mathematical formulation involves:

Profit = ∑ SP(j) × X(j) - ∑ ( PCl(j) × L(j) + PCm(j) × M(j) + PCh(j) × H(j) )

### Naive Approach

- Executes the optimization model with different scenarios of raw material availability.
- Analyzes variations in profits to understand the impact of uncertainty.

### Robust Optimization Approach

- Incorporates uncertainty bounds within a deterministic framework.
- Controls the degree of conservatism to ensure feasible solutions under specified uncertainty levels.


def normal_perturbation(original_value):
    return original_value * (1 + 0.5 * np.random.normal(0, 0.2))

# Example usage
original_value = 100
uniform_value = uniform_perturbation(original_value)
normal_value = normal_perturbation(original_value)



###Results

### Naive Approach

- Demonstrates variability in profits under different scenarios of raw material availability.
- Provides insights into the impact of uncertainty but lacks systematic control over robustness.

### Robust Optimization

- Offers a computationally efficient solution that incorporates uncertainty bounds within a deterministic framework.
- Balances optimality and resilience, allowing decision-makers to control the degree of conservatism and ensure feasible solutions under specified uncertainty levels.

## Future Work

- **Refinement of Robust Optimization Model**: Explore further refinement and fine-tuning of the robust optimization model, including adjustments in formulation parameters and additional constraints.
- **Integration of Real-Time Data**: Investigate incorporating real-time data to update the optimization model dynamically, enhancing the decision-making process under changing conditions.
- **Comparative Analysis**: Conduct a comprehensive comparative analysis between various uncertainty modeling approaches, such as stochastic programming and fuzzy programming, to understand their strengths and limitations.
- **Application to Specific Industries**: Extend the research to focus on specific industries, tailoring optimization models to their unique challenges, such as electrical power generation, reservoir operation, or inventory management.
- **Incorporation of Multi-Objective Optimization**: Explore the integration of multi-objective optimization techniques to balance conflicting goals, considering factors like risk mitigation and resource utilization alongside profit.
- **Machine Learning Integration**: Investigate the use of machine learning techniques to predict and adapt to uncertainty patterns, enhancing the robustness of the optimization model in dynamic environments.
  
### Naive Approach

- Executes the optimization model with different scenarios of raw material availability.
- Analyzes variations in profits to understand the impact of uncertainty.

### Robust Optimization Approach

- Incorporates uncertainty bounds within a deterministic framework.
- Controls the degree of conservatism to ensure feasible solutions under specified uncertainty levels.

## Code Snippets

### Naive Approach: Multiple Runs with Different Scenarios

```python
# Example: Naive approach with different raw material scenarios
import numpy as np

# Random perturbation for uniform distribution
def uniform_perturbation(original_value):
    return original_value * (1 + 0.5 * np.random.uniform(-1, 1))

# Random perturbation for normal distribution
def normal_perturbation(original_value):
    return original_value * (1 + 0.5 * np.random.normal(0, 0.2))

# Example usage
original_value = 100
uniform_value = uniform_perturbation(original_value)
normal_value = normal_perturbation(original_value)

print(f"Uniform perturbed value: {uniform_value}")
print(f"Normal perturbed value: {normal_value}")
# Example: Robust optimization approach using linear programming
from scipy.optimize import linprog

# Coefficients for the objective function (profit)
c = [-1, -2]  # Negative because linprog performs minimization

# Inequality constraints (A_ub * x <= b_ub)
A_ub = [[-1, 1], [2, 1]]
b_ub = [1, 6]

# Bounds for variables
x_bounds = [(0, None), (0, None)]

# Solving the linear program
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[x_bounds])

print(f"Optimal solution: {result.x}")
print(f"Optimal value: {result.fun}")

