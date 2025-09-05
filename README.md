# Black-Scholes Options Pricing Model

A comprehensive Python implementation of the Black-Scholes model for European options pricing, extended with Monte Carlo simulations for exotic options and detailed analysis tools.

## Project Overview

This project implements the Black-Scholes formula to price European call and put options, extends it to Monte Carlo simulations for pricing exotic options, and compares analytical vs. numerical methods for option valuation.

### Features

- **Black-Scholes Formula**: Complete implementation for European call and put options
- **Option Greeks**: Delta, Gamma, Theta, Vega, and Rho calculations
- **Monte Carlo Simulations**: For European and exotic options pricing
- **Exotic Options**: Asian options, Barrier options (up-and-out, down-and-out, up-and-in, down-and-in)
- **Sensitivity Analysis**: Price and Greeks sensitivity to various parameters
- **Comparison Tools**: Analytical vs. numerical method comparison
- **Visualization**: Comprehensive plotting and analysis tools

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

- `numpy>=1.21.0` - Numerical computing
- `matplotlib>=3.5.0` - Plotting and visualization
- `scipy>=1.7.0` - Scientific computing (for normal distribution)
- `pandas>=1.3.0` - Data manipulation
- `seaborn>=0.11.0` - Statistical data visualization

## Quick Start

### Basic Usage

```python
from black_scholes import BlackScholes

# Initialize parameters
S = 100.0  # Current stock price
K = 100.0  # Strike price
T = 1.0    # Time to maturity (years)
r = 0.05   # Risk-free rate
sigma = 0.2  # Volatility

# Create Black-Scholes instance
bs = BlackScholes(S, K, T, r, sigma)

# Calculate option prices
call_price = bs.call_price()
put_price = bs.put_price()

print(f"Call Option Price: ${call_price:.4f}")
print(f"Put Option Price: ${put_price:.4f}")
```

### Running the Demo

```bash
python demo.py
```

This will run a comprehensive demonstration showing:
- Basic pricing
- Greeks calculation
- Monte Carlo simulations
- Exotic options pricing
- Sensitivity analysis
- Comparison between methods

## API Reference

### BlackScholes Class

The main class for Black-Scholes pricing.

#### Constructor
```python
BlackScholes(S, K, T, r, sigma)
```

**Parameters:**
- `S` (float): Current stock price
- `K` (float): Strike price
- `T` (float): Time to maturity (in years)
- `r` (float): Risk-free interest rate
- `sigma` (float): Volatility of the underlying asset

#### Methods

- `call_price()`: Calculate European call option price
- `put_price()`: Calculate European put option price
- `greeks()`: Calculate all option Greeks

### MonteCarloPricing Class

Monte Carlo simulation for options pricing.

#### Constructor
```python
MonteCarloPricing(S, K, T, r, sigma, n_simulations=100000)
```

**Parameters:**
- `S` (float): Current stock price
- `K` (float): Strike price
- `T` (float): Time to maturity (in years)
- `r` (float): Risk-free interest rate
- `sigma` (float): Volatility of the underlying asset
- `n_simulations` (int): Number of Monte Carlo simulations

#### Methods

- `european_call_price()`: Calculate European call option price
- `european_put_price()`: Calculate European put option price
- `asian_call_price()`: Calculate Asian call option price
- `barrier_call_price(barrier, barrier_type)`: Calculate barrier call option price

### OptionPricingComparison Class

Compare analytical vs. numerical methods.

#### Constructor
```python
OptionPricingComparison(S, K, T, r, sigma, n_simulations=100000)
```

#### Methods

- `compare_european_options()`: Compare Black-Scholes vs Monte Carlo
- `plot_price_sensitivity(param_name, param_range, option_type)`: Plot price sensitivity
- `plot_greeks(param_name, param_range, greek_name, option_type)`: Plot Greeks sensitivity

## Examples

### Example 1: Basic Pricing

```python
from black_scholes import BlackScholes

# Create option
bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)

# Get prices
call = bs.call_price()
put = bs.put_price()

print(f"Call: ${call:.4f}, Put: ${put:.4f}")
```

### Example 2: Greeks Analysis

```python
from black_scholes import BlackScholes

bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
greeks = bs.greeks()

print(f"Delta: {greeks['call_delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
print(f"Vega: {greeks['vega']:.4f}")
```

### Example 3: Monte Carlo Pricing

```python
from black_scholes import MonteCarloPricing

mc = MonteCarloPricing(S=100, K=100, T=1, r=0.05, sigma=0.2, n_simulations=100000)

# European options
call_price, call_std = mc.european_call_price()
put_price, put_std = mc.european_put_price()

print(f"Call: ${call_price:.4f} ± ${call_std:.4f}")
print(f"Put: ${put_price:.4f} ± ${put_std:.4f}")

# Exotic options
asian_call, asian_std = mc.asian_call_price()
barrier_call, barrier_std = mc.barrier_call_price(barrier=120, barrier_type="up_and_out")

print(f"Asian Call: ${asian_call:.4f} ± ${asian_std:.4f}")
print(f"Barrier Call: ${barrier_call:.4f} ± ${barrier_std:.4f}")
```

### Example 4: Sensitivity Analysis

```python
import numpy as np
from black_scholes import OptionPricingComparison

# Create comparison instance
comparison = OptionPricingComparison(S=100, K=100, T=1, r=0.05, sigma=0.2)

# Plot price sensitivity to stock price
S_range = np.linspace(50, 150, 50)
comparison.plot_price_sensitivity('S', S_range, 'call')

# Plot Delta sensitivity to stock price
comparison.plot_greeks('S', S_range, 'delta', 'call')
```

## Mathematical Background

### Black-Scholes Formula

The Black-Scholes formula for a European call option is:

```
C = S * N(d1) - K * e^(-r*T) * N(d2)
```

Where:
- `d1 = (ln(S/K) + (r + σ²/2)*T) / (σ*√T)`
- `d2 = d1 - σ*√T`
- `N(x)` is the cumulative standard normal distribution

For a European put option:

```
P = K * e^(-r*T) * N(-d2) - S * N(-d1)
```

### Option Greeks

- **Delta**: Rate of change of option price with respect to underlying asset price
- **Gamma**: Rate of change of delta with respect to underlying asset price
- **Theta**: Rate of change of option price with respect to time
- **Vega**: Rate of change of option price with respect to volatility
- **Rho**: Rate of change of option price with respect to interest rate

### Monte Carlo Method

The Monte Carlo method generates random price paths using geometric Brownian motion:

```
S(t+dt) = S(t) * exp((r - σ²/2)*dt + σ*√dt*Z)
```

Where `Z` is a standard normal random variable.

## File Structure

```
Black Scholes Model
1. black_scholes.py      # Main implementation
2. demo.py              # Comprehensive demo script
3. example.py     
4. test_black_scholes.py

```

## References

1. Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of political economy, 81(3), 637-654.
2. Hull, J. C. (2017). Options, futures, and other derivatives. Pearson.
3. Wilmott, P. (2013). Paul Wilmott on quantitative finance. John Wiley & Sons.

## Acknowledgments

This implementation is based on the classic Black-Scholes model and extends it with modern numerical methods for comprehensive options analysis.
