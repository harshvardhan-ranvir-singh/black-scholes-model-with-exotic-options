#!/usr/bin/env python3
"""
Simple example script demonstrating the Black-Scholes implementation.
This script shows basic usage and can be run immediately after installation.
"""

from black_scholes import BlackScholes, MonteCarloPricing, OptionPricingComparison
import numpy as np

def main():
    """Run a simple example."""
    print("Black-Scholes Options Pricing - Simple Example")
    print("=" * 50)
    print()
    
    # Example parameters
    S = 100.0  # Current stock price
    K = 100.0  # Strike price
    T = 1.0    # Time to maturity (1 year)
    r = 0.05   # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)
    
    print("Parameters:")
    print(f"  Stock Price (S): ${S}")
    print(f"  Strike Price (K): ${K}")
    print(f"  Time to Maturity (T): {T} years")
    print(f"  Risk-free Rate (r): {r*100}%")
    print(f"  Volatility (σ): {sigma*100}%")
    print()
    
    # 1. Black-Scholes Pricing
    print("1. Black-Scholes Pricing")
    print("-" * 30)
    
    bs = BlackScholes(S, K, T, r, sigma)
    call_price = bs.call_price()
    put_price = bs.put_price()
    
    print(f"Call Option Price: ${call_price:.4f}")
    print(f"Put Option Price:  ${put_price:.4f}")
    print()
    
    # 2. Option Greeks
    print("2. Option Greeks")
    print("-" * 30)
    
    greeks = bs.greeks()
    print(f"Call Delta: {greeks['call_delta']:.4f}")
    print(f"Put Delta:  {greeks['put_delta']:.4f}")
    print(f"Gamma:      {greeks['gamma']:.4f}")
    print(f"Vega:       {greeks['vega']:.4f}")
    print()
    
    # 3. Monte Carlo Pricing
    print("3. Monte Carlo Pricing (10,000 simulations)")
    print("-" * 30)
    
    mc = MonteCarloPricing(S, K, T, r, sigma, n_simulations=10000)
    mc_call, mc_call_std = mc.european_call_price()
    mc_put, mc_put_std = mc.european_put_price()
    
    print(f"Call Option: ${mc_call:.4f} ± ${mc_call_std:.4f}")
    print(f"Put Option:  ${mc_put:.4f} ± ${mc_put_std:.4f}")
    print()
    
    # 4. Method Comparison
    print("4. Method Comparison")
    print("-" * 30)
    
    comparison = OptionPricingComparison(S, K, T, r, sigma, n_simulations=10000)
    results = comparison.compare_european_options()
    
    call_diff = results['call']['difference']
    put_diff = results['put']['difference']
    
    print(f"Call Option Difference: ${call_diff:.4f}")
    print(f"Put Option Difference:  ${put_diff:.4f}")
    print()
    
    # 5. Exotic Options
    print("5. Exotic Options (Monte Carlo)")
    print("-" * 30)
    
    asian_call, asian_std = mc.asian_call_price()
    barrier_call, barrier_std = mc.barrier_call_price(barrier=120, barrier_type="up_and_out")
    
    print(f"Asian Call Option:        ${asian_call:.4f} ± ${asian_std:.4f}")
    print(f"Barrier Call (Up-and-Out): ${barrier_call:.4f} ± ${barrier_std:.4f}")
    print()
    
    # 6. Sensitivity Analysis
    print("6. Sensitivity Analysis")
    print("-" * 30)
    
    # Test different stock prices
    stock_prices = [80, 90, 100, 110, 120]
    print("Call Option Prices at Different Stock Prices:")
    for S_test in stock_prices:
        bs_test = BlackScholes(S_test, K, T, r, sigma)
        call_test = bs_test.call_price()
        print(f"  S=${S_test:3d}: Call=${call_test:.4f}")
    
    print()
    print("=" * 50)
    print("Example completed successfully!")
    print("Run 'python demo.py' for a comprehensive demonstration.")

if __name__ == "__main__":
    main()
