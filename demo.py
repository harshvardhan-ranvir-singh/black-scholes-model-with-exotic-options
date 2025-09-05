#!/usr/bin/env python3
"""
Comprehensive demo script for the Black-Scholes Options Pricing Model.
This script demonstrates various features including:
- Basic European options pricing
- Greeks calculation
- Monte Carlo simulations
- Exotic options pricing
- Sensitivity analysis
- Comparison between analytical and numerical methods
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from black_scholes import BlackScholes, MonteCarloPricing, OptionPricingComparison
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def demo_basic_pricing():
    """Demonstrate basic Black-Scholes pricing."""
    print("=" * 60)
    print("BASIC BLACK-SCHOLES PRICING DEMO")
    print("=" * 60)
    
    # Example parameters
    S = 100.0  # Current stock price
    K = 100.0  # Strike price
    T = 1.0    # Time to maturity (1 year)
    r = 0.05   # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)
    
    print(f"Parameters:")
    print(f"  Stock Price (S): ${S}")
    print(f"  Strike Price (K): ${K}")
    print(f"  Time to Maturity (T): {T} years")
    print(f"  Risk-free Rate (r): {r*100}%")
    print(f"  Volatility (σ): {sigma*100}%")
    print()
    
    # Create Black-Scholes instance
    bs = BlackScholes(S, K, T, r, sigma)
    
    # Calculate prices
    call_price = bs.call_price()
    put_price = bs.put_price()
    
    print("Black-Scholes Prices:")
    print(f"  Call Option Price: ${call_price:.4f}")
    print(f"  Put Option Price: ${put_price:.4f}")
    print()
    
    # Verify put-call parity
    put_call_parity = call_price - put_price - S + K * np.exp(-r * T)
    print(f"Put-Call Parity Check: {put_call_parity:.6f} (should be ~0)")
    print()

def demo_greeks():
    """Demonstrate Greeks calculation."""
    print("=" * 60)
    print("GREEKS CALCULATION DEMO")
    print("=" * 60)
    
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    bs = BlackScholes(S, K, T, r, sigma)
    greeks = bs.greeks()
    
    print("Option Greeks:")
    print(f"  Call Delta: {greeks['call_delta']:.4f}")
    print(f"  Put Delta:  {greeks['put_delta']:.4f}")
    print(f"  Gamma:      {greeks['gamma']:.4f}")
    print(f"  Call Theta: {greeks['call_theta']:.4f}")
    print(f"  Put Theta:  {greeks['put_theta']:.4f}")
    print(f"  Vega:       {greeks['vega']:.4f}")
    print(f"  Call Rho:   {greeks['call_rho']:.4f}")
    print(f"  Put Rho:    {greeks['put_rho']:.4f}")
    print()

def demo_monte_carlo():
    """Demonstrate Monte Carlo pricing."""
    print("=" * 60)
    print("MONTE CARLO PRICING DEMO")
    print("=" * 60)
    
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    n_simulations = 100000
    
    print(f"Running {n_simulations:,} Monte Carlo simulations...")
    
    mc = MonteCarloPricing(S, K, T, r, sigma, n_simulations)
    
    # European options
    mc_call, mc_call_std = mc.european_call_price()
    mc_put, mc_put_std = mc.european_put_price()
    
    print("European Options (Monte Carlo):")
    print(f"  Call Option: ${mc_call:.4f} ± ${mc_call_std:.4f}")
    print(f"  Put Option:  ${mc_put:.4f} ± ${mc_put_std:.4f}")
    print()
    
    # Exotic options
    asian_call, asian_std = mc.asian_call_price()
    barrier_up_out, barrier_std = mc.barrier_call_price(barrier=120, barrier_type="up_and_out")
    barrier_down_out, barrier_std2 = mc.barrier_call_price(barrier=80, barrier_type="down_and_out")
    
    print("Exotic Options (Monte Carlo):")
    print(f"  Asian Call:           ${asian_call:.4f} ± ${asian_std:.4f}")
    print(f"  Barrier Up-and-Out:   ${barrier_up_out:.4f} ± ${barrier_std:.4f}")
    print(f"  Barrier Down-and-Out: ${barrier_down_out:.4f} ± ${barrier_std2:.4f}")
    print()

def demo_comparison():
    """Demonstrate comparison between analytical and numerical methods."""
    print("=" * 60)
    print("ANALYTICAL vs NUMERICAL COMPARISON DEMO")
    print("=" * 60)
    
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    comparison = OptionPricingComparison(S, K, T, r, sigma, n_simulations=100000)
    
    results = comparison.compare_european_options()
    
    print("Comparison Results:")
    print(f"Call Option:")
    print(f"  Black-Scholes: ${results['call']['black_scholes']:.4f}")
    print(f"  Monte Carlo:   ${results['call']['monte_carlo']:.4f} ± {results['call']['monte_carlo_std']:.4f}")
    print(f"  Difference:    ${results['call']['difference']:.4f} ({results['call']['difference_pct']:.2f}%)")
    print()
    print(f"Put Option:")
    print(f"  Black-Scholes: ${results['put']['black_scholes']:.4f}")
    print(f"  Monte Carlo:   ${results['put']['monte_carlo']:.4f} ± {results['put']['monte_carlo_std']:.4f}")
    print(f"  Difference:    ${results['put']['difference']:.4f} ({results['put']['difference_pct']:.2f}%)")
    print()

def demo_sensitivity_analysis():
    """Demonstrate sensitivity analysis with plots."""
    print("=" * 60)
    print("SENSITIVITY ANALYSIS DEMO")
    print("=" * 60)
    
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    comparison = OptionPricingComparison(S, K, T, r, sigma)
    
    # Create parameter ranges
    S_range = np.linspace(50, 150, 50)
    sigma_range = np.linspace(0.1, 0.5, 50)
    T_range = np.linspace(0.1, 2.0, 50)
    
    print("Generating sensitivity plots...")
    
    # Plot 1: Price vs Stock Price
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    call_prices = []
    put_prices = []
    for S_val in S_range:
        bs = BlackScholes(S_val, K, T, r, sigma)
        call_prices.append(bs.call_price())
        put_prices.append(bs.put_price())
    
    plt.plot(S_range, call_prices, 'b-', label='Call', linewidth=2)
    plt.plot(S_range, put_prices, 'r-', label='Put', linewidth=2)
    plt.axvline(x=K, color='k', linestyle='--', alpha=0.7, label='Strike Price')
    plt.xlabel('Stock Price ($)')
    plt.ylabel('Option Price ($)')
    plt.title('Option Price vs Stock Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Price vs Volatility
    plt.subplot(1, 3, 2)
    call_prices = []
    put_prices = []
    for sigma_val in sigma_range:
        bs = BlackScholes(S, K, T, r, sigma_val)
        call_prices.append(bs.call_price())
        put_prices.append(bs.put_price())
    
    plt.plot(sigma_range, call_prices, 'b-', label='Call', linewidth=2)
    plt.plot(sigma_range, put_prices, 'r-', label='Put', linewidth=2)
    plt.xlabel('Volatility')
    plt.ylabel('Option Price ($)')
    plt.title('Option Price vs Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Price vs Time to Maturity
    plt.subplot(1, 3, 3)
    call_prices = []
    put_prices = []
    for T_val in T_range:
        bs = BlackScholes(S, K, T_val, r, sigma)
        call_prices.append(bs.call_price())
        put_prices.append(bs.put_price())
    
    plt.plot(T_range, call_prices, 'b-', label='Call', linewidth=2)
    plt.plot(T_range, put_prices, 'r-', label='Put', linewidth=2)
    plt.xlabel('Time to Maturity (years)')
    plt.ylabel('Option Price ($)')
    plt.title('Option Price vs Time to Maturity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Sensitivity analysis plots generated!")
    print()

def demo_greeks_analysis():
    """Demonstrate Greeks analysis with plots."""
    print("=" * 60)
    print("GREEKS ANALYSIS DEMO")
    print("=" * 60)
    
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    comparison = OptionPricingComparison(S, K, T, r, sigma)
    
    # Create parameter ranges
    S_range = np.linspace(50, 150, 50)
    sigma_range = np.linspace(0.1, 0.5, 50)
    
    print("Generating Greeks analysis plots...")
    
    # Plot 1: Delta vs Stock Price
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    call_deltas = []
    put_deltas = []
    for S_val in S_range:
        bs = BlackScholes(S_val, K, T, r, sigma)
        greeks = bs.greeks()
        call_deltas.append(greeks['call_delta'])
        put_deltas.append(greeks['put_delta'])
    
    plt.plot(S_range, call_deltas, 'b-', label='Call Delta', linewidth=2)
    plt.plot(S_range, put_deltas, 'r-', label='Put Delta', linewidth=2)
    plt.axvline(x=K, color='k', linestyle='--', alpha=0.7, label='Strike Price')
    plt.xlabel('Stock Price ($)')
    plt.ylabel('Delta')
    plt.title('Delta vs Stock Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Gamma vs Stock Price
    plt.subplot(1, 3, 2)
    gammas = []
    for S_val in S_range:
        bs = BlackScholes(S_val, K, T, r, sigma)
        greeks = bs.greeks()
        gammas.append(greeks['gamma'])
    
    plt.plot(S_range, gammas, 'g-', label='Gamma', linewidth=2)
    plt.axvline(x=K, color='k', linestyle='--', alpha=0.7, label='Strike Price')
    plt.xlabel('Stock Price ($)')
    plt.ylabel('Gamma')
    plt.title('Gamma vs Stock Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Vega vs Volatility
    plt.subplot(1, 3, 3)
    vegas = []
    for sigma_val in sigma_range:
        bs = BlackScholes(S, K, T, r, sigma_val)
        greeks = bs.greeks()
        vegas.append(greeks['vega'])
    
    plt.plot(sigma_range, vegas, 'm-', label='Vega', linewidth=2)
    plt.xlabel('Volatility')
    plt.ylabel('Vega')
    plt.title('Vega vs Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Greeks analysis plots generated!")
    print()

def demo_monte_carlo_convergence():
    """Demonstrate Monte Carlo convergence."""
    print("=" * 60)
    print("MONTE CARLO CONVERGENCE DEMO")
    print("=" * 60)
    
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    
    # Test different numbers of simulations
    n_sims = [1000, 5000, 10000, 50000, 100000, 200000]
    call_prices = []
    call_stds = []
    
    print("Testing Monte Carlo convergence...")
    for n in n_sims:
        mc = MonteCarloPricing(S, K, T, r, sigma, n)
        price, std = mc.european_call_price()
        call_prices.append(price)
        call_stds.append(std)
        print(f"  n={n:6d}: Price=${price:.4f}, Std=${std:.4f}")
    
    # Black-Scholes price for comparison
    bs = BlackScholes(S, K, T, r, sigma)
    bs_price = bs.call_price()
    
    print(f"\nBlack-Scholes price: ${bs_price:.4f}")
    print()
    
    # Plot convergence
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.errorbar(n_sims, call_prices, yerr=call_stds, fmt='o-', capsize=5, capthick=2)
    plt.axhline(y=bs_price, color='r', linestyle='--', label='Black-Scholes')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Call Option Price ($)')
    plt.title('Monte Carlo Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.subplot(1, 2, 2)
    plt.semilogx(n_sims, call_stds, 'o-', linewidth=2)
    plt.xlabel('Number of Simulations')
    plt.ylabel('Standard Error ($)')
    plt.title('Monte Carlo Standard Error')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Convergence analysis complete!")
    print()

def main():
    """Run all demonstrations."""
    print("BLACK-SCHOLES OPTIONS PRICING MODEL - COMPREHENSIVE DEMO")
    print("=" * 80)
    print()
    
    try:
        # Run all demos
        demo_basic_pricing()
        demo_greeks()
        demo_monte_carlo()
        demo_comparison()
        demo_sensitivity_analysis()
        demo_greeks_analysis()
        demo_monte_carlo_convergence()
        
        print("=" * 80)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("Please check your installation and try again.")

if __name__ == "__main__":
    main()
