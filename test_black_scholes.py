#!/usr/bin/env python3
"""
Test script for the Black-Scholes implementation.
This script verifies that the implementation works correctly by testing
various scenarios and edge cases.
"""

import numpy as np
from black_scholes import BlackScholes, MonteCarloPricing, OptionPricingComparison
import warnings
warnings.filterwarnings('ignore')

def test_basic_pricing():
    """Test basic Black-Scholes pricing."""
    print("Testing basic pricing...")
    
    # Test case 1: At-the-money option
    bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
    call_price = bs.call_price()
    put_price = bs.put_price()
    
    # Verify put-call parity
    put_call_parity = call_price - put_price - 100 + 100 * np.exp(-0.05)
    assert abs(put_call_parity) < 1e-10, f"Put-call parity failed: {put_call_parity}"
    
    print(f"  ✓ At-the-money: Call=${call_price:.4f}, Put=${put_price:.4f}")
    
    # Test case 2: In-the-money call
    bs_itm = BlackScholes(S=110, K=100, T=1, r=0.05, sigma=0.2)
    call_itm = bs_itm.call_price()
    put_itm = bs_itm.put_price()
    
    assert call_itm > put_itm, "ITM call should be more valuable than ITM put"
    assert call_itm > call_price, "ITM call should be more valuable than ATM call"
    
    print(f"  ✓ In-the-money: Call=${call_itm:.4f}, Put=${put_itm:.4f}")
    
    # Test case 3: Out-of-the-money call
    bs_otm = BlackScholes(S=90, K=100, T=1, r=0.05, sigma=0.2)
    call_otm = bs_otm.call_price()
    put_otm = bs_otm.put_price()
    
    assert call_otm < call_price, "OTM call should be less valuable than ATM call"
    assert put_otm > put_price, "OTM put should be more valuable than ATM put"
    
    print(f"  ✓ Out-of-the-money: Call=${call_otm:.4f}, Put=${put_otm:.4f}")
    
    print("  ✓ Basic pricing tests passed!")

def test_greeks():
    """Test Greeks calculation."""
    print("Testing Greeks...")
    
    bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
    greeks = bs.greeks()
    
    # Test Delta bounds
    assert 0 <= greeks['call_delta'] <= 1, f"Call delta out of bounds: {greeks['call_delta']}"
    assert -1 <= greeks['put_delta'] <= 0, f"Put delta out of bounds: {greeks['put_delta']}"
    
    # Test Delta relationship
    assert abs(greeks['call_delta'] - greeks['put_delta'] - 1) < 1e-10, "Delta relationship failed"
    
    # Test Gamma is positive
    assert greeks['gamma'] > 0, f"Gamma should be positive: {greeks['gamma']}"
    
    # Test Vega is positive
    assert greeks['vega'] > 0, f"Vega should be positive: {greeks['vega']}"
    
    print(f"  ✓ Call Delta: {greeks['call_delta']:.4f}")
    print(f"  ✓ Put Delta: {greeks['put_delta']:.4f}")
    print(f"  ✓ Gamma: {greeks['gamma']:.4f}")
    print(f"  ✓ Vega: {greeks['vega']:.4f}")
    print("  ✓ Greeks tests passed!")

def test_monte_carlo():
    """Test Monte Carlo pricing."""
    print("Testing Monte Carlo pricing...")
    
    # Test with known parameters
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    mc = MonteCarloPricing(S, K, T, r, sigma, n_simulations=10000)
    
    # Test European options
    call_price, call_std = mc.european_call_price()
    put_price, put_std = mc.european_put_price()
    
    assert call_price > 0, f"Call price should be positive: {call_price}"
    assert put_price > 0, f"Put price should be positive: {put_price}"
    assert call_std > 0, f"Call std should be positive: {call_std}"
    assert put_std > 0, f"Put std should be positive: {put_std}"
    
    print(f"  ✓ European Call: ${call_price:.4f} ± ${call_std:.4f}")
    print(f"  ✓ European Put: ${put_price:.4f} ± ${put_std:.4f}")
    
    # Test exotic options
    asian_call, asian_std = mc.asian_call_price()
    barrier_call, barrier_std = mc.barrier_call_price(barrier=120, barrier_type="up_and_out")
    
    assert asian_call > 0, f"Asian call price should be positive: {asian_call}"
    assert barrier_call >= 0, f"Barrier call price should be non-negative: {barrier_call}"
    
    print(f"  ✓ Asian Call: ${asian_call:.4f} ± ${asian_std:.4f}")
    print(f"  ✓ Barrier Call: ${barrier_call:.4f} ± ${barrier_std:.4f}")
    
    print("  ✓ Monte Carlo tests passed!")

def test_comparison():
    """Test comparison between methods."""
    print("Testing method comparison...")
    
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    comparison = OptionPricingComparison(S, K, T, r, sigma, n_simulations=10000)
    
    results = comparison.compare_european_options()
    
    # Test that differences are reasonable (within 5%)
    call_diff_pct = results['call']['difference_pct']
    put_diff_pct = results['put']['difference_pct']
    
    assert call_diff_pct < 5, f"Call difference too large: {call_diff_pct}%"
    assert put_diff_pct < 5, f"Put difference too large: {put_diff_pct}%"
    
    print(f"  ✓ Call difference: {call_diff_pct:.2f}%")
    print(f"  ✓ Put difference: {put_diff_pct:.2f}%")
    print("  ✓ Comparison tests passed!")

def test_edge_cases():
    """Test edge cases."""
    print("Testing edge cases...")
    
    # Test very short maturity
    bs_short = BlackScholes(S=100, K=100, T=0.001, r=0.05, sigma=0.2)
    call_short = bs_short.call_price()
    put_short = bs_short.put_price()
    
    assert call_short >= 0, f"Short maturity call should be non-negative: {call_short}"
    assert put_short >= 0, f"Short maturity put should be non-negative: {put_short}"
    
    print(f"  ✓ Short maturity: Call=${call_short:.4f}, Put=${put_short:.4f}")
    
    # Test very high volatility
    bs_high_vol = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=1.0)
    call_high_vol = bs_high_vol.call_price()
    put_high_vol = bs_high_vol.put_price()
    
    assert call_high_vol > 0, f"High volatility call should be positive: {call_high_vol}"
    assert put_high_vol > 0, f"High volatility put should be positive: {put_high_vol}"
    
    print(f"  ✓ High volatility: Call=${call_high_vol:.4f}, Put=${put_high_vol:.4f}")
    
    # Test zero interest rate
    bs_zero_r = BlackScholes(S=100, K=100, T=1, r=0.0, sigma=0.2)
    call_zero_r = bs_zero_r.call_price()
    put_zero_r = bs_zero_r.put_price()
    
    assert call_zero_r > 0, f"Zero rate call should be positive: {call_zero_r}"
    assert put_zero_r > 0, f"Zero rate put should be positive: {put_zero_r}"
    
    print(f"  ✓ Zero interest rate: Call=${call_zero_r:.4f}, Put=${put_zero_r:.4f}")
    
    print("  ✓ Edge case tests passed!")

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("RUNNING BLACK-SCHOLES TESTS")
    print("=" * 60)
    print()
    
    try:
        test_basic_pricing()
        print()
        test_greeks()
        print()
        test_monte_carlo()
        print()
        test_comparison()
        print()
        test_edge_cases()
        print()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"Test failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
