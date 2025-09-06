import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Union, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class BlackScholes:
    """
    Black-Scholes model implementation for European options pricing.
    
    The Black-Scholes model is a mathematical model for pricing European-style options.
    It assumes that the underlying asset follows a geometric Brownian motion.
    """
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float):
        """
        Initialize Black-Scholes parameters.
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to maturity (in years)
            r (float): Risk-free interest rate
            sigma (float): Volatility of the underlying asset
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        
    def _calculate_d1_d2(self) -> Tuple[float, float]:
        """Calculate d1 and d2 parameters for Black-Scholes formula."""
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2
    
    def call_price(self) -> float:
        """
        Calculate European call option price using Black-Scholes formula.
        
        Returns:
            float: Call option price
        """
        d1, d2 = self._calculate_d1_d2()
        call_price = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        return call_price
    
    def put_price(self) -> float:
        """
        Calculate European put option price using Black-Scholes formula.
        
        Returns:
            float: Put option price
        """
        d1, d2 = self._calculate_d1_d2()
        put_price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        return put_price
    
    def greeks(self) -> dict:
        """
        Calculate option Greeks (Delta, Gamma, Theta, Vega, Rho).
        
        Returns:
            dict: Dictionary containing all Greeks
        """
        d1, d2 = self._calculate_d1_d2()
        
        # Delta
        call_delta = norm.cdf(d1)
        put_delta = call_delta - 1
        
        # Gamma (same for both call and put)
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        
        # Theta
        call_theta = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) 
                     - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        put_theta = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) 
                    + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))
        
        # Vega (same for both call and put)
        vega = self.S * norm.pdf(d1) * np.sqrt(self.T)
        
        # Rho
        call_rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        put_rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)
        
        return {
            'call_delta': call_delta,
            'put_delta': put_delta,
            'gamma': gamma,
            'call_theta': call_theta,
            'put_theta': put_theta,
            'vega': vega,
            'call_rho': call_rho,
            'put_rho': put_rho
        }


class MonteCarloPricing:
    """
    Monte Carlo simulation for options pricing, including exotic options.
    """
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, n_simulations: int = 100000):
        """
        Initialize Monte Carlo parameters.
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to maturity (in years)
            r (float): Risk-free interest rate
            sigma (float): Volatility of the underlying asset
            n_simulations (int): Number of Monte Carlo simulations
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_simulations = n_simulations
    
    def _generate_price_paths(self, n_steps: int = 252) -> np.ndarray:
        """
        Generate stock price paths using geometric Brownian motion.
        
        Args:
            n_steps (int): Number of time steps in the path
            
        Returns:
            np.ndarray: Array of stock price paths
        """
        dt = self.T / n_steps
        random_shocks = np.random.normal(0, 1, (self.n_simulations, n_steps))
        
        # Generate price paths
        price_paths = np.zeros((self.n_simulations, n_steps + 1))
        price_paths[:, 0] = self.S
        
        for t in range(1, n_steps + 1):
            price_paths[:, t] = price_paths[:, t-1] * np.exp(
                (self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * random_shocks[:, t-1]
            )
        
        return price_paths
    
    def european_call_price(self) -> Tuple[float, float]:
        """
        Calculate European call option price using Monte Carlo.
        
        Returns:
            Tuple[float, float]: (price, standard_error)
        """
        price_paths = self._generate_price_paths()
        final_prices = price_paths[:, -1]
        
        # Calculate payoffs
        payoffs = np.maximum(final_prices - self.K, 0)
        
        # Discount to present value
        option_prices = np.exp(-self.r * self.T) * payoffs
        
        # Calculate mean and standard error
        mean_price = np.mean(option_prices)
        std_error = np.std(option_prices) / np.sqrt(self.n_simulations)
        
        return mean_price, std_error
    
    def european_put_price(self) -> Tuple[float, float]:
        """
        Calculate European put option price using Monte Carlo.
        
        Returns:
            Tuple[float, float]: (price, standard_error)
        """
        price_paths = self._generate_price_paths()
        final_prices = price_paths[:, -1]
        
        # Calculate payoffs
        payoffs = np.maximum(self.K - final_prices, 0)
        
        # Discount to present value
        option_prices = np.exp(-self.r * self.T) * payoffs
        
        # Calculate mean and standard error
        mean_price = np.mean(option_prices)
        std_error = np.std(option_prices) / np.sqrt(self.n_simulations)
        
        return mean_price, std_error
    
    def asian_call_price(self, n_steps: int = 252) -> Tuple[float, float]:
        """
        Calculate Asian call option price using Monte Carlo.
        Asian options depend on the average price over the option's lifetime.
        
        Args:
            n_steps (int): Number of time steps for averaging
            
        Returns:
            Tuple[float, float]: (price, standard_error)
        """
        price_paths = self._generate_price_paths(n_steps)
        
        # Calculate average prices
        average_prices = np.mean(price_paths, axis=1)
        
        # Calculate payoffs
        payoffs = np.maximum(average_prices - self.K, 0)
        
        # Discount to present value
        option_prices = np.exp(-self.r * self.T) * payoffs
        
        # Calculate mean and standard error
        mean_price = np.mean(option_prices)
        std_error = np.std(option_prices) / np.sqrt(self.n_simulations)
        
        return mean_price, std_error
    
    def barrier_call_price(self, barrier: float, barrier_type: str = "up_and_out") -> Tuple[float, float]:
        """
        Calculate barrier call option price using Monte Carlo.
        
        Args:
            barrier (float): Barrier level
            barrier_type (str): Type of barrier ("up_and_out", "down_and_out", "up_and_in", "down_and_in")
            
        Returns:
            Tuple[float, float]: (price, standard_error)
        """
        price_paths = self._generate_price_paths()
        
        # Check barrier conditions
        if barrier_type == "up_and_out":
            # Option is knocked out if price goes above barrier
            barrier_hit = np.any(price_paths > barrier, axis=1)
            payoffs = np.where(barrier_hit, 0, np.maximum(price_paths[:, -1] - self.K, 0))
        elif barrier_type == "down_and_out":
            # Option is knocked out if price goes below barrier
            barrier_hit = np.any(price_paths < barrier, axis=1)
            payoffs = np.where(barrier_hit, 0, np.maximum(price_paths[:, -1] - self.K, 0))
        elif barrier_type == "up_and_in":
            # Option is activated only if price goes above barrier
            barrier_hit = np.any(price_paths > barrier, axis=1)
            payoffs = np.where(barrier_hit, np.maximum(price_paths[:, -1] - self.K, 0), 0)
        elif barrier_type == "down_and_in":
            # Option is activated only if price goes below barrier
            barrier_hit = np.any(price_paths < barrier, axis=1)
            payoffs = np.where(barrier_hit, np.maximum(price_paths[:, -1] - self.K, 0), 0)
        else:
            raise ValueError("Invalid barrier_type. Must be one of: up_and_out, down_and_out, up_and_in, down_and_in")
        
        # Discount to present value
        option_prices = np.exp(-self.r * self.T) * payoffs
        
        # Calculate mean and standard error
        mean_price = np.mean(option_prices)
        std_error = np.std(option_prices) / np.sqrt(self.n_simulations)
        
        return mean_price, std_error


class OptionPricingComparison:
    """
    Compare analytical vs numerical methods for option pricing.
    """
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, n_simulations: int = 100000):
        """
        Initialize comparison parameters.
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to maturity (in years)
            r (float): Risk-free interest rate
            sigma (float): Volatility of the underlying asset
            n_simulations (int): Number of Monte Carlo simulations
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_simulations = n_simulations
        
        # Initialize pricing methods
        self.black_scholes = BlackScholes(S, K, T, r, sigma)
        self.monte_carlo = MonteCarloPricing(S, K, T, r, sigma, n_simulations)
    
    def compare_european_options(self) -> dict:
        """
        Compare Black-Scholes vs Monte Carlo for European options.
        
        Returns:
            dict: Comparison results
        """
        # Black-Scholes prices
        bs_call = self.black_scholes.call_price()
        bs_put = self.black_scholes.put_price()
        
        # Monte Carlo prices
        mc_call, mc_call_std = self.monte_carlo.european_call_price()
        mc_put, mc_put_std = self.monte_carlo.european_put_price()
        
        return {
            'call': {
                'black_scholes': bs_call,
                'monte_carlo': mc_call,
                'monte_carlo_std': mc_call_std,
                'difference': abs(bs_call - mc_call),
                'difference_pct': abs(bs_call - mc_call) / bs_call * 100
            },
            'put': {
                'black_scholes': bs_put,
                'monte_carlo': mc_put,
                'monte_carlo_std': mc_put_std,
                'difference': abs(bs_put - mc_put),
                'difference_pct': abs(bs_put - mc_put) / bs_put * 100
            }
        }


def main():
    """
    Main function to demonstrate the Black-Scholes implementation.
    """
    # Example parameters
    S = 100.0  # Current stock price
    K = 100.0  # Strike price
    T = 1.0    # Time to maturity (1 year)
    r = 0.05   # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)
    
    print("Black-Scholes Options Pricing Model")
    print("=" * 40)
    print(f"Stock Price (S): ${S}")
    print(f"Strike Price (K): ${K}")
    print(f"Time to Maturity (T): {T} years")
    print(f"Risk-free Rate (r): {r*100}%")
    print(f"Volatility (σ): {sigma*100}%")
    print()
    
    # Black-Scholes pricing
    bs = BlackScholes(S, K, T, r, sigma)
    call_price = bs.call_price()
    put_price = bs.put_price()
    
    print("Black-Scholes Prices:")
    print(f"Call Option Price: ${call_price:.4f}")
    print(f"Put Option Price: ${put_price:.4f}")
    print()
    
    # Greeks
    greeks = bs.greeks()
    print("Option Greeks:")
    print(f"Call Delta: {greeks['call_delta']:.4f}")
    print(f"Put Delta: {greeks['put_delta']:.4f}")
    print(f"Gamma: {greeks['gamma']:.4f}")
    print(f"Call Theta: {greeks['call_theta']:.4f}")
    print(f"Put Theta: {greeks['put_theta']:.4f}")
    print(f"Vega: {greeks['vega']:.4f}")
    print(f"Call Rho: {greeks['call_rho']:.4f}")
    print(f"Put Rho: {greeks['put_rho']:.4f}")
    print()
    
    # Monte Carlo pricing
    mc = MonteCarloPricing(S, K, T, r, sigma, n_simulations=100000)
    mc_call, mc_call_std = mc.european_call_price()
    mc_put, mc_put_std = mc.european_put_price()
    
    print("Monte Carlo Prices (100,000 simulations):")
    print(f"Call Option Price: ${mc_call:.4f} ± ${mc_call_std:.4f}")
    print(f"Put Option Price: ${mc_put:.4f} ± ${mc_put_std:.4f}")
    print()
    
    # Comparison
    comparison = OptionPricingComparison(S, K, T, r, sigma, n_simulations=100000)
    results = comparison.compare_european_options()
    
    print("Comparison (Black-Scholes vs Monte Carlo):")
    print(f"Call Option Difference: ${results['call']['difference']:.4f} ({results['call']['difference_pct']:.2f}%)")
    print(f"Put Option Difference: ${results['put']['difference']:.4f} ({results['put']['difference_pct']:.2f}%)")
    print()
    
    # Exotic options
    print("Exotic Options (Monte Carlo):")
    asian_call, asian_std = mc.asian_call_price()
    barrier_call, barrier_std = mc.barrier_call_price(barrier=120, barrier_type="up_and_out")
    
    print(f"Asian Call Option Price: ${asian_call:.4f} ± ${asian_std:.4f}")
    print(f"Barrier Call Option Price (Up-and-Out, Barrier=120): ${barrier_call:.4f} ± ${barrier_std:.4f}")


if __name__ == "__main__":
    main()
