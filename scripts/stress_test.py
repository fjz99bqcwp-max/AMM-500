#!/usr/bin/env python3
"""
Monte Carlo and Stress Testing for AMM-500 Strategy

Simulates various market scenarios to validate strategy robustness:
- Volatility spikes (50%, 100%, 200%)
- Flash crashes (5%, 10%, 20% instant drops)
- Funding rate shocks (0.1%, 0.5%, 1% per 8h)
- API outages (30s, 60s, 120s downtime)
- High correlation periods (trending markets)
- Liquidity crises (wide spreads, thin orderbook)

Usage:
    python scripts/stress_test.py --scenarios all --runs 1000
    python scripts/stress_test.py --scenarios crash --runs 500
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from src.config import Config
from src.backtest import RealisticBacktest
from src.data_fetcher import DataFetcher


@dataclass
class StressScenario:
    """Defines a stress test scenario."""
    name: str
    description: str
    price_shock: float = 0.0  # % instant price change
    volatility_multiplier: float = 1.0  # Multiply IV by this
    spread_multiplier: float = 1.0  # Multiply spread by this
    funding_rate_shock: float = 0.0  # Additional funding rate per 8h
    api_outage_seconds: int = 0  # Simulate API downtime
    duration_days: int = 7  # How long scenario lasts


# Define stress scenarios
SCENARIOS = {
    "baseline": StressScenario(
        name="Baseline",
        description="Normal market conditions (control)",
        duration_days=30
    ),
    
    "vol_spike_50": StressScenario(
        name="Volatility Spike 50%",
        description="IV increases 50% (minor stress)",
        volatility_multiplier=1.5,
        duration_days=7
    ),
    
    "vol_spike_100": StressScenario(
        name="Volatility Spike 100%",
        description="IV doubles (major stress)",
        volatility_multiplier=2.0,
        duration_days=7
    ),
    
    "vol_spike_200": StressScenario(
        name="Volatility Spike 200%",
        description="IV triples (extreme stress)",
        volatility_multiplier=3.0,
        duration_days=7
    ),
    
    "flash_crash_5": StressScenario(
        name="Flash Crash 5%",
        description="Instant 5% price drop",
        price_shock=-0.05,
        volatility_multiplier=2.0,
        spread_multiplier=3.0,
        duration_days=1
    ),
    
    "flash_crash_10": StressScenario(
        name="Flash Crash 10%",
        description="Instant 10% price drop",
        price_shock=-0.10,
        volatility_multiplier=3.0,
        spread_multiplier=5.0,
        duration_days=1
    ),
    
    "flash_crash_20": StressScenario(
        name="Flash Crash 20%",
        description="Instant 20% price drop (Black Swan)",
        price_shock=-0.20,
        volatility_multiplier=5.0,
        spread_multiplier=10.0,
        duration_days=1
    ),
    
    "funding_shock_01": StressScenario(
        name="Funding Shock 0.1%",
        description="High funding rate: 0.1% per 8h",
        funding_rate_shock=0.001,
        duration_days=7
    ),
    
    "funding_shock_05": StressScenario(
        name="Funding Shock 0.5%",
        description="Extreme funding rate: 0.5% per 8h",
        funding_rate_shock=0.005,
        duration_days=7
    ),
    
    "funding_shock_10": StressScenario(
        name="Funding Shock 1.0%",
        description="Crisis funding rate: 1.0% per 8h",
        funding_rate_shock=0.010,
        duration_days=7
    ),
    
    "api_outage_30s": StressScenario(
        name="API Outage 30s",
        description="30-second API downtime",
        api_outage_seconds=30,
        duration_days=7
    ),
    
    "api_outage_60s": StressScenario(
        name="API Outage 60s",
        description="60-second API downtime",
        api_outage_seconds=60,
        duration_days=7
    ),
    
    "api_outage_120s": StressScenario(
        name="API Outage 120s",
        description="120-second API downtime (critical)",
        api_outage_seconds=120,
        duration_days=7
    ),
    
    "liquidity_crisis": StressScenario(
        name="Liquidity Crisis",
        description="Spreads widen 5x, low volume",
        spread_multiplier=5.0,
        volatility_multiplier=2.0,
        duration_days=7
    ),
    
    "trending_market": StressScenario(
        name="Strong Trending Market",
        description="High correlation, directional move",
        volatility_multiplier=1.5,
        funding_rate_shock=0.003,
        duration_days=7
    ),
    
    "perfect_storm": StressScenario(
        name="Perfect Storm",
        description="Crash + vol spike + funding shock + API outage",
        price_shock=-0.15,
        volatility_multiplier=4.0,
        spread_multiplier=8.0,
        funding_rate_shock=0.008,
        api_outage_seconds=60,
        duration_days=3
    ),
}


class StressTester:
    """Runs Monte Carlo simulations and stress tests."""
    
    def __init__(self, config_path: str = None):
        self.config = Config.load(config_path)
        self.fetcher = DataFetcher(self.config)
        
    def generate_stressed_data(
        self,
        scenario: StressScenario,
        base_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply stress scenario to market data."""
        data = base_data.copy()
        
        # Apply instant price shock
        if scenario.price_shock != 0:
            shock_point = len(data) // 2  # Middle of dataset
            multiplier = 1 + scenario.price_shock
            data.loc[shock_point:, ['open', 'high', 'low', 'close']] *= multiplier
            print(f"  Applied {scenario.price_shock*100:+.1f}% price shock at candle {shock_point}")
        
        # Increase volatility
        if scenario.volatility_multiplier != 1.0:
            # Add noise proportional to multiplier
            noise_factor = (scenario.volatility_multiplier - 1.0) * 0.01
            for col in ['open', 'high', 'low', 'close']:
                noise = np.random.normal(0, data[col].std() * noise_factor, len(data))
                data[col] += noise
            print(f"  Volatility multiplied by {scenario.volatility_multiplier}x")
        
        # Widen spreads
        if scenario.spread_multiplier != 1.0:
            # Store in metadata for backtest to use
            data.attrs['spread_multiplier'] = scenario.spread_multiplier
            print(f"  Spreads widened by {scenario.spread_multiplier}x")
        
        # Add funding rate shock
        if scenario.funding_rate_shock != 0:
            data.attrs['funding_shock'] = scenario.funding_rate_shock
            print(f"  Funding rate shock: {scenario.funding_rate_shock*100:+.3f}% per 8h")
        
        # Simulate API outage
        if scenario.api_outage_seconds > 0:
            data.attrs['api_outage_seconds'] = scenario.api_outage_seconds
            print(f"  Simulating {scenario.api_outage_seconds}s API outage")
        
        return data
    
    def run_monte_carlo(
        self,
        scenario_name: str,
        num_runs: int = 1000
    ) -> Dict:
        """Run Monte Carlo simulation for a scenario."""
        print(f"\n{'='*60}")
        print(f"Monte Carlo: {scenario_name} ({num_runs} runs)")
        print(f"{'='*60}")
        
        scenario = SCENARIOS[scenario_name]
        print(f"Scenario: {scenario.description}")
        
        # Generate base synthetic data
        print(f"\nGenerating {scenario.duration_days}-day synthetic data...")
        base_data = self.fetcher.generate_synthetic_data(
            days=scenario.duration_days,
            volatility=0.15  # Base 15% annualized vol
        )
        
        results = []
        
        for run in range(num_runs):
            if (run + 1) % 100 == 0:
                print(f"  Run {run + 1}/{num_runs}...", end='\r')
            
            # Apply stress with random seed
            np.random.seed(run)
            stressed_data = self.generate_stressed_data(scenario, base_data)
            
            # Run backtest
            backtest = RealisticBacktest(self.config, stressed_data)
            metrics = backtest.run()
            
            results.append({
                'run': run,
                'roi': metrics.total_return,
                'sharpe': metrics.sharpe_ratio,
                'max_dd': metrics.max_drawdown,
                'trades': metrics.total_trades,
                'win_rate': metrics.win_rate,
                'final_equity': metrics.final_equity,
                'maker_pct': metrics.maker_fill_rate,
            })
        
        print(f"\nCompleted {num_runs} runs")
        
        # Aggregate statistics
        df_results = pd.DataFrame(results)
        
        stats = {
            'scenario': scenario_name,
            'description': scenario.description,
            'num_runs': num_runs,
            'duration_days': scenario.duration_days,
            
            # ROI stats
            'roi_mean': df_results['roi'].mean(),
            'roi_std': df_results['roi'].std(),
            'roi_min': df_results['roi'].min(),
            'roi_max': df_results['roi'].max(),
            'roi_percentile_5': df_results['roi'].quantile(0.05),
            'roi_percentile_95': df_results['roi'].quantile(0.95),
            
            # Sharpe stats
            'sharpe_mean': df_results['sharpe'].mean(),
            'sharpe_std': df_results['sharpe'].std(),
            'sharpe_min': df_results['sharpe'].min(),
            'sharpe_percentile_5': df_results['sharpe'].quantile(0.05),
            
            # Drawdown stats
            'max_dd_mean': df_results['max_dd'].mean(),
            'max_dd_worst': df_results['max_dd'].max(),
            'max_dd_percentile_95': df_results['max_dd'].quantile(0.95),
            
            # Failure rates
            'failure_rate_dd_2pct': (df_results['max_dd'] > 0.02).sum() / num_runs,
            'failure_rate_negative_roi': (df_results['roi'] < 0).sum() / num_runs,
            'failure_rate_sharpe_below_1': (df_results['sharpe'] < 1.0).sum() / num_runs,
            
            # Trade stats
            'trades_mean': df_results['trades'].mean(),
            'trades_std': df_results['trades'].std(),
            
            'raw_results': df_results,
        }
        
        return stats
    
    def print_results(self, stats: Dict):
        """Print formatted results."""
        print(f"\n{'='*60}")
        print(f"RESULTS: {stats['scenario']}")
        print(f"{'='*60}")
        print(f"Description: {stats['description']}")
        print(f"Runs: {stats['num_runs']} x {stats['duration_days']} days")
        print(f"\n--- ROI Statistics ---")
        print(f"Mean ROI: {stats['roi_mean']*100:+.2f}%")
        print(f"Std Dev: {stats['roi_std']*100:.2f}%")
        print(f"Range: {stats['roi_min']*100:+.2f}% to {stats['roi_max']*100:+.2f}%")
        print(f"5th-95th Percentile: {stats['roi_percentile_5']*100:+.2f}% to {stats['roi_percentile_95']*100:+.2f}%")
        
        print(f"\n--- Sharpe Ratio ---")
        print(f"Mean Sharpe: {stats['sharpe_mean']:.2f}")
        print(f"Std Dev: {stats['sharpe_std']:.2f}")
        print(f"Min Sharpe: {stats['sharpe_min']:.2f}")
        print(f"5th Percentile: {stats['sharpe_percentile_5']:.2f}")
        
        print(f"\n--- Drawdown Risk ---")
        print(f"Mean Max DD: {stats['max_dd_mean']*100:.2f}%")
        print(f"Worst Case DD: {stats['max_dd_worst']*100:.2f}%")
        print(f"95th Percentile DD: {stats['max_dd_percentile_95']*100:.2f}%")
        
        print(f"\n--- Failure Rates ---")
        print(f"DD > 2%: {stats['failure_rate_dd_2pct']*100:.1f}%")
        print(f"Negative ROI: {stats['failure_rate_negative_roi']*100:.1f}%")
        print(f"Sharpe < 1.0: {stats['failure_rate_sharpe_below_1']*100:.1f}%")
        
        print(f"\n--- Trading Activity ---")
        print(f"Avg Trades: {stats['trades_mean']:.0f} ± {stats['trades_std']:.0f}")
        
        # Risk assessment
        print(f"\n--- RISK ASSESSMENT ---")
        if stats['failure_rate_dd_2pct'] < 0.05:
            print(f"✅ Drawdown Risk: LOW ({stats['failure_rate_dd_2pct']*100:.1f}% chance of DD>2%)")
        elif stats['failure_rate_dd_2pct'] < 0.20:
            print(f"⚠️  Drawdown Risk: MEDIUM ({stats['failure_rate_dd_2pct']*100:.1f}% chance of DD>2%)")
        else:
            print(f"❌ Drawdown Risk: HIGH ({stats['failure_rate_dd_2pct']*100:.1f}% chance of DD>2%)")
        
        if stats['failure_rate_negative_roi'] < 0.10:
            print(f"✅ Loss Risk: LOW ({stats['failure_rate_negative_roi']*100:.1f}% chance of loss)")
        elif stats['failure_rate_negative_roi'] < 0.30:
            print(f"⚠️  Loss Risk: MEDIUM ({stats['failure_rate_negative_roi']*100:.1f}% chance of loss)")
        else:
            print(f"❌ Loss Risk: HIGH ({stats['failure_rate_negative_roi']*100:.1f}% chance of loss)")
        
        if stats['sharpe_mean'] >= 1.5:
            print(f"✅ Risk-Adjusted Return: EXCELLENT (Sharpe {stats['sharpe_mean']:.2f})")
        elif stats['sharpe_mean'] >= 1.0:
            print(f"✅ Risk-Adjusted Return: GOOD (Sharpe {stats['sharpe_mean']:.2f})")
        elif stats['sharpe_mean'] >= 0.5:
            print(f"⚠️  Risk-Adjusted Return: ACCEPTABLE (Sharpe {stats['sharpe_mean']:.2f})")
        else:
            print(f"❌ Risk-Adjusted Return: POOR (Sharpe {stats['sharpe_mean']:.2f})")
    
    def plot_results(self, stats: Dict, output_path: str = None):
        """Generate visualization of Monte Carlo results."""
        df = stats['raw_results']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Monte Carlo: {stats['scenario']} ({stats['num_runs']} runs)", 
                     fontsize=14, fontweight='bold')
        
        # ROI distribution
        axes[0, 0].hist(df['roi'] * 100, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(stats['roi_mean'] * 100, color='red', linestyle='--', 
                          label=f"Mean: {stats['roi_mean']*100:.2f}%")
        axes[0, 0].axvline(0, color='black', linestyle='-', linewidth=0.5)
        axes[0, 0].set_xlabel('ROI (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('ROI Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Sharpe distribution
        axes[0, 1].hist(df['sharpe'], bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[0, 1].axvline(stats['sharpe_mean'], color='red', linestyle='--',
                          label=f"Mean: {stats['sharpe_mean']:.2f}")
        axes[0, 1].axvline(1.0, color='orange', linestyle=':', label='Target: 1.0')
        axes[0, 1].set_xlabel('Sharpe Ratio')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Sharpe Ratio Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Max DD distribution
        axes[1, 0].hist(df['max_dd'] * 100, bins=50, edgecolor='black', alpha=0.7, color='red')
        axes[1, 0].axvline(stats['max_dd_mean'] * 100, color='darkred', linestyle='--',
                          label=f"Mean: {stats['max_dd_mean']*100:.2f}%")
        axes[1, 0].axvline(2.0, color='black', linestyle=':', label='Limit: 2%')
        axes[1, 0].set_xlabel('Max Drawdown (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Max Drawdown Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Scatter: ROI vs Sharpe
        axes[1, 1].scatter(df['sharpe'], df['roi'] * 100, alpha=0.3, s=10)
        axes[1, 1].set_xlabel('Sharpe Ratio')
        axes[1, 1].set_ylabel('ROI (%)')
        axes[1, 1].set_title('Risk-Return Relationship')
        axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].axvline(1.0, color='orange', linestyle=':', alpha=0.5)
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {output_path}")
        else:
            plt.show()
    
    def run_all_scenarios(self, num_runs: int = 100, quick: bool = False):
        """Run all stress scenarios."""
        if quick:
            # Quick test with fewer runs
            scenarios = ['baseline', 'flash_crash_10', 'vol_spike_100', 'perfect_storm']
            num_runs = 50
        else:
            scenarios = list(SCENARIOS.keys())
        
        all_results = {}
        
        for scenario in scenarios:
            results = self.run_monte_carlo(scenario, num_runs)
            self.print_results(results)
            all_results[scenario] = results
        
        # Summary comparison
        print(f"\n{'='*80}")
        print(f"STRESS TEST SUMMARY - All Scenarios")
        print(f"{'='*80}")
        print(f"{'Scenario':<25} {'Mean ROI':<12} {'Mean Sharpe':<12} {'Worst DD':<12} {'Loss Rate':<12}")
        print(f"{'-'*80}")
        
        for scenario, results in all_results.items():
            print(f"{scenario:<25} "
                  f"{results['roi_mean']*100:>10.2f}% "
                  f"{results['sharpe_mean']:>11.2f} "
                  f"{results['max_dd_worst']*100:>10.2f}% "
                  f"{results['failure_rate_negative_roi']*100:>10.1f}%")
        
        return all_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Monte Carlo & Stress Testing for AMM-500')
    parser.add_argument('--scenarios', nargs='+', 
                       choices=list(SCENARIOS.keys()) + ['all', 'quick'],
                       default=['baseline'],
                       help='Scenarios to test')
    parser.add_argument('--runs', type=int, default=100,
                       help='Number of Monte Carlo runs per scenario')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    parser.add_argument('--output', type=str, default='data/stress_test_results.csv',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    tester = StressTester()
    
    if 'all' in args.scenarios:
        results = tester.run_all_scenarios(num_runs=args.runs, quick=False)
    elif 'quick' in args.scenarios:
        results = tester.run_all_scenarios(num_runs=50, quick=True)
    else:
        results = {}
        for scenario in args.scenarios:
            result = tester.run_monte_carlo(scenario, num_runs=args.runs)
            tester.print_results(result)
            results[scenario] = result
            
            if args.plot:
                plot_path = f"data/stress_test_{scenario}.png"
                tester.plot_results(result, plot_path)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    
    summary_data = []
    for scenario, result in results.items():
        summary_data.append({
            'scenario': scenario,
            'description': result['description'],
            'runs': result['num_runs'],
            'roi_mean': result['roi_mean'],
            'roi_std': result['roi_std'],
            'sharpe_mean': result['sharpe_mean'],
            'max_dd_worst': result['max_dd_worst'],
            'failure_dd_2pct': result['failure_rate_dd_2pct'],
            'failure_loss': result['failure_rate_negative_roi'],
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
