import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for professional academic/report plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Ensure the plots directory exists
os.makedirs('../plots', exist_ok=True)


def plot_tail_latency():
    """Plot 5: Tail Latency Percentile Curve"""
    # Load 512MB Locust data
    csv_path = "../results/results_vs_512mb/Locust_2026-05-30-18h33_locustfile.py_https___3fxen53kw4.execute-api.us-east-1.amazonaws.com_Prod_requests-512mb4.csv"
    df = pd.read_csv(csv_path)

    # Filter out the 'Aggregated' row
    df_endpoints = df[df['Name'] != 'Aggregated']

    percentiles = ['50%', '66%', '75%', '80%',
                   '90%', '95%', '98%', '99%', '99.9%', '100%']
    percentile_labels = ['50th', '66th', '75th', '80th',
                         '90th', '95th', '98th', '99th', '99.9th', 'Max']

    plt.figure(figsize=(10, 5))
    markers = ['o', 's', '^', 'D']

    for idx, (i, row) in enumerate(df_endpoints.iterrows()):
        y_vals = row[percentiles].values
        # Clean up the name for a nicer legend (e.g., '/Prod/predict/lgbm' -> 'lgbm')
        name = row['Name'].replace('/Prod/predict/', '')
        plt.plot(percentile_labels, y_vals,
                 marker=markers[idx % len(markers)], label=name, linewidth=2)

    plt.yscale('log')
    plt.title('Tail Latency: Response Time Percentile Curve (512MB)',
              fontsize=14, pad=15)
    plt.xlabel('Percentile', fontsize=12)
    plt.ylabel('Response Time (ms) - Log Scale', fontsize=12)
    plt.legend(title='Endpoints')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("../plots/plot_5_tail_latency_curve.png", dpi=300)
    print("Saved: plot_5_tail_latency_curve.png")


def plot_duration_distribution():
    """Plot 6: Violin Plot of Lambda Execution Durations"""
    # Load CloudWatch data for 256, 512, 1024
    cw_256 = pd.read_csv(
        '../results/results_vs_265mb/logs-insights-results-256mb4.csv')
    cw_512 = pd.read_csv(
        '../results/results_vs_512mb/logs-insights-results-512mb4.csv')
    cw_1024 = pd.read_csv(
        '../results/results_vs_1024mb/logs-insights-results-1024mb4.csv')

    # Add a column to identify the memory size
    cw_256['Memory'] = '256 MB'
    cw_512['Memory'] = '512 MB'
    cw_1024['Memory'] = '1024 MB'

    # Combine dataframes
    df_cw = pd.concat([cw_256, cw_512, cw_1024])

    # Filter out the extreme 1% anomalies (like huge cold starts) so the violin shape is readable
    q99 = df_cw['@duration'].quantile(0.99)
    df_cw_filtered = df_cw[df_cw['@duration'] < q99]

    plt.figure(figsize=(9, 5))
    sns.violinplot(x='Memory', y='@duration', data=df_cw_filtered,
                   palette='muted', inner='quartile')
    plt.title('Distribution of Lambda Execution Durations (Excl. Extremes)',
              fontsize=14, pad=15)
    plt.xlabel('Allocated Memory', fontsize=12)
    plt.ylabel('Duration (ms)', fontsize=12)
    plt.tight_layout()
    plt.savefig("../plots/plot_6_duration_distribution.png", dpi=300)
    print("Saved: plot_6_duration_distribution.png")


def plot_cost_vs_performance():
    """Plot 7: Cost vs Performance Pareto Curve"""
    # AWS Lambda Pricing (x86 architecture) is roughly $0.0000166667 for every GB-second.
    # Therefore, cost per ms calculation:
    cost_per_ms = [0.0000000042, 0.0000000083,
                   0.0000000167]  # 256, 512, 1024 respectively
    mem_sizes = ['256 MB', '512 MB', '1024 MB']

    # Read CloudWatch Data for billed duration
    cw_256 = pd.read_csv(
        '../results/results_vs_265mb/logs-insights-results-256mb4.csv')
    cw_512 = pd.read_csv(
        '../results/results_vs_512mb/logs-insights-results-512mb4.csv')
    cw_1024 = pd.read_csv(
        '../results/results_vs_1024mb/logs-insights-results-1024mb4.csv')

    avg_billed_durations = [cw_256['@billedDuration'].mean(),
                            cw_512['@billedDuration'].mean(), cw_1024['@billedDuration'].mean()]

    # Calculate cost per 1 million invocations
    cost_per_1m = [(d * c * 1_000_000)
                   for d, c in zip(avg_billed_durations, cost_per_ms)]

    # Read Locust Data for 99th Percentile User Latency (Aggregated)
    locust_256 = pd.read_csv(
        '../results/results_vs_265mb/Locust_2026-05-30-18h55_locustfile.py_https___3fxen53kw4.execute-api.us-east-1.amazonaws.com_Prod_requests-256mb4.csv')
    locust_512 = pd.read_csv(
        '../results/results_vs_512mb/Locust_2026-05-30-18h33_locustfile.py_https___3fxen53kw4.execute-api.us-east-1.amazonaws.com_Prod_requests-512mb4.csv')
    locust_1024 = pd.read_csv(
        '../results/results_vs_1024mb/Locust_2026-05-30-19h17_locustfile.py_https___3fxen53kw4.execute-api.us-east-1.amazonaws.com_Prod_requests-1024mb4.csv')

    p99_lat = [
        locust_256[locust_256['Name'] == 'Aggregated']['99%'].values[0],
        locust_512[locust_512['Name'] == 'Aggregated']['99%'].values[0],
        locust_1024[locust_1024['Name'] == 'Aggregated']['99%'].values[0]
    ]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=cost_per_1m, y=p99_lat, s=200, hue=mem_sizes,
                    palette='deep', edgecolor='black', legend=False)
    plt.plot(cost_per_1m, p99_lat, linestyle='--', color='gray', alpha=0.5)

    # Annotate points
    for i, mem in enumerate(mem_sizes):
        plt.annotate(mem, (cost_per_1m[i], p99_lat[i]), xytext=(
            10, 5), textcoords='offset points', fontweight='bold')

    plt.title('Cost vs. Performance Trade-off', fontsize=14, pad=15)
    plt.xlabel('Estimated Cost per 1 Million Invocations (USD)', fontsize=12)
    plt.ylabel('99th Percentile End-User Latency (ms)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("../plots/plot_7_cost_vs_performance.png", dpi=300)
    print("Saved: plot_7_cost_vs_performance.png")


if __name__ == "__main__":
    plot_tail_latency()
    plot_duration_distribution()
    plot_cost_vs_performance()
    print("Advanced plots generated successfully!")
