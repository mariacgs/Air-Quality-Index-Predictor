import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Impostazioni di stile per grafici professionali
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# --- FUNZIONI DI ESTRAZIONE DATI ---


def get_locust_stats(filepath):
    df = pd.read_csv(filepath)
    agg = df[df['Name'] == 'Aggregated'].iloc[0]
    return agg['50%'], agg['99%']


def get_cw_duration(filepath):
    df = pd.read_csv(filepath)
    return df['@duration'].mean(), df['@billedDuration'].mean()


def get_cw_coldstart(filepath):
    df = pd.read_csv(filepath)
    cold_starts = df['@initDuration'].dropna()
    return cold_starts.max() if len(cold_starts) > 0 else 0

# --- ESTRAZIONE DINAMICA DAI FILE CSV ---
# Nota: I percorsi iniziano con '../' perché lo script viene eseguito da dentro 'src/'


# 1. Dati per Vertical Scaling
cw_256 = '../results/results_vs_265mb/logs-insights-results-256mb4.csv'
cw_512 = '../results/results_vs_512mb/logs-insights-results-512mb4.csv'
cw_1024 = '../results/results_vs_1024mb/logs-insights-results-1024mb4.csv'

locust_256 = '../results/results_vs_265mb/Locust_2026-05-30-18h55_locustfile.py_https___3fxen53kw4.execute-api.us-east-1.amazonaws.com_Prod_requests-256mb4.csv'
locust_512 = '../results/results_vs_512mb/Locust_2026-05-30-18h33_locustfile.py_https___3fxen53kw4.execute-api.us-east-1.amazonaws.com_Prod_requests-512mb4.csv'
locust_1024 = '../results/results_vs_1024mb/Locust_2026-05-30-19h17_locustfile.py_https___3fxen53kw4.execute-api.us-east-1.amazonaws.com_Prod_requests-1024mb4.csv'

_, billed_256 = get_cw_duration(cw_256)
_, billed_512 = get_cw_duration(cw_512)
_, billed_1024 = get_cw_duration(cw_1024)

med_256, p99_256 = get_locust_stats(locust_256)
med_512, p99_512 = get_locust_stats(locust_512)
med_1024, p99_1024 = get_locust_stats(locust_1024)

# 2. Dati per Framework Showdown
cw_lgbm = '../results/results_lgbm_only_512mb_50/logs-insights-results-lgbm-50.csv'
cw_xgb = '../results/results_xgb_baseline_only_50/logs-insights-results-xgb_baseline-50.csv'
cw_arpa_xgb = '../results/results_xgb_arpa_only_50/logs-insights-results-xgb_arpa-50.csv'
cw_arpa_lgbm = '../results/results_lgbm_arpa_only_50/logs-insights-results-lgbm_arpa-50.csv'

dur_lgbm, _ = get_cw_duration(cw_lgbm)
dur_xgb, _ = get_cw_duration(cw_xgb)
dur_lgbm_arpa, _ = get_cw_duration(cw_arpa_lgbm)
dur_arpa_xgb, _ = get_cw_duration(cw_arpa_xgb)

# 3. Dati per Cold Start (Spike)
cw_spike = '../results/results_coldstart_80/logs-insights-results-coldstart-60.csv'
max_cold_spike = get_cw_coldstart(cw_spike)
dur_warm_spike, _ = get_cw_duration(cw_spike)


# --- GENERAZIONE GRAFICI ---

def plot_billed_duration():
    """Plot 1: Cloud Execution Time vs Memory Allocation"""
    memory = ['256 MB', '512 MB', '1024 MB']
    billed_duration = [billed_256, billed_512, billed_1024]

    plt.figure(figsize=(8, 5))
    sns.lineplot(x=memory, y=billed_duration, marker='o',
                 color='b', linewidth=2.5, markersize=10)

    plt.title('AWS Lambda Execution Time by Memory Allocation',
              fontsize=14, pad=15)
    plt.xlabel('Allocated Memory (MiB)', fontsize=12)
    plt.ylabel('Average Billed Duration (ms)', fontsize=12)
    plt.ylim(0, max(billed_duration) * 1.2)

    for i, v in enumerate(billed_duration):
        plt.text(i, v + 0.3, f"{v:.2f} ms", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('../plots/plot_1_vertical_scaling_compute.png', dpi=300)
    print("Saved: plot_1_vertical_scaling_compute.png")


def plot_user_latency():
    """Plot 2: User Latency across Memory Sizes"""
    memory = ['256 MB', '512 MB', '1024 MB']
    median_rt = [med_256, med_512, med_1024]
    p99_rt = [p99_256, p99_512, p99_1024]

    x = np.arange(len(memory))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, median_rt, width,
                    label='Median (50%)', color='#4C72B0')
    rects2 = ax.bar(x + width/2, p99_rt, width,
                    label='99th Percentile', color='#DD8452')

    ax.set_ylabel('Response Time (ms)')
    ax.set_title('User-Facing Latency by Memory Allocation',
                 fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(memory)
    ax.legend()
    ax.set_ylim(0, max(p99_rt) * 1.2)

    ax.bar_label(rects1, padding=3, fmt='%.0f ms')
    ax.bar_label(rects2, padding=3, fmt='%.0f ms')

    plt.tight_layout()
    plt.savefig('../plots/plot_2_user_latency.png', dpi=300)
    print("Saved: plot_2_user_latency.png")


def plot_framework_showdown():
    """Plot 3: LightGBM vs XGBoost Baseline vs ARPA XGBoost vs LightGBM ARPA"""
    frameworks = ['LightGBM',
                  'XGBoost', 'XGBoost ARPA', 'LightGBM ARPA']
    # dur_arpa_xgb da definire se disponibile
    duration = [dur_lgbm, dur_xgb, dur_arpa_xgb, dur_lgbm_arpa]

    plt.figure(figsize=(6, 5))
    ax = sns.barplot(x=frameworks, y=duration, palette='viridis')

    plt.title('Framework Computational Footprint (512MB)', fontsize=14, pad=15)
    plt.ylabel('Average Execution Duration (ms)', fontsize=12)
    plt.ylim(0, max(duration) * 1.2)

    for i, v in enumerate(duration):
        ax.text(i, v + 0.1, f"{v:.2f} ms", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('../plots/plot_3_framework_showdown.png', dpi=300)
    print("Saved: plot_3_framework_showdown.png")


def plot_cold_start():
    """Plot 4: The Cold Start Penalty"""
    states = ['Warm Execution', 'Cold Start (Flash Spike)']
    times = [dur_warm_spike, max_cold_spike]

    plt.figure(figsize=(6, 5))
    ax = sns.barplot(x=states, y=times, palette=['#55A868', '#C44E52'])

    plt.title('The Serverless Cold Start Penalty', fontsize=14, pad=15)
    plt.ylabel('AWS Lambda Init + Duration (ms)', fontsize=12)
    plt.yscale('log')

    for i, v in enumerate(times):
        ax.text(i, v * 1.2, f"{v:.2f} ms", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('../plots/plot_4_cold_start_penalty.png', dpi=300)
    print("Saved: plot_4_cold_start_penalty.png")


if __name__ == "__main__":
    plot_billed_duration()
    plot_user_latency()
    plot_framework_showdown()
    plot_cold_start()
    print("All plots generated successfully dynamically from CSVs!")
