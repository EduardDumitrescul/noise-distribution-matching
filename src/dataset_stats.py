import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
import loader

def display_pie_chart(data: dict, filename):
    values = [x.shape[0] for x in data.values()]
    labels = data.keys()

    def make_autopct(all_values):
        def my_autopct(pct):
            total = sum(all_values)
            val = int(round(pct * total / 100.0))
            return '{p:.1f}%\n({v:d})'.format(p=pct, v=val)

        return my_autopct

    plt.figure(figsize=(8, 6))

    plt.pie(
        values,
        labels=labels,
        autopct=make_autopct(values),
    )

    plt.title(filename)
    plt.axis('equal')

    plt.savefig(filename)
    plt.show()


def display_label_distribution(df, filename):
    label_counts = df['label'].value_counts().sort_index()

    plt.figure(figsize=(4, 4))

    plt.pie(
        label_counts.values,
        labels=label_counts.index,
        autopct='%1.1f%%',  # Show percentages with 1 decimal place
        startangle=90  # Rotate start to vertical
    )

    plt.title(filename)
    plt.savefig(filename)
    plt.show()


def analyze_and_plot_histograms(df, filename):
    pixel_sum = 0.0
    pixel_sum_sq = 0.0
    pixel_count = 0
    global_min = np.inf
    global_max = -np.inf

    image_stats = {
        'means': [],
        'stds': [],
        'mins': [],
        'maxs': []
    }

    print(f"Processing {len(df)} images...")

    for filename in tqdm(df['id']):
        img = loader.load_image(filename)
        flat = img.flatten().astype(np.float64)

        current_min = np.min(flat)
        current_max = np.max(flat)
        if current_min < global_min: global_min = current_min
        if current_max > global_max: global_max = current_max

        pixel_sum += np.sum(flat)
        pixel_sum_sq += np.sum(flat ** 2)
        pixel_count += len(flat)

        image_stats['means'].append(np.mean(flat))
        image_stats['stds'].append(np.std(flat))
        image_stats['mins'].append(current_min)
        image_stats['maxs'].append(current_max)

    total_mean = pixel_sum / pixel_count
    # Variance = E[x^2] - (E[x])^2
    total_var = (pixel_sum_sq / pixel_count) - (total_mean ** 2)
    total_std = np.sqrt(total_var)

    print(f"Global Stats -> Mean: {total_mean:.2f}, Std: {total_std:.2f}, Min: {global_min}, Max: {global_max}")

    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    subplots = [
        (1, 0, 'means', 'Distribution of Image Means'),
        (1, 1, 'stds', 'Distribution of Image Contrast (Std)'),
        (0, 0, 'mins', 'Distribution of Image Mins'),
        (0, 1, 'maxs', 'Distribution of Image Maxs')
    ]

    for row, col, key, title in subplots:
        ax = plt.subplot2grid((2, 2), (row, col))
        sns.histplot(image_stats[key], kde=True, ax=ax, color='steelblue')
        ax.set_title(title)
        ax.set_xlabel(key.capitalize())

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
