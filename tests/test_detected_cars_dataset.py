import os

import matplotlib.pyplot as plt
import seaborn as sns
import os
from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))


def test_plot_prices():
    sns.set(color_codes=True)

    df = DetectedCarDataset.get_data_frame(os.path.join(
        os.environ['HOME'],
        'datasets/detected-cars/more_than_4000_detected_per_make/feed.json'))

    makes = df['make'].unique().tolist()
    i = 0
    for make in makes:
        models = df[df['make'] == make]['model'].unique().tolist()
        num_rows = len(models)
        num_cols = 1
        fig, axes = plt.subplots(num_rows, num_cols, sharex=True, figsize=(25, 30))

        for idx, model in enumerate(models):
            ax = axes[idx]
            make_model_items = df[(df.make == make) & (df.model == model)].filter(items=['price', 'dofr'])
            sns.scatterplot(x="dofr", y="price", data=make_model_items, ax=ax, color=".3")
            ax.set_title(f"{model} - {make}")
        plt.show()
        i += 1

        if i > 5:
            break
