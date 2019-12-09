import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os


def _get_df():
    images = []
    makes = []
    models = []
    prices = []
    date_of_first_registrations = []
    xs = []
    ys = []
    ws = []
    hs = []
    i = 0
    # with open('/home/peter/datasets/detected-cars/2019-12-02T12-44-20/feed.json') as f:
    with open('/home/peter/datasets/detected-cars/more_than_4000_detected_per_make/feed.json') as f:
        for line in f:
            # {"image": "images/75be2a0d2743d7c670cba48c416af6ae00d37171.jpg", "make": "Peugeot", "model": "Camper", "price": 3200, "date_of_first_registration": 1999, "bbox": [0.5305101871490479, 0.4644054174423218, 0.9032427072525024, 0.5885359048843384]}
            obj = json.loads(line)
            images.append(obj['image'])
            makes.append(obj['make'])
            models.append(obj['model'])
            prices.append(float(obj['price']))
            date_of_first_registrations.append(int(obj['date_of_first_registration']))
            xs.append(float(obj['bbox'][0]))
            ys.append(float(obj['bbox'][1]))
            ws.append(float(obj['bbox'][2]))
            hs.append(float(obj['bbox'][3]))
            i += 1
            # if i > 10000:
            #     break

    return pd.DataFrame({  # 'images': images,
        'make': pd.Categorical(makes),
        'model': pd.Categorical(models),
        'price': prices,
        'dofr': date_of_first_registrations,
        # 'x': xs,
        # 'y': ys,
        # 'w': ws,
        # 'h': hs,
    })


HERE = os.path.dirname(os.path.realpath(__file__))


def test_detected_cars_dataset():
    df = _get_df()
    print(df)
    plt.figure()
    makes = df['make'].value_counts()
    # makes.where(lambda x: x > 4000).dropna().plot(kind='bar')
    makes.plot(kind='bar')
    # df['make'].value_counts().plot(kind='bar')
    plt.show()

    # with open(os.path.join(HERE, 'output/makes_with_more_than_4000_detected_cars.csv'), 'w') as f:
    #     f.writelines('\n'.join(makes.index.tolist()))


def test_seaborn():
    sns.set(style="whitegrid")

    # Load the example Titanic dataset
    titanic = sns.load_dataset("titanic")

    # Draw a nested barplot to show survival for class and sex
    g = sns.catplot(x="class", y="survived", hue="sex", data=titanic,
                    height=6, kind="bar", palette="muted")
    g.despine(left=True)
    g.set_ylabels("survival probability")
    plt.show()


def test_plot_prices():
    sns.set(color_codes=True)

    df = _get_df()

    makes = df['make'].unique().tolist()
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
