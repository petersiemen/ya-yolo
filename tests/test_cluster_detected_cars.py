from .context import *


def test_kmeans_with_one_model():
    df = DetectedCarDataset.get_data_frame(
        '/home/peter/datasets/detected-cars/more_than_4000_detected_per_make/feed.json')

    supra = df[(df['make'] == 'Toyota') & (df['model'] == 'Supra')]

    detected_cars = list(DetectedCar.from_data_frame(supra))
    detected_cars.sort(key=lambda i: i.date_of_first_registration)

    labels = cluster(detected_cars, 2)

    plt.scatter([detected_car.date_of_first_registration for detected_car in detected_cars],
                [detected_car.price for detected_car in detected_cars],
                c=labels,
                s=50, cmap='viridis')
    plt.show()

    clusters = DetectedCar.to_price_clusters(detected_cars, labels)

    variance = sum_of_variance(clusters)
    print(variance)


def test_kmeans_with_more_makes_and_models():
    df = DetectedCarDataset.get_data_frame(
        '/home/peter/datasets/detected-cars/more_than_4000_detected_per_make/feed.json')

    makes = df['make'].unique().tolist()
    max_makes = 5
    max_models = 5
    make_i = 0

    for make in makes:
        models = df[df['make'] == make]['model'].unique().tolist()
        model_i = 0
        for model in models:

            make_model = df[(df['make'] == make) & (df['model'] == model)]
            detected_cars = list(DetectedCar.from_data_frame(make_model))
            detected_cars.sort(key=lambda i: i.date_of_first_registration)

            print(f"cluster {make} -  {model}")
            if len(detected_cars) > 2:
                #labels = cluster(detected_cars, 2)
                labels = find_best_cluster(detected_cars,1,3)

                plt.scatter([detected_car.date_of_first_registration for detected_car in detected_cars],
                            [detected_car.price for detected_car in detected_cars],
                            c=labels,
                            s=50, cmap='viridis')
                plt.title(f"{model} - {make}")
                plt.show()
                model_i += 1
                if model_i > max_models:
                    continue
            else:
                print(f"skipping  {make} -  {model}")
        make_i += 1
        if make_i > max_makes:
            break


def test_find_best_cluster():
    df = DetectedCarDataset.get_data_frame(
        '/home/peter/datasets/detected-cars/more_than_4000_detected_per_make/feed.json')

    supra = df[(df['make'] == 'Toyota') & (df['model'] == 'Supra')]

    detected_cars = list(DetectedCar.from_data_frame(supra))
    detected_cars.sort(key=lambda i: i.date_of_first_registration)
    labels = find_best_cluster(detected_cars, 1, 3)
    plt.scatter([detected_car.date_of_first_registration for detected_car in detected_cars],
                [detected_car.price for detected_car in detected_cars],
                c=labels,
                s=50, cmap='viridis')
    plt.show()

