from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))


def test_simple_car_dataset():
    image_and_target_transform = Compose([
        SquashResize(416),
        CocoToTensor()
    ])

    to_pil_image = transforms.Compose([
        ToPILImage()
    ])

    batch_size = 2
    dataset = SimpleCarDataset(root_dir='/home/peter/datasets/simple_cars/2019-08-23T10-22-54',
                               transforms=image_and_target_transform, batch_size=batch_size)

    data_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)
    limit = 10

    for batch_i, (images, annotations, _) in enumerate(data_loader):

        for b_i in range(batch_size):
            image = images[b_i]
            make = annotations[0]['make'][b_i]
            price = annotations[0]['price'][b_i]
            dofr = annotations[0]['date_of_first_registration'][b_i]

            pil_image = to_pil_image(image)
            plt.imshow(pil_image)
            print(make)
            print(price)
            print(dofr)
            plt.show()

        if batch_i > limit:
            break


def test_detected_car_dataset():
    image_and_target_transform = Compose([
        SquashResize(416),
        CocoToTensor()
    ])

    batch_size = 2
    dataset = DetectedCarDataset(json_file=os.path.join(HERE, 'resources/cars.json'),
                                 transforms=image_and_target_transform, batch_size=batch_size)

    data_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)
    limit = 10

    for batch_i, (images, annotations, _) in enumerate(data_loader):

        if len(images) != batch_size:
            print("skipping batch {}. Size {} does not equal expected batch size {}".format(batch_i, len(images),
                                                                                            batch_size))
            continue

        ground_truth_boxes = dataset.get_ground_truth_boxes(annotations)

        plot_batch(None, ground_truth_boxes, images, None)
        if batch_i > limit:
            break


def test_detected_car_make_dataset():
    image_and_target_transform = Compose([
        SquashResize(416),
        CocoToTensor()
    ])

    batch_size = 2
    dataset = DetectedCareMakeDataset(json_file=os.path.join(HERE, 'resources/cars.json'),
                                      transforms=image_and_target_transform, batch_size=batch_size)

    data_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)
    limit = 10

    for batch_i, (images, annotations, _) in enumerate(data_loader):

        if len(images) != batch_size:
            print("skipping batch {}. Size {} does not equal expected batch size {}".format(batch_i, len(images),
                                                                                            batch_size))
            continue

        ground_truth_boxes = dataset.get_ground_truth_boxes(annotations)

        plot_batch(None, ground_truth_boxes, images, dataset._class_names)
        if batch_i > limit:
            break


def test_voc_dataset():
    image_and_target_transform = Compose([
        SquashResize(416),
        CocoToTensor()
    ])
    batch_size = 2
    dataset = YaYoloVocDataset(root_dir='/home/peter/datasets/VOC',
                               batch_size=batch_size,
                               transforms=image_and_target_transform)

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    batch_i, (images, annotations, image_paths) = next(enumerate(dataloader))
    # ground_truth_boxes = dataset.get_ground_truth_boxes(annotations)

    pil_image = to_pil_image(images[0])

    plt.imshow(pil_image)
    plt.show()
    pil_image = to_pil_image(images[1])
    plt.imshow(pil_image)
    plt.show()
