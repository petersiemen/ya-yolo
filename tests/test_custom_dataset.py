from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))
COCO_IMAGES_DIR = os.path.join(HERE, '../../../datasets/coco-small/images/val2014')
COCO_ANNOTATIONS_FILE = os.path.join(HERE, '../../../datasets/coco-small/annotations/instances_val2014_10_per_category.json')


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
    dataset = DetectedCarDataset(json_file='/home/peter/datasets/detected-cars/more_than_4000_detected_per_make/train.json',
                                 transforms=image_and_target_transform, batch_size=batch_size)

    data_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size, collate_fn=dataset.collate_fn)
    limit = 10

    for batch_i, (images, ground_truth_boxes, _) in enumerate(data_loader):

        if len(images) != batch_size:
            print("skipping batch {}. Size {} does not equal expected batch size {}".format(batch_i, len(images),
                                                                                            batch_size))
            continue


        plot_batch(None, ground_truth_boxes, images, None)
        if batch_i > limit:
            break


def test_detected_car_make_dataset():
    image_and_target_transform = Compose([
        SquashResize(416),
        CocoToTensor()
    ])

    batch_size = 2
    dataset = DetectedCareMakeDataset(json_file=os.path.join(HERE, 'output/detected-cars-small/feed.json'),
                                      transforms=image_and_target_transform, batch_size=batch_size)

    data_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size, collate_fn=dataset.collate_fn)
    limit = 10

    for batch_i, (images, ground_truth_boxes, _) in enumerate(data_loader):

        if len(images) != batch_size:
            print("skipping batch {}. Size {} does not equal expected batch size {}".format(batch_i, len(images),
                                                                                            batch_size))
            continue


        plot_batch(None, ground_truth_boxes, images, dataset._class_names)
        if batch_i > limit:
            break


def test_coco_dataset():
    batch_size = 3
    image_and_target_transform = Compose([
        SquashResize(416),
        CocoToTensor()
    ])

    dataset = YaYoloCocoDataset(images_dir=COCO_IMAGES_DIR, annotations_file=COCO_ANNOTATIONS_FILE,
                                transforms=image_and_target_transform,
                                batch_size=batch_size)

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    batch_i, (images, ground_truth_boxes, image_paths) = next(enumerate(dataloader))

    plot_batch(None, ground_truth_boxes, images, dataset.class_names)


def test_voc_dataset():
    image_and_target_transform = Compose([
        SquashResize(416),
        CocoToTensor()
    ])
    batch_size = 3
    namesfile = os.path.join(HERE, '../cfg/coco.names')
    class_names = load_class_names(namesfile)

    dataset = YaYoloVocDataset(root_dir='/home/peter/datasets/PascalVOC2012',
                               batch_size=batch_size,
                               transforms=image_and_target_transform,
                               image_set='val',
                               download=True,
                               class_names=class_names)

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    batch_i, (images, ground_truth_boxes, image_paths) = next(enumerate(dataloader))

    plot_batch(None, ground_truth_boxes, images, class_names)


def test_image_net():
    image_and_target_transform = Compose([
        SquashResize(416),
        CocoToTensor()
    ])
    batch_size = 2
    dataset = YaYoloImageNetDataset(root='/home/peter/datasets/ImageNet2012',
                                    batch_size=batch_size,
                                    transforms=image_and_target_transform,
                                    download=True)

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    batch_i, (images, annotations, image_paths) = next(enumerate(dataloader))

    pil_image = to_pil_image(images[0])

    plt.imshow(pil_image)
    plt.show()
    pil_image = to_pil_image(images[1])
    plt.imshow(pil_image)
    plt.show()
