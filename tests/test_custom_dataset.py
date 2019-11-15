from .context import *


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

            pil_image = to_pil_image(image)
            plt.imshow(pil_image)
            print(make)
            plt.show()

        if batch_i > limit:
            break
