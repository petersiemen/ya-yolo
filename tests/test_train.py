from torch.utils.tensorboard import SummaryWriter


from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))
COCO_IMAGES_DIR = os.path.join(HERE, '../../../datasets/coco-small/images/val2014')
COCO_ANNOTATIONS_FILE = os.path.join(HERE, '../../../datasets/coco-small/annotations/instances_val2014_10_per_category.json')



def test_get_indices_for_center_of_ground_truth_bounding_boxes__for_no_annotations():
    ground_truth_boxes = torch.tensor([[], []])
    grid_sizes = [13, 26, 52]
    indices = get_indices_for_center_of_ground_truth_bounding_boxes(ground_truth_boxes, grid_sizes)
    assert indices.shape == (2, 0)


def test_training():
    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
    namesfile = os.path.join(HERE, '../cfg/coco.names')
    model_dir = os.path.join(HERE, 'models')
    class_names = load_class_names(namesfile)

    batch_size = 2
    lr = 0.001
    model = Yolo(cfg_file=cfg_file, class_names=class_names, batch_size=batch_size)
    model.load_weights(weight_file)

    image_and_target_transform = Compose([
        SquashResize(416),
        ToTensor()
    ])

    dataset = YaYoloCocoDataset(images_dir=COCO_IMAGES_DIR, annotations_file=COCO_ANNOTATIONS_FILE,
                                transforms=image_and_target_transform,
                                batch_size=batch_size)

    summary_writer = SummaryWriter(comment=f' batch_size={batch_size} lr={lr}')

    train(model=model,
          dataset=dataset,
          model_dir=model_dir,
          summary_writer=summary_writer,
          epochs=1,
          lr=lr,
          conf_thres=0.9,
          nms_thres=0.5,
          iou_thres=0.5,
          lambda_coord=5,
          lambda_no_obj=0.5,
          gradient_accumulations=1,
          clip_gradients=True,
          limit=3,
          debug=True,
          print_every=10,
          save_every=1)

    summary_writer.close()


def test_training_car_makes():
    image_and_target_transform = Compose([
        SquashResize(416),
        ToTensor()
    ])
    batch_size = 2
    dataset = DetectedCareMakeDataset(
        json_file=os.path.join(os.environ['HOME'], 'datasets/detected-cars/more_than_4000_detected_per_make/train.json'),
        transforms=image_and_target_transform, batch_size=batch_size)

    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    #weight_file = os.path.join(HERE, '../cfg/yolov3.weights')

    class_names = load_class_names(os.path.join(os.environ['HOME'], 'datasets/detected-cars/more_than_4000_detected_per_make/makes.csv'))
    model_dir = os.path.join(HERE, 'models')

    lr = 0.001
    model = Yolo(cfg_file=cfg_file, class_names=class_names, batch_size=batch_size)
    #model.load_weights(weight_file)
    model.load_state_dict(
        torch.load(os.path.join(HERE, '../models/yolo__num_classes_80__epoch_2_batch_7500.pt'), map_location=DEVICE))


    summary_writer = SummaryWriter(comment=f' batch_size={batch_size} lr={lr}')
    train(model=model,
          dataset=dataset,
          model_dir=model_dir,
          summary_writer=summary_writer,
          epochs=1,
          lr=lr,
          conf_thres=0.9,
          nms_thres=0.7,
          iou_thres=0.5,
          lambda_coord=5,
          lambda_no_obj=0.5,
          gradient_accumulations=1,
          limit=3,
          debug=True,
          print_every=10,
          save_every=1,
          log_to_neptune=False)
