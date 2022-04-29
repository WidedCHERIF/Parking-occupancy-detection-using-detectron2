# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import _init_paths
import sys
import argparse
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

def parse_args():

    parser = argparse.ArgumentParser(description='Train and test Detectron2')
    parser.add_argument('--mode', dest='mode',
                      help='choose to train or to test',
                      default='train', type=str)
    parser.add_argument('--lr', dest='learning_rate', default= '0.0125', type=int)
    parser.add_argument('--it', dest='max_iteration', default= '1500', type=int)
    parser.add_argument('--workers', dest='number_of_workers', default= '2', type=int)
    parser.add_argument('--ims_per_batch', dest='number_of_workers', default= '4', type=int)
    parser.add_argument('--eval_period', dest='evaluation_period', default= '500', type=int)
    parser.add_argument('--batch_size', dest='batch_size', default= '256', type=int)
    args = parser.parse_args()
    return args

def register_data():
    register_coco_instances("PKLot_train", {}, "/Detection-using-Detectron2/train/_annotations.coco.json", "/Detection-using-Detectron2/train")
    register_coco_instances("PKLot_test", {}, "/Detection-using-Detectron2/valid/_annotations.coco.json", "/Detection-using-Detectron2/valid")
    register_coco_instances("PKLot_valid", {}, "/Detection-using-Detectron2/test/_annotations.coco.json", "/Detection-using-Detectron2/test")

def visualize_data():
    dataset_dicts = DatasetCatalog.get("PKLot_train")
    for d in random.sample(dataset_dicts, 3):
        print(d["annotations"])
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("PKLot_train"), scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2_imshow(out.get_image()[:, :, ::-1])

def config():

    args = parse_args()
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("PKLot_train",)
    cfg.DATASETS.TEST = ("PKLot_valid",)
    cfg.DATALOADER.NUM_WORKERS = args.number_of_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.BASE_LR = args.learning_rate # pick a good LR
    cfg.SOLVER.MAX_ITER = args.max_iteration    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch_size  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 
    cfg.TEST.EVAL_PERIOD = args.evaluation_period
    return cfg

def train():
    cfg = config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

def test():
    cfg = config()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    test_set = DatasetCatalog.get("PKLot_test")
    for d in random.sample(test_set,7):
        im = cv2.imread(d["file_name"])
        outputs = predictor(
            im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=MetadataCatalog.get("PKLot_test"),
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)
        plt.imsave(os.path.join(os.path.join(cfg.OUTPUT_DIR, 'visualization'), str(d["image_id"]) + '.png'), img)

if __name__ == '__main__':
    args = parse_args()
    register_data()
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()


