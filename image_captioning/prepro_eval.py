import os
import sys
sys.path.append("coco-caption")
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import argparse


def main(args):
    coco = COCO(args.reference_file_path)
    cocoRes = coco.loadRes(args.candidate_file_path)

    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    for metric, score in cocoEval.eval.items():
        print('%s: %.3f' % (metric, score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--reference_file_path', type=str, default='data/annotations/captions_val2014.json')
    parser.add_argument('--candidate_file_path', type=str, default='')

    args = parser.parse_args()

    main(args)
