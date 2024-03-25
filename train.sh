python train.py --epochs 300 --batch-size 4 --img-size 640 --cfg ./models/yolov5s-final.yaml --worker 4 --data data/coco128_Charis.yaml --name Charis_final --project my_exps --cache ram
out='Charis'
python train.py --epochs 300 --batch-size 4 --img-size 640 --cfg ./models/yolov5s-final.yaml --worker 4 --data data/coco128_OrCaScore.yaml --name OrCaScore_final --project my_exps --cache ram
out='OrCaScore'
# python train.py --epochs 300 --batch-size 8 --img-size 640 --cfg ./models/yolov5s-bff.yaml --worker 4 --data data/coco128_Charis.yaml --name Charis_bff --project my_exps --cache ram
# out='Charis'
# python train.py --epochs 300 --batch-size 8 --img-size 640 --cfg ./models/yolov5s-bff.yaml --worker 4 --data data/coco128_OrCaScore.yaml --name OrCaScore_bff --project my_exps --cache ram
# out='OrCaScore'
# python train.py --epochs 300 --batch-size 8 --img-size 640 --cfg ./models/yolov5s-qf.yaml --worker 4 --data data/coco128_Charis.yaml --name Charis_qf --project my_exps --cache ram
# out='Charis'
# python train.py --epochs 300 --batch-size 8 --img-size 640 --cfg ./models/yolov5s-qf.yaml --worker 4 --data data/coco128_OrCaScore.yaml --name OrCaScore_qf --project my_exps --cache ram
# out='OrCaScore'
python train.py --epochs 300 --batch-size 4 --img-size 640 --cfg ./models/yolov5s-gold.yaml --worker 4 --data data/coco128_Charis.yaml --name Charis_gold --project my_exps --cache ram
out='Charis'
python train.py --epochs 300 --batch-size 4 --img-size 640 --cfg ./models/yolov5s-gold.yaml --worker 4 --data data/coco128_OrCaScore.yaml --name OrCaScore_gold --project my_exps --cache ram
out='OrCaScore'
# nohup sh train.sh > train.log 2>&1 &
# tail -f train.log