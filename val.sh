python val.py --weights /root/my/v5-medical/my_exps/Charis_final/weights/best.pt --batch-size 8 --data data/coco128_Charis.yaml  
out='Charis'
python val.py --weights /root/my/v5-medical/my_exps/OrCaScore_final/weights/best.pt --batch-size 8  --data data/coco128_OrCaScore.yaml 
out='OrCaScore'
python val.py --weights /root/my/v5-medical/my_exps/Charis_gold/weights/best.pt --batch-size 8 --data data/coco128_Charis.yaml 
out='Charis'
python val.py --weights /root/my/v5-medical/my_exps/OrCaScore_gold/weights/best.pt --batch-size 8 --data data/coco128_OrCaScore.yaml 
out='OrCaScore'