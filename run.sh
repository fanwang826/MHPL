# Table 2        A->C,P,R
~/anaconda3/envs/pytorch/bin/python MHPL_source.py --trte full --output ckps/source/ --da uda --gpu_id 0 --dset office-home --max_epoch 50 --s 0 --t 1
~/anaconda3/envs/pytorch/bin/python MHPL_target.py --beta 0.3 --ratio 0.05 --alpha 3.0 --KK 20 --da uda --dset office-home --gpu_id 0 --s 0 --t 1 --output_src ckps/source/ --output ckps/target/

# Table 3        A->D,W
~/anaconda3/envs/pytorch/bin/python MHPL_source.py --trte full --output ckps/source/ --da uda --gpu_id 0 --dset office --max_epoch 100 --s 0 --t 1
~/anaconda3/envs/pytorch/bin/python MHPL_target.py --beta 0.3 --ratio 0.05 --alpha 3.0 --KK 9 --da uda --dset office --gpu_id 0 --s 0 --t 1 --output_src ckps/source/ --output ckps/target/

# Table 4       VisDA-C
~/anaconda3/envs/pytorch/bin/python MHPL_source.py --trte full --output ckps/source/ --da uda --gpu_id 0 --dset VISDA-C --net resnet101 --lr 1e-3 --max_epoch 10 --s 0 --t 1
~/anaconda3/envs/pytorch/bin/python MHPL_target.py --beta 0.3 --ratio 0.05 --alpha 25.0 --KK 5 --da uda --dset VISDA-C --gpu_id 0 --s 0 --t 1 --output_src ckps/source/ --output ckps/target/ --net resnet101 --lr 1e-3


# implementation details Table 1        A->C,P,R on Alexnet and Vgg
~/anaconda3/envs/pytorch/bin/python MHPL_source.py --trte full --output ckps/source/ --da uda --gpu_id 0 --dset office-home --net ale --max_epoch 50 --s 0 --t 1
~/anaconda3/envs/pytorch/bin/python MHPL_target.py --beta 0.3 --ratio 0.05 --alpha 3.0 --KK 20 --da uda --dset office-home --net ale --gpu_id 0 --s 0 --t 1 --output_src ckps/source/ --output ckps/target/
~/anaconda3/envs/pytorch/bin/python MHPL_source.py --trte full --output ckps/source/ --da uda --gpu_id 0 --dset office-home --net vgg16 --max_epoch 50 --s 0 --t 1
~/anaconda3/envs/pytorch/bin/python MHPL_target.py --beta 0.3 --ratio 0.05 --alpha 3.0 --KK 20 --da uda --dset office-home --net vgg16 --gpu_id 0 --s 0 --t 1 --output_src ckps/source/ --output ckps/target/

# implementation details Table 1        A->D,W on Alexnet and Vgg
~/anaconda3/envs/pytorch/bin/python MHPL_source.py --trte full --output ckps/source/ --da uda --gpu_id 0 --dset office --net ale  --max_epoch 100 --s 0 --t 1
~/anaconda3/envs/pytorch/bin/python MHPL_target.py --beta 0.3 --ratio 0.05 --alpha 3.0 --KK 9 --da uda --dset office --net ale --gpu_id 0 --s 0 --t 1 --output_src ckps/source/ --output ckps/target/
~/anaconda3/envs/pytorch/bin/python MHPL_source.py --trte full --output ckps/source/ --da uda --gpu_id 0 --dset office --net vgg16  --max_epoch 100 --s 0 --t 1
~/anaconda3/envs/pytorch/bin/python MHPL_target.py --beta 0.3 --ratio 0.05 --alpha 3.0 --KK 9 --da uda --dset office --net vgg16 --gpu_id 0 --s 0 --t 1 --output_src ckps/source/ --output ckps/target/
