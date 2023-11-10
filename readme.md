# implementation for **MHPL (Minimum happy points)**
 

### Prerequisites:
- python == 3.7.3
- pytorch == 1.0.1.post2
- torchvision == 0.2.2
- numpy, scipy, sklearn, PIL, argparse, tqdm

### Dataset:

- Please manually download the datasets [Office](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view), [VisDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) from the official websites, and modify the path of images in each '.txt' under the folder './object/data/'. The source model of office-home and visda can be downloaded in this [Url](https://drive.google.com/drive/folders/1eiJtky4seNApOSYJiGrDywfJbCBp_3sb)


### Training:
	
##### Active Source free domain adaptation (ASFDA) on the Office/ Office-Home dataset
	- Train model on the source domain **A** (**s = 0**), we view the full source data as a test set.
    ```python
    cd object/
    python MHPL_source.py --trte full --da uda --output ckps/source/ --gpu_id 0 --dset office --max_epoch 100 --s 0 --t 1
    ```
	
	- Adaptation to other target domains **D and W**, respectively
    ```python
    python MHPL_target.py --beta 0.3 --ratio 0.05 --alpha 3.0 --da uda --output_src ckps/source/ --output ckps/target/ --gpu_id 0 --dset office --s 0 --t 1  
    ```
   
##### Active Source free domain adaptation (ASFDA) on the VisDA-C dataset
	- Synthetic-to-real 
    ```python
    cd object/
	 python MHPL_source.py --trte full --output ckps/source/ --da uda --gpu_id 0 --dset VISDA-C --net resnet101 --lr 1e-3 --max_epoch 10 --s 0 --t 1
	 python MHPL_target.py --beta 0.3 --ratio 0.05 --alpha 25.0 --da uda --dset VISDA-C --gpu_id 0 --s 0 --t 1 --output_src ckps/source/ --output ckps/target/ --net resnet101 --lr 1e-3
	 ```
	
