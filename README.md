* Requirements

    ----------
    
    * Dependencies:
		* Python 3.6
        * pytorch-0.4.0  http://pytorch.org/  
            `pip install torch`
        * librosa  
            `pip install librosa`
        * scipy, sklearn, numpy, pandas...  
            `pip install $package`
   
    * Datasets  
        * VoxCeleb1
            - [homepage](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) 
            - [GoogleDrive](https://drive.google.com/drive/folders/1__Ob2AUuAdzVDRCVhKtKSvGmTKHUKTuR)
        * Reddots   
            - [homepage](https://sites.google.com/site/thereddotsproject/)
        * SpeechCommand
            - [homapage](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html)
        * dataframes for above datasets
            - [GoogleDrive](https://drive.google.com/open?id=17LwA37xbMft4ciqHKh0ntxf-Q79T0vL2)
 
* Directory Structure

    ----------

    
* Demos

    ----------
    * conda environment

           source activate {env_name} (for montreal it is "pytorch") 

    
    * si_model_train
    
           CUDA_VISIBLE_DEVICES=0  python si_model_train.py -batch 128  -dataset voxc12_mfcc30 -model tdnn_xvector -loss softmax -inFr 800 -spFr 200 800 -lrs 0.1 -nep 100 -cuda
            
        CUDA_VISIBLE_DEVICES define the used gpu's number (0~3)

            Montreal's GPU ordering is like below.
            cuda:0 -> P100 (appears as #2 on nvidia-smi)
            cuda:1 -> TITAN X (Pascal) (#0 on nvidia-smi)
            cuda:2 -> TITAN X (Pascal) (#1 on nvidia-smi)
            cuda:3 -> TITAN Xp              (#3 on nvidia-smi)

        for more details for arguments Type:
        
            python si_model_train.py -h  

    * si_model_eval (for confirming si validation accuracy)
    
            python si_model_eval.py -batch 128 -dataset voxc12_mfcc30 -model tdnn_xvector -loss softmax -inFr 800 -spFr 800 -cuda
            
    * sv_model_test.py  
       
            python sv_model_test.py -batch 128  -dataset voxc12_mfcc30 -model tdnn_xvector -loss softmax -inFr 800 -spFr 800 -cuda
