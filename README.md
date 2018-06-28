* Requirements

    ----------
    
    * Dependencies:
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
            - soon
 
* Directory Structure

    ----------
    
        sv_system  
        ├── train                           # speaker identification model training (d-vector)   
        ├── sv_score                        # speaker verification scoring (Equal Error Rate)  
        ├── data                            # handling the datasets  
        ├── utils                           # including an util for parser  
        ├── si_model_train.py               # script for si_model training  
        ├── sv_score_reddots.py             # script for sv_score on reddots dataset  
        ├── sv_score_voxc.py                # script for sv_score on voxceleb dataset   
        └── README.md                       # this file  
    
* Demos

    ----------
    
    * si_model_train
    
            python si_model_train.py -batch 128  -dataset voxc -model CTdnnModel -loss softmax -inFr 100 -spFr 10 -stFr 3  -lrs 0.1 -nep 300 -version 1 -cuda
        
        for more details for arguments Type:
        
            python si_model_train.py -h  
            
    * sv_score  
       
            python sv_score_voxc.py -model ResNet34 -dataset voxc -input_file models/si_voxc_ResNet34.pt -inFr 400 -spFr 400 -cuda
