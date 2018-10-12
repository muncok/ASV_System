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
		├── README.md
		├── __init__.py
		├── data									# handling the datasets  
		│   ├── __init__.py
		│   ├── data_utils.py
		│   ├── dataloader.py
		│   ├── dataset.py
		│   ├── manage_audio.py
		│   ├── prototypical_batch_sampler.py
		│   └── verification_batch_sampler.py
		├── eval									# speaker verification scoring (Equal Error Rate)  # speaker verification scoring (Equal Error Rate)  
		│   ├── score_utils.py
		│   └── sv_test.py
		├── model
		│   ├── ResNet34.py
		│   ├── __init__.py
		│   ├── auxModels.py
		│   ├── gcnnModel.py
		│   ├── lstmModel.py
		│   ├── model.py
		│   ├── model_utils.py
		│   ├── resNet34Models.py
		│   ├── speechModel.py
		│   └── tdnnModel.py
		├── si_model_eval.py
		├── si_model_train.py						# script for si_model training  
		├── sv_model_test.py
		├── sv_system_tree.txt
		├── train									# speaker identification model training (d-vector)   
		│   ├── __init__.py
		│   ├── angularLoss.py
		│   ├── mixup_train.py
		│   ├── prototypical_loss.py
		│   ├── prototypical_train.py
		│   ├── si_train.py
		│   ├── train_utils.py
		│   └── verification_loss.py
		└── utils									# including an util for parser  
			├── __init__.py
			└── parser.py    

    
* Demos

    ----------
    
    * si_model_train
    
            python si_model_train.py -batch 128  -dataset voxc12_mfcc30 -model tdnn_xvector -loss softmax -inFr 800 -spFr 200 800 -lrs 0.1 -nep 100 -cuda
        
        for more details for arguments Type:
        
            python si_model_train.py -h  

    * si_model_eval (for confirming si validation accuracy)
    
            python si_model_eval.py -batch 128 -dataset voxc12_mfcc30 -model tdnn_xvector -loss softmax -inFr 800 -spFr 800 -cuda
            
    * sv_model_test.py  
       
            python sv_model_test.py -batch 128  -dataset voxc12_mfcc30 -model tdnn_xvector -loss softmax -inFr 800 -spFr 800 -cuda
