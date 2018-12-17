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
        .
        ├── README.md
        ├── __init__.py
        ├── core
        ├── data
        │   ├── __init__.py
        │   ├── __pycache__
        │   │   ├── __init__.cpython-36.pyc
        │   │   ├── data_utils.cpython-36.pyc
        │   │   ├── dataloader.cpython-36.pyc
        │   │   ├── dataset.cpython-36.pyc
        │   │   ├── manage_audio.cpython-36.pyc
        │   │   ├── prototypical_batch_sampler.cpython-36.pyc
        │   │   └── verification_batch_sampler.cpython-36.pyc
        │   ├── data_utils.py
        │   ├── dataloader.py
        │   ├── dataset.py
        │   ├── manage_audio.py
        │   ├── prototypical_batch_sampler.py
        │   └── verification_batch_sampler.py
        ├── dataset
        │   ├── gcommand -> /dataset/SV_sets/gcommand/
        │   └── voxceleb12 -> /dataset/SV_sets/voxceleb12/
        ├── embeddings -> /dataset/SV_sets/embeddings/
        ├── eval
        │   ├── __pycache__
        │   │   ├── score_utils.cpython-36.pyc
        │   │   └── sv_test.cpython-36.pyc
        │   ├── score_utils.py
        │   └── sv_test.py
        ├── extract_embeds.py
        ├── extract_embeds_diff_lengths.py
        ├── extract_embeds_sub_uttrs.py
        ├── kaldi_utils
        │   ├── compute_min_dcf.py
        │   ├── feat2ark.py
        │   ├── make_spk2utt.py
        │   ├── prepare_for_eer.py
        │   └── run.pl
        ├── model
        │   ├── ResNet34.py
        │   ├── __init__.py
        │   ├── __pycache__
        │   │   ├── ResNet34.cpython-36.pyc
        │   │   ├── __init__.cpython-36.pyc
        │   │   ├── auxModels.cpython-36.pyc
        │   │   ├── model.cpython-36.pyc
        │   │   ├── model_utils.cpython-36.pyc
        │   │   ├── resNet34Models.cpython-36.pyc
        │   │   ├── speechModel.cpython-36.pyc
        │   │   ├── tdnnModel.cpython-36.pyc
        │   │   └── wide_resnet.cpython-36.pyc
        │   ├── auxModels.py
        │   ├── center_loss.py
        │   ├── gcnnModel.py
        │   ├── lstmModel.py
        │   ├── model.py
        │   ├── model_utils.py
        │   ├── resNet34Models.py
        │   ├── speechModel.py
        │   ├── tdnnModel.py
        │   └── wide_resnet.py
        ├── models
        │   └── voxc2_fbank64_vad
        ├── path.sh
        ├── run_cos.sh
        ├── run_enroll_test_plda.sh
        ├── run_plda.sh
        ├── saved_models -> /dataset/SV_sets/models/
        ├── si_model_eval.py
        ├── si_model_summary.py
        ├── si_model_train.py
        ├── si_model_train_cosLr.py
        ├── si_model_train_length_schedule.py
        ├── sv_model_test.py
        ├── sv_score_reddots.py
        ├── train
        │   ├── __init__.py
        │   ├── __pycache__
        │   │   ├── __init__.cpython-36.pyc
        │   │   ├── angularLoss.cpython-36.pyc
        │   │   ├── si_train.cpython-36.pyc
        │   │   └── train_utils.cpython-36.pyc
        │   ├── angularLoss.py
        │   ├── mixup_train.py
        │   ├── prototypical_loss.py
        │   ├── prototypical_train.py
        │   ├── si_train.py
        │   ├── train_utils.py
        │   └── verification_loss.py
        └── utils
            ├── __init__.py
            ├── __pycache__
            │   ├── __init__.cpython-36.pyc
            │   └── parser.cpython-36.pyc
            └── parser.py

    
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
