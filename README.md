## TODO List

- [x] basic training 
- [x] model save and load
- [x] all catergory for identification (1250 claases) -- but low acc, about 33%
- [x] learning optimization -- now following paper settings
- [x] speaker verification (basic: siamense network)
- [ ] speaker model improvement
- [ ] Convolution? LSTM?
- [x] shuffle inputs 
- [x] audio read, random clipping for each samples 
- [x] make use of ASVspoof2017 dataset 

## Installation:

* Dependencies:
    * pytorch
        
        http://pytorch.org/
    
    * torch-audio
            
            git clone https://github.com/pytorch/audio.git
            cd audio
            pip install cffi
            python setup.py install
        
    * scipy

* Dataset
    * VoxCeleb

    |                 | dev     | test  |
    |-----------------|---------|-------|
    | # of speakers   | 1,211   | 40    |
    | # of videos     | 21,819  | 677   |
    | # of utterances | 139,124 | 6,255 |

    http://www.robots.ox.ac.uk/~vgg/data/voxceleb/

    * ASVspoof2017

    |                         | dev    | train  | eval   | total  |
    |-------------------------|--------|--------|--------|--------|
    | # of speakers           | 8      | 10     | 24     | 42     |
    | # of replay sessions    | 10     | 6      | 161    | 177    |
    | # of replay configs     | 10     | 3      | 110    | 123    |
    | # of genuine utterances | 760    | 1508   | 1298   | 3566   |
    | # of replay utterances  | 950    | 1508   | 12008  | 14466  |

    https://datashare.is.ed.ac.uk/handle/10283/2778
