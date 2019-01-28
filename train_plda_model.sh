#!/bin/bash
export KALDI_ROOT=/host/projects/kaldi
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

stage=1

 if [ $# != 1 ]; then
   echo "Usage: $0 <data_dir>"
   echo "e.g.: $0 voxc1_fbank64_embeds"
   exit 1;
 fi

data_dir=$1
train_dir=$data_dir/plda_train


if [ $stage -le 0 ]; then
    mkdir -p $train_dir

    # transform npy format to ark format
    kaldi_utils/feat2ark.py -key_in $data_dir/voxc12_keys.pkl -embed_in $data_dir/voxc12_embeds.npy -output $train_dir
    # compute dvector mean for following global mean subtraction
    ivector-mean ark:$train_dir/feats.ark $train_dir/mean.vec || exit 1;
fi

if [ $stage -le 1 ]; then
    # This script uses LDA to decrease the dimensionality prior to PLDA.
    lda_dim=200
    kaldi_utils/run.pl $data_dir/lda_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$train_dir/feats.scp ark:- |" \
    ark:$data_dir/utt2spk $train_dir/transform.mat || exit 1;
fi


if [ $stage -le 2 ]; then
    #compute plda model
    kaldi_utils/run.pl $train_dir/log/plda.log \
        ivector-compute-plda ark:$data_dir/spk2utt \
        "ark:ivector-subtract-global-mean ark:${train_dir}/feats.ark ark:- | transform-vec $train_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        $train_dir/plda || exit 1;
fi
