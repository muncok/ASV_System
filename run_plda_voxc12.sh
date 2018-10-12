#!/bin/bash

export KALDI_ROOT=/host/projects/kaldi
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

stage=1

data_dir=$1
train_dir=$data_dir/plda_train
test_dir=$data_dir/plda_test
trials=dataset/voxceleb12/trials/voxceleb12_sv
scores_dir=$data_dir/plda_scores

if [ $stage -le 1 ]; then

mkdir -p $train_dir
mkdir -p $test_dir
cp dataset/voxceleb12/trials/spk2utt $train_dir
cp dataset/voxceleb12/trials/utt2spk $train_dir

kaldi_utils/feat2ark.py -key_in $1/si_keys.pkl -embed_in $1/si_embeds.npy -output $train_dir
kaldi_utils/feat2ark.py -key_in $1/sv_keys.pkl -embed_in $1/sv_embeds.npy -output $test_dir

# compute dvector mean for following global mean subtraction
ivector-mean ark:$train_dir/feats.ark $train_dir/mean.vec || exit 1;

fi

if [ $stage -le 2 ]; then

# This script uses LDA to decrease the dimensionality prior to PLDA.
lda_dim=200
kaldi_utils/run.pl $data_dir/lda_train/log/lda.log \
ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
"ark:ivector-subtract-global-mean scp:$train_dir/feats.scp ark:- |" \
ark:$train_dir/utt2spk $train_dir/transform.mat || exit 1;

fi


if [ $stage -le 3 ]; then

#compute plda model
kaldi_utils/run.pl $train_dir/log/plda.log \
    ivector-compute-plda ark:$train_dir/spk2utt \
    "ark:ivector-subtract-global-mean ark:${train_dir}/feats.ark ark:- | transform-vec $train_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $train_dir/plda || exit 1;

#run.pl $train_dir/log/plda.log \
    #ivector-compute-plda ark:$train_dir/spk2utt \
    #"ark:ivector-subtract-global-mean ark:${train_dir}/feats.ark ark:- | ivector-normalize-length ark:- ark:- |" \
    #$train_dir/plda || exit 1;
fi

if [ $stage -le 4 ]; then

# plda scoring
mkdir -p $scores_dir/log
kaldi_utils/run.pl $scores_dir/log/plda_scoring.log \
  ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 ${train_dir}/plda - |" \
    "ark:ivector-subtract-global-mean ${train_dir}/mean.vec ark:${test_dir}/feats.ark ark:- | transform-vec $train_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${train_dir}/mean.vec ark:${test_dir}/feats.ark ark:- | transform-vec $train_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$trials' | cut -d\  --fields=1,2 |" $scores_dir/plda_scores || exit 1;

#run.pl $scores_dir/log/plda_scoring.log \
  #ivector-plda-scoring --normalize-length=true \
    #"ivector-copy-plda --smoothing=0.0 ${train_dir}/plda - |" \
    #"ark:ivector-subtract-global-mean ${train_dir}/mean.vec ark:${test_dir}/feats.ark ark:- | ivector-normalize-length ark:- ark:- |" \
    #"ark:ivector-subtract-global-mean ${train_dir}/mean.vec ark:${test_dir}/feats.ark ark:- | ivector-normalize-length ark:- ark:- |" \
    #"cat '$trials' | cut -d\  --fields=1,2 |" $scores_dir/plda_scores || exit 1;

# comput eer and minDCFs
eer=`compute-eer <(kaldi_utils/prepare_for_eer.py $trials $scores_dir/plda_scores) 2> /dev/null`
mindcf1=`kaldi_utils/compute_min_dcf.py --p-target 0.01 $scores_dir/plda_scores $trials 2> /dev/null`
mindcf2=`kaldi_utils/compute_min_dcf.py --p-target 0.001 $scores_dir/plda_scores $trials 2> /dev/null`
echo "EER: $eer%"
echo "minDCF(p-target=0.01): $mindcf1"
echo "minDCF(p-target=0.001): $mindcf2"

fi
