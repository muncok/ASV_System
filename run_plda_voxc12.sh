#!/bin/bash

export KALDI_ROOT=/host/projects/kaldi
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

stage=1

dataset=$1
data_dir=$2
train_dir=$data_dir/plda_train
test_dir=$data_dir/plda_test
scores_dir=$data_dir/scores

if [ $stage -le 1 ]; then

mkdir -p $train_dir
mkdir -p $test_dir

if [ "$dataset" == "voxc" ]; then
    cp dataset/voxceleb12/trials/spk2utt $train_dir
    cp dataset/voxceleb12/trials/utt2spk $train_dir
    trials=dataset/voxceleb12/trials/voxceleb12_sv
elif [ "$dataset" == "gcommand" ]; then
    cp dataset/gcommand/kaldi_files/spk2utt $train_dir
    cp dataset/gcommand/kaldi_files/utt2spk $train_dir
    trials=dataset/gcommand/kaldi_files/gcommand_trial
else
    echo not supported dataset
fi

kaldi_utils/feat2ark.py -key_in $data_dir/si_keys.pkl -embed_in $data_dir/si_embeds.npy -output $train_dir
kaldi_utils/feat2ark.py -key_in $data_dir/sv_keys.pkl -embed_in $data_dir/sv_embeds.npy -output $test_dir

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

fi

if [ $stage -le 4 ]; then
# cosine scoring
kaldi_utils/run.pl $scores_dir/log/cosine_scoring.log \
  cat $trials \| awk '{print $1" "$2}' \| \
  ivector-compute-dot-products - \
    "ark:ivector-subtract-global-mean ${train_dir}/mean.vec ark:${test_dir}/feats.ark ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${train_dir}/mean.vec ark:${test_dir}/feats.ark ark:- | ivector-normalize-length ark:- ark:- |" \
   $scores_dir/cosine_scores || exit 1;

# lda scoring
kaldi_utils/run.pl $scores_dir/log/lda_scoring.log \
  cat $trials \| awk '{print $1" "$2}' \| \
  ivector-compute-dot-products - \
    "ark:ivector-subtract-global-mean ${train_dir}/mean.vec ark:${test_dir}/feats.ark ark:- | transform-vec $train_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${train_dir}/mean.vec ark:${test_dir}/feats.ark ark:- | transform-vec $train_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
   $scores_dir/lda_scores || exit 1;

# plda scoring
mkdir -p $scores_dir/log
kaldi_utils/run.pl $scores_dir/log/plda_scoring.log \
  ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 ${train_dir}/plda - |" \
    "ark:ivector-subtract-global-mean ${train_dir}/mean.vec ark:${test_dir}/feats.ark ark:- | transform-vec $train_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${train_dir}/mean.vec ark:${test_dir}/feats.ark ark:- | transform-vec $train_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$trials' | cut -d\  --fields=1,2 |" $scores_dir/plda_scores || exit 1;

# comput eer and minDCFs
eer=`compute-eer <(kaldi_utils/prepare_for_eer.py $trials $scores_dir/cosine_scores) 2> /dev/null`
mindcf1=`kaldi_utils/compute_min_dcf.py --p-target 0.01 $scores_dir/cosine_scores $trials 2> /dev/null`
mindcf2=`kaldi_utils/compute_min_dcf.py --p-target 0.001 $scores_dir/cosine_scores $trials 2> /dev/null`
echo "=========COSINE========="
echo "EER: $eer%"
echo "minDCF(p-target=0.01): $mindcf1"
echo "minDCF(p-target=0.001): $mindcf2"

# comput eer and minDCFs
eer=`compute-eer <(kaldi_utils/prepare_for_eer.py $trials $scores_dir/lda_scores) 2> /dev/null`
mindcf1=`kaldi_utils/compute_min_dcf.py --p-target 0.01 $scores_dir/lda_scores $trials 2> /dev/null`
mindcf2=`kaldi_utils/compute_min_dcf.py --p-target 0.001 $scores_dir/lda_scores $trials 2> /dev/null`
echo "=========LDA========="
echo "EER: $eer%"
echo "minDCF(p-target=0.01): $mindcf1"
echo "minDCF(p-target=0.001): $mindcf2"

# comput eer and minDCFs
eer=`compute-eer <(kaldi_utils/prepare_for_eer.py $trials $scores_dir/plda_scores) 2> /dev/null`
mindcf1=`kaldi_utils/compute_min_dcf.py --p-target 0.01 $scores_dir/plda_scores $trials 2> /dev/null`
mindcf2=`kaldi_utils/compute_min_dcf.py --p-target 0.001 $scores_dir/plda_scores $trials 2> /dev/null`
echo "=========PLDA========="
echo "EER: $eer%"
echo "minDCF(p-target=0.01): $mindcf1"
echo "minDCF(p-target=0.001): $mindcf2"

fi

if [ $stage -le 5 ]; then
# extract feature after LDA
#copy-vector "ark:ivector-subtract-global-mean ${train_dir}/mean.vec ark:${train_dir}/feats.ark ark:- | transform-vec $train_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    #ark:${train_dir}/lda_feats.ark
copy-vector "ark:ivector-subtract-global-mean ${train_dir}/mean.vec ark:${test_dir}/feats.ark ark:- | transform-vec $train_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ark:${data_dir}/lda_feats.ark
fi
