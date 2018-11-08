#!/bin/bash

export KALDI_ROOT=/host/projects/kaldi
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

stage=2

dataset=$1
data_dir=$2
train_dir=$data_dir/plda_train
test_dir=$data_dir/plda_test
scores_dir=$data_dir/scores


if [ "$dataset" == "voxc1" ] || "$dataset" == "voxc12" ]; then
    cp dataset/voxceleb12/kaldi_files/spk2utt $data_dir
    cp dataset/voxceleb12/kaldi_files/utt2spk $data_dir
    trials=dataset/voxceleb12/trials/voxceleb12_sv
elif [ "$dataset" == "voxc2" ]; then
    cp dataset/voxceleb12/kaldi_files/spk2utt $data_dir
    cp dataset/voxceleb12/kaldi_files/utt2spk $data_dir
    trials=dataset/voxceleb12/trials/voxceleb12_sv
elif [ "$dataset" == "gcommand" ]; then
    cp dataset/gcommand/kaldi_files/spk2utt $data_dir
    cp dataset/gcommand/kaldi_files/utt2spk $data_dir
    trials=dataset/gcommand/kaldi_files/gcommand_30spks_sv
else
    echo not supported dataset
    exit 1;

fi

if [ $stage -le 1 ]; then

mkdir -p $train_dir
mkdir -p $test_dir

kaldi_utils/feat2ark.py -key_in $data_dir/si_keys.pkl -embed_in $data_dir/si_embeds.npy -output $train_dir
kaldi_utils/feat2ark.py -key_in $data_dir/sv_keys.pkl -embed_in $data_dir/sv_embeds.npy -output $test_dir

# compute dvector mean for following global mean subtraction
ivector-mean ark:$train_dir/feats.ark $train_dir/mean.vec || exit 1;

fi



if [ $stage -le 2 ]; then

# cosine scoring
kaldi_utils/run.pl $scores_dir/log/cosine_scoring.log \
  cat $trials \| awk '{print $1" "$2}' \| \
  ivector-compute-dot-products - \
    "ark:ivector-subtract-global-mean ${train_dir}/mean.vec ark:${test_dir}/feats.ark ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${train_dir}/mean.vec ark:${test_dir}/feats.ark ark:- | ivector-normalize-length ark:- ark:- |" \
   $scores_dir/cosine_scores || exit 1;


# comput eer and minDCFs
echo ""
echo ""
echo "embeddings: $2"
echo "trials: $trials"
eer=`compute-eer <(kaldi_utils/prepare_for_eer.py $trials $scores_dir/cosine_scores) 2> /dev/null`
mindcf1=`kaldi_utils/compute_min_dcf.py --p-target 0.01 $scores_dir/cosine_scores $trials 2> /dev/null`
mindcf2=`kaldi_utils/compute_min_dcf.py --p-target 0.001 $scores_dir/cosine_scores $trials 2> /dev/null`
echo "=========COSINE========="
echo "EER: $eer%"
echo "minDCF(p-target=0.01): $mindcf1"
echo "minDCF(p-target=0.001): $mindcf2"
fi
