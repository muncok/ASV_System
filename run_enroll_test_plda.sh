#!/bin/bash

export KALDI_ROOT=/host/projects/kaldi
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

stage=$3

dataset=$1
data_dir=$2
out_dir=$2/kaldi
enroll_dir=$out_dir/enroll/
test_dir=$out_dir/test/
scores_dir=$out_dir/scores


if [ "$dataset" == "voxc1" ] || "$dataset" == "voxc12" ]; then
    spk2utt=dataset/voxceleb12/kaldi_files/spk2utt
    utt2spk=dataset/voxceleb12/kaldi_files/utt2spk
    trials=dataset/voxceleb12/trials/voxceleb12_sv
else
    echo not supported dataset
    exit 1;

fi

if [ $stage -le 1 ]; then

mkdir -p $enroll_dir/si_feat/
mkdir -p $enroll_dir/sv_feat/
mkdir -p $test_dir/sv_feat/

kaldi_utils/feat2ark.py -key_in $data_dir/si_keys.pkl -embed_in $data_dir/si_embeds.npy -output $enroll_dir/si_feat
kaldi_utils/feat2ark.py -key_in $data_dir/sv_keys.pkl -embed_in $data_dir/sv_enroll_embeds.npy -output $enroll_dir/sv_feat
kaldi_utils/feat2ark.py -key_in $data_dir/sv_keys.pkl -embed_in $data_dir/sv_test_embeds.npy -output $test_dir/sv_feat

# compute dvector mean for following global mean subtraction
ivector-mean ark:$enroll_dir/si_feat/feats.ark $out_dir/mean.vec || exit 1;

fi

if [ $stage -le 2 ]; then

# This script uses LDA to decrease the dimensionality prior to PLDA.
lda_dim=200
kaldi_utils/run.pl $out_dir/lda_train/log/lda.log \
ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
"ark:ivector-subtract-global-mean scp:$enroll_dir/si_feat/feats.scp ark:- |" \
ark:$utt2spk $out_dir/transform.mat || exit 1;

fi


if [ $stage -le 3 ]; then

#compute plda model
kaldi_utils/run.pl $dout_dir/log/plda.log \
    ivector-compute-plda ark:$spk2utt \
    "ark:ivector-subtract-global-mean ark:$enroll_dir/si_feat/feats.ark ark:- | transform-vec $out_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $out_dir/plda || exit 1;

fi

if [ $stage -le 4 ]; then

# cosine scoring
kaldi_utils/run.pl $scores_dir/log/cosine_scoring.log \
  cat $trials \| awk '{print $1" "$2}' \| \
  ivector-compute-dot-products - \
    "ark:ivector-subtract-global-mean $out_dir/mean.vec ark:$enroll_dir/sv_feat/feats.ark ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $out_dir/mean.vec ark:$test_dir/sv_feat/feats.ark ark:- | ivector-normalize-length ark:- ark:- |" \
   $scores_dir/cosine_scores || exit 1;

# lda scoring
kaldi_utils/run.pl $scores_dir/log/lda_scoring.log \
  cat $trials \| awk '{print $1" "$2}' \| \
  ivector-compute-dot-products - \
    "ark:ivector-subtract-global-mean $out_dir/mean.vec ark:$enroll_dir/sv_feat/feats.ark ark:- | transform-vec $out_dir/transform.mat ark:- ark:-  | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $out_dir/mean.vec ark:$test_dir/sv_feat/feats.ark ark:- | transform-vec $out_dir/transform.mat ark:- ark:-  | ivector-normalize-length ark:- ark:- |" \
   $scores_dir/lda_scores || exit 1;

# plda scoring
mkdir -p $scores_dir/log
kaldi_utils/run.pl $scores_dir/log/plda_scoring.log \
  ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $out_dir/plda - |" \
    "ark:ivector-subtract-global-mean $out_dir/mean.vec ark:$enroll_dir/sv_feat/feats.ark ark:- | transform-vec $out_dir/transform.mat ark:- ark:-  | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $out_dir/mean.vec ark:$test_dir/sv_feat/feats.ark ark:- | transform-vec $out_dir/transform.mat ark:- ark:-  | ivector-normalize-length ark:- ark:- |" \
    "cat '$trials' | cut -d\  --fields=1,2 |" $scores_dir/plda_scores || exit 1;

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
echo ""
echo ""

fi

#if [ $stage -le 5 ]; then
 ##extract feature after LDA
##copy-vector "ark:ivector-subtract-global-mean ${train_dir}/mean.vec ark:${train_dir}/feats.ark ark:- | transform-vec $train_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ##ark:${train_dir}/lda_feats.ark
#copy-vector "ark:ivector-subtract-global-mean ${train_dir}/mean.vec ark:${test_dir}/feats.ark ark:- | transform-vec $train_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    #ark:${data_dir}/lda_feats.ark
#fi
