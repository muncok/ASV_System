#!/bin/bash

export KALDI_ROOT=/host/projects/kaldi
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C


 if [ $# != 2 ]; then
   echo "Usage: $0 <plda_model_dir> <test_dir>"
   echo "e.g.: $0 voxc1_fbank64_embeds/plda_train voices_fbank64_embeds"
   exit 1;
 fi

model_dir=$1
test_dir=$2
scores_dir=$test_dir/scores

trials=datasets/voices/trials/voices_sv

# transform sv_embeds.npy to sv_embeds.ark
kaldi_utils/feat2ark.py -key_in $test_dir/sv_keys.pkl -embed_in $test_dir/sv_embeds.npy -output $test_dir

## cosine scoring
#kaldi_utils/run.pl $scores_dir/log/cosine_scoring.log \
#  cat $trials \| awk '{print $1" "$2}' \| \
#  ivector-compute-dot-products - \
#    "ark:ivector-subtract-global-mean ${model_dir}/mean.vec ark:${test_dir}/feats.ark ark:- | ivector-normalize-length ark:- ark:- |" \
#    "ark:ivector-subtract-global-mean ${model_dir}/mean.vec ark:${test_dir}/feats.ark ark:- | ivector-normalize-length ark:- ark:- |" \
#   $scores_dir/cosine_scores || exit 1;

## lda scoring
#kaldi_utils/run.pl $scores_dir/log/lda_scoring.log \
#  cat $trials \| awk '{print $1" "$2}' \| \
#  ivector-compute-dot-products - \
#    "ark:ivector-subtract-global-mean ${model_dir}/mean.vec ark:${test_dir}/feats.ark ark:- | transform-vec $model_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
#    "ark:ivector-subtract-global-mean ${model_dir}/mean.vec ark:${test_dir}/feats.ark ark:- | transform-vec $model_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
#   $scores_dir/lda_scores || exit 1;

# plda scoring
mkdir -p $scores_dir/log
kaldi_utils/run.pl $scores_dir/log/plda_scoring.log \
  ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 ${model_dir}/plda - |" \
    "ark:ivector-subtract-global-mean ${model_dir}/mean.vec ark:${test_dir}/feats.ark ark:- | transform-vec $model_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${model_dir}/mean.vec ark:${test_dir}/feats.ark ark:- | transform-vec $model_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$trials' | cut -d\  --fields=1,2 |" $scores_dir/plda_scores || exit 1;

# comput eer and minDCFs
echo ""
echo ""
echo "embeddings: $2"
echo "trials: $trials"

#eer=`compute-eer <(kaldi_utils/prepare_for_eer.py $trials $scores_dir/cosine_scores) 2> /dev/null`
#mindcf1=`kaldi_utils/compute_min_dcf.py --p-target 0.01 $scores_dir/cosine_scores $trials 2> /dev/null`
#mindcf2=`kaldi_utils/compute_min_dcf.py --p-target 0.001 $scores_dir/cosine_scores $trials 2> /dev/null`
#echo "=========COSINE========="
#echo "EER: $eer%"
#echo "minDCF(p-target=0.01): $mindcf1"
#echo "minDCF(p-target=0.001): $mindcf2"

## comput eer and minDCFs
#eer=`compute-eer <(kaldi_utils/prepare_for_eer.py $trials $scores_dir/lda_scores) 2> /dev/null`
#mindcf1=`kaldi_utils/compute_min_dcf.py --p-target 0.01 $scores_dir/lda_scores $trials 2> /dev/null`
#mindcf2=`kaldi_utils/compute_min_dcf.py --p-target 0.001 $scores_dir/lda_scores $trials 2> /dev/null`
#echo "=========LDA========="
#echo "EER: $eer%"
#echo "minDCF(p-target=0.01): $mindcf1"
#echo "minDCF(p-target=0.001): $mindcf2"

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
