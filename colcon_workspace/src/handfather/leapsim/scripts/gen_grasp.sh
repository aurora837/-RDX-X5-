
# --------------------------------------------------------

SCALE=$1
GRASP_CACHE=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

python3 train.py task=LeapHandGrasp task.env.baseObjScale=$SCALE \
task.env.grasp_cache_name=$GRASP_CACHE test=true pipeline=cpu test=true train.params.config.player.games_num=10000000000000000 task.env.episodeLength=50 \
${EXTRA_ARGS}
