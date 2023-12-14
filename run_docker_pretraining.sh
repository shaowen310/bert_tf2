docker build -t berttrain .

if [[ "$(docker ps -a -q -f name=berttrainins 2>/dev/null)" == "" ]]; then
  docker container run -dit \
    --name berttrainins \
    --mount type=bind,source="$(pwd)",target=/workspace/code \
    --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    berttrain:latest
fi

docker exec -it berttrainins bash run_pretraining.sh
