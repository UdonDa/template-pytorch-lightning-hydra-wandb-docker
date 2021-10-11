docker run --gpus all \
     --rm -it --name template_$(($RANDOM % 1000 + 1000)) \
     --shm-size=64g \
     -v $PWD:/src \
     -v /srv/datasets/CelebA/Img/img_align_celeba_png:/src/datasets/celeba \
     template bash
