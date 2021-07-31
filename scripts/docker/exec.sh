docker run --gpus 0,1,2,3,4,5,6,7,8 \
     --rm -it --name template_$(($RANDOM % 1000 + 1000)) \
     --shm-size=64g \
     -v $PWD:/src \
     -v /srv/datasets/CelebA/Img/img_align_celeba_png:/src/datasets/celeba \
     template bash