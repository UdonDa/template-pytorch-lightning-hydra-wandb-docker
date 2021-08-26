CUDA_VISIBLE_DEVICES=$1 \
    python main.py gpus=-1 \
        experiment_name=DCGAN_Celeba \
        versions=001 \
        solver=src.solver.gan_from_z \
        dataset=celeba \
        epochs=5 \
        optimizer.betas=[0.5,0.999]
