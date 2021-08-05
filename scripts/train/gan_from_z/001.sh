python main.py gpus=[3] \
    experiment_name=DCGAN_Celeba \
    versions=001 \
    solver=src.solver.gan_from_z \
    dataset=celeba \
    epochs=5