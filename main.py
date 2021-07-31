import hydra
import importlib
from omegaconf import DictConfig
# from src.solver.experiment import Experiment
from hydra.utils import instantiate


from src.dataset.data_module import DataModule

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig) -> None:

    exp = getattr(
            importlib.import_module(config.solver), 'Solver')(
                config
            )

    # exp.run()


if __name__ == "__main__":
    main()
