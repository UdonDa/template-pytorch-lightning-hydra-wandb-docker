import hydra
import importlib
from omegaconf import DictConfig

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig) -> None:

    exp = getattr(
            importlib.import_module(config.solver), 'Solver')(
                config
            )

    exp.run()


if __name__ == "__main__":
    main()
