import hydra
from hydra.core.config_store import ConfigStore

from conf.config import LinkageConfig


conf_store = ConfigStore.instance()
conf_store.store(name='linkage_config', node= LinkageConfig)

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: LinkageConfig):
    """_summary_
    """
    print(cfg.loc_paths)
    
    

if __name__ == "__main__":
    main()