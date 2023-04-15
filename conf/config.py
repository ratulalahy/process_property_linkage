from dataclasses import dataclass

@dataclass
class TrainTestParams:
    test_size: float = 0.2
    random_state: int   = 42
    
@dataclass
class LocPaths:
    file_log: str
    dir_data: str
    f_name_data_csv: str    
    
@dataclass
class LinkageConfig:
    train_test_param: TrainTestParams
    loc_paths: LocPaths