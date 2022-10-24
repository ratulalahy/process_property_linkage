from dataclasses import dataclass

@dataclass
class TrainTestParams:
    test_size: float
    random_state: int
    
@dataclass
class LocPaths:
    file_log: str
    dir_data: str
    f_name_data_csv: str    
    
@dataclass
class LinkageConfig:
    train_test_param: TrainTestParams
    loc_paths: LocPaths