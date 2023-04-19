from pathlib import Path
from dataclasses import dataclass
import os
from typing import List, Tuple, Optional, Dict

@dataclass
class UtilsCommon:
    @staticmethod
    def get_project_root() -> Path:
        return Path(__file__).parent.parent.parent
    
    
@dataclass
class Features:
    @staticmethod
    def get_all_column_name_map() -> Dict:
        return   {
            'Date Completed / Initials':'initials',  
    '  Print ID #': 'print_id', 
    'Printhead ID #': 'print_head_id',
    'Line Width (um)': 'line_width', 
    'Substrate Thickness (um)': 'substrate_thickness',
    'Cartridge Print Height': 'print_height', 
    'Nozzle Voltage': 'nozzle_voltage',
    'Drop Spacing (um)': 'drop_spacing',
    'Nozzle #': 'nozzle_id',
    'Jetting Frequency (kHz)': 'jetting_freq', 
    'Distance  (um)': 'distance',
    'Time (us)': 'time', 
    'Velocity (m/s) ':'velocity', 
    'Notes': 'notes', 
    'Platen  Temperature': 'platen_temperature',
    'Cartridge Temperature': 'cartridge_temperature',
    'Waveform File': 'waveform_file', 
    'Number of Layers': 'no_of_layers',
    'Ink  Viscosity (cP) (η)': 'ink_visco_cp',
    'Ink  Viscosity (Pa*s) (η)': 'ink_visco_pas',
    'Surface  Tension (dyne/cm) (γ)': 'surface_tension_dyne_cm',
    'Surface Tension (N/m) (γ)': 'surface_tension_n_m',
    'Ink  Density (ρ=g/L)': 'ink _density',
    'Nozzle  Diameter (um)': 'nozzle_diameter',
    'Particle size (nm)' : 'particle_size',
    'z-Number  ( √γρD/ η )': 'z_number',
    'Line Width': 'line_width',
    'Roughness': 'roughness',
    'Overspray': 'overspray',
    'Thickness': 'thickness',
    'conductivity': 'conductivity',
    'Print ID': 'print_id',
    'Print Height': 'print_height',
    'Drop Spacing': 'drop_spacing', 
    }
    
    @staticmethod
    def get_target_columns() -> Optional[Tuple[str, str, str]]:
        """[summary]
        
        Returns:
            List: List of target columns
        """
        return ('line_width', 'roughness', 'overspray')
    
    @staticmethod
    def get_feature_columns() -> List[str]:
        """[summary]
        
        Returns:
            List: List of feature columns
        """
        return ['distance', 'time', 'velocity', 'ink_visco_cp', 'surface_tension_dyne_cm', 'ink _density']
    
    @staticmethod
    def get_nan_features() -> List[str]:
        """_summary_

        Returns:
            List: List of features which have nan values
        """
        return ['platen_temperature', 'cartridge_temperature', 'Unnamed: 26', 'thickness', 'conductivity']
    
    @staticmethod
    def get_id_features() -> List[str]:
        """_summary_

        Returns:
            List: List of features which are ids
        """
        return ['print_id', 'print_head_id', 'nozzle_id']
    
    @staticmethod
    def get_meta_data_features() -> List[str]:
        """_summary_

        Returns:
            List: List of features which are meta data (used for pprinting)
        """
        return ['initials', 'notes', 'waveform_file']
    
    @staticmethod
    def get_const_value_features() -> List[str]:
        """_summary_

        Returns:
            List: List of features which are constant values
        """
        return ['nozzle_diameter', 'jetting_freq', 'substrate_thickness','no_of_layers', 'particle_size']
     
    @staticmethod
    def get_corelated_features() -> List[str]:
        return ['z_number',  'ink_visco_pas', 'surface_tension_n_m', 'nozzle_voltage', 'drop_spacing', 'print_height']
        