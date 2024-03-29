import json
import os
from typing import List, Dict, Any


class Config:
    """Container for a loaded configuration.

    Assumes that the main configuration contains a key "path_hypotheses".
    Hypotheses are loaded at initialization time from this path and made accessible
    in the configuration under the key "hypotheses".
    """

    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        with open('paths.json') as fin:
            main_paths = json.load(fin)
        self.config['dataset']['path'] = os.path.join(main_paths['data_dir'], self.config['dataset']['path'])
        self.config['path_out'] = os.path.join(main_paths['output_dir'], self.config['path_out'])
        if 'predictor' in self.config:
            if 'checkpoint' in self.config['predictor'] and self.config['predictor']['checkpoint'] is not None:
                self.config['predictor']['checkpoint'] = os.path.join(main_paths['output_dir'], self.config['predictor']['checkpoint'])
            if not self.config['prediction_pipeline'].get('no_nli') or 'monoling_target_lang' in self.config['path_out'] or 'X' in self.config['path_out']:
                # fix for diff storages
                self.config['path_out'] = self.config['path_out'].replace('scratch0', 'scratch1')
                if 'checkpoint' in self.config['predictor'] and self.config['predictor']['checkpoint'] is not None:
                    self.config['predictor']['checkpoint'] = self.config['predictor']['checkpoint'].replace('scratch0', 'scratch1')
                if not self.config['prediction_pipeline'].get('no_nli'):
                    self.config['path_hypotheses'] = os.path.join(main_paths['data_dir'], self.config['path_hypotheses'])
                    with open(self.config['path_hypotheses']) as fin:
                        self.config['hypotheses'] = json.load(fin)
        elif 'predictors' in self.config:
            for predictor_id, predictor_dict in self.config['predictors'].items():
                if 'checkpoint' in predictor_dict and predictor_dict['checkpoint'] is not None:
                    self.config['predictors'][predictor_id]['checkpoint'] = os.path.join(main_paths['output_dir'], self.config['predictors'][predictor_id]['checkpoint'])
                if not self.config['prediction_pipeline'].get('no_nli') or 'monoling_target_lang' in self.config['path_out']:
                    # fix for diff storages
                    self.config['path_out'] = self.config['path_out'].replace('scratch0', 'scratch1')
                    if 'checkpoint' in predictor_dict and predictor_dict['checkpoint'] is not None:
                        self.config['predictors'][predictor_id]['checkpoint'] = self.config['predictors'][predictor_id]['checkpoint'].replace('scratch0', 'scratch1')
                    if not self.config['prediction_pipeline'].get('no_nli'):
                        self.config['path_hypotheses'] = os.path.join(main_paths['data_dir'], self.config['path_hypotheses'])
                        with open(self.config['path_hypotheses']) as fin:
                            self.config['hypotheses'] = json.load(fin)

    def __getitem__(self, key):
        return self.config[key]

    def get_from_key_list(self, key_list: List[str]) -> Any:
        """Get a sub-config or config item from a key-list."""
        key_list = [key for key in key_list]
        key_list.insert(0, 'hypotheses')
        sub_config = self.config
        for key in key_list:
            sub_config = sub_config[key]
        return sub_config

    def get_hypos_for_section(self, sec_group: str, sec_name: str):
        """Get the hypothesis or multiple hypotheses for a pipeline section."""
        hypotheses_keys = self.config['prediction_pipeline'][sec_group][sec_name]['hypotheses_keys']
        results = self.get_from_key_list(hypotheses_keys)
        return results
