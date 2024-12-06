# config.py
import os
import yaml
import os.path as osp

def get_config(dir='C:/Users/leejo/OneDrive/바탕 화면/HGNN/config/config.py'):
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return os.path.sep.join(seq)

    def concat(loader, node):
        seq = loader.construct_sequence(node)
        seq = [str(tmp) for tmp in seq]
        return ''.join(seq)

    yaml.add_constructor('!join', join)
    yaml.add_constructor('!concat', concat)
    with open(dir, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    check_dirs(cfg)

    return cfg

def check_dir(folder, mk_dir=True):
    if not osp.exists(folder):
        if mk_dir:
            print(f'making direction {folder}!')
            os.mkdir(folder)
        else:
            raise Exception(f'Not exist direction {folder}')

def check_dirs(cfg):
    check_dir(cfg['data_root'], mk_dir=False)
    check_dir(cfg['result_root'])
    check_dir(cfg['ckpt_folder'])
    check_dir(cfg['result_sub_folder'])

