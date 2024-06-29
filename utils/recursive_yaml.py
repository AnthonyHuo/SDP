import yaml, os
from yamlinclude import YamlIncludeConstructor
fpath = os.path.dirname(os.path.dirname(__file__))
Path = lambda p:os.path.join(fpath,p)
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)
def read_yaml(path):
    p = Path(path)
    with open(p) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data
def write_yaml(data, path):
    p = Path(path)
    with open(p, 'w') as f:
        yaml.dump(data, f) 

if __name__ == '__main__':
    path = 'config/base.yaml'
    t = read_yaml(path)
    # write to yaml
    with open('config/tmp/full.yaml', 'w') as f:
        yaml.dump(t, f)