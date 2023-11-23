from pathlib import Path


def get_project_root():
    return Path(__file__).parent.parent


def get_data_path():
    raw = (get_project_root() / 'data' / 'raw').absolute()
    if not raw.exists():
        raw.mkdir(parents=True)
    return raw
