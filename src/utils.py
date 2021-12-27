from pathlib import Path
def get_project_path():
    return Path(__file__).parent.parent

print(get_project_path())