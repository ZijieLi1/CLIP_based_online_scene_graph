from setuptools import setup, find_packages

setup(
    name="online_scene_graph",
    version="0.1.0",
    description="Online open-vocabulary object map with find() and planner snapshots",
    packages=find_packages(),
    package_data={"online_scene_graph": ["hydra_configs/**/*", "dataset/dataconfigs/**/*", "*.txt", "*.json"]},
)
