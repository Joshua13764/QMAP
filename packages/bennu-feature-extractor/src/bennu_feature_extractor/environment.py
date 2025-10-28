from typing import Dict, List, Set
from bennu_feature_extractor.logger_factory import get_logger
from bennu_feature_extractor.environment_tools.env_cluster_base import EnvCluster
from pathlib import Path
from logging import Logger
from prefect import get_run_logger
import attrs

@attrs.define(frozen=False, cache_hash=True)
class Environment():
    self.clusters : Dict[str, EnvCluster] = {}
    
    @property
    def logger(self) -> Logger:
        return get_run_logger()
    
    def add_cluster(self, cluster : EnvCluster) -> None:
        self.clusters[cluster.name] = cluster
        self.logger.info(f"Added cluster with {len(cluster.files)} files to environment. Total clusters: {len(self.clusters)}.")

    def add_clusters(self, clusters : List[EnvCluster]) -> None:
        for cluster in clusters:
            self.add_cluster(cluster)

    def get_total_size(self) -> int:
        total_size = sum(cluster.get_total_size() for cluster in self.clusters.values())
        self.logger.info(f"Calculated total size of environment: {total_size} bytes across {len(self.clusters)} clusters.")
        return total_size
    
    def delete_all_clusters(self) -> None:
        for cluster in self.clusters.values():
            cluster.delete()
        self.clusters.clear()
        self.logger.info("Deleted all clusters from environment.")

    def check_all_metadata(self) -> bool:
        all_valid = all(cluster.check_metadata() for cluster in self.clusters.values())
        if all_valid:
            self.logger.info("All clusters have valid metadata.")
        else:
            self.logger.warning("Some clusters have invalid metadata.")
        return all_valid
    
    def add_cluster_from_folder(self, folder_path : Path, virtual_path : Path) -> None:
        cluster = EnvCluster.from_folder(folder_path=folder_path, virtual_path=virtual_path)
        self.add_cluster(cluster)
    
    @staticmethod
    def merge_environments(envs : List['Environment']) -> 'Environment':
        merged_env = Environment()
        [merged_env.add_clusters(list(env.clusters.values())) for env in envs]

        merged_env.logger.info(f"Merged {len(envs)} environments into one with {len(merged_env.clusters)} clusters.")
        return merged_env