"""Data loading workflow step for loading and preparing datasets."""

import asyncio
from typing import Any

from benchmark.core.logging import get_logger
from benchmark.interfaces.orchestration_interfaces import WorkflowContext, WorkflowStep


class DataLoadingStep(WorkflowStep):
    """Workflow step for loading and preparing datasets."""

    def __init__(self) -> None:
        self.logger = get_logger("data_loading_step")

    async def execute(self, context: WorkflowContext) -> dict[str, Any]:
        """Execute data loading step with parallel loading and error handling."""
        self.logger.info("Starting data loading step for experiment %s", context.experiment_id)

        data_service = context.services.get("data")
        if not data_service:
            raise Exception("Data service not available")

        datasets_config = context.config.get("datasets", [])
        if not datasets_config:
            self.logger.info("No datasets specified, skipping data loading")
            return {"datasets_loaded": 0, "datasets": [], "loading_results": {}}

        loaded_datasets = {}
        loading_results = {}
        dataset_info = []

        # Load datasets with concurrency control
        semaphore = asyncio.Semaphore(3)  # Limit concurrent dataset loading

        async def load_single_dataset(dataset_config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            """Load a single dataset with error handling."""
            async with semaphore:
                dataset_id = dataset_config.get("id")
                if not dataset_id:
                    raise Exception("Dataset configuration missing 'id' field")

                try:
                    self.logger.info("Loading dataset: %s", dataset_id)

                    # Load dataset using data service
                    load_response = await data_service.load_dataset(
                        dataset_id=dataset_id,
                        loader_type=dataset_config.get("loader_type", "local"),
                        config=dataset_config.get("config", {}),
                    )

                    if not load_response.success:
                        raise Exception(
                            f"Failed to load dataset {dataset_id}: {load_response.error}"
                        )

                    dataset = load_response.data
                    loaded_datasets[dataset_id] = dataset

                    # Get dataset information
                    try:
                        info_response = await data_service.get_dataset_info(dataset_id)
                        dataset_metadata = info_response.data if info_response.success else {}
                    except Exception as e:
                        self.logger.warning("Failed to get dataset info for %s: %s", dataset_id, e)
                        dataset_metadata = {}

                    # Extract dataset statistics
                    samples_count = dataset_metadata.get(
                        "samples_count", len(dataset.get("samples", []))
                    )
                    data_splits = dataset_metadata.get("splits", {})
                    features = dataset_metadata.get("features", [])

                    loading_results[dataset_id] = {
                        "status": "success",
                        "samples_count": samples_count,
                        "data_splits": data_splits,
                        "features": features,
                        "preprocessing_applied": dataset_config.get("preprocessing", []),
                        "loader_type": dataset_config.get("loader_type", "local"),
                    }

                    dataset_info.append(
                        {
                            "id": dataset_id,
                            "size": samples_count,
                            "features": features,
                            "loader_type": dataset_config.get("loader_type"),
                            "splits": data_splits,
                        }
                    )

                    self.logger.info(
                        "Successfully loaded dataset: %s (%d samples)", dataset_id, samples_count
                    )
                    return dataset_id, dataset

                except Exception as e:
                    error_msg = str(e)
                    self.logger.error("Failed to load dataset %s: %s", dataset_id, error_msg)

                    loading_results[dataset_id] = {
                        "status": "failed",
                        "error": error_msg,
                        "loader_type": dataset_config.get("loader_type", "local"),
                    }

                    raise Exception(f"Data loading failed for dataset {dataset_id}: {e}") from e

        # Execute dataset loading tasks
        loading_tasks = [load_single_dataset(ds_config) for ds_config in datasets_config]

        try:
            # Wait for all datasets to load
            dataset_results = await asyncio.gather(*loading_tasks, return_exceptions=True)

            # Process results and handle partial failures
            successful_loads = 0
            failed_loads = 0

            for result in dataset_results:
                if isinstance(result, Exception):
                    failed_loads += 1
                    self.logger.error("Dataset loading failed: %s", result)
                else:
                    successful_loads += 1

            if successful_loads == 0:
                raise Exception("All dataset loading attempts failed")

            # Store loaded datasets in context resources
            context.resources["loaded_datasets"] = loaded_datasets

            # Calculate total samples across all loaded datasets
            total_samples = sum(info.get("size", 0) for info in dataset_info)

            # Generate loading summary
            result = {  # type: ignore
                "datasets_loaded": len(loaded_datasets),
                "datasets": dataset_info,
                "loading_results": loading_results,
                "total_samples": total_samples,
                "successful_loads": successful_loads,
                "failed_loads": failed_loads,
                "dataset_types": list(
                    {
                        res.get("loader_type", "unknown")
                        for res in loading_results.values()
                        if res["status"] == "success"
                    }
                ),
            }

            self.logger.info(
                "Data loading step completed: %d datasets loaded, %d failed, %d total samples",
                successful_loads,
                failed_loads,
                total_samples,
            )

            return result  # type: ignore

        except Exception as e:
            # Cleanup any partially loaded datasets
            cleanup_tasks = []
            for dataset_id in loaded_datasets:
                try:
                    cleanup_task = data_service.cleanup_dataset(dataset_id)
                    cleanup_tasks.append(cleanup_task)
                except Exception as cleanup_error:
                    self.logger.warning(
                        "Failed to cleanup dataset %s: %s", dataset_id, cleanup_error
                    )

            if cleanup_tasks:
                try:
                    await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                    self.logger.info("Cleaned up %d datasets after failure", len(cleanup_tasks))
                except Exception as cleanup_error:
                    self.logger.warning("Error during dataset cleanup: %s", cleanup_error)

            self.logger.error("Data loading step failed: %s", e)
            raise

    def get_step_name(self) -> str:
        """Get step name."""
        return "data_loading"

    def get_dependencies(self) -> list[str]:
        """Get required service dependencies."""
        return ["data"]

    def get_estimated_duration_seconds(self) -> float:
        """Estimate step duration."""
        return 180.0  # 3 minutes for data loading with multiple datasets
