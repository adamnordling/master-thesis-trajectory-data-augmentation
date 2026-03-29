import datetime
import os
import platform
import time
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None


class PerformanceTracker:
    """Tracks execution time across the pipeline.
    Categorizes timings into Global steps and Dataset-specific missions.
    """

    def __init__(self) -> None:
        """Initializes the performance tracker."""
        self.timings: dict[str, float] = {}
        self.data: dict[str, Any] = {"start_time_utc": datetime.datetime.utcnow().isoformat()}
        self.start_times: dict[str, float] = {}
        self.start("total_run")

    def start(self, key: str) -> None:
        """Starts a timer for a given key. Use descriptive keys like 'prep', 'extraction', 'training_datasetA', etc."""
        self.start_times[key] = time.time()

    def stop(self, key: str) -> None:
        """Stops the timer for a given key and records the duration. If the key was not started, it will be ignored."""
        if key in self.start_times:
            duration = time.time() - self.start_times[key]
            self.timings[key] = duration
            del self.start_times[key]

    def set_info(self, key: str, value: Any) -> None:
        """Sets additional information about the run, such as number of workers used, dataset names, etc. This is for contextual data that doesn't fit into timings."""
        self.data[key] = value

    def _gather_system_info(self) -> None:
        self.data["system_platform"] = platform.system()
        self.data["processor_architecture"] = platform.machine()
        if psutil:
            self.data["physical_cores"] = psutil.cpu_count(logical=False)
            self.data["logical_cores"] = psutil.cpu_count(logical=True)
            self.data["total_ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
        else:
            self.data["physical_cores"] = "N/A"
            self.data["logical_cores"] = os.cpu_count()
            self.data["total_ram_gb"] = "N/A"

    def generate_report(self, dataset_names: list[str]) -> None:
        """Generates a grouped performance report.
        Groups timings by dataset for a clean mission-based audit.
        """
        self.stop("total_run")
        self._gather_system_info()

        output_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "output", "performance")
        )
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_prefix = "-".join(sorted(dataset_names)) if len(dataset_names) > 1 else dataset_names[0]
        report_path = os.path.join(output_dir, f"{timestamp}_{report_prefix}_performance_audit.txt")

        report_lines = [
            "=" * 80,
            "PIPELINE PERFORMANCE AUDIT",
            "=" * 80,
            f"Audit Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Duration:  {self.timings.get('total_run', 0.0):.2f} seconds",
            "-" * 80,
            "SYSTEM ENVIRONMENT:",
            f"  - OS: {self.data.get('system_platform')} | Cores: {self.data.get('logical_cores')} | RAM: {self.data.get('total_ram_gb')} GB",
            f"  - Workers: {self.data.get('parallel_workers_used')}",
            "=" * 80,
            "",
        ]

        # --- SECTION 1: GLOBAL STEPS (Prep/Extraction) ---
        report_lines.append("GLOBAL PIPELINE STEPS (Shared Across All Datasets)")
        report_lines.append("-" * 50)

        global_keys = [k for k in self.timings.keys() if not any(ds in k for ds in dataset_names) and k != "total_run"]
        if global_keys:
            for k in sorted(global_keys):
                name = k.replace("_", " ").title()
                report_lines.append(f"  {name:<45}: {self.timings[k]:>10.2f}s")
        else:
            report_lines.append("  No global steps recorded.")
        report_lines.append("")

        # --- SECTION 2: DATASET SPECIFIC MISSIONS ---
        for ds in dataset_names:
            report_lines.append(f"DATASET MISSION: {ds.upper()}")
            report_lines.append("-" * 50)

            # Find all keys that belong to this dataset (mission logic)
            ds_keys = [k for k in self.timings.keys() if f"_{ds}" in k]

            if ds_keys:
                for k in sorted(ds_keys):
                    # Clean the name: Remove the dataset suffix and underscores
                    clean_name = k.replace(f"_{ds}", "").replace("_", " ").title()
                    report_lines.append(f"  {clean_name:<45}: {self.timings[k]:>10.2f}s")
            else:
                report_lines.append(f"  No specific steps recorded for {ds}.")

            report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append("END OF AUDIT")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        try:
            import colorama

            print(f"\n{colorama.Fore.GREEN}Performance audit saved to:{colorama.Style.RESET_ALL} {report_path}")
        except ImportError:
            print(f"\nPerformance audit saved to: {report_path}")
