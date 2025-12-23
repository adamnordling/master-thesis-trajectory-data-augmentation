import time
import os
import platform
import datetime
from typing import Dict, List

# psutil is a powerful library for system monitoring.
# It is highly recommended for detailed performance reports.
try:
    import psutil
except ImportError:
    print("Warning: 'psutil' library not found. System info will be limited.")
    print("For full performance reporting, please run: pip install psutil")
    psutil = None


class PerformanceTracker:
    """
    A class to track and report on the performance of the entire pipeline run.
    It measures execution time of key stages and gathers system information.
    """

    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.data: Dict[str, any] = {'start_time_utc': datetime.datetime.utcnow().isoformat()}
        self.start_times: Dict[str, float] = {}
        self.start('total_run')

    def start(self, key: str):
        """Starts a timer for a specific pipeline stage."""
        self.start_times[key] = time.time()

    def stop(self, key: str):
        """Stops a timer and records the duration."""
        if key in self.start_times:
            duration = time.time() - self.start_times[key]
            self.timings[key] = duration
            del self.start_times[key]  # Remove timer to prevent re-stopping

    def set_info(self, key: str, value: any):
        """Stores a piece of non-time-related information."""
        self.data[key] = value

    def _gather_system_info(self):
        """Collects static information about the execution environment."""
        self.data['system_platform'] = platform.system()
        self.data['processor_architecture'] = platform.machine()
        self.data['processor_details'] = platform.processor()

        if psutil:
            self.data['physical_cores'] = psutil.cpu_count(logical=False)
            self.data['logical_cores'] = psutil.cpu_count(logical=True)
            self.data['total_ram_gb'] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        else:
            self.data['physical_cores'] = "N/A (psutil not installed)"
            self.data['logical_cores'] = os.cpu_count()
            self.data['total_ram_gb'] = "N/A (psutil not installed)"

    def generate_report(self, dataset_names: List[str]):
        """
        Calculates final metrics, formats them into a report, and saves to a file.
        This version is multi-dataset aware.
        """
        self.stop('total_run')
        self._gather_system_info()

        # --- Create a dynamic, multi-dataset-aware filename and path ---
        if len(dataset_names) > 1:
            # For multiple datasets, create a combined name and save in the parent 'reports' dir
            report_name_prefix = "-".join(sorted(dataset_names))
            output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'output', 'analysis', 'reports')
        else:
            # For a single dataset, save it inside its specific subfolder
            report_name_prefix = dataset_names[0]
            output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'output', 'analysis', 'reports',
                                      report_name_prefix)

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"{timestamp}_{report_name_prefix}_performance_report.txt")

        # --- Build the Report String ---
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"PERFORMANCE AND EXECUTION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Target Dataset(s): {', '.join(dataset_names)}")
        report_lines.append(f"Report Generated (Local Time): {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("-" * 80)

        report_lines.append(f"Execution Environment:")
        report_lines.append(
            f"  - System: {self.data.get('system_platform', 'N/A')} ({self.data.get('processor_architecture', 'N/A')})")
        report_lines.append(f"  - Processor: {self.data.get('processor_details', 'N/A')}")
        report_lines.append(f"  - Physical Cores: {self.data.get('physical_cores', 'N/A')}")
        report_lines.append(f"  - Logical Cores: {self.data.get('logical_cores', 'N/A')}")
        report_lines.append(f"  - Parallel Workers Used: {self.data.get('parallel_workers_used', 'N/A')}")
        report_lines.append(f"  - Total RAM (GB): {self.data.get('total_ram_gb', 'N/A')}")
        report_lines.append("-" * 80)

        report_lines.append(f"Overall Timing:")
        total_seconds = self.timings.get('total_run', 0)
        report_lines.append(
            f"  - Total Pipeline Duration: {total_seconds:.2f} seconds ({total_seconds / 60:.2f} minutes)")
        report_lines.append("-" * 80)

        report_lines.append("Detailed Stage Timings:")
        for key, duration in sorted(self.timings.items()):
            if key != 'total_run':
                stage_name = key.replace('_', ' ').title()
                report_lines.append(f"  - {stage_name:<40}: {duration:>10.2f} seconds")
        report_lines.append("-" * 80)

        report_lines.append("Notes on Parallelism:")
        report_lines.append("  - Timings reflect the total wall-clock time for processing all listed datasets.")
        report_lines.append("  - A direct measurement of 'time saved' is not possible in a single run, but this")
        report_lines.append("    report confirms the scale of parallelization used.")
        report_lines.append("=" * 80)

        # --- Save the Report ---
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))

        # Use color codes for the final print statement
        try:
            import colorama
            print(f"\n{colorama.Fore.GREEN}Performance report saved to:{colorama.Style.RESET_ALL}\n  {report_path}")
        except ImportError:
            print(f"\nPerformance report saved to:\n  {report_path}")