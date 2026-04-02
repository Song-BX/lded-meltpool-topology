from __future__ import annotations

from pathlib import Path

from preprocess import main as preprocess_main
from reconstruct_q import main as reconstruct_main
from statistics import main as statistics_main
from plot_figures import main as plot_main


if __name__ == "__main__":
    preprocess_main()
    reconstruct_main()
    statistics_main()
    plot_main()
