#!/usr/bin/env python3
"""
Launcher: runs the OS-Ken controller using the system os_ken.cmd.manager
(which has the CLI/manager entry point) while ensuring all virtualenv
packages (networkx, Flask, etc.) are available on the path.

Usage:
    python run_controller.py controller/garro_controller.py --observe-links
"""
import sys
import os

# 1. Paths
HERE      = os.path.dirname(os.path.abspath(__file__))
site_pkgs = os.path.join(HERE, "garro_env", "lib", "python3.12", "site-packages")
sys_dist  = "/usr/lib/python3/dist-packages"

# 2. Put system os_ken FIRST so os_ken.cmd.manager resolves correctly
if sys_dist not in sys.path:
    sys.path.insert(0, sys_dist)

# 3. Append venv site-packages so networkx, Flask, torch etc. are found
if site_pkgs not in sys.path:
    sys.path.append(site_pkgs)

# 4. Launch the controller manager (same as osken-manager)
from os_ken.cmd.manager import main  # noqa: E402
sys.exit(main())
