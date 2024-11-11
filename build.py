#!/usr/bin/env python
import os
import subprocess
import shutil
import sys

# Configuration globals with environment variable defaults
PACKAGE_NAME = os.getenv("PACKAGE_NAME", "transformers")
PACKAGE_VERSION = os.getenv("PACKAGE_VERSION", "4.37.0")
dir_path = os.path.dirname(os.path.realpath(__file__))
PATCH_FILES_DIR = os.getenv("PATCH_FILES_DIR", os.path.join(dir_path, "transformers_patch"))
BUILD_DIR = os.getenv("BUILD_DIR", os.path.join(dir_path, "build"))
VENV_PATH = os.getenv("VENV_PATH", "")
DEBUG = False

# Print usage information
def print_usage():
    print("Usage: python build.py [--package-name <name> | -p <name>]")
    print("                       [--package-version <version> | -V <version>]")
    print("                       [--patch-files-dir <dir> | -d <dir>]")
    print("                       [--build-dir <dir> | -b <dir>]")
    print("                       [--venv-path <path> | -e <path>] [-y] [-v | --verbose]")
    print("\nEnvironment variables:")
    print("  PACKAGE_NAME, PACKAGE_VERSION, PATCH_FILES_DIR, BUILD_DIR, VENV_PATH")
    print("\nThis script fetches, patches, and installs transformers.\n")

# Check if a virtual environment is active
def get_active_venv_path():
    return os.environ.get("VIRTUAL_ENV")

# Parse command-line arguments
def parse_arguments():
    global PACKAGE_NAME, PACKAGE_VERSION, PATCH_FILES_DIR, BUILD_DIR, VENV_PATH, DEBUG
    skip_confirmation = False

    args = sys.argv[1:]
    while args:
        arg = args.pop(0)
        if arg in ('--package-name', '-p'):
            PACKAGE_NAME = args.pop(0)
        elif arg in ('--package-version', '-V'):
            PACKAGE_VERSION = args.pop(0)
        elif arg in ('--patch-files-dir', '-d'):
            PATCH_FILES_DIR = args.pop(0)
        elif arg in ('--build-dir', '-b'):
            BUILD_DIR = args.pop(0)
        elif arg in ('--venv-path', '-e'):
            VENV_PATH = args.pop(0)
        elif arg == '-y':
            skip_confirmation = True
        elif arg in ('-v', '--verbose'):
            DEBUG = True
        else:
            print_usage()
            sys.exit(1)

    if not PACKAGE_NAME:
        print("Error: PACKAGE_NAME is required.")
        print_usage()
        sys.exit(1)

    if not VENV_PATH:
        active_venv = get_active_venv_path()
        if active_venv:
            VENV_PATH = active_venv
            print(f"Note: No virtual environment path provided. Using the currently active virtual environment at {VENV_PATH}.")
        else:
            print("Warning: No virtual environment detected and no VENV_PATH provided. The package will be installed globally.")

    if DEBUG:
        print(f"DEBUG: PACKAGE_NAME={PACKAGE_NAME}")
        print(f"DEBUG: PACKAGE_VERSION={PACKAGE_VERSION}")
        print(f"DEBUG: PATCH_FILES_DIR={PATCH_FILES_DIR}")
        print(f"DEBUG: BUILD_DIR={BUILD_DIR}")
        print(f"DEBUG: VENV_PATH={VENV_PATH}")

    return skip_confirmation

# Clean the specific package files from the build directory
def clean_package_files():
    if not os.path.exists(BUILD_DIR):
        return
    package_prefix = f"{PACKAGE_NAME}-{PACKAGE_VERSION}"
    for item in os.listdir(BUILD_DIR):
        item_path = os.path.join(BUILD_DIR, item)
        if item.startswith(package_prefix) and os.path.isdir(item_path):
            if DEBUG:
                print(f"DEBUG: Removing directory {item_path}")
            shutil.rmtree(item_path)

# Step 1: Fetch the package source code into a build directory
def fetch_package_source(package_name, version=""):
    os.makedirs(BUILD_DIR, exist_ok=True)
    package_spec = f"{package_name}=={version}" if version else package_name
    tar_file = f"{package_name}-{version}.tar.gz"

    if not os.path.exists(os.path.join(BUILD_DIR, tar_file)):
        download_command = [
            "pip", "download", package_spec, "--no-binary=:all:", "--no-deps", "-d", BUILD_DIR, "--find-links", BUILD_DIR
        ]
        if DEBUG:
            download_command.append("-v")
            print(f"DEBUG: Running {' '.join(download_command)}")
        subprocess.run(download_command, check=True)

    if DEBUG:
        print(f"DEBUG: Extracting {tar_file}")
    subprocess.run(["tar", "xzf", os.path.join(BUILD_DIR, tar_file), "-C", BUILD_DIR], check=True)
    return os.path.join(BUILD_DIR, tar_file.replace(".tar.gz", ""))

# Step 2: Replace files with patches
def replace_files(package_dir, patch_dir):
    for file in os.listdir(patch_dir):
        src_path = os.path.join(patch_dir, file)
        dest_path = os.path.join(package_dir, file)
        if os.path.isfile(src_path):
            dest_dir = os.path.dirname(dest_path)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(src_path, dest_path)
            if DEBUG:
                print(f"DEBUG: Replaced {dest_path} with {src_path}")
            else:
                print(f"Replaced {dest_path} with {src_path}")
        else:
            replace_files(os.path.join(package_dir, file), src_path)

# Step 3: Install the patched package using pip
def install_patched_package(package_dir, venv_path):
    python_executable = f"{venv_path}/bin/pip" if venv_path else "pip"
    install_command = [python_executable, "install", package_dir]
    if DEBUG:
        install_command.append("-v")
        print(f"DEBUG: Running {' '.join(install_command)}")
    subprocess.run(install_command, check=True)

if __name__ == "__main__":
    skip_confirmation = parse_arguments()

    if not skip_confirmation:
        print_usage()
        confirmation = input("Do you want to proceed with the installation? (y/N): ")
        if confirmation.lower() != 'y':
            print("Installation aborted.")
            sys.exit(0)

    # Clean only the extracted package files from the build directory
    clean_package_files()

    package_source_dir = fetch_package_source(PACKAGE_NAME, PACKAGE_VERSION)
    replace_files(package_source_dir, PATCH_FILES_DIR)
    install_patched_package(package_source_dir, VENV_PATH)
    print(f"{PACKAGE_NAME} {PACKAGE_VERSION if PACKAGE_VERSION else ''} has been patched and installed.")
