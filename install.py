#!/usr/bin/env python3
"""
HieraticAI One-Click Installer
==============================

This script automatically installs all requirements for HieraticAI and sets up the environment.
Works on Windows, macOS, and Linux.

Usage:
    python install.py

What it does:
1. Checks system requirements
2. Creates virtual environment 
3. Installs all dependencies
4. Verifies installation
5. Creates launch scripts
"""

import os
import sys
import subprocess
import platform
import urllib.request
from pathlib import Path


class HieraticAIInstaller:
    def __init__(self):
        self.system = platform.system().lower()
        
        # Prevent macOS from creating metadata files
        if self.system == "darwin":
            self.setup_macos_environment()
        
        self.python_cmd = self.find_python_command()
        self.pip_cmd = self.find_pip_command()
        self.project_dir = Path(__file__).parent
        self.venv_dir = self.project_dir / "hieratic_env"
        
    def find_python_command(self):
        """Find the correct Python command on this system."""
        # On Windows, try 'py' first as it's the Python Launcher
        if self.system == "windows":
            cmd_order = ['py', 'python', 'python3']
        else:
            cmd_order = ['python3', 'python', 'py']
            
        for cmd in cmd_order:
            try:
                result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
                if result.returncode == 0 and 'Python 3.' in result.stdout:
                    return cmd
            except FileNotFoundError:
                continue
        return None
    
    def find_pip_command(self):
        """Find the correct pip command on this system."""
        for cmd in ['pip', 'pip3']:
            try:
                result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    return cmd
            except FileNotFoundError:
                continue
        return None
    
    def get_python_version(self):
        """Get the Python version for path construction."""
        return f"{sys.version_info.major}.{sys.version_info.minor}"
    
    def setup_macos_environment(self):
        """Set up environment variables to prevent macOS metadata file creation."""
        # Prevent .DS_Store files
        os.environ['COPYFILE_DISABLE'] = '1'
        # Prevent AppleDouble files (._*)
        os.environ['APPLEFILE'] = '0'
        # Additional prevention for tar operations
        os.environ['TAR_OPTIONS'] = '--exclude="._*" --exclude=".DS_Store"'
        # Prevent extended attributes
        os.environ['NOEXTATTR'] = '1'
        # Prevent resource forks
        os.environ['_COPYFILE_DISABLE'] = '1'
        
        print("Configured environment to prevent macOS metadata files")
    
    def prevent_metadata_creation(self):
        """Additional steps to prevent metadata file creation during operations."""
        if self.system == "darwin":
            # Create .noindex file to prevent Spotlight indexing
            noindex_file = self.venv_dir / ".noindex"
            try:
                noindex_file.touch(exist_ok=True)
                print("Created .noindex file to prevent Spotlight indexing")
            except Exception as e:
                print(f"Warning: Could not create .noindex file: {e}")
            
            # Create .fseventsd directory to prevent filesystem event logging
            try:
                fseventsd_dir = self.venv_dir / ".fseventsd"
                fseventsd_dir.mkdir(exist_ok=True)
                (fseventsd_dir / "no_log").touch(exist_ok=True)
                print("Created filesystem event prevention")
            except Exception as e:
                print(f"Warning: Could not create fseventsd prevention: {e}")
    
    def create_venv_no_metadata(self):
        """Create virtual environment with macOS metadata prevention."""
        print("Running: Creating virtual environment (macOS optimized)...")
        
        # Set up environment to prevent metadata creation
        env = os.environ.copy()
        env.update({
            'COPYFILE_DISABLE': '1',
            'APPLEFILE': '0',
            '_COPYFILE_DISABLE': '1',  # Additional prevention
            'NOEXTATTR': '1'  # Prevent extended attributes
        })
        
        try:
            result = subprocess.run(
                [self.python_cmd, '-m', 'venv', str(self.venv_dir)],
                capture_output=True, text=True, env=env
            )
            
            if result.returncode == 0:
                print("Success!")
                return True
            else:
                print(f"Error: {result.stderr.strip()}")
                return False
                
        except Exception as e:
            print(f"Error creating virtual environment: {e}")
            return False
    
    def clean_macos_metadata(self):
        """Remove macOS metadata files that can cause Unicode issues in virtual environments."""
        if self.system != "darwin":
            return  # Only run on macOS
            
        try:
            # Remove ._ files that macOS creates on external drives
            subprocess.run(
                ['find', str(self.venv_dir), '-name', '._*', '-delete'],
                capture_output=True, text=True
            )
            
            # Also remove .DS_Store files
            subprocess.run(
                ['find', str(self.venv_dir), '-name', '.DS_Store', '-delete'],
                capture_output=True, text=True
            )
            
            # Clean up any corrupted .pth files that might contain binary data
            subprocess.run(
                ['find', str(self.venv_dir), '-name', '*.pth', '-exec', 'file', '{}', ';'],
                capture_output=True, text=True
            )
            
            print("Cleaned up macOS metadata files")
        except Exception as e:
            print(f"Warning: Could not clean metadata files: {e}")
    
    def safe_remove_directory(self, directory):
        """Safely remove a directory, handling platform-specific issues."""
        import shutil
        try:
            # First, try to remove macOS metadata files (macOS only)
            if self.system == "darwin":
                try:
                    subprocess.run(['find', str(directory), '-name', '._*', '-delete'], 
                                 capture_output=True)
                    subprocess.run(['find', str(directory), '-name', '.DS_Store', '-delete'], 
                                 capture_output=True)
                except:
                    pass  # Ignore errors here
            
            # Then remove the directory
            shutil.rmtree(directory)
        except Exception as e:
            print(f"Warning: Error removing directory {directory}: {e}")
            # Force removal if standard removal fails
            if self.system == "windows":
                try:
                    subprocess.run(['rmdir', '/s', '/q', str(directory)], check=True, shell=True)
                except subprocess.CalledProcessError:
                    print(f"Could not remove {directory}. Please remove manually.")
            elif self.system == "darwin":
                try:
                    subprocess.run(['rm', '-rf', str(directory)], check=True)
                except subprocess.CalledProcessError:
                    print(f"Could not remove {directory}. Please remove manually.")

    def print_step(self, step, description):
        """Print installation step with formatting."""
        print(f"\nStep {step}: {description}")
        print("="* 50)

    def run_command(self, command, description="", check=True):
        """Run a command with error handling."""
        if description:
            print(f"Running: {description}...")
        
        try:
            # Ensure environment variables are passed to subprocesses
            env = os.environ.copy()
            if self.system == "darwin":
                env.update({
                    'COPYFILE_DISABLE': '1',
                    'APPLEFILE': '0',
                    'TAR_OPTIONS': '--exclude="._*" --exclude=".DS_Store"'
                })
            
            if isinstance(command, str):
                result = subprocess.run(command, shell=True, check=check, 
                                      capture_output=True, text=True, env=env)
            else:
                result = subprocess.run(command, check=check, 
                                      capture_output=True, text=True, env=env)
            
            if result.returncode == 0:
                print(f"Success!")
                if result.stdout.strip():
                    print(f"Output: {result.stdout.strip()}")
            else:
                print(f"Warning: {result.stderr.strip()}")
            
            return result.returncode == 0
            
        except subprocess.CalledProcessError as e:
            print(f"Error: Command '{' '.join(e.cmd) if hasattr(e, 'cmd') else 'unknown'}' returned non-zero exit status {e.returncode}.")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"Command output: {e.stderr}")
            elif hasattr(e, 'output') and e.output:
                print(f"Command output: {e.output}")
            return False
        except UnicodeDecodeError as e:
            print(f"Unicode error (common on external drives): {e}")
            print("This usually indicates corrupted Python environment files.")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False

    def check_requirements(self):
        """Check if system meets requirements."""
        self.print_step(1, "Checking System Requirements")
        
        # Check Python
        if not self.python_cmd:
            print("ERROR: Python 3.8+ not found!")
            if self.system == "windows":
                print("Please install Python from: https://python.org/downloads/")
                print("Make sure to check 'Add Python to PATH' during installation.")
            elif self.system == "darwin":
                print("Please install Python from: https://python.org/downloads/")
                print("Or use Homebrew: brew install python3")
            else:  # Linux
                print("Please install Python using your system package manager:")
                print("  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv")
                print("  CentOS/RHEL: sudo yum install python3 python3-pip")
                print("  Arch: sudo pacman -S python python-pip")
            return False
        
        # Check Python version
        result = subprocess.run([self.python_cmd, '--version'], 
                              capture_output=True, text=True)
        version_str = result.stdout.strip()
        print(f"Found {version_str}")
        
        # Extract version number
        version_parts = version_str.split()[1].split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if major < 3 or (major == 3 and minor < 8):
            print("Python 3.8+ required!")
            return False
        
        # Check pip
        if not self.pip_cmd:
            print("pip not found!")
            return False
        
        print(f"Found pip")
        
        # Check disk space (approximate)
        free_space = self.get_free_space()
        if free_space < 5:  # 5GB
            print(f"Warning: Only {free_space:.1f}GB free space. Recommended: 5GB+")
        else:
            print(f"Sufficient disk space: {free_space:.1f}GB free")
        
        return True

    def get_free_space(self):
        """Get free disk space in GB."""
        try:
            import shutil
            # Use shutil.disk_usage for cross-platform compatibility
            _, _, free_bytes = shutil.disk_usage('.')
            return free_bytes / (1024**3)  # Convert to GB
        except:
            return float('inf')  # Unknown, assume sufficient

    def create_virtual_environment(self):
        """Create Python virtual environment."""
        self.print_step(2, "Creating Virtual Environment")
        
        if self.venv_dir.exists():
            print("Virtual environment already exists. Removing old one...")
            self.safe_remove_directory(self.venv_dir)
        
        # Create virtual environment with metadata prevention
        if self.system == "darwin":
            success = self.create_venv_no_metadata()
        else:
            success = self.run_command(
                [self.python_cmd, '-m', 'venv', str(self.venv_dir)],
                "Creating virtual environment"
            )
        
        if not success:
            print("Failed to create virtual environment!")
            return False
        
        # Prevent and clean up macOS metadata files that can cause Unicode issues
        if self.system == "darwin":
            self.prevent_metadata_creation()
            self.clean_macos_metadata()
        
        # Get activation command and executables
        if self.system == "windows":
            self.activate_cmd = str(self.venv_dir / "Scripts" / "activate.bat")
            self.venv_python = str(self.venv_dir / "Scripts" / "python.exe")
            self.venv_pip = str(self.venv_dir / "Scripts" / "pip.exe")
        else:
            self.activate_cmd = f"source {self.venv_dir / 'bin' / 'activate'}"
            self.venv_python = str(self.venv_dir / "bin" / "python")
            self.venv_pip = str(self.venv_dir / "bin" / "pip")
        
        return True

    def install_dependencies(self):
        """Install all Python dependencies."""
        self.print_step(3, "Installing Dependencies")
        
        requirements_file = self.project_dir / "requirements.txt"
        if not requirements_file.exists():
            print("requirements.txt not found!")
            return False
        
        print("This may take 10-15 minutes depending on internet speed...")
        
        # Clean metadata files before pip operations (macOS fix)
        if self.system == "darwin":
            self.clean_macos_metadata()
        
        # Test the virtual environment python first
        test_success = self.run_command(
            [self.venv_python, '-c', 'print("Environment test successful")'],
            "Testing virtual environment",
            check=False
        )
        
        if not test_success:
            print("Virtual environment appears corrupted. Recreating...")
            self.safe_remove_directory(self.venv_dir)
            
            # Recreate environment
            self.run_command(
                [self.python_cmd, '-m', 'venv', str(self.venv_dir)],
                "Recreating virtual environment"
            )
            
            if self.system == "darwin":
                self.clean_macos_metadata()
        
        # Upgrade pip first
        success = self.run_command(
            [self.venv_python, '-m', 'pip', 'install', '--upgrade', 'pip'],
            "Upgrading pip"
        )
        
        if not success:
            print("Warning: Could not upgrade pip, continuing anyway...")
        
        # Install requirements
        success = self.run_command(
            [self.venv_pip, 'install', '-r', str(requirements_file)],
            "Installing Python packages"
        )
        
        return success

    def verify_installation(self):
        """Verify that all key dependencies are installed."""
        self.print_step(4, "Verifying Installation")
        
        # Clean up any metadata files that might have been created during installation
        if self.system == "darwin":
            self.clean_macos_metadata()
        
        # Key packages to check
        packages = ['streamlit', 'torch', 'cv2', 'numpy', 'pandas']
        
        for package in packages:
            if package == 'cv2':
                import_name = 'cv2'
                package_name = 'opencv-python'
            else:
                import_name = package
                package_name = package
            
            # Try alternative import method if standard method fails
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            if self.system == "windows":
                site_packages_path = f"hieratic_env/Lib/site-packages"
            else:
                site_packages_path = f"hieratic_env/lib/python{python_version}/site-packages"
            
            success = self.run_command(
                [self.venv_python, '-S', '-c', f'import sys; sys.path.insert(0, "{site_packages_path}"); import {import_name}; print("{package_name}: OK")'],
                f"Checking {package_name}",
                check=False
            )
            
            # Fallback to standard method if -S method fails
            if not success:
                success = self.run_command(
                    [self.venv_python, '-c', f'import {import_name}; print(f"{package_name}: OK")'],
                    f"Checking {package_name} (fallback)",
                    check=False
                )
            
            if not success:
                print(f"{package_name} not properly installed!")
                return False
        
        print("All dependencies verified!")
        return True

    def create_launch_scripts(self):
        """Create easy launch scripts for the user."""
        self.print_step(5, "Creating Launch Scripts")
        
        if self.system == "windows":
            self.create_windows_launcher()
        else:
            self.create_unix_launcher()
        
        print("Launch scripts created!")
        return True

    def create_windows_launcher(self):
        """Create Windows batch file launcher."""
        launcher_content = f'''@echo off
echo Starting HieraticAI...
cd /d "{self.project_dir}"
call "{self.venv_dir}\\Scripts\\activate.bat"
streamlit run tools\\validation\\prediction_validator.py
pause
'''
        
        launcher_path = self.project_dir / "start_hieratic_ai.bat"
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)
        
        print(f"Created launcher: {launcher_path}")
        print("Double-click 'start_hieratic_ai.bat' to run HieraticAI")

    def create_unix_launcher(self):
        """Create Unix shell script launcher."""
        launcher_content = f'''#!/bin/bash
echo "Starting HieraticAI..."
cd "{self.project_dir}"
source "{self.venv_dir}/bin/activate"
streamlit run tools/validation/prediction_validator.py
'''
        
        launcher_path = self.project_dir / "start_hieratic_ai.sh"
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)
        
        # Make executable
        os.chmod(launcher_path, 0o755)
        
        print(f"Created launcher: {launcher_path}")
        print(f"Run './start_hieratic_ai.sh' to start HieraticAI")

    def install(self):
        """Run the complete installation process."""
        print("HieraticAI Installation Starting...")
        print(f"System: {platform.system()} {platform.machine()}")
        print(f"Installation directory: {self.project_dir}")
        
        # Apply macOS metadata prevention globally for this process
        if self.system == "darwin":
            print("Applying macOS metadata file prevention...")
        
        steps = [
            self.check_requirements,
            self.create_virtual_environment,
            self.install_dependencies,
            self.verify_installation,
            self.create_launch_scripts
        ]
        
        for i, step in enumerate(steps, 1):
            try:
                if not step():
                    print(f"\nInstallation failed at step {i}")
                    print("Please report this issue with error details to:")
                    print("https://github.com/MargotBelot/HieraticAI/issues")
                    return False
            except KeyboardInterrupt:
                print("\nInstallation cancelled by user")
                return False
            except Exception as e:
                print(f"\nUnexpected error in step {i}: {e}")
                return False
        
        self.print_success_message()
        return True

    def print_success_message(self):
        """Print success message with next steps."""
        print("\n"+ "="*60)
        print("INSTALLATION COMPLETE!")
        print("="*60)
        print("\nNext Steps:")
        
        if self.system == "windows":
            print("1. Double-click 'start_hieratic_ai.bat' to launch")
        else:
            print("1. Run './start_hieratic_ai.sh' to launch")
        
        print("2. Your browser will open to http://localhost:8501")
        print("3. Start validating hieratic characters!")
        print("\nTips:")
        print("- The first launch may take 30-60 seconds")
        print("- Keep the terminal/command prompt window open while using")
        print("- Press Ctrl+C in terminal to stop the application")
        
        print("\nDocumentation:")
        print("- Getting Started Guide: GETTING_STARTED.md")
        print("- Technical Guide: TECHNICAL_GUIDE.md")
        
        print("\nNeed help?")
        print("- GitHub Issues: https://github.com/MargotBelot/HieraticAI/issues")


def main():
    """Main installation function."""
    installer = HieraticAIInstaller()
    
    try:
        success = installer.install()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n Installation cancelled")
        sys.exit(1)


if __name__ == "__main__":
    main()
