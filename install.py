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
        self.python_cmd = self.find_python_command()
        self.pip_cmd = self.find_pip_command()
        self.project_dir = Path(__file__).parent
        self.venv_dir = self.project_dir / "hieratic_env"
        
    def find_python_command(self):
        """Find the correct Python command on this system."""
        for cmd in ['python', 'python3', 'py']:
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

    def print_step(self, step, description):
        """Print installation step with formatting."""
        print(f"\n🔧 Step {step}: {description}")
        print("="* 50)

    def run_command(self, command, description="", check=True):
        """Run a command with error handling."""
        if description:
            print(f"⏳ {description}...")
        
        try:
            if isinstance(command, str):
                result = subprocess.run(command, shell=True, check=check, 
                                      capture_output=True, text=True)
            else:
                result = subprocess.run(command, check=check, 
                                      capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Success!")
                if result.stdout.strip():
                    print(f"Output: {result.stdout.strip()}")
            else:
                print(f"Warning: {result.stderr.strip()}")
            
            return result.returncode == 0
            
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            print(f"Command output: {e.stderr}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False

    def check_requirements(self):
        """Check if system meets requirements."""
        self.print_step(1, "Checking System Requirements")
        
        # Check Python
        if not self.python_cmd:
            print("Python 3.8+ not found!")
            print("Please install Python from: https://python.org/downloads/")
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
            if self.system == "windows":
                free_bytes = os.statvfs('.').f_frsize * os.statvfs('.').f_bavail
            else:
                statvfs = os.statvfs('.')
                free_bytes = statvfs.f_frsize * statvfs.f_bavail
            return free_bytes / (1024**3)  # Convert to GB
        except:
            return float('inf')  # Unknown, assume sufficient

    def create_virtual_environment(self):
        """Create Python virtual environment."""
        self.print_step(2, "Creating Virtual Environment")
        
        if self.venv_dir.exists():
            print("📁 Virtual environment already exists. Removing old one...")
            import shutil
            shutil.rmtree(self.venv_dir)
        
        success = self.run_command(
            [self.python_cmd, '-m', 'venv', str(self.venv_dir)],
            "Creating virtual environment"
        )
        
        if not success:
            print("Failed to create virtual environment!")
            return False
        
        # Get activation command
        if self.system == "windows":
            self.activate_cmd = str(self.venv_dir / "Scripts"/ "activate.bat")
            self.venv_python = str(self.venv_dir / "Scripts"/ "python.exe")
            self.venv_pip = str(self.venv_dir / "Scripts"/ "pip.exe")
        else:
            self.activate_cmd = f"source {self.venv_dir / 'bin' / 'activate'}"
            self.venv_python = str(self.venv_dir / "bin"/ "python")
            self.venv_pip = str(self.venv_dir / "bin"/ "pip")
        
        return True

    def install_dependencies(self):
        """Install all Python dependencies."""
        self.print_step(3, "Installing Dependencies")
        
        requirements_file = self.project_dir / "requirements.txt"
        if not requirements_file.exists():
            print("requirements.txt not found!")
            return False
        
        print("📦 This may take 10-15 minutes depending on internet speed...")
        print("☕ Perfect time for a coffee break!")
        
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
        
        # Key packages to check
        packages = ['streamlit', 'torch', 'cv2', 'numpy', 'pandas']
        
        for package in packages:
            if package == 'cv2':
                import_name = 'cv2'
                package_name = 'opencv-python'
            else:
                import_name = package
                package_name = package
            
            success = self.run_command(
                [self.venv_python, '-c', f'import {import_name}; print(f"{package_name}: OK")'],
                f"Checking {package_name}",
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
        print("💡 Double-click 'start_hieratic_ai.bat' to run HieraticAI")

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
        print(f"💡 Run './start_hieratic_ai.sh' to start HieraticAI")

    def install(self):
        """Run the complete installation process."""
        print("🚀 HieraticAI Installation Starting...")
        print(f"🖥️  System: {platform.system()} {platform.machine()}")
        print(f"📁 Installation directory: {self.project_dir}")
        
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
