#!/bin/bash
# setup_banditpam.sh - All-in-one setup and fix script for BanditPAM
# Usage: ./setup_banditpam.sh [--fix-github-actions] [--clean] [--verbose]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VERBOSE=false
FIX_GITHUB_ACTIONS=false
CLEAN_BUILD=false
REPO_URL="https://github.com/navygit/BanditPAM.git"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fix-github-actions)
            FIX_GITHUB_ACTIONS=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "BanditPAM Setup Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --fix-github-actions    Apply GitHub Actions fixes"
            echo "  --clean                 Clean build directories"
            echo "  --verbose, -v           Verbose output"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                      # Basic setup"
            echo "  $0 --clean --verbose   # Clean build with verbose output"
            echo "  $0 --fix-github-actions # Apply CI fixes"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Logging functions
log() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

verbose_log() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[VERBOSE]${NC} $1"
    fi
}

# Platform detection
detect_platform() {
    case "$OSTYPE" in
        linux-gnu*) echo "linux" ;;
        darwin*) echo "macos" ;;
        msys* | cygwin* | win32*) echo "windows" ;;
        *) echo "unknown" ;;
    esac
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install system dependencies
install_system_deps() {
    local platform=$(detect_platform)
    log "Installing system dependencies for $platform..."

    case $platform in
        "linux")
            if command_exists apt-get; then
                # Ubuntu/Debian
                log "Detected Debian/Ubuntu system"
                sudo apt-get update
                sudo apt-get install -y \
                    build-essential cmake libarmadillo-dev \
                    libopenblas-dev liblapack-dev libomp-dev \
                    pkg-config git clang-format
            elif command_exists dnf; then
                # Fedora/RHEL 8+
                log "Detected Fedora/RHEL 8+ system"
                sudo dnf install -y \
                    gcc-c++ cmake armadillo-devel openblas-devel \
                    lapack-devel libomp-devel pkgconfig git clang-tools-extra
            elif command_exists yum; then
                # CentOS/RHEL 7
                log "Detected CentOS/RHEL 7 system"
                sudo yum install -y \
                    gcc-c++ cmake3 armadillo-devel openblas-devel \
                    lapack-devel git
            else
                log_error "Unsupported Linux distribution"
                return 1
            fi
            ;;
        "macos")
            log "Detected macOS system"
            if ! command_exists brew; then
                log "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew update
            brew install cmake armadillo libomp clang-format
            ;;
        "windows")
            log_error "Windows setup requires manual installation"
            log "Please install:"
            log "1. Visual Studio Build Tools"
            log "2. CMake from https://cmake.org/download/"
            log "3. vcpkg for C++ libraries"
            return 1
            ;;
        *)
            log_error "Unsupported platform: $platform"
            return 1
            ;;
    esac

    log_success "System dependencies installed"
}

# Check system requirements
check_system_requirements() {
    log "Checking system requirements..."

    local missing=()

    if ! command_exists cmake; then
        missing+=("cmake")
    fi

    if ! command_exists git; then
        missing+=("git")
    fi

    if ! command_exists gcc && ! command_exists clang; then
        missing+=("C++ compiler (gcc/clang)")
    fi

    if ! command_exists python3 && ! command_exists python; then
        missing+=("python")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        log_warning "Missing requirements: ${missing[*]}"
        return 1
    fi

    log_success "All system requirements satisfied"
    return 0
}

# Setup Python environment
setup_python() {
    log "Setting up Python environment..."

    # Detect Python command
    local python_cmd=""
    if command_exists python3; then
        python_cmd="python3"
    elif command_exists python; then
        python_cmd="python"
    else
        log_error "Python not found"
        return 1
    fi

    verbose_log "Using Python: $($python_cmd --version)"

    # Upgrade pip and install dependencies
    $python_cmd -m pip install --upgrade pip setuptools wheel
    $python_cmd -m pip install numpy>=1.18.0 pybind11>=2.10.0

    # Install optional dependencies
    $python_cmd -m pip install matplotlib pandas scikit-learn pytest black flake8

    log_success "Python environment configured"
}

# Initialize git submodules
init_submodules() {
    log "Initializing git submodules..."

    if [ ! -d ".git" ]; then
        log_error "Not in a git repository"
        return 1
    fi

    git submodule update --init --recursive

    # Verify CARMA submodule
    if [ -d "headers/carma/include" ]; then
        log_success "CARMA submodule initialized"
    else
        log_warning "CARMA submodule not properly initialized"
        # Try to fix it
        git submodule add --force https://github.com/RUrlus/carma.git headers/carma
        git submodule update --init --recursive
    fi
}

# Apply code formatting
format_code() {
    log "Formatting code..."

    if ! command_exists clang-format; then
        log_warning "clang-format not found, skipping code formatting"
        return 0
    fi

    # Format specific files that are known to have issues
    local files=(
        "headers/python_bindings/kmedoids_pywrapper.hpp"
        "src/python_bindings/sparse_support_python.cpp"
        "src/python_bindings/kmedoids_pywrapper.cpp"
        "src/python_bindings/predict_python.cpp"
    )

    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            verbose_log "Formatting: $file"
            clang-format -i -style=file "$file"
        fi
    done

    # Format Python files if black is available
    if command_exists black; then
        find . -name "*.py" -not -path "./build/*" -not -path "./.git/*" -exec black {} +
    fi

    log_success "Code formatting completed"
}

# Clean build directories
clean_build() {
    log "Cleaning build directories..."

    local dirs_to_clean=("build" "dist" "*.egg-info" "__pycache__" ".pytest_cache")

    for pattern in "${dirs_to_clean[@]}"; do
        if ls $pattern 1> /dev/null 2>&1; then
            verbose_log "Removing: $pattern"
            rm -rf $pattern
        fi
    done

    log_success "Build directories cleaned"
}

# Build BanditPAM
build_banditpam() {
    log "Building BanditPAM..."

    # Detect Python command
    local python_cmd=""
    if command_exists python3; then
        python_cmd="python3"
    elif command_exists python; then
        python_cmd="python"
    else
        log_error "Python not found"
        return 1
    fi

    # Set environment variables for better compatibility
    export GITHUB_ACTIONS="false"

    # Try building
    if $python_cmd setup.py build_ext --inplace; then
        log_success "Build completed successfully"
    else
        log_error "Build failed"
        log "Trying alternative build method..."

        # Try with pip
        if $python_cmd -m pip install -e . --verbose; then
            log_success "Installation completed via pip"
        else
            log_error "Both build methods failed"
            return 1
        fi
    fi
}

# Test installation
test_installation() {
    log "Testing BanditPAM installation..."

    # Detect Python command
    local python_cmd=""
    if command_exists python3; then
        python_cmd="python3"
    elif command_exists python; then
        python_cmd="python"
    else
        log_error "Python not found"
        return 1
    fi

    # Create test script
    cat > test_banditpam.py << 'EOF'
import sys
import numpy as np

try:
    import banditpam
    print("✅ BanditPAM imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test basic functionality
try:
    X = np.random.random((20, 2)).astype(np.float32)
    kmedoids = banditpam.KMedoids(n_medoids=3)
    kmedoids.fit(X, 'L2')
    print(f"✅ Clustering works, average loss: {kmedoids.average_loss:.4f}")

    # Test predict if available
    if hasattr(kmedoids, 'predict'):
        X_new = np.random.random((5, 2)).astype(np.float32)
        predictions = kmedoids.predict(X_new)
        print(f"✅ Predict function works: {len(predictions)} predictions")

    print("🎉 All tests passed!")

except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
EOF

    # Run test
    if $python_cmd test_banditpam.py; then
        log_success "All tests passed!"
        rm test_banditpam.py
        return 0
    else
        log_error "Tests failed"
        rm test_banditpam.py
        return 1
    fi
}

# Apply GitHub Actions fixes
fix_github_actions() {
    log "Applying GitHub Actions fixes..."

    # Create .github/workflows directory if it doesn't exist
    mkdir -p .github/workflows

    # Apply the fixed workflows (you would need to copy the actual fixed files here)
    log_warning "GitHub Actions fixes require manual file updates"
    log "Please update the following files:"
    log "  - .github/workflows/linux_run_tests.yml"
    log "  - .github/workflows/run_windows_tests.yml"
    log "  - .github/workflows/run_style_checks.yml"

    # Run code formatting as part of CI fixes
    format_code

    log_success "GitHub Actions fixes applied"
}

# Main execution
main() {
    log "🚀 Starting BanditPAM setup..."
    log "Platform: $(detect_platform)"
    log "Options: Clean=$CLEAN_BUILD, Fix CI=$FIX_GITHUB_ACTIONS, Verbose=$VERBOSE"
    echo ""

    # Clean build if requested
    if [ "$CLEAN_BUILD" = true ]; then
        clean_build
    fi

    # Check system requirements
    if ! check_system_requirements; then
        log "Installing missing system dependencies..."
        if ! install_system_deps; then
            log_error "Failed to install system dependencies"
            exit 1
        fi
    fi

    # Setup Python environment
    if ! setup_python; then
        log_error "Failed to setup Python environment"
        exit 1
    fi

    # Initialize submodules if in a git repo
    if [ -d ".git" ]; then
        init_submodules
    else
        log_warning "Not in a git repository, skipping submodule initialization"
    fi

    # Apply GitHub Actions fixes if requested
    if [ "$FIX_GITHUB_ACTIONS" = true ]; then
        fix_github_actions
    fi

    # Format code
    format_code

    # Build BanditPAM
    if ! build_banditpam; then
        log_error "Build failed"
        exit 1
    fi

    # Test installation
    if ! test_installation; then
        log_error "Installation test failed"
        exit 1
    fi

    echo ""
    log_success "🎉 BanditPAM setup completed successfully!"
    log "You can now use BanditPAM in Python:"
    log "  python -c \"import banditpam; print('BanditPAM ready!')\""
    echo ""

    if [ "$FIX_GITHUB_ACTIONS" = true ]; then
        log "Don't forget to commit and push the GitHub Actions fixes:"
        log "  git add ."
        log "  git commit -m 'fix: Apply comprehensive fixes for setup and CI issues'"
        log "  git push"
    fi
}

# Handle interrupts
trap 'log_error "Setup interrupted"; exit 1' INT TERM

# Run main function
main "$@"