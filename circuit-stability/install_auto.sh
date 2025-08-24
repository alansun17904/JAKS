#!/bin/bash

# ============================================================================
# Circuit Stability Research Framework - Automated Installation Script
# ============================================================================
# Non-interactive version for automated deployments, CI/CD, and containers

set -euo pipefail

# Source the main install script functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/install.sh"

# Override main function for non-interactive mode
main() {
    echo -e "${PURPLE}"
    echo "============================================================================"
    echo "  Circuit Stability Research Framework - Automated Installation"
    echo "============================================================================"
    echo -e "${NC}"
    echo "Installing complete environment for circuit stability research..."
    echo
    
    # Run installation steps without prompts
    detect_system
    setup_conda
    install_system_dependencies
    create_environment
    install_pytorch
    install_ml_packages
    setup_project
    
    # Verify installation
    if verify_installation; then
        print_usage_info
        log_success "🚀 Automated installation completed successfully!"
    else
        log_error "Installation completed with some issues. See verification output above."
        exit 1
    fi
}

# Run automated installation
main "$@"