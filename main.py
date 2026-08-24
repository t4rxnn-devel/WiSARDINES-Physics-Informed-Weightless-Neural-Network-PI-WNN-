import sys
from pipeline import run_system_verification
from wisard_engine import PurePhysicsInformedWiSARD
from config import WiSARDPhysicsConfig
from tests import NISTStatisticalSuite

def main() -> None:
    """Production runtime entry point execution hook."""
    print("=== Start Pure WiSARD Framework Execution ===")
    try:
        # Run the primary operational training and validation dataset evaluation pipeline
        run_system_verification()
        
        # Instantiate an isolated hardware layout configuration state to verify addressing entropy
        cfg = WiSARDPhysicsConfig()
        wisard_instance = PurePhysicsInformedWiSARD(cfg)
        
        # Trigger the NIST cryptographic statistical validation framework checks
        validator = NISTStatisticalSuite(wisard_instance)
        validator.execute_all_verification_checks()
        
    except Exception as error:
        print(f"Critical System Crash within WiSARD engine: {str(error)}", file=sys.stderr)
        sys.exit(1)
    print("=== System Environment Gracefully Terminated ===")

if __name__ == "__main__":
    main()
