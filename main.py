import sys
from pipeline import run_system_verification

def main() -> None:
    """Production runtime entry point execution hook."""
    print("=== Start Pure WiSARD Framework Execution ===")
    try:
        run_system_verification()
    except Exception as error:
        print(f"Critical System Crash within WiSARD engine: {str(error)}", file=sys.stderr)
        sys.exit(1)
    print("=== System Environment Gracefully Terminated ===")

if __name__ == "__main__":
    main()
