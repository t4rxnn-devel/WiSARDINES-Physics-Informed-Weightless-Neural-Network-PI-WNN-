import sys
from config import WiSARDPhysicsConfig
from pipeline import NaturalTrainingPipeline
from wisard_engine import PurePhysicsInformedWiSARD
from tests import NISTStatisticalSuite

def main() -> None:
    print("=== Start Pure WiSARD Framework Execution ===")
    try:
        cfg = WiSARDPhysicsConfig()
        
        pipeline = NaturalTrainingPipeline(cfg)
        pipeline.execute_natural_stream(total_epochs=5, samples_per_epoch=20000)
        
        validator = NISTStatisticalSuite(pipeline.engine)
        validator.execute_all_verification_checks()
        
    except Exception as error:
        print(f"Critical System Crash within WiSARD engine: {str(error)}", file=sys.stderr)
        sys.exit(1)
    print("=== System Environment Gracefully Terminated ===")

if __name__ == "__main__":
    main()

