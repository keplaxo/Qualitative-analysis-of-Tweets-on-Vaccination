"""Initiate the independent public-reference schema preflight only; CPU-only."""
import os,sys
from advance_v2 import STAGES,state,submit
def main():
    if not (os.getenv("KAGGLE_API_TOKEN") or
            (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))):
        raise RuntimeError("Kaggle authentication not available")
    step=STAGES[2]
    current=state(step["ref"])
    if current=="MISSING_OR_INACCESSIBLE":
        submit(step)
    else:
        print("EXTERNAL_PREFLIGHT_NOT_RESTARTED",current)
if __name__=="__main__":
    try:main()
    except Exception as e:
        print("EXTERNAL_PREFLIGHT_BLOCKED: "+type(e).__name__,file=sys.stderr)
        sys.exit(1)
