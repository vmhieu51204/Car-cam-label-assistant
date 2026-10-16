"""Local UI: python run_mock.py [--port 8000]. Install requirements-mock.txt first."""
import argparse
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--env-file", help="Optional local credentials file; used only when the user starts SAM3 tracking")
    parser.add_argument("--kaggle-command", help="Path to an installed Kaggle CLI")
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    if args.kaggle_command:
        import os
        os.environ["KAGGLE_COMMAND"] = args.kaggle_command
    print(f"Local mock: http://127.0.0.1:{args.port} (saved video results; offline replay)")
    uvicorn.run("local_mock:app", host="127.0.0.1", port=args.port)
