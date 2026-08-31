"""Run the server: python -m maya_csm [--host 0.0.0.0] [--port 8000]"""

import argparse

import uvicorn

from .config import Settings
from .server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="maya_csm")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(Settings.from_env()), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
