"""
Runs the web interface version of Chemprop.
This allows for training and predicting in a web browser.
"""

import getpass
import os
import torch.serialization
import argparse
torch.serialization.add_safe_globals([argparse.Namespace])

from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

from chemprop.web.app import app, auth, db
from chemprop.web.utils import clear_temp_folder, set_root_folder


class WebArgs(Tap):
    host: str = os.environ.get('CHEMPROP_HOST', '127.0.0.1')  # Host IP address (override per machine via CHEMPROP_HOST)
    port: int = int(os.environ.get('CHEMPROP_PORT', '5000'))  # Port (override per machine via CHEMPROP_PORT)
    debug: bool = False  # Whether to run in debug mode
    demo: bool = False  # Display only demo features
    initdb: bool = False  # Initialize Database
    set_password: str = None  # Create/update the password for this username, then exit (no server start)
    root_folder: str = None  # Root folder where web data and checkpoints will be saved (defaults to chemprop/web/app)


def run_web(args: WebArgs) -> None:
    app.config['DEMO'] = args.demo

    # Set up root folder and subfolders
    set_root_folder(
        app=app,
        root_folder=args.root_folder,
        create_folders=True
    )
    app.secret_key = auth.load_or_create_secret_key(app.config['SECRET_KEY_PATH'])

    db.init_app(app)

    if args.initdb or not os.path.isfile(app.config['DB_PATH']):
        with app.app_context():
            db.init_db()
            print("-- INITIALIZED DATABASE --")

    # Account management: set a password and exit without starting the server.
    if args.set_password:
        password = getpass.getpass(f'New password for "{args.set_password}": ')
        if password != getpass.getpass('Confirm password: '):
            raise SystemExit('Passwords did not match.')
        with app.app_context():
            auth.set_password(args.set_password, password)
            db.get_or_create_user(args.set_password)
        print(f'-- PASSWORD SET FOR "{args.set_password}" --')
        return

    clear_temp_folder(app=app)

    app.run(host=args.host, port=args.port, debug=args.debug)


def chemprop_web() -> None:
    """Runs the Chemprop website locally.

    This is the entry point for the command line command :code:`chemprop_web`.
    """
    run_web(args=WebArgs().parse_args())
