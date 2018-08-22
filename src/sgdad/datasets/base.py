import logging

from orion.core.utils import module_import

log = logging.getLogger(__name__)


def load_modules_parser(main_parser):
    """Search through the `cli` folder for any module containing a `get_parser` function"""
    modules = module_import.load_modules_in_path('sgdad.datasets',
                                                 lambda m: hasattr(m, 'add_subparser'))

    for module in modules:
        getattr(module, 'add_subparser')(main_parser.get_subparsers())


def main(argv=None):
    """Entry point for `orion.core` functionality."""
    # Fetch experiment name, user's script path and command line arguments
    # Use `-h` option to show help

    main_parser = OrionArgsParser()

    load_modules_parser(main_parser)

    main_parser.execute(argv)

    return 0


if __name__ == "__main__":
    main()
