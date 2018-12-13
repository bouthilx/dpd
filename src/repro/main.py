from repro.dataset.base import build_dataset


def main(argv=None):
    print(build_dataset('mnist'))


if __name__ == "__main__":
    main()
