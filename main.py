from unsupervised import run_unsupervised
from supervised_ import run_supervised
from supB import run_supB
from cnn import run_cnn


def main():

    print("Starting ML pipeline")

    run_unsupervised()
    run_supervised()
    run_supB()
    run_cnn()

    print("Pipeline complete")


if __name__ == "__main__":
    main()
