# Main file for the project.
# Author: Sara Knezevic
# Email: lost.func@gmail.com

from main_training import autoencoder_training
from main_classify import classifier_training
from main_perturb import perturb

def main():
    # autoencoder_training()
    classifier_training()
    # perturb()

if __name__ == '__main__':
    main()