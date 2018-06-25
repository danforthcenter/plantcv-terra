from plantcv import plantcv as pcv
import sys
import cv2


def main():
    #command line arguments
    source_file = sys.argv[1]
    matrix_file = sys.argv[2]
    output_path = sys.argv[3]


    source_img = cv2.imread(source_file)
    transform_matrix = pcv.transform.load_matrix(matrix_file)

    corrected_img = pcv.transform.apply_transformation_matrix(source_img, source_img, transform_matrix)

    pcv.print_image(corrected_img, output_path)


if __name__ == "__main__":
    main()
