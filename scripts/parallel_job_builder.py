#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import argparse
import time
import datetime
from subprocess import call
#import plantcv.parallel as pcvp
import re
from dateutil.parser import parse as dt_parser
import json
from math import ceil

# Parse command-line arguments
###########################################
def options():
    """Parse command line options.

    Args:

    Returns:
        argparse object.
    Raises:
        IOError: if dir does not exist.
        IOError: if pipeline does not exist.
        IOError: if the metadata file SnapshotInfo.csv does not exist in dir when flat is False.
        ValueError: if adaptor is not phenofront or dbimportexport.
        ValueError: if a metadata field is not supported.
    """
    # Job start time
    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print("Starting run " + start_time + '\n', file=sys.stderr)

    # These are metadata types that PlantCV deals with.
    # Values are default values in the event the metadata is missing
    valid_meta = {
        # Camera settings
        'camera': 'none',
        'imgtype': 'none',
        'zoom': 'none',
        'exposure': 'none',
        'gain': 'none',
        'frame': 'none',
        'lifter': 'none',
        # Date-Time
        'timestamp': None,
        # Sample attributes
        'id': 'none',
        'plantbarcode': 'none',
        'treatment': 'none',
        'cartag': 'none',
        # Experiment attributes
        'measurementlabel': 'none',
        # Other
        'other': 'none'
    }
    parser = argparse.ArgumentParser(description='Parallel imaging processing with PlantCV.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dir", help='Input directory containing images or snapshots.', required=True)
    parser.add_argument("-a", "--adaptor",
                        help='Image metadata reader adaptor. PhenoFront metadata is stored in a CSV file and the '
                             'image file name. For the filename option, all metadata is stored in the image file '
                             'name. Current adaptors: phenofront, filename', default="phenofront")
    parser.add_argument("-p", "--pipeline", help='Pipeline script file.', required=True)
    parser.add_argument("-i", "--outdir", help='Output directory for images. Not required by all pipelines.',
                        default=".")
    parser.add_argument("-t", "--type", help='Image format type (extension).', default="png")
    parser.add_argument('-c', '--config', help= "file of matrix assignments per image", required= True)
    parser.add_argument("-g", "--group", help= "condor accounting group", required= True)
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        raise IOError("Directory does not exist: {0}".format(args.dir))
    if not os.path.exists(args.pipeline):
        raise IOError("File does not exist: {0}".format(args.pipeline))
    if args.adaptor is 'phenofront':
        if not os.path.exists(os.path.join(args.dir, 'SnapshotInfo.csv')):
            raise IOError(
                'The snapshot metadata file SnapshotInfo.csv does not exist in {0}. '
                'Perhaps you meant to use a different adaptor?'.format(
                    args.dir))
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)


    return args


def metadata_parser(data_dir, output_dir, file_type="png"):
    """Reads metadata the input data directory.

    Args:
        data_dir:     Input data directory.
        file_type:    Image filetype extension (e.g. png).

    Returns:
        images:       List of file paths for source_images

    """

    # Images
    images = []

    # Check whether there is a snapshot metadata file or not
    if os.path.exists(os.path.join(data_dir, "SnapshotInfo.csv")):
        # Open the SnapshotInfo.csv file
        csvfile = open(os.path.join(data_dir, 'SnapshotInfo.csv'), 'rU')
        csvout = open(os.path.join(output_dir,'SnapshotInfo.csv'), 'w')
        # Read the first header line
        header = csvfile.readline()
        csvout.write(header)
        header = header.rstrip('\n')

        # Remove whitespace from the field names
        header = header.replace(" ", "")

        # Table column order
        cols = header.split(',')
        colnames = {}
        for i, col in enumerate(cols):
            colnames[col] = i

        # Read through the CSV file
        for row in csvfile:
            row = row.rstrip('\n')
            data = row.split(',')
            img_list = data[colnames['tiles']]
            if img_list[:-1] == ';':
                img_list = img_list[:-1]
            if len(img_list)==0:
                csvout.write(row + '\n')
            imgs = img_list.split(';')
            kept_tiles = []
            for img in imgs:
                if len(img) != 0:
                    dirpath = os.path.join(data_dir, 'snapshot' + data[colnames['id']])
                    destpath = os.path.join(output_dir, 'snapshot' + data[colnames['id']])
                    if not os.path.exists(destpath):
                        os.mkdir(destpath)
                    filename = img + '.' + file_type
                    source = os.path.join(os.path.abspath(dirpath), filename)
                    if not os.path.exists(os.path.join(dirpath, filename)):
                        continue
                        # raise IOError("Something is wrong, file {0}/{1} does not exist".format(dirpath, filename))
                if "VIS_SV" in img:
                    images.append(source)
                    kept_tiles.append(img)
            data[-1] = ";".join(map(str,kept_tiles))
            csvout.write(",".join(map( str, data)) + '\n')

    return images

    ###########################################

def create_jobfile(jobfile, outdir, exe, arguments, group=None):
    jobfile.write("universe = vanilla\n")
    jobfile.write("getenv = true\n")
    jobfile.write("request_cpus = 1\n")
    jobfile.write("output_dir = " + outdir + "\n")
    if group:
        # If CONDOR_GROUP was defined, define the job accounting group
        jobfile.write("accounting_group = " + group + '\n')
    jobfile.write("executable = " + exe + '\n')
    jobfile.write("arguments = " + arguments + '\n')
    jobfile.write("log = $(output_dir)/$(Cluster).$(Process).bottleneck-distance.log\n")
    jobfile.write("error = $(output_dir)/$(Cluster).$(Process).bottleneck-distance.error\n")
    jobfile.write("output = $(output_dir)/$(Cluster).$(Process).bottleneck-distance.out\n")
    # jobfile.write("queue\n")



###########################################

# Main
###########################################
def main():
    """Main program.

    Args:

    Returns:

    Raises:

    """

    # Get options
    args = options()

    # Variables
    ###########################################
    config_file = open(args.config, 'r')
    config = json.load(config_file)
    config_file.close()

    # Open log files
    #error_log = pcvp.file_writer('error.log')

    # Read image file names
    ###########################################
    images = metadata_parser(data_dir=args.dir, output_dir=args.outdir, file_type=args.type)
    ###########################################

    # Process images
    ###########################################
    # Job builder
    print("Building job list... ", file=sys.stderr)
    # Create DAGman file
    dagman = open(os.path.basename(args.dir) + '.dag', 'w')

    # Job counter
    job_num = 0

    # Number of batches
    batches = int(ceil(len(images) / 1000.0))

    for batch in range(0, batches):
        # Create job batch (cluster) file
        bname = os.path.basename(args.dir) + ".batch." + str(batch) + ".condor"
        batchfile = open(bname, "w")
        # Initialize batch condor file
        if not os.path.exists("logs"):
            os.mkdir("logs")
        create_jobfile(batchfile, "logs", args.pipeline, "$(job_args)", args.group)

        for job in range(job_num, job_num + 1000.0):
            if job == len(images):
                break
            matrix = ""
            if "z1000" in images[job]:
                matrix = config["z1000"]
            elif "z2000" in images[job]:
                matrix = config["z2000"]
            else:
                matrix = config["z1"]
            file_parts = os.path.split(images[job])
            batchfile.write("job_args = " + images[job] + " " + matrix + " " + os.path.join(args.outdir, file_parts[-2], file_parts[-1]) + "\n")
            batchfile.write("queue\n")
        job_num += args.numjobs

        # Add job batch file to the DAGman file
        dagman.write("JOB batch" + str(batch) + " " + bname + "\n")
    dagman.close()

###########################################

if __name__ == '__main__':
    main()
