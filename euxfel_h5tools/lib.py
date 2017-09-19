import fabio
import h5py


def stat(files, multiline=False):
    """gather basic info about file
    :param files: a list of filenames to check
    :param split: print each information on a new line
    """
    first_train = float('inf')
    last_train = 0
    total_size = 0
    total_entries = 0
    instruments = set()
    invalid = []

    for filename in files:
        out = ""
        try:
            xfel_file = h5py.File(filename, 'r')
            sfile = "File: '{}'".format(xfel_file.filename)

            size_mb = xfel_file.fid.get_filesize() / 1000000
            ssize = "Size: {} MB".format(size_mb)

            inst = xfel_file["INSTRUMENT"]
            sinst = "Instruments: {}".format(", ".join(inst))

            f_train = xfel_file["INDEX/trainId"][0]
            sf_train = "First Train: {}".format(f_train)

            l_train = xfel_file["INDEX/trainId"][-1]
            sl_train = "Last Train: {}".format(l_train)

            entries = len(xfel_file["INDEX/trainId"])
            sentries = "Entries: {}".format(entries)

            paths = list(xfel_file["METADATA/dataSourceId"])
            header_path = [p for p in paths if p.endswith(b"header")][0]
            pulses = xfel_file[header_path + b"/pulseCount"][0]
            spulses = "Pulses per Train: {}".format(pulses)

            instruments.update(inst)
            total_size += size_mb
            total_entries += entries

            if f_train <= first_train:
                first_train = f_train
            if l_train >= last_train:
                last_train = l_train

            if multiline:
                out = ("{f}\n{size}\n{entries}\n"
                       "{pulses}\n{first}\n{last}\n"
                       "{instruments}".format(f=sfile,
                                              size=ssize,
                                              entries=sentries,
                                              pulses=spulses,
                                              first=sf_train,
                                              last=sl_train,
                                              instruments=sinst))
            else:
                out = ("{f}\n\t{size}\t{entries}\t"
                       "{pulses}\n\t{first}\t{last}\n\t"
                       "{instruments}\n".format(f=sfile,
                                                size=ssize,
                                                entries=sentries,
                                                pulses=spulses,
                                                first=sf_train,
                                                last=sl_train,
                                                instruments=sinst))

        except (OSError, IOError, KeyError):
            # The errors could be:
            #  - OSError: not an HDF5 file
            #  - IOError: truncated file
            #  - KeyError: one of the keys used was not found,
            #                therefore not a EuXFEL specific file
            out = "{}: not an EuXFEL HDF5 file\n".format(filename)
            invalid.append(filename)

        print(out)

    if len(files) > 1:
        if multiline:
            total = "Total Files: {}\n".format(len(files) - len(invalid))
            total += "Total File Size: {} MB\n".format(total_size)
            total += "First Train: {}\n".format(first_train)
            total += "Last Train: {}\n".format(last_train)
            total += "Instruments: {}\n".format(", ".join(instruments))
        else:
            total = "---\n"
            total += "Total Files: {}\t".format(len(files) - len(invalid))
            total += "Total File Size: {} MB\n".format(total_size)
            total += "First Train: {}\t".format(first_train)
            total += "Last Train: {}\n".format(last_train)
            total += "Instruments: {}\n".format(", ".join(instruments))
            total += "-" * 3

        print(total)

    if invalid:
        print("These are not valid files: {}".format(", ".join(invalid)))


def rec_print_h5_level(ds, indent=0, maxlen=100):
    """Given an h5 data structure, iterate through it (from S Hauf)"""

    for k in list(ds.keys())[:maxlen]:
        print(" "*indent+k)
        if isinstance(ds[k], h5py.Group):
            rec_print_h5_level(ds[k], indent+4, maxlen)
        else:
            print(" "*indent + k)


def h5_to_cbf(in_h5file, cbf_filename, index, header=None):
    """Conversion from numpy array to cbf binary image"""
    if header is not None:
        header = header
    else:
        header = {}
    try:
        tmpf = h5py.File(in_h5file, 'r')
    except IOError:
        print("Check input file.")
        return

    paths = list(tmpf["METADATA/dataSourceId"])
    image_path = [p for p in paths if p.endswith(b"image")][0]
    images = tmpf[image_path + b"/data"][index]
    img_reduced = images[0, ...]
    cbf_out = fabio.cbfimage.cbfimage(header=header, data=img_reduced)
    cbf_out.write(cbf_filename)
