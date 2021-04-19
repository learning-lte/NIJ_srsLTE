import sys
import matplotlib.pyplot as plt
import scipy
import numpy as np
import argparse


def lte_main(args):

    raw_data = scipy.fromfile(open(args.base_path + "raw_data" + args.fileName + ".txt"), dtype = scipy.complex64)
    raw_delim = 999999+0j
    ant2_delim = 999999.1+0j
    csi_delim = 999999.2+0j
    rsrp_delim = 999999.3+0j
    tti_delim = 999999.4 + 0j
    frame_count = np.count_nonzero((raw_data == raw_delim))
    ant2_count = np.count_nonzero((raw_data == ant2_delim))
    csi_count = np.count_nonzero((raw_data == csi_delim))
    rsrp_count = np.count_nonzero((raw_data == rsrp_delim))
    tti_count = np.count_nonzero((raw_data == tti_delim))

    if (frame_count != csi_count or frame_count != rsrp_count or frame_count != ant2_count or frame_count != tti_count):
        Exception("Error: Uneven number of Frames, CSI, and RSRP!")
    if (not args.debug):
        skip_samp1_ind = np.where(raw_data==tti_delim)[0][100] + 1
        raw_data = raw_data[skip_samp1_ind:]
    MIN_DESIRED_CSI_LEN = 100
    working_frame = []
    curr_min_size = 25000
    curr_min_ant2_size = 25000
    curr_min_csi_size = 1000
    training_raw_frames = np.empty((1, curr_min_size), np.complex64)
    training_csi_frames = np.empty((1, curr_min_csi_size), np.complex64)
    training_ant2_frames = np.empty((1, curr_min_size), np.complex64)
    training_rsrp = np.empty((1,1), np.float)
    training_tti = np.empty((1, 1), np.uint32)
    delete_frame_flag = False

    for sample in raw_data:
        if (training_tti.shape == args.num_save):
            break
        if sample == np.complex64(raw_delim):
            if (args.debug):
                print("Length of Ant1 Raw Frame: ", len(working_frame))
            if (args.plot_frames and len(training_rsrp) > 000):
                test_plots(working_frame, np.correlate(working_frame, working_frame, 'same'))

            working_frame, curr_min_size, training_raw_frames = maintain_shape(working_frame, curr_min_size, training_raw_frames)
            raw_frame = np.array(working_frame)
            #raw_angle = np.angle(raw_frame, deg=True)
            training_raw_frames = np.vstack([training_raw_frames, raw_frame])
            working_frame = []
        elif sample == np.complex64(ant2_delim):
            if (args.debug):
                print("Length of Ant2 Raw Frame: ", len(working_frame))
            if (args.plot_frames and len(training_rsrp) > 000):
                test_plots(working_frame, np.correlate(working_frame, raw_frame, 'same'))

            working_frame, curr_min_ant2_size, training_ant2_frames = maintain_shape(working_frame, curr_min_ant2_size, training_ant2_frames)
            ant2_frame = np.array(working_frame)
            training_ant2_frames = np.vstack([training_ant2_frames, ant2_frame])
            #get_AoA(raw_frame, ant2_frame)
            working_frame = []
        elif sample == np.complex64(csi_delim):
            if (args.debug):
                print("Length of CSI Frame: ", len(working_frame))
                if (args.plot_frames and len(training_rsrp) > 000):
                    test_plots(working_frame)
            if (len(working_frame) < MIN_DESIRED_CSI_LEN):
                delete_frame_flag = True
                working_frame = []
                continue

            #working_frame = working_frame[12:]
            working_frame, curr_min_csi_size, training_csi_frames = maintain_shape(working_frame, curr_min_csi_size, training_csi_frames)
            csi_frame = np.array(working_frame)
            training_csi_frames = np.vstack([training_csi_frames, csi_frame])
            working_frame = []
        elif sample == np.complex64(rsrp_delim):
            if (len(working_frame) != 1):
                Exception("Extra Elements captured with RSRP")
            rsrp = np.array(float(working_frame[0]))
            if (args.debug):
                print("RSRP in dbm:", float(working_frame[0]))

            training_rsrp = np.vstack((training_rsrp, rsrp))
            working_frame = []

        elif sample == np.complex64(tti_delim):
            if (len(working_frame) != 1):
                Exception("Extra Elements captured with TTI")
            tti = np.array(int(float(working_frame[0])))
            if (args.debug):
                print("TTI:", int(float(working_frame[0])))
            if (delete_frame_flag):
                training_raw_frames = training_raw_frames[:-1]
                training_ant2_frames = training_ant2_frames[:-1]
                training_rsrp = training_rsrp[:-1]
                delete_frame_flag = False
                working_frame = []
                if (args.debug):
                    print("Deleted frame")
                continue

            training_tti = np.vstack((training_tti, tti))
            working_frame = []

        else:
            working_frame.append(sample)

    save2np(args, training_raw_frames, training_csi_frames, training_rsrp, training_ant2_frames, training_tti)

def save2np(args, training_frames, training_csi, training_pow, training_ant2=np.array([]), training_num=np.array([])):
    if (training_frames.shape[0] != training_csi.shape[0] or training_frames.shape[0] != training_pow.shape[0]) or training_frames.shape[0] != training_ant2.shape[0] or training_frames.shape[0] != training_num.shape[0]:
        Exception("Samples missing attributes before save")
    training_frames = training_frames[1:,:]
    training_csi = training_csi[1:,:]
    training_pow = training_pow[1:]
    training_ant2 = training_ant2[1:,:]
    training_num = training_num[1:]
    final_size = training_frames.shape[0]
    dev_labels = np.full((final_size), args.dev, dtype=np.uint8)
    loc_labels = np.full((final_size,2), list(args.loc), dtype=np.float16)
    rx_labels = np.full((final_size), args.rx, dtype=np.uint8)
    save_path = args.base_path + "LTE" + str(args.dev) + "." + str(args.loc[0]) + "." + str(args.loc[1]) + "." + str(args.rx)
    np.savez(save_path, rawTrain=training_frames, ant2Train=training_ant2, csiTrain=training_csi, powTrain=training_pow, frameNums=training_num,
             dev_labels=dev_labels, loc_labels=loc_labels, rx_labels=rx_labels)

    print("Number of Frames Captured:", final_size)
    print("Done")

def maintain_shape(working_frame, min_size, frame_array):
    frame_size = len(working_frame)
    if min_size > frame_size:
        min_size = frame_size
        frame_array = frame_array[:, :frame_size]
    elif min_size < frame_size:
        working_frame = working_frame[:min_size]
    return working_frame, min_size, frame_array

def test_plots(raw_frame, corr_frame=[]):
    fig, ax = plt.subplots(4)
    fig.suptitle('CrossCorrelation, Magnitude, Power, Freq Plots')
    ax[0].plot(corr_frame, 'y')
    ax[1].plot(list(map(lambda x: abs(x), raw_frame)), 'b')
    ax[2].plot(list(map(lambda x: abs(x) ** 2, raw_frame)), 'r')
    ax[3].plot(np.fft.fftshift(np.fft.fft(raw_frame)), 'g')
    plt.show()
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes 5 files (file sink, file sink header, parse data, and '
                                                 'CSI data and organizes them into a text file where each lline corresponds to a single frame',
                                     version=1.0)
    parser.add_argument('-file', action='store', default='', dest='fileName',
                        help='File suffix at the end of each file')
    parser.add_argument('-base_path', action='store', default="/home/nick/srsLTE/", dest='base_path',
                        help='Path to all files')
    parser.add_argument('-p', action='store_true', default=False, dest='plot_frames',
                        help='Plots frames in matplotlib to appear on screen')
    parser.add_argument('-n', action='store', type=int, default=float('inf'), dest='num_save',
                        help='The upper bound of how many frames will be saved to file')
    parser.add_argument('-d', action='store_true', default=False, dest='debug', help='Turns on debug mode')
    parser.add_argument('-no_csi', action='store_false', default=True, dest='got_csi', help='No CSI data for set')
    parser.add_argument('-no_power', action='store_false', default=True, dest='got_power',
                        help='No RSSI/RSRP data for set')
    parser.add_argument('-dev', action='store', type=int, default=None, dest='dev',
                        help='Class number of the transmitting device')
    parser.add_argument('-loc', nargs='+', action='store', type=int, default=(None, None), dest='loc',
                        help='Location point of the transmitting device')
    parser.add_argument('-rx', action='store', type=int, default=None, dest='rx',
                        help='Class number of the receiving device')
    args = parser.parse_args()

    lte_main(args)
