import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
# from detection.detect import detect
from .tracking.utils import in_frame, init_trackers, resize_external_detections, write_tracking_results_to_file
from .tools.video_readers import FramesWithInfo #, IterableFrameReader
from .tools.optical_flow import compute_flow
# from tools.misc import load_model
from .tracking.trackers import get_tracker
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment
import copy


class Display:

    def __init__(self, on, interactive=True):
        self.on = on
        if on:
            self.fig, self.ax0 = plt.subplots(figsize=(20,10))
            self.colors =  plt.rcParams['axes.prop_cycle'].by_key()['color']

        self.interactive = interactive
        if on and interactive: plt.ion()
        self.legends = []
        self.plot_count = 0
        self.video_writer = None


    def display(self, trackers):

        something_to_show = False
        for tracker_nb, tracker in enumerate(trackers):
            if tracker.enabled:
                tracker.fill_display(self, tracker_nb)
                something_to_show = True

        self.ax0.imshow(self.latest_frame_to_show)
        # self.ax1.imshow(self.latest_frame_to_show)

        # if len(self.latest_detections):
        #     self.ax0.scatter(self.latest_detections[:, 0], self.latest_detections[:, 1], c='r', s=40)

        if something_to_show:
            self.ax0.xaxis.tick_top()

            # plt.legend(handles=self.legends)
            self.fig.canvas.draw()
            self.ax0.set_axis_off()
            # self.ax1.set_axis_off()
            plt.autoscale(True)
            plt.tight_layout()
            if self.interactive:
                plt.show()
                while not plt.waitforbuttonpress():
                    continue
            else:
                figure = plt.gcf()
                b = figure.axes[0].get_window_extent()
                img = np.array(figure.canvas.buffer_rgba())
                img = img[int(b.y0):int(b.y1),int(b.x0):int(b.x1),:]
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                if self.video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(filename='algorithm_demo.mp4',
                                                    apiPreference=cv2.CAP_FFMPEG,
                                                    fourcc=fourcc,
                                                    fps=11.98,
                                                    frameSize=img.shape[:-1][::-1],
                                                    params=None)
                self.video_writer.write(img)
            self.ax0.cla()
            # self.ax1.cla()
            self.legends = []
            self.plot_count+=1

    def update_detections_and_frame(self, latest_detections, frame):
        self.latest_detections = latest_detections
        self.latest_frame_to_show = cv2.cvtColor(cv2.resize(frame, self.display_shape), cv2.COLOR_BGR2RGB)

    def update_flow(self, flow):
        self.flow = flow
        self.mask = np.zeros_like(self.latest_frame_to_show)

        # Sets image saturation to maximum
        self.mask[..., 1] = 255

    def close(self):
        if self.interactive:
            self.video_writer.release()

class DisplaySpecificFrames:
    def __init__(self, args, reader):
        self.colors =  plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.legends = []
        self.plot_count = 0
        self.display_shape = (reader.output_shape[0] // args.downsampling_factor, reader.output_shape[1] // args.downsampling_factor)

    def display(self, frame, trackers, ax0):
        self.ax0 = ax0

        frame = cv2.cvtColor(cv2.resize(frame, self.display_shape), cv2.COLOR_BGR2RGB)
        self.ax0.imshow(frame)

        for tracker_nb, tracker in enumerate(trackers):
            if tracker.enabled:
                tracker.fill_display(self, tracker_nb)
        self.ax0.set_axis_off()


display = None

def build_confidence_function_for_trackers(trackers, flow01):
    tracker_nbs = []
    confidence_functions = []
    for tracker_nb, tracker in enumerate(trackers):
        if tracker.enabled:
            tracker_nbs.append(tracker_nb)
            confidence_functions.append(tracker.build_confidence_function(flow01))
    return tracker_nbs, confidence_functions

def associate_detections_to_trackers(args, detections_for_frame, trackers, flow01):
    tracker_nbs, confidence_functions = build_confidence_function_for_trackers(trackers, flow01)
    assigned_trackers = [None]*len(detections_for_frame)
    if len(tracker_nbs):
        cost_matrix = np.zeros(shape=(len(detections_for_frame),len(tracker_nbs)))
        for detection_nb, detection in enumerate(detections_for_frame):
            for tracker_id, confidence_function in enumerate(confidence_functions):
                score = confidence_function(detection)
                if score > args.confidence_threshold:
                    cost_matrix[detection_nb,tracker_id] = score
                else:
                    cost_matrix[detection_nb,tracker_id] = 0
        row_inds, col_inds = linear_sum_assignment(cost_matrix,maximize=True)
        for row_ind, col_ind in zip(row_inds, col_inds):
            if cost_matrix[row_ind,col_ind] > args.confidence_threshold: assigned_trackers[row_ind] = tracker_nbs[col_ind]

    return assigned_trackers

def track_video(reader, detections, args, engine, transition_variance, observation_variance, return_trackers=False):


    if args.display == 0:
        display = Display(on=False, interactive=True)
    elif args.display == 1:
        display = Display(on=True, interactive=True)
    elif args.display == 2:
        display = Display(on=True, interactive=False)

    init = False
    trackers = dict()
    frame_nb = 0
    frame0 = next(reader)
    detections_for_frame = detections[frame_nb]
    frame_to_trackers = {}
    flow01 = None
    max_distance = euclidean(reader.output_shape, np.array([0,0]))
    delta = 0.05*max_distance

    if display.on:
        display.display_shape = (reader.output_shape[0] // args.downsampling_factor, reader.output_shape[1] // args.downsampling_factor)
        display.update_detections_and_frame(detections_for_frame, frame0)

    if len(detections_for_frame):
        trackers = init_trackers(engine, detections_for_frame, frame_nb, transition_variance, observation_variance, delta)
        init = True

    if display.on:
        display.display(trackers)
    frame_to_trackers[0] = copy.deepcopy(trackers)

    print('Tracking...')
    for frame_nb in range(1,len(detections)):

        detections_for_frame = detections[frame_nb]
        frame1 = next(reader)
        if display.on:
            display.update_detections_and_frame(detections_for_frame, frame1)

        if not init:
            if len(detections_for_frame):
                trackers = init_trackers(engine, detections_for_frame, frame_nb, transition_variance, observation_variance, delta)
                init = True

        else:

            new_trackers = []
            flow01 = compute_flow(frame0, frame1, args.downsampling_factor)

            if len(detections_for_frame):

                assigned_trackers = associate_detections_to_trackers(args, detections_for_frame, trackers, flow01)

                for detection, assigned_tracker in zip(detections_for_frame, assigned_trackers):
                    if in_frame(detection, flow01.shape[:-1]):
                        if assigned_tracker is None :
                            new_trackers.append(engine(frame_nb, detection, transition_variance, observation_variance, delta))
                        else:
                            trackers[assigned_tracker].update(detection, flow01, frame_nb)

            for tracker in trackers:
                tracker.update_status(flow01)

            if len(new_trackers):
                trackers.extend(new_trackers)

        if display.on: display.display(trackers)
        frame_to_trackers[frame_nb] = copy.deepcopy(trackers)
        frame0 = frame1.copy()


    results = []
    tracklets = [tracker.tracklet for tracker in trackers]

    for tracker_nb, associated_detections in enumerate(tracklets):
        for associated_detection in associated_detections:
            results.append((associated_detection[0], tracker_nb, associated_detection[1][0], associated_detection[1][1]))

    results = sorted(results, key=lambda x: x[0])

    if display.on and not display.interactive: display.close()
    if args.display == 0:
        display = Display(on=False, interactive=True)
    elif args.display == 1:
        display = Display(on=True, interactive=True)
    elif args.display == 2:
        display = Display(on=True, interactive=False)

    print('Tracking done.')
    if return_trackers:
        return results, frame_to_trackers
    else:
        return results

def track(args):

    transition_variance = np.load(os.path.join(args.noise_covariances_path, 'transition_variance.npy'))
    observation_variance = np.load(os.path.join(args.noise_covariances_path, 'observation_variance.npy'))

    engine = get_tracker(args.algorithm)

    if args.external_detections:
        print('USING EXTERNAL DETECTIONS')

        sequence_names = next(os.walk(args.data_dir))[1]

        for sequence_name in sequence_names:
            print(f'---Processing {sequence_name}')
            with open(os.path.join(args.data_dir,sequence_name,'saved_detections.pickle'),'rb') as f:
                detections = pickle.load(f)
            with open(os.path.join(args.data_dir,sequence_name,'saved_frames.pickle'),'rb') as f:
                frames = pickle.load(f)

            ratio = 4
            reader = FramesWithInfo(frames)
            detections = resize_external_detections(detections, ratio)

            print('Tracking...')
            results = track_video(reader, detections, args, engine, transition_variance, observation_variance)

            output_filename = os.path.join(args.output_dir, sequence_name)
            write_tracking_results_to_file(results, ratio_x=ratio, ratio_y=ratio, output_filename=output_filename)

    # else:
    #     print(f'USING INTERNAL DETECTOR, detection threshold at {args.detection_threshold}.')

    #     print('---Loading model...')
    #     model = load_model(args.model_weights)
    #     print('Model loaded.')

    #     def detector(frame): return detect(frame, threshold=args.detection_threshold,
    #                                             model=model)


    #     video_filenames = [video_filename for video_filename in os.listdir(args.data_dir) if video_filename.endswith('.mp4')]

    #     for video_filename in video_filenames:
    #         print(f'---Processing {video_filename}')
    #         reader = IterableFrameReader(os.path.join(args.data_dir,video_filename), skip_frames=args.skip_frames, output_shape=args.output_shape)

    #         print('Detections...')
    #         detections = get_detections_for_video(reader, detector)
    #         reader.init()

    #         print('Tracking...')
    #         results = track_video(reader, detections, args, engine, transition_variance, observation_variance)

    #         output_filename = os.path.join(args.output_dir, video_filename.split('.')[0] +'.txt')
    #         input_shape = reader.input_shape
    #         output_shape = reader.output_shape
    #         ratio_y = input_shape[0] / (output_shape[0] // args.downsampling_factor)
    #         ratio_x = input_shape[1] / (output_shape[1] // args.downsampling_factor)
    #         write_tracking_results_to_file(results, ratio_x=ratio_x, ratio_y=ratio_y, output_filename=output_filename)

def build_image_trackers(frames, list_trackers, args, reader):
    n = len(frames)
    assert n == len(list_trackers)
    plt.figure(figsize=(16,10))
    axs = [plt.subplot(1,n,i+1) for i in range(n)]

    for ax, frame, trackers in zip(axs, frames, list_trackers):
        display = DisplaySpecificFrames(args, reader)
        display.display(frame, trackers, ax)
    plt.subplots_adjust(wspace=0, hspace=0)
    return plt.gcf()


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Tracking')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--detection_threshold', type=float, default=0.33)
    parser.add_argument('--confidence_threshold', type=float, default=0.2)
    parser.add_argument('--model_weights', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--downsampling_factor', type=int, default=1)
    parser.add_argument('--algorithm', type=str, default='Kalman')
    parser.add_argument('--noise_covariances_path',type=str)
    parser.add_argument('--skip_frames',type=int,default=0)
    parser.add_argument('--output_shape',type=str,default='960,544')
    parser.add_argument('--external_detections',action='store_true')
    parser.add_argument('--display', type=int, default=0)
    args = parser.parse_args()

    if args.display == 0:
        display = Display(on=False, interactive=True)
    elif args.display == 1:
        display = Display(on=True, interactive=True)
    elif args.display == 2:
        display = Display(on=True, interactive=False)

    args.output_shape = tuple(int(s) for s in args.output_shape.split(','))

    track(args)

default_args = argparse.Namespace()
default_args.data_dir = 'data/external_detections'
default_args.detection_threshold = 0.33
default_args.confidence_threshold = 0.5
default_args.model_weights = ''
default_args.output_dir = 'surfnet/results'
default_args.downsampling_factor = 1
default_args.algorithm = 'EKF'
default_args.noise_covariances_path = 'surfnet/data/tracking_parameters'
default_args.skip_frames = 0
default_args.output_shape = (960,540)
default_args.external_detections = True
default_args.display = 0
