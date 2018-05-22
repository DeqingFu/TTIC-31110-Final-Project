import os
import glob
import soundfile as sf
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def make_example(seq_len, spec_feat, labels):
    # Feature lists for the sequential features of the example
    feats_list  = [tf.train.Feature(float_list=tf.train.FloatList(value=spec_feat))]
    feat_dict = {"feats": tf.train.FeatureList(feature=feats_list)}
    
    sequence_feats = tf.train.FeatureLists(feature_list=feat_dict)

    # Context features for the entire sequence
    len_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[seq_len]))
    label_feat = tf.train.Feature(float_list=tf.train.FloatList(value=labels))

    context_feats = tf.train.Features(feature={"seq_len": len_feat,
                                               "labels": label_feat})

    ex = tf.train.SequenceExample(context=context_feats,
                                  feature_lists=sequence_feats)

    return ex.SerializeToString()


def process_data(partition):
    feats = {}
    transcripts = {}
    utt_len = {}  # Required for sorting the utterances based on length
    cnt = 0
    for filename in glob.iglob(os.path.abspath(partition+'/*.wav.trn'), recursive=True):
        with open(filename, 'r') as file:
            cnt += 1
            file_dir = os.path.abspath(AUDIO_PATH + file.readline()[3:-5])
            print("sub-processing file number", cnt, "in", os.path.basename(partition))
            features_name = file_dir + ".data"
            labels_name = file_dir + ".zh"
            features = np.loadtxt(os.path.abspath(features_name), delimiter=",")
            labels = np.loadtxt(os.path.abspath(labels_name), delimiter=",")
            feats[file_dir] = features
            utt_len[file_dir] = feats[file_dir].shape[0]
            transcripts[file_dir] = labels.tolist()
    return feats, transcripts, utt_len


def create_records():
    for partition in sorted(glob.glob(os.path.abspath(AUDIO_PATH+'/*'))):
        if os.path.basename(partition) in ["data", "README.TXT", "lm_phone", "lm_word", "processed"]:
            continue
        print('Processing' + partition)
        feats, transcripts, utt_len = process_data(partition)
        sorted_utts = sorted(utt_len, key=utt_len.get)
        
        # bin into groups of 100 frames.
        max_t = int(utt_len[sorted_utts[-1]]/100)
        min_t = int(utt_len[sorted_utts[0]]/100)

        # Create destination directory
        write_dir = os.path.abspath(AUDIO_PATH + 'processed/' + partition.split('/')[-1])
        if tf.gfile.Exists(write_dir):
            tf.gfile.DeleteRecursively(write_dir)
        tf.gfile.MakeDirs(write_dir)

        if os.path.basename(partition) == 'train':
            # Create multiple TFRecords based on utterance length for training
            writer = {}
            count = {}
            print('Processing training files...')
            for i in range(min_t, max_t+1):
                filename = os.path.abspath(os.path.join(write_dir, 'train' + '_' + str(i) + '.tfrecords'))
                writer[i] = tf.python_io.TFRecordWriter(filename)
                count[i] = 0

            for utt in tqdm(sorted_utts):
                example = make_example(utt_len[utt], feats[utt].tolist(), transcripts[utt])
                index = int(utt_len[utt]/100)
                writer[index].write(example)
                count[index] += 1

            for i in range(min_t, max_t+1):
                writer[i].close()
            print(count)

            # Remove bins which have fewer than 20 utterances
            for i in range(min_t, max_t+1):
                if count[i] < 20:
                    os.remove(os.path.abspath(os.path.join(write_dir, 'train' + '_' + str(i) + '.tfrecords')))
        else:
            # Create single TFRecord for dev and test partition
            filename = os.path.abspath(os.path.join(write_dir, os.path.basename(write_dir) + '.tfrecords'))
            print('Creating', filename)
            record_writer = tf.python_io.TFRecordWriter(filename)
            for utt in sorted_utts:
                #print(feats[utt].tolist())
                example = make_example(utt_len[utt], feats[utt].tolist(), transcripts[utt])
                record_writer.write(example)
            record_writer.close()
            print('Processed '+str(len(sorted_utts))+' audio files')

AUDIO_PATH = './data_thchs30/'

if __name__ == '__main__':
    create_records()
