import tensorflow as tf

# Open the TFRecord file
filename = "../../data/urmp_vn_solo_ddsp_conditioning_train_unbatched.tfrecord-00056-of-00064"
raw_dataset = tf.data.TFRecordDataset(filename)

num_items = raw_dataset.reduce(0, lambda x, _: x + 1)

print(num_items.numpy())

#Parse the example proto data
for raw_record in raw_dataset:
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    # Print the structure of the example proto
    # print(example)

    features = example.features

    # Print the feature names and their values
    for key in features.feature:
        feature = features.feature[key]
        values = feature.bytes_list.value if feature.HasField('bytes_list') else feature.float_list.value
        print(f'{key}, {values.shape}')