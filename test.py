from metai.dataset import ScwdsDataset

ds = ScwdsDataset(data_path="data/samples.jsonl", is_train=True, test_set="TestSetB")

for metadata, input_data, target_data, target_mask, input_mask in ds:
    print(input_data.shape, target_data.shape, target_mask.shape, input_mask.shape)
    break