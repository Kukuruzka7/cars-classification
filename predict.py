from addict import Dict
import pandas as pd
import torch
from training.dataset import CarDataset
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

if __name__ == '__main__':
    model = timm.create_model('timm/tf_efficientnetv2_s.in21k', pretrained=True)

    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # ct = 0
    # for child in model.blocks.children():
    #     ct += 1
    #     if ct == 11:
    #         for param in child.parameters():
    #             param.requires_grad = True
    #
    # outputs_attrs = 196
    # num_inputs = model.head.in_features
    # last_layer = torch.nn.Linear(num_inputs, outputs_attrs)
    # model.head_drop = torch.nn.Dropout(p=0.3)
    # model.head = last_layer

    args = Dict()
    args.device = "mps"
    args.img_size = 320
    args.model_path = "models/efficientnetv2_gpu.pth"
    args.output_file = "models/efficientnetv2_gpu_sub.csv"

    test_dataset = CarDataset(df_filename="../data/intro-dl-2024/sample_submission.csv",
                              phase="test",
                              dataroot="../data/intro-dl-2024",
                              img_size=args.img_size)

    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=8)

    model.load_state_dict(torch.load(args.model_path))
    model = model.to(args.device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for features, _, in tqdm(test_loader):
            outputs = model(features.to(args.device))
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())

    df = pd.read_csv('../data/intro-dl-2024/sample_submission.csv')
    df['label'] = predictions
    df.to_csv(args.output_file, index=False)
