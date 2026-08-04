import torch
import wandb
import argparse
import transformers

import numpy as np
import pandas as pd

from torch import nn
from tqdm import tqdm
from datetime import datetime
from sklearn.utils import shuffle
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

from models.lstm_atn import ChronosLSTM
from src.chronos import ChronosPipeline
from utils.utils import save_qualitative_results
from utils.fordA_utils import fordADatasetGenerator
from utils.PPOC_utils import PPOC_dataset_generator
from utils.swat_utils import swat_dataset_generator
from utils.mitBIH_utils import mitBIH_dataset_generator
from utils.cwru_utilities import cwru_dataset_generator
from utils.waveForm_utils import waveForm_dataset_generator
from utils.twoLeadECG_utils import twoLeadECG_dataset_generator
from utils.strawberry_utils import strawberry_dataset_generator
from utils.twoPatterns_utils import twoPatterns_dataset_generator
from utils.landingGear_utils import landingGear_dataset_generator
from utils.utils import get_chronos_cache_embeddings, save_chronos_cache_embeddings
from utils.uWaveGestureLibraryY_utils import uWaveGestureLibraryY_dataset_generator
from utils.smallKitchenAppliances_utils import smallKitchenAppliances_dataset_generator


def create_chronos_embeddings(data:pd.DataFrame, batch_size:int, device:torch.device='cpu'):
    feature_columns = [col for col in data.columns if col != 'class']
    dataframe_dict = {col:[] for col in feature_columns}

    chronos = ChronosPipeline.from_pretrained(
                    "amazon/chronos-t5-large",
                    device_map=device,
                    torch_dtype=torch.float32,
            )

    def to_list_of_dicts(embedings:torch.tensor, scale:torch.tensor):
        embeddings_np = embedings.detach().cpu().numpy()
        scale_np = scale.detach().cpu().numpy()
        return [{"embeddings":e,"scale":b} for e,b in zip(embeddings_np,scale_np)]

    for i in tqdm(range(0, data.shape[0], batch_size)):
        batch_data = data.iloc[i:i+batch_size]

        for col in feature_columns:
            batch_data_col = None

            try:
                batch_data_col = torch.tensor(np.stack(np.array(batch_data[col].values)), dtype=torch.float32)
            except Exception as e:
                batch_data_col = [torch.tensor(np.array(item), dtype=torch.float32) for item in batch_data[col].values]

            chronos_emb = chronos.embed(batch_data_col)
            dataframe_dict[col] = dataframe_dict[col] + to_list_of_dicts(chronos_emb[0],chronos_emb[1])

    dataframe_dict["class"] = data["class"].tolist()

    del chronos
    torch.cuda.empty_cache()

    return pd.DataFrame(dataframe_dict, index=data.index)


def train_model(model, criterion, optimizer, train_X, train_Y, num_epochs, device, batch_size, wandb_run=None, name=''):
    print('Start training the model [...]')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_X), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for i in range(0, train_X.shape[0], batch_size):
                batch_data = train_X.iloc[i:i+batch_size]

                if batch_data.shape[0] < 2: # For batch normalization
                    continue

                optimizer.zero_grad()
                output = model(batch_data)
                class_b = torch.tensor(np.array(train_Y.iloc[i:i+batch_size]),
                                       device=device, dtype=torch.float).view(-1,1)
                loss = criterion(output, class_b)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix({'loss':running_loss/(i+1)})
                pbar.update(batch_size)

        wandb_run.log({f'{name}/train_loss':running_loss/(i+1), 'epoch':epoch})


def evaluate_model(model, criterion, test_X, test_Y, device, batch_size, wandb_run=None, name=''):
    print('Start evaluating the model [...]')
    model.eval()

    correct = 0
    running_loss = 0
    output_concat = []
    with torch.no_grad():
        with tqdm(total=len(test_X), desc='Evaluating', unit='batch') as pbar:
            for i in range(0, test_X.shape[0], batch_size):
                batch_data = test_X.iloc[i:i+batch_size]

                output = model(batch_data)
                class_b = torch.tensor(np.array(test_Y.iloc[i:i+batch_size]),
                                       device=device, dtype=torch.float).view(-1,1)
                loss = criterion(output, class_b)
                pred = (output > 0.5).float()
                batch_correct = (pred == class_b).sum().item()
                correct += batch_correct
                running_loss += loss.item()
                output_concat.append(output.cpu().numpy())
                pbar.set_postfix({'accuracy': batch_correct/len(batch_data), 'loss': running_loss/(i+1) })
                pbar.update(batch_size)

    wandb_run.log({f'{name}/test_loss':running_loss/(i+1), "epoch": 1})
    print(f'{name}/test_loss:{running_loss/(i+1)}')

    output_concat = np.concatenate(output_concat).flatten()

    FPR,TPR, roc_thresholds = roc_curve(test_Y, output_concat)
    optimal_idx = np.argmax(TPR - FPR)
    optimal_threshold = roc_thresholds[optimal_idx]

    pred = (output_concat > optimal_threshold).astype(float)
    correct = (pred == test_Y).sum().item()
    optimal_accuracy = correct / len(test_Y)

    wandb_run.log({f'{name}/test_accuracy':optimal_accuracy, "epoch": 1})
    print(f'{name}/test_accuracy: {optimal_accuracy}')

    print(f"{name}/AUC:{roc_auc_score(test_Y, output_concat)}")
    wandb_run.log({f'{name}/AUC':roc_auc_score(test_Y, output_concat), "epoch": 1})
    print(f"{name}/AP:{average_precision_score(test_Y, output_concat)}")
    wandb_run.log({f'{name}/AP':average_precision_score(test_Y, output_concat), "epoch": 1})

    return pred


def main(args):
    print(f'Arguments: {args}')

    # Set seed for reproducibility.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    transformers.set_seed(args.seed)

    device = torch.device(f'cuda:{args.gpu_idx}' if torch.cuda.is_available() else 'cpu')

    wandb_run = None
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M")

    wandb.init(
            entity='intelligoml',
            project='chronos_ad',
            name=f'{args.model}_{dt_string}_LR-{args.learning_rate}_BS-{args.batch_size}_EP-{args.num_epochs}_CL-{args.context}',
            config=args,
            mode = "online" if args.use_wandb else "disabled",
            group=args.dataset_name
        )
    wandb_run = wandb

    dataGenerator = None
    if args.dataset_name.lower() == 'cwru':
        dataGenerator = cwru_dataset_generator(args.dataset_path, args.context)
    elif args.dataset_name.lower() == 'landinggear':
        dataGenerator = landingGear_dataset_generator(args.dataset_path, args.context)
    elif args.dataset_name.lower() == 'forda':
        dataGenerator = fordADatasetGenerator(args.dataset_path, args.context)
    elif args.dataset_name.lower() == 'ppoc':
        dataGenerator = PPOC_dataset_generator(args.dataset_path, args.context)
    elif args.dataset_name.lower() == 'twoleadecg':
        dataGenerator = twoLeadECG_dataset_generator(args.dataset_path, args.context)
    elif args.dataset_name.lower() == 'strawberry':
        dataGenerator = strawberry_dataset_generator(args.dataset_path, args.context)
    elif args.dataset_name.lower() == 'twopatterns':
        dataGenerator = twoPatterns_dataset_generator(args.dataset_path, args.context)
    elif args.dataset_name.lower() == 'uwave':
        dataGenerator = uWaveGestureLibraryY_dataset_generator(args.dataset_path, args.context)
    elif args.dataset_name.lower() == 'smallkitchen':
        dataGenerator = smallKitchenAppliances_dataset_generator(args.dataset_path, args.context)
    elif args.dataset_name.lower() == 'waveform':
        dataGenerator = waveForm_dataset_generator(args.dataset_path, args.context)
    elif args.dataset_name.lower() == 'swat':
        dataGenerator = swat_dataset_generator(args.dataset_path, args.context, args.seed, sampling_rate=0.8)
    elif args.dataset_name.lower() == 'mit-bih':
        dataGenerator = mitBIH_dataset_generator(args.dataset_path, args.seed)
    else:
        raise Exception("Error: Wrong dataset name!")

    for fileName, data in dataGenerator:
        embeddings = get_chronos_cache_embeddings(fileName, args.dataset_name, args.context)
        if embeddings is None:
            embeddings = create_chronos_embeddings(data, args.batch_size, device )
            save_chronos_cache_embeddings(embeddings, fileName, args.dataset_name, args.context)

        embeddings = shuffle(embeddings, random_state=args.seed)

        train_X, test_X, train_y, test_y = train_test_split(embeddings.loc[:, embeddings.columns != 'class'],
                                                embeddings['class'],test_size=0.2, stratify=embeddings['class'], random_state=args.seed)

        class_counts = np.bincount(np.astype(np.array(train_y), int))  # Count occurrences of each class.
        num_classes = len(class_counts)

        weights = class_counts.sum() / (num_classes * class_counts)
        pos_weight = torch.tensor(weights[1], dtype=torch.float32)

        model = None
        if args.model.lower() == 'chronoslstm':
            model = ChronosLSTM(device, train_X.shape[1], dataset=args.dataset_name).to(device)
        else:
            raise Exception("Error: Invalid model name!")

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        train_model(model, criterion, optimizer, train_X, train_y, args.num_epochs,
                    device, args.batch_size, wandb_run, fileName)

        predictions = evaluate_model(model, criterion, test_X, test_y, device, args.batch_size, wandb_run, fileName)

        if args.qualitative_results:
            test_x_orignal = data.loc[test_X.index, data.columns != 'class']
            test_y_orignal = data.loc[test_X.index, data.columns == 'class']
            print("Saving qualitative results [...]")
            save_qualitative_results(test_x_orignal, test_y_orignal, predictions, args.dataset_name + '_' + fileName)
            print("Qualitative results saved!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # General parameters.
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--results', type=str, default='')
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging.')
    parser.add_argument('--qualitative_results', action='store_true', help='Enable qualitative results.')

    # Model specific parameters.
    parser.add_argument('--model', type=str, required=True)         # Options: ChronosLSTM.
    parser.add_argument('--context', type=int, default=256)

    # Training specific parameters.
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    # Dataset specific parameters.
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)

    args = parser.parse_args()
    main(args)
