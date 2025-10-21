import torch
from model import AdvancedRNN, gru_lstm
from train import train
from loader import get_loaders
from load import load_data
from report_end import  make_predictions
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import os 



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dict_feature_eng = dict(use_microstructure=True,
                use_imbalance=True,
                use_price_dynamics=True,
                use_momentum=True,
                momentum_windows=[5, 20],
                use_flux=True,
                use_directional=True,
                use_time=True,
                use_log=True,
                corr_threshold=None)
dict_load_data = dict(dummy = True, normalize=True, filter=True, print_shape= True, threshold = 10, dict_feature_eng = dict_feature_eng)
dict_loaders = dict(batch_size=32, test_size=0.1, seed=42, shuffle = True)
params = {
    "num_features": 16,
    "num_classes": 24,
    "embed_features": [0, 2, 3, 4],
    "num_embed_features": [6, 3, 2, 2],
    "encode_features": [1],
    "embedding_dim": 8,
    "hidden_dim": 256,  # renommé d_hidden -> hidden_dim
    "num_layers": 1,
    "dropout": 0.15,
    "rnn_type": "LSTM",
    "attention": True,
    "use_transformer": True,
    "nhead": 4,
    "num_transformer_layers": 2,
    "device": device,
}


training_params =  {
    "lr": 1e-4,
    "weight_decay": 3e-4,
    "gamma": 0.6,
    "step_size": 5,
    "num_epochs": 100,
    "early_stop_n": 10,
    "patience": 10,
    "device": device, 
    "scheduler": {
        "mode": "min",
        "factor": 0.6,
        "patience": 3,
        "verbose": True
    }
}


def compile (model, dict_load_data, dict_loaders, model_params, training_params,list_file = []):
    loaded_data = load_data(**dict_load_data)
    train_loader,val_loader,test_loader = get_loaders(loaded_data = loaded_data, **dict_loaders)
    
    model = AdvancedRNN(**model_params)
    if "/kaggle/input/best_mod/pytorch/default/1/best_mod_cfm.pth" in list_file:
        state_dict = torch.load("/kaggle/input/best_mod/pytorch/default/1/best_mod_cfm.pth", map_location='cuda')  # ou 'cpu'
        model.load_state_dict(state_dict)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


    # === Optimiseur === #
    optimizer = Adam(model.parameters(), lr=training_params["lr"], weight_decay=training_params["weight_decay"])

    # === Scheduler === #
    scheduler = ReduceLROnPlateau(optimizer, **training_params["scheduler"])

    # === Entraînement === #
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=training_params["num_epochs"],
        device=training_params["device"],
        patience=training_params["patience"],
        n=training_params["early_stop_n"]
    )

    MODEL_OUT = '/kaggle/working/'
    torch.save(model.state_dict(), os.path.join(MODEL_OUT,'best_mod_cfm.pth'))
    Y = make_predictions(model,test_loader)

    print(Y)
    return model 