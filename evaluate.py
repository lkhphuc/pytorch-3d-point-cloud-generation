import options
import utils
from trainer import Validator

if __name__ == "__main__":

    print("=======================================================")
    print("Evaluate / generate 3D Point Cloud generation model.")
    print("=======================================================")

    cfg = options.get_arguments()
    cfg.batchSize = cfg.inputViewN
    # cfg.chunkSize = 50

    RESULTS_PATH = f"results/{cfg.model}_{cfg.experiment}"
    utils.make_folder(RESULTS_PATH)

    dataloaders = utils.make_data_fixed(cfg)
    test_dataset = dataloaders[1].dataset

    model = utils.build_structure_generator(cfg).to(cfg.device)

    validator = Validator(cfg, test_dataset) 

    hist = validator.eval(model)
    hist.to_csv(f"{RESULTS_PATH}.csv", index=False)
