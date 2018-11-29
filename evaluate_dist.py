import options
import utils
from trainer import Validator

if __name__ == "__main__":

    print("=======================================================")
    print("Evaluate distance of 3D Point Cloud generation model.")
    print("=======================================================")

    cfg = options.get_arguments()
    cfg.batchSize = cfg.inputViewN
    # cfg.chunkSize = 50

    EXPERIMENT = f"{cfg.model}_{cfg.experiment}"
    RESULTS_PATH = f"results/{EXPERIMENT}"
    utils.make_folder(RESULTS_PATH)

    dataloaders = utils.make_data_fixed(cfg)
    test_dataset = dataloaders[1].dataset

    model = utils.build_structure_generator(cfg).to(cfg.device)

    validator = Validator(cfg, test_dataset) 

    hist = validator.eval_dist(model)
    hist.to_csv(f"{RESULTS_PATH}_testerror.csv", index=False)
